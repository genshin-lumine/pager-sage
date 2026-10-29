"""RRF 公式、真实候选回放与页面参考资料检查；模型调用均被替代。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

WORKSPACE = Path(__file__).resolve().parents[1]

import faiss
import numpy as np

from paper_sage.retrieval.dual import merge_candidates
from paper_sage.retrieval.index_store import save_index
from paper_sage.projects.store import get_index_dir, save_project
from paper_sage.retrieval.fusion import calculate_rrf_score, fuse_rrf
import test_dual_retriever as dual_tests


def candidate(label, ranks):
    return {
        "chunk": {
            "project_id": "project", "document_id": "paper", "chunk_id": label,
            "source": "paper.pdf", "pdf_page": 1, "text": f"Text {label}",
        },
        "retrievals": {
            method: {"rank": rank, "score": 100.0 if method == "bm25" else 0.9}
            for method, rank in ranks.items()
        },
    }


def saved_keyword_case():
    report = json.loads((WORKSPACE / "data/evaluations/day22_retrieval_comparison.json").read_text(encoding="utf-8"))
    return next(item for item in report["results"] if item["case_id"] == "day22_keyword")


class RankFusionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            candidate("A", {"dense": 1}),
            candidate("B", {"dense": 2, "bm25": 1}),
            candidate("C", {"bm25": 2}),
        ]

    def test_hand_calculations_for_two_constants(self):
        for constant, expected in [(1, [1 / 2, 5 / 6, 1 / 3]), (2, [1 / 3, 7 / 12, 1 / 4])]:
            with self.subTest(constant=constant):
                for item, score in zip(self.candidates, expected):
                    self.assertAlmostEqual(calculate_rrf_score(item["retrievals"], constant), score)
                results = fuse_rrf(self.candidates, rank_constant=constant)
                self.assertEqual([r["chunk"]["chunk_id"] for r in results], ["B", "A", "C"])

    def test_saved_candidates_promote_bm25_only_hardware_chunk(self):
        case = saved_keyword_case()
        pool = merge_candidates({"dense": case["dense"], "bm25": case["bm25"]})
        results = fuse_rrf(pool)
        self.assertEqual([r["chunk"]["chunk_id"].rsplit(":", 2)[-2:] for r in results], [
            ["p7", "c4"], ["p2", "c3"], ["p7", "c3"],
        ])
        self.assertAlmostEqual(results[0]["score"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(results[1]["score"], 1 / 62 + 1 / 63)
        self.assertAlmostEqual(results[2]["score"], 1 / 61)
        self.assertEqual(set(results[2]["retrievals"]), {"bm25"})
        by_id = {item["chunk"]["chunk_id"]: item for item in pool}
        for result in results:
            original = by_id[result["chunk"]["chunk_id"]]
            self.assertEqual(result["chunk"], original["chunk"])
            self.assertEqual(result["retrievals"], original["retrievals"])

    def test_original_scores_do_not_change_fusion(self):
        before = fuse_rrf(self.candidates)
        for item in self.candidates:
            for retrieval in item["retrievals"].values():
                retrieval["score"] = -123456.0
        after = fuse_rrf(self.candidates)
        self.assertEqual(
            [(r["chunk"]["chunk_id"], r["score"]) for r in before],
            [(r["chunk"]["chunk_id"], r["score"]) for r in after],
        )

    def test_one_route_and_empty_candidates(self):
        self.assertEqual(fuse_rrf([]), [])
        self.assertEqual(calculate_rrf_score({}), 0.0)
        for method in ("dense", "bm25"):
            pool = [candidate("late", {method: 2}), candidate("early", {method: 1})]
            self.assertEqual([r["chunk"]["chunk_id"] for r in fuse_rrf(pool)], ["early", "late"])

    def test_ties_are_deterministic_and_top_k_is_applied_after_sort(self):
        pool = [candidate("Z", {"dense": 1}), candidate("A", {"bm25": 1})]
        self.assertEqual(fuse_rrf(pool), fuse_rrf(list(reversed(pool))))
        self.assertEqual(fuse_rrf(pool, top_k=1)[0]["chunk"]["chunk_id"], "A")
        results = fuse_rrf(self.candidates, top_k=2)
        self.assertEqual([r["chunk"]["chunk_id"] for r in results], ["B", "A"])
        self.assertEqual([r["rank"] for r in results], [1, 2])

    def test_output_changes_do_not_mutate_original_candidates(self):
        before = deepcopy(self.candidates)
        results = fuse_rrf(self.candidates)
        results[0]["chunk"]["source"] = "changed.pdf"
        results[0]["retrievals"]["dense"]["rank"] = 99
        self.assertEqual(self.candidates, before)

    def test_invalid_constants_limits_and_ranks_are_rejected(self):
        for value in [0, -1, True, 1.5, "2"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    calculate_rrf_score({}, rank_constant=value)
                with self.assertRaises(ValueError):
                    fuse_rrf([], rank_constant=value)
                with self.assertRaises(ValueError):
                    fuse_rrf([], top_k=value)
                with self.assertRaises(ValueError):
                    calculate_rrf_score({"dense": {"rank": value}})
        for retrievals in [[], {"dense": {}}, {"dense": None}]:
            with self.subTest(retrievals=retrievals), self.assertRaises(ValueError):
                calculate_rrf_score(retrievals)

    def test_duplicate_or_unranked_candidates_are_rejected(self):
        for pool in [
            [self.candidates[0], deepcopy(self.candidates[0])],
            [candidate("A", {})],
            [candidate("", {"dense": 1})],
        ]:
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                fuse_rrf(pool)


class RRFPageTests(dual_tests.DualRetrievalScopeTests):
    """继承七组范围与页面检查，再验证最终截取与旧历史兼容。"""

    def seed_full_paper(self):
        case = saved_keyword_case()
        pool = merge_candidates({"dense": case["dense"], "bm25": case["bm25"]})
        document_id = "saved_paper"
        source = Path("data/papers") / self.project_a / document_id / "paper.pdf"
        source.parent.mkdir(parents=True)
        source.write_bytes((WORKSPACE / pool[0]["chunk"]["source"]).read_bytes())
        # 扩大到六条后，仍让 p7:c3 成为 BM25 独有候选，合并后超过 RRF 上限。
        similarities = {
            "p7:c4": 0.9, "p2:c3": 0.8, "p10:c3": 0.7,
            "p2:c2": 0.6, "p2:c1": 0.5, "p1:c4": 0.4, "p7:c3": -0.1,
        }
        # BM25 的统计语料也保持为完整论文，不能只用四条候选重建。
        chunks = dual_tests.scope_tests.split_pages(
            dual_tests.scope_tests.load_pdf_pages(source.as_posix(), document_id), self.project_a
        )
        self.assertEqual(len(chunks), 51)
        vectors = []
        for chunk in chunks:
            suffix = ":".join(chunk["chunk_id"].split(":")[-2:])
            similarity = similarities.get(suffix, 0.0)
            vector = np.zeros(384, dtype="float32")
            vector[0], vector[1] = np.sqrt(1 - similarity ** 2), similarity
            vectors.append(vector)
        index = faiss.IndexFlatIP(384)
        index.add(np.asarray(vectors, dtype="float32"))
        save_index(index, chunks, get_index_dir(self.project_a, document_id, "v1"), {
            "model_name": "intfloat/multilingual-e5-small", "normalize_embeddings": True,
            "passage_prefix": "passage: ", "query_prefix": "query: ",
        })
        save_project({"project_id": self.project_a, "name": "RRF test", "documents": [
            {"document_id": document_id, "source": source.as_posix(), "index_version": "v1"},
        ]})
        quote = "We trained our models on one machine with 8 NVIDIA P100 GPUs."
        return document_id, quote

    def test_fused_sources_survive_identity_rerank_and_reference_limit(self):
        document_id, quote = self.seed_full_paper()

        def simulated_hardware_answer(messages, api_key):
            self.assertEqual(api_key, "test-only-key")
            self.requests.append(deepcopy(messages))
            self.assertIn(quote, messages[-1]["content"])
            return {"answer": quote, "evidence": [quote], "insufficient_reason": ""}

        self.question = "NVIDIA P100"
        with patch.object(sys.modules['paper_sage.llm.client'], "answer_response", side_effect=simulated_hardware_answer):
            at = self.app(self.project_a, [document_id])
            self.ask(at, "8 NVIDIA P100 GPUs")
        answer = at.session_state["chat_session"]["messages"][-1]
        self.assertEqual(len(answer["retrieval_channels"]["dense"]), 6)
        self.assertEqual(len(answer["retrieval_channels"]["bm25"]), 6)
        self.assertEqual(len(answer["retrieval_candidates"]), 9)
        self.assertEqual(len(answer["retrieval_results"]), 3)
        suffixes = [":".join(r["chunk"]["chunk_id"].split(":")[-2:]) for r in answer["retrieval_results"]]
        self.assertEqual(suffixes, ["p7:c4", "p2:c3", "p2:c2"])
        rejected_id = f"{self.project_a}:{document_id}:s800:o100:p10:c3"
        self.assertNotIn(rejected_id, self.requests[-1][-1]["content"])
        self.assertFalse(any(rejected_id in caption.value for caption in at.caption))
        self.assertTrue(any("RRF 分数：0.032522" == caption.value for caption in at.caption))
        self.assertEqual(answer["fusion_config"], {"rank_constant": 60, "recall_top_k": 6, "rrf_top_k": 6, "final_top_k": 3})
        self.assertEqual(len(answer["rrf_results"]), 6)
        self.record_case("fused_sources_survive_identity_rerank")

    def test_legacy_dense_history_keeps_label_and_new_answer_uses_rerank(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        session = deepcopy(at.session_state["chat_session"])
        old_answer = session["messages"][-1]
        old_answer["retrieval_results"] = deepcopy(old_answer["retrieval_channels"]["dense"])
        old_answer.pop("reference_method")
        old_answer.pop("fusion_config")
        for key in ("rrf_results", "rerank_config", "rerank_elapsed_seconds", "retrieval_trace"):
            old_answer.pop(key)
        at.session_state["chat_session"] = session
        before_calls = (len(self.requests), self.encode_calls, len(self.rerank_calls))
        self.checked(at.run())
        self.assertEqual((len(self.requests), self.encode_calls, len(self.rerank_calls)), before_calls)
        self.assertTrue(any("本轮回答使用 Dense" in caption.value for caption in at.caption))
        self.assertFalse(any("RRF 分数" in caption.value for caption in at.caption))

        self.clear_legacy_session_before_new_query(at, session)
        self.ask(at, self.fixture_a["expected_code"])
        self.assertEqual([item["role"] for item in self.requests[-1]],
                         ["system", "user"])
        self.assertTrue(all(set(message) == {"role", "content"} for message in self.requests[-1]))
        messages = at.session_state["chat_session"]["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["reference_method"], "rerank")
        captions = [caption.value for caption in at.caption]
        self.assertFalse(any("本轮回答使用 Dense" in value for value in captions))
        self.assertTrue(any("本轮回答使用 Rerank" in value for value in captions))
        self.record_case("legacy_dense_history_and_new_rerank_answer")
        self.observations[-1]["legacy_history_was_injected_for_compatibility_check"] = True


if __name__ == "__main__":
    unittest.main(verbosity=2)
