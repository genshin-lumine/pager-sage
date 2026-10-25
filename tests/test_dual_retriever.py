"""双路接口、候选合并与页面范围检查；不调用真实模型。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import dual_retriever
from dual_retriever import merge_candidates, retrieve_dual
from project_store import load_project
from retrieval_scope import load_scope_index
from rag_context import build_reference_text
import test_project_scope as scope_tests


def make_chunk(chunk_id="project:paper:p1:c1", text="NVIDIA P100"):
    return {
        "project_id": "project", "document_id": "paper",
        "chunk_id": chunk_id, "source": "paper.pdf", "pdf_page": 1, "text": text,
    }


def hit(chunk, rank=1, score=0.5):
    return {"rank": rank, "score": score, "chunk": chunk}


class DualRetrieverTests(unittest.TestCase):
    def test_saved_candidates_preserve_both_routes_without_mutation(self):
        report = json.loads((WORKSPACE / "data/evaluations/day22_retrieval_comparison.json").read_text(encoding="utf-8"))
        case = next(item for item in report["results"] if item["case_id"] == "day22_keyword")
        routes = {"dense": case["dense"], "bm25": case["bm25"]}
        before = deepcopy(routes)
        candidates = merge_candidates(routes)
        self.assertEqual(len(candidates), 4)
        by_id = {item["chunk"]["chunk_id"]: item for item in candidates}
        self.assertEqual(len(by_id), 4)
        for method, results in routes.items():
            for result in results:
                candidate = by_id[result["chunk"]["chunk_id"]]
                self.assertEqual(candidate["chunk"], result["chunk"])
                self.assertEqual(candidate["retrievals"][method], {
                    "rank": result["rank"], "score": result["score"],
                })
        candidates[0]["retrievals"]["dense"]["rank"] = 99
        self.assertEqual(routes, before)

    def test_empty_route_and_same_text_at_different_positions(self):
        first = make_chunk()
        second = make_chunk("project:paper:p2:c1")
        second["pdf_page"] = 2
        results = [hit(first), hit(second, rank=2)]
        self.assertEqual(merge_candidates({"dense": [], "bm25": []}), [])
        for method in ("dense", "bm25"):
            routes = {"dense": [], "bm25": []}
            routes[method] = results
            candidates = merge_candidates(routes)
            self.assertEqual([c["chunk"] for c in candidates], [first, second])
            self.assertTrue(all(set(c["retrievals"]) == {method} for c in candidates))

    def test_duplicate_route_or_conflicting_source_is_rejected(self):
        first = make_chunk()
        with self.assertRaisesRegex(ValueError, "同一路"):
            merge_candidates({"dense": [hit(first), hit(first, rank=2)], "bm25": []})
        changed = {**first, "source": "another.pdf"}
        with self.assertRaisesRegex(ValueError, "不同的内容或来源"):
            merge_candidates({"dense": [hit(first)], "bm25": [hit(changed)]})

    def test_invalid_scope_metadata_is_rejected_before_retrieval(self):
        chunk = make_chunk()
        for chunks, count in [([chunk], 2), ([chunk, deepcopy(chunk)], 2)]:
            with self.subTest(count=count, chunks=len(chunks)):
                with patch.object(dual_retriever, "retrieve_dense") as dense:
                    with patch.object(dual_retriever, "retrieve_bm25") as bm25:
                        with self.assertRaises(ValueError):
                            retrieve_dual("P100", object(), SimpleNamespace(ntotal=count), {"chunks": chunks})
                        dense.assert_not_called()
                        bm25.assert_not_called()

    def test_each_route_must_return_original_scope_records(self):
        original = make_chunk()
        for method in ("dense", "bm25"):
            for field, value in [("project_id", "other_project"), ("source", "other.pdf"), ("chunk_id", "other:paper:p1:c1")]:
                with self.subTest(method=method, field=field):
                    routes = {"dense": [hit(original)], "bm25": [hit(original)]}
                    routes[method] = [hit({**original, field: value})]
                    with patch.object(dual_retriever, "retrieve_dense", return_value=routes["dense"]):
                        with patch.object(dual_retriever, "retrieve_bm25", return_value=routes["bm25"]):
                            with self.assertRaisesRegex(ValueError, "当前检索范围"):
                                retrieve_dual("P100", object(), SimpleNamespace(ntotal=1), {"chunks": [original]})


class DualRetrievalScopeTests(scope_tests.ProjectScopeTests):
    """复用四组范围场景，增加两路归属、合并、页面及参考资料检查。"""

    def setUp(self):
        super().setUp()
        # 英文词项让两路都返回候选；中文无词项另有专门场景。
        self.question = "verification code"

    def ask(self, at, expected_code):
        results = super().ask(at, expected_code)
        answer = at.session_state["chat_session"]["messages"][-1]
        channels = answer["retrieval_channels"]
        candidates = answer["retrieval_candidates"]
        self.assertEqual(set(channels), {"dense", "bm25"})
        self.assertEqual(answer["reference_method"], "rerank")
        self.assertLessEqual(len(results), 3)
        self.assertEqual([r["rank"] for r in results], list(range(1, len(results) + 1)))
        self.assertEqual([r["score"] for r in results], sorted((r["score"] for r in results), reverse=True))
        project_id, documents = at.session_state["chat_session"]["scope_key"]
        selected_ids = [record[0] for record in documents]
        registered = {doc["document_id"]: doc for doc in load_project(project_id)["documents"]}
        by_id = {candidate["chunk"]["chunk_id"]: candidate for candidate in candidates}
        fused_by_id = {item["chunk"]["chunk_id"]: item for item in answer["rrf_results"]}
        self.assertLessEqual(len(fused_by_id), 6)
        self.assertEqual(len(fused_by_id), len(answer["rrf_results"]))
        self.assertGreaterEqual(answer["rerank_elapsed_seconds"], 0)
        self.assertEqual(
            self.rerank_calls[-1],
            [[self.question, item["chunk"]["text"]] for item in answer["rrf_results"]],
        )
        self.assertEqual(len(by_id), len(candidates))
        expected_ids = set()
        for method, route in channels.items():
            scope_tests.assert_results_in_scope(route, project_id, selected_ids)
            for result in route:
                chunk = result["chunk"]
                expected_ids.add(chunk["chunk_id"])
                self.assertEqual(chunk["source"], registered[chunk["document_id"]]["source"])
                candidate = by_id[chunk["chunk_id"]]
                self.assertEqual(candidate["chunk"], chunk)
                self.assertEqual(candidate["retrievals"][method], {
                    "rank": result["rank"], "score": result["score"],
                })
        self.assertEqual(set(by_id), expected_ids)
        for result in results:
            candidate = by_id[result["chunk"]["chunk_id"]]
            self.assertEqual(result["chunk"], candidate["chunk"])
            self.assertEqual(result["retrievals"], candidate["retrievals"])
            fused = fused_by_id[result["chunk"]["chunk_id"]]
            self.assertEqual(result["rrf_rank"], fused["rank"])
            self.assertEqual(result["rrf_score"], fused["score"])
            self.assertEqual(result["score"], result["rerank_score"])
        reference = build_reference_text(results).strip()
        self.assertEqual(
            self.requests[-1][-1]["content"],
            f"<reference>\n{reference}\n</reference>\n\n问题：{self.question}",
        )
        self.assertTrue(all(set(message) == {"role", "content"} for message in self.requests[-1]))
        captions = [caption.value for caption in at.caption]
        self.assertIn(
            f"Dense：{len(channels['dense'])} 条 · BM25：{len(channels['bm25'])} 条 · 去重后：{len(candidates)} 个片段",
            captions,
        )
        self.assertTrue(any("本轮回答使用 Rerank" in value for value in captions))
        self.queries[-1].update({
            "question": self.question,
            "channels": {
                method: [
                    {"chunk_id": r["chunk"]["chunk_id"], "source": r["chunk"]["source"], "rank": r["rank"], "score": r["score"]}
                    for r in route
                ] for method, route in channels.items()
            },
            "unique_candidate_count": len(candidates),
            "reference_method": "rerank",
            "rrf_input_chunk_ids": list(fused_by_id),
            "reference_chunk_ids": [r["chunk"]["chunk_id"] for r in results],
        })
        return results

    def test_empty_bm25_rerun_and_clear(self):
        self.question = "这份测试文档的校验码是什么？"
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        session = deepcopy(at.session_state["chat_session"])
        self.assertEqual(session["messages"][-1]["retrieval_channels"]["bm25"], [])
        self.assertTrue(any("BM25 未找到" in info.value for info in at.info))
        before = (len(self.requests), self.encode_calls, len(self.rerank_calls))
        self.checked(at.run())
        self.assertEqual(at.session_state["chat_session"], session)
        self.assertEqual((len(self.requests), self.encode_calls, len(self.rerank_calls)), before)
        clear_button = next(widget for widget in at.button if widget.label == "清空会话")
        self.checked(clear_button.click().run())
        self.assert_empty_chat(at)
        self.assertEqual((len(self.requests), self.encode_calls, len(self.rerank_calls)), before)
        self.assertEqual(self.rerank_model_instances, 1)
        self.record_case("empty_bm25_rerun_and_clear")

    def test_bm25_only_candidate_is_visible_and_used_as_reference(self):
        self.question = "217"
        at = self.app(self.project_a, [self.aid, self.bid])
        original_dense = dual_retriever.retrieve_dense

        def one_dense_candidate(**kwargs):
            # 控制 Dense 只取一个，让本例明确出现 BM25 独有候选。
            kwargs["top_k"] = 1
            return original_dense(**kwargs)

        with patch.object(dual_retriever, "retrieve_dense", side_effect=one_dense_candidate):
            self.ask(at, self.fixture_b["expected_code"])
        answer = at.session_state["chat_session"]["messages"][-1]
        self.assertEqual(len(answer["retrieval_candidates"]), 2)
        bm25_chunk = answer["retrieval_channels"]["bm25"][0]["chunk"]
        self.assertEqual(bm25_chunk["document_id"], self.aid)
        self.assertIn(bm25_chunk["chunk_id"], self.requests[-1][-1]["content"])
        self.assertTrue(any(bm25_chunk["chunk_id"] in caption.value for caption in at.caption))
        self.record_case("bm25_only_candidate_used_after_rerank")
        self.observations[-1]["controlled_dense_top_k"] = 1

    def test_out_of_scope_result_blocks_model_and_preserves_history(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = deepcopy(at.session_state["chat_session"])
        requests_before = len(self.requests)
        reranks_before = len(self.rerank_calls)
        _, foreign_metadata = load_scope_index(load_project(self.project_b), [self.bid])
        foreign_chunk = foreign_metadata["chunks"][0]
        with patch.object(dual_retriever, "retrieve_bm25", return_value=[hit(foreign_chunk)]):
            next(widget for widget in at.text_input if str(widget.key).startswith("question_")).set_value("verification code")
            next(widget for widget in at.button if widget.label == "提交问题").click().run()
        self.assertFalse(at.exception, [item.message for item in at.exception])
        self.assertTrue(any("当前检索范围" in error.value for error in at.error))
        self.assertEqual(len(self.requests), requests_before)
        self.assertEqual(len(self.rerank_calls), reranks_before)
        self.assertEqual(at.session_state["chat_session"], before)
        self.assertFalse(any(foreign_chunk["chunk_id"] in caption.value for caption in at.caption))
        self.record_case("out_of_scope_result_rejected")
        self.observations[-1]["rejected_attempts"] = 1


if __name__ == "__main__":
    unittest.main(verbosity=2)
