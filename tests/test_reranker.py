"""精排的来源保留、排序及失败拦截检查；不加载真实模型。"""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from reranker import rerank_candidates, get_rerank_cache_folder
import reranker
import test_rank_fusion as rrf_tests


def fused_candidates():
    return [
        {
            "rank": rank,
            "score": 1 / (60 + rank),
            "chunk": {
                "project_id": "project", "document_id": "paper",
                "chunk_id": label, "source": "paper.pdf", "pdf_page": rank,
                "text": f"Text {label}",
            },
            "retrievals": {"dense": {"rank": rank, "score": 0.9 - rank / 100}},
        }
        for rank, label in enumerate(["A", "B", "C"], start=1)
    ]


class RerankerTests(unittest.TestCase):
    def test_local_cache_settings_are_optional_and_paths_are_validated(self):
        with tempfile.TemporaryDirectory(prefix="paper-sage-rerank-config-") as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
            self.assertTrue(root.name.startswith("paper-sage-rerank-config-"))
            config = root / "config.local.toml"
            self.assertIsNone(get_rerank_cache_folder(config))
            config.write_text('[rerank]\ncache_folder = ""\n', encoding="utf-8")
            self.assertIsNone(get_rerank_cache_folder(config))
            config.write_text('[rerank]\ncache_folder = "model-cache"\n', encoding="utf-8")
            self.assertEqual(get_rerank_cache_folder(config), str(root / "model-cache"))
            config.write_text('[rerank]\ncache_folder = 123\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                get_rerank_cache_folder(config)

    def test_scores_reorder_full_records_before_limit_and_preserve_originals(self):
        candidates = fused_candidates()
        before = deepcopy(candidates)
        model = Mock()
        model.predict.return_value = np.array([-10.0, -1.0, -2.0])
        results = rerank_candidates("question", candidates, model, top_k=2)
        self.assertEqual([item["chunk"]["chunk_id"] for item in results], ["B", "C"])
        self.assertEqual([item["rank"] for item in results], [1, 2])
        self.assertEqual([item["rrf_rank"] for item in results], [2, 3])
        model.predict.assert_called_once_with(
            [["question", "Text A"], ["question", "Text B"], ["question", "Text C"]],
            batch_size=3, show_progress_bar=False, convert_to_numpy=True,
        )
        for result, original in zip(results, before[1:], strict=True):
            self.assertEqual(result["chunk"], original["chunk"])
            self.assertEqual(result["retrievals"], original["retrievals"])
            self.assertEqual(result["rrf_score"], original["score"])
            self.assertEqual(result["score"], result["rerank_score"])
        results[0]["chunk"]["source"] = "changed.pdf"
        results[0]["retrievals"]["dense"]["rank"] = 99
        self.assertEqual(candidates, before)

    def test_ties_keep_rrf_order_and_limit_can_exceed_candidate_count(self):
        model = Mock()
        model.predict.return_value = np.array([-2.0, -2.0, -2.0])
        results = rerank_candidates("question", fused_candidates(), model, top_k=10)
        self.assertEqual([item["chunk"]["chunk_id"] for item in results], ["A", "B", "C"])

    def test_empty_candidates_skip_model(self):
        model = Mock()
        self.assertEqual(rerank_candidates("question", [], model), [])
        model.predict.assert_not_called()

    def test_malformed_or_nonfinite_scores_are_rejected_without_mutation(self):
        bad_scores = [
            [1, 2], [1, 2, 3, 4], [[1], [2], [3]], 1.0,
            [1, float("nan"), 3], [1, float("inf"), 3],
            ["1", "2", "3"], [True, False, True],
        ]
        for scores in bad_scores:
            with self.subTest(scores=scores):
                candidates = fused_candidates()
                before = deepcopy(candidates)
                model = Mock()
                model.predict.return_value = scores
                with self.assertRaises(ValueError):
                    rerank_candidates("question", candidates, model)
                self.assertEqual(candidates, before)

    def test_invalid_inputs_are_rejected_before_model_call(self):
        model = Mock()
        for question in [None, "", "  "]:
            with self.subTest(question=question), self.assertRaises(ValueError):
                rerank_candidates(question, fused_candidates(), model)
        for top_k in [0, -1, True, 1.5]:
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                rerank_candidates("question", fused_candidates(), model, top_k=top_k)
        for field, value in [("chunk_id", ""), ("chunk_id", "B"), ("text", " ")]:
            candidates = fused_candidates()
            candidates[0]["chunk"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                rerank_candidates("question", candidates, model)
        for field, value in [("rank", True), ("rank", 0), ("score", float("nan")), ("score", True)]:
            candidates = fused_candidates()
            candidates[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                rerank_candidates("question", candidates, model)
        model.predict.assert_not_called()

    def test_prediction_failure_propagates_without_mutating_candidates(self):
        candidates = fused_candidates()
        before = deepcopy(candidates)
        model = Mock()
        model.predict.side_effect = TimeoutError("simulated scoring failure")
        with self.assertRaises(TimeoutError):
            rerank_candidates("question", candidates, model)
        self.assertEqual(candidates, before)


class RerankPageTests(rrf_tests.RRFPageTests):
    """继承九组范围与历史场景，增加精排改序、失败及旧 RRF 历史检查。"""

    def test_rerank_can_promote_sixth_rrf_candidate_into_reference(self):
        document_id, quote = self.seed_full_paper()
        self.question = "NVIDIA P100"
        def promote_last(pairs):
            scores = -np.arange(1, len(pairs) + 1, dtype="float32")
            scores[-1] = 0.0
            return scores

        self.rerank_score_fn = promote_last

        def simulated_answer(messages, api_key):
            self.assertEqual(api_key, "test-only-key")
            self.requests.append(deepcopy(messages))
            self.assertIn(quote, messages[-1]["content"])
            return {"answer": quote, "evidence": [quote], "insufficient_reason": ""}

        with patch.object(sys.modules["llm"], "ask_deepseek", side_effect=simulated_answer):
            at = self.app(self.project_a, [document_id])
            results = self.ask(at, "8 NVIDIA P100 GPUs")
        answer = at.session_state["chat_session"]["messages"][-1]
        self.assertEqual(len(self.rerank_calls[-1]), 6)
        self.assertEqual([r["rrf_rank"] for r in results], [6, 1, 2])
        self.assertEqual([r["rank"] for r in results], [1, 2, 3])
        excluded_id = answer["rrf_results"][2]["chunk"]["chunk_id"]
        promoted_id = answer["rrf_results"][5]["chunk"]["chunk_id"]
        self.assertNotIn(excluded_id, self.requests[-1][-1]["content"])
        self.assertIn(promoted_id, self.requests[-1][-1]["content"])
        self.assertFalse(any(excluded_id in caption.value for caption in at.caption))
        self.assertTrue(any("精排排名 1（原 RRF 排名 6）" in item.value for item in at.markdown))
        self.record_case("controlled_rerank_promotes_sixth_candidate")
        self.observations[-1]["final_original_rrf_ranks"] = [r["rrf_rank"] for r in results]

    def test_rerank_failures_preserve_history_and_skip_answer_model(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = deepcopy(at.session_state["chat_session"])
        before_requests = len(self.requests)
        for kind in ("runtime_failure", "nan_score"):
            with self.subTest(kind=kind):
                self.rerank_failure = RuntimeError("simulated scoring failure") if kind == "runtime_failure" else None
                self.rerank_score_fn = (lambda pairs: np.full(len(pairs), np.nan)) if kind == "nan_score" else None
                next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(self.question)
                next(w for w in at.button if w.label == "提交问题").click().run()
                self.assertFalse(at.exception, [item.message for item in at.exception])
                self.assertTrue(any("论文精排失败" in item.value for item in at.error))
                self.assertEqual(at.session_state["chat_session"], before)
                self.assertEqual(len(self.requests), before_requests)
        self.record_case("rerank_failure_and_invalid_score_preserve_history")
        self.observations[-1]["rejected_attempts"] = 2

    def test_missing_rerank_cache_skips_answer_and_history_save(self):
        at = self.app(self.project_a, [self.aid])
        before = deepcopy(at.session_state["chat_session"])
        with patch.object(reranker, "load_reranker", side_effect=OSError("missing test cache")):
            next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(self.question)
            next(w for w in at.button if w.label == "提交问题").click().run()
        self.assertFalse(at.exception, [item.message for item in at.exception])
        self.assertTrue(any("精排资源加载失败" in item.value for item in at.error))
        self.assertEqual(at.session_state["chat_session"], before)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.rerank_calls, [])
        self.record_case("missing_rerank_cache")

    def test_no_candidates_skip_rerank_loading_and_answer(self):
        at = self.app(self.project_a, [self.aid])
        before = deepcopy(at.session_state["chat_session"])
        with patch("dual_retriever.retrieve_dense", return_value=[]), patch("dual_retriever.retrieve_bm25", return_value=[]):
            next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(self.question)
            next(w for w in at.button if w.label == "提交问题").click().run()
        self.assertFalse(at.exception, [item.message for item in at.exception])
        self.assertTrue(any("没有可用的论文片段" in item.value for item in at.warning))
        self.assertEqual(at.session_state["chat_session"], before)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.rerank_calls, [])
        self.assertEqual(self.rerank_model_instances, 0)
        self.record_case("empty_retrieval_skips_rerank")

    def test_legacy_rrf_history_keeps_label_and_new_answer_uses_rerank(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        session = deepcopy(at.session_state["chat_session"])
        old_answer = session["messages"][-1]
        old_answer["reference_method"] = "rrf"
        old_answer["fusion_config"]["recall_top_k"] = 3
        old_answer["retrieval_results"] = deepcopy(old_answer["rrf_results"][:3])
        old_answer["fusion_config"].pop("rrf_top_k")
        for key in ("rrf_results", "rerank_config", "rerank_elapsed_seconds", "retrieval_trace"):
            old_answer.pop(key)
        at.session_state["chat_session"] = session
        before_calls = (len(self.requests), self.encode_calls, len(self.rerank_calls))
        self.checked(at.run())
        self.assertEqual((len(self.requests), self.encode_calls, len(self.rerank_calls)), before_calls)
        self.assertTrue(any("本轮回答使用 RRF" in item.value for item in at.caption))
        self.assertFalse(any("精排分数" in item.value for item in at.caption))
        next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(self.question)
        self.checked(next(w for w in at.button if w.label == "提交问题").click().run())
        self.assertEqual(len(self.requests[-1]), 4)
        self.assertTrue(all(set(item) == {"role", "content"} for item in self.requests[-1]))
        messages = at.session_state["chat_session"]["messages"]
        self.assertEqual(messages[1], old_answer)
        self.assertEqual(messages[3]["reference_method"], "rerank")
        self.assertTrue(any("本轮回答使用 RRF" in item.value for item in at.caption))
        self.assertTrue(any("本轮回答使用 Rerank" in item.value for item in at.caption))
        self.record_case("legacy_rrf_history_and_new_rerank_answer")
        self.observations[-1]["legacy_history_was_injected_for_compatibility_check"] = True


if __name__ == "__main__":
    unittest.main()
