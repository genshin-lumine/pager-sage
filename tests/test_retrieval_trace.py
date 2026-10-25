"""检索阶段快照、未入选候选及页面生命周期；使用控制模型。"""

from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import dual_retriever
import retrieval_trace
from retrieval_trace import run_stage, build_candidate_paths
from rank_fusion import fuse_rrf
from reranker import rerank_candidates
import test_dual_retriever as dual_tests
import test_reranker as rerank_tests


class RetrievalTraceTests(unittest.TestCase):
    def test_timing_snapshot_and_failure_do_not_reexecute_operation(self):
        results = [dual_tests.hit(dual_tests.make_chunk())]
        operation = Mock(return_value=results)
        records = []
        with patch.object(retrieval_trace, "perf_counter", side_effect=[10, 10.125]):
            returned = run_stage(records, "dense", 51, operation)
        operation.assert_called_once_with()
        self.assertIs(returned, results)
        self.assertEqual(records[0]["elapsed_seconds"], 0.125)
        snapshot = deepcopy(records)
        results[0]["chunk"]["text"] = "changed"
        self.assertEqual(records, snapshot)
        with self.assertRaises(RuntimeError):
            run_stage(records, "bm25", 51, Mock(side_effect=RuntimeError("failed")))
        self.assertEqual(records, snapshot)
        self.assertIs(run_stage(None, "dense", 51, operation), results)
        self.assertEqual(operation.call_count, 2)

    def test_optional_dual_trace_keeps_interface_and_rejects_foreign_snapshots(self):
        chunk = dual_tests.make_chunk()
        hits = [dual_tests.hit(chunk)]
        args = ("P100", object(), SimpleNamespace(ntotal=1), {"chunks": [chunk]})
        for trace_enabled in (False, True):
            records = [] if trace_enabled else None
            with patch.object(dual_retriever, "retrieve_dense", return_value=hits) as dense:
                with patch.object(dual_retriever, "retrieve_bm25", return_value=[]) as bm25:
                    routes = dual_retriever.retrieve_dual(*args, stage_records=records)
            self.assertEqual(routes, {"dense": hits, "bm25": []})
            dense.assert_called_once()
            bm25.assert_called_once()
            if trace_enabled:
                self.assertEqual([r["stage"] for r in records], ["dense", "bm25"])
                self.assertEqual([r["output_count"] for r in records], [1, 0])
                self.assertTrue(all(r["elapsed_seconds"] >= 0 for r in records))
        records = []
        foreign = dual_tests.hit({**chunk, "source": "foreign.pdf"})
        with patch.object(dual_retriever, "retrieve_dense", return_value=hits):
            with patch.object(dual_retriever, "retrieve_bm25", return_value=[foreign]):
                with self.assertRaises(ValueError):
                    dual_retriever.retrieve_dual(*args, stage_records=records)
        self.assertEqual(records, [])

    def test_id_paths_distinguish_missing_route_rrf_limit_and_final_limit(self):
        a, b, c = [dual_tests.make_chunk(label) for label in "ABC"]
        routes = {
            "dense": [dual_tests.hit(a), dual_tests.hit(b, rank=2)],
            "bm25": [dual_tests.hit(b), dual_tests.hit(c, rank=2)],
        }
        records = []
        merged = run_stage(records, "merge", 4, lambda: dual_retriever.merge_candidates(routes))
        fused = run_stage(records, "rrf", 3, lambda: fuse_rrf(merged, top_k=2))
        model = Mock()
        model.predict.return_value = [-2, -1]
        ranked = run_stage(records, "rerank", 2, lambda: rerank_candidates("P100", fused, model, top_k=2))
        run_stage(records, "select", 2, lambda: ranked[:1])
        rows = build_candidate_paths(records)
        self.assertEqual([r["chunk_id"] for r in rows], ["A", "B", "C"])
        self.assertIsNone(rows[0]["bm25_rank"])
        self.assertEqual((rows[0]["rrf_rank"], rows[0]["rerank_rank"]), (2, 1))
        self.assertEqual([r["final_selected"] for r in rows], [True, False, False])
        self.assertEqual([r["selection_status"] for r in rows], [
            "selected", "excluded_by_final_top_k", "excluded_by_rrf",
        ])
        self.assertIsNone(rows[2]["dense_rank"])
        self.assertIsNone(rows[2]["rerank_score"])
        self.assertEqual(json.loads(json.dumps(rows, allow_nan=False)), rows)
        model.predict.assert_called_once()


class RetrievalTracePageTests(rerank_tests.RerankPageTests):
    """复用十四组页面场景，对每次成功提交追加完整追踪断言。"""

    def ask(self, at, expected_code):
        results = super().ask(at, expected_code)
        answer = at.session_state["chat_session"]["messages"][-1]
        trace = answer["retrieval_trace"]
        self.assertEqual(trace["scope_key"], at.session_state["chat_session"]["scope_key"])
        self.assertEqual(trace["question"], self.question)
        self.assertEqual([r["stage"] for r in trace["stages"]], [
            "dense", "bm25", "merge", "rrf", "rerank", "select",
        ])
        stages = {r["stage"]: r for r in trace["stages"]}
        for stage, expected in {
            "dense": answer["retrieval_channels"]["dense"],
            "bm25": answer["retrieval_channels"]["bm25"],
            "merge": answer["retrieval_candidates"], "rrf": answer["rrf_results"],
            "select": results,
        }.items():
            self.assertEqual(stages[stage]["results"], expected)
        project_id, documents = trace["scope_key"]
        selected_ids = [doc[0] for doc in documents]
        for record in trace["stages"]:
            self.assertEqual(record["output_count"], len(record["results"]))
            self.assertTrue(isfinite(record["elapsed_seconds"]))
            self.assertGreaterEqual(record["elapsed_seconds"], 0)
            dual_tests.scope_tests.assert_results_in_scope(record["results"], project_id, selected_ids)
        self.assertEqual(stages["dense"]["input_count"], stages["bm25"]["input_count"])
        self.assertEqual(stages["merge"]["input_count"], sum(stages[s]["output_count"] for s in ("dense", "bm25")))
        self.assertEqual(stages["rrf"]["input_count"], stages["merge"]["output_count"])
        self.assertEqual(stages["rerank"]["input_count"], stages["rrf"]["output_count"])
        self.assertEqual(stages["rerank"]["output_count"], stages["rrf"]["output_count"])
        self.assertEqual(stages["select"]["input_count"], stages["rerank"]["output_count"])
        self.assertEqual(stages["rerank"]["results"][:3], results)
        self.assertEqual(stages["rerank"]["elapsed_seconds"], answer["rerank_elapsed_seconds"])
        self.assertTrue(all(isfinite(value) and value >= 0 for value in trace["other_elapsed_seconds"].values()))
        self.assertEqual(len(self.rerank_calls), len(self.requests))
        rows = trace["candidate_paths"]
        by_id = {r["chunk"]["chunk_id"]: r for r in stages["rerank"]["results"]}
        final_ids = {r["chunk"]["chunk_id"] for r in results}
        self.assertEqual([r["chunk_id"] for r in rows], [r["chunk"]["chunk_id"] for r in answer["retrieval_candidates"]])
        for row in rows:
            self.assertEqual(row["final_selected"], row["chunk_id"] in final_ids)
            ranked = by_id.get(row["chunk_id"])
            if ranked is None:
                self.assertIsNone(row["rrf_rank"])
                self.assertIsNone(row["rerank_rank"])
                self.assertIsNone(row["rerank_score"])
                self.assertEqual(row["selection_status"], "excluded_by_rrf")
            else:
                self.assertEqual(row["rerank_rank"], ranked["rank"])
                self.assertEqual(row["rerank_score"], ranked["rerank_score"])
            if not row["final_selected"]:
                self.assertNotIn(row["chunk_id"], self.requests[-1][-1]["content"])
        self.assertTrue(any(exp.label == "查看本轮检索流程" for exp in at.expander))
        # 直接解码页面发送的 Arrow 数据，避免 AppTest 的 pandas 反序列化类型冲突。
        table = pa.RecordBatchStreamReader(at.dataframe[-1].proto.arrow_data.data).read_all().to_pylist()
        self.assertEqual([r["片段标识"] for r in table], [r["chunk_id"] for r in rows])
        self.assertEqual(sum(r["最终状态"] == "已入选参考资料" for r in table), len(results))
        self.assertTrue(any("是否支持回答" in info.value for info in at.info))
        json.dumps(trace, allow_nan=False)
        self.queries[-1]["trace"] = deepcopy(trace)
        return results

    def test_legacy_rerank_has_no_fabricated_trace_and_new_query_gets_trace(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        session = deepcopy(at.session_state["chat_session"])
        session["messages"][-1].pop("retrieval_trace")
        at.session_state["chat_session"] = session
        before_calls = (len(self.requests), self.encode_calls, len(self.rerank_calls))
        self.checked(at.run())
        self.assertEqual(at.session_state["chat_session"], session)
        self.assertEqual((len(self.requests), self.encode_calls, len(self.rerank_calls)), before_calls)
        self.assertFalse(any(exp.label == "查看本轮检索流程" for exp in at.expander))
        next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(self.question)
        self.checked(next(w for w in at.button if w.label == "提交问题").click().run())
        messages = at.session_state["chat_session"]["messages"]
        self.assertNotIn("retrieval_trace", messages[1])
        self.assertIn("retrieval_trace", messages[3])
        self.assertEqual(len(self.rerank_calls), 2)
        self.assertTrue(all(set(m) == {"role", "content"} for m in self.requests[-1]))
        self.record_case("legacy_rerank_trace_not_fabricated")

    def test_answer_failure_does_not_save_partial_trace(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = deepcopy(at.session_state["chat_session"])
        error_type = sys.modules["llm"].IncompleteAnswerError
        with patch.object(sys.modules["llm"], "ask_deepseek", side_effect=error_type("simulated incomplete answer")):
            next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(self.question)
            next(w for w in at.button if w.label == "提交问题").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("simulated incomplete answer" in item.value for item in at.error))
        self.assertEqual(at.session_state["chat_session"], before)
        self.assertEqual(len(self.rerank_calls), 2)
        self.record_case("answer_failure_preserves_existing_trace")


if __name__ == "__main__":
    unittest.main(verbosity=2)
