"""比较流程的执行顺序、延后加载、范围隔离和失败记录；全部使用模拟响应。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


from langchain_core.messages import AIMessage, ToolMessage

from paper_sage.comparison.plan import PLAN_PROMPT_VERSION, build_comparison_plan_messages
from paper_sage.comparison.runner import ComparisonRunError, run_comparison
from paper_sage.retrieval.tools import EMPTY_REFERENCE_TEXT
from paper_sage.retrieval.context import build_reference_text


class ComparisonRunnerTests(unittest.TestCase):
    def setUp(self):
        self.documents = [{"document_id": doc_id, "source": f"{doc_id}.pdf"} for doc_id in ("a", "b")]
        self.question = "Compare training data and results"
        self.plan = {"dimensions": [
            {"name": "训练数据", "query": "Which training datasets are used?"},
            {"name": "实验结果", "query": "What test scores are reported?"},
        ], "clarification_question": ""}
        self.events = []
        self.model = Mock()
        self.model.bind.return_value.invoke.side_effect = self.respond
        self.loader = Mock(side_effect=self.load_tools)
        self.wrong_source = False

    def saved_plan(self, *, history=None):
        info = [{"document_id": doc["document_id"], "name": doc["source"]} for doc in self.documents]
        messages = build_comparison_plan_messages(self.question, info, history=history)
        return {
            "prompt_version": PLAN_PROMPT_VERSION, "input": json.loads(messages[-1].content),
            "request_messages": [{"role": m.type, "content": m.content} for m in messages],
            "raw_response": json.dumps(self.plan), "plan": deepcopy(self.plan), "finish_reason": "stop",
            "tool_calls": [], "invalid_tool_calls": [],
        }

    def respond(self, messages):
        payload = json.loads(messages[-1].content)
        if "groups" not in payload:
            self.events.append("plan")
            result = self.plan
        else:
            self.events.append("generate")
            rows = []
            for group in payload["groups"]:
                papers, cited = [], []
                for paper in group["papers"]:
                    ids = [chunk["chunk_id"] for chunk in paper["chunks"]]
                    cited.extend(ids)
                    papers.append({"document_id": paper["document_id"],
                                   "findings": [{"text": "Test finding", "chunk_ids": ids}] if ids else [],
                                   "missing_information": "Missing in supplied text" if not ids else ""})
                rows.append({"dimension": group["dimension"], "papers": papers,
                             "comparison": {"text": "Test comparison", "chunk_ids": cited} if all(p["findings"] for p in papers) else None,
                             "comparison_limit": "Simulated test; no factual evaluation"})
            result = {"comparisons": rows}
        return AIMessage(content=json.dumps(result), response_metadata={"finish_reason": "stop"})

    def load_tools(self):
        self.events.append("load_tools")
        return {doc["document_id"]: self.make_tool(doc) for doc in self.documents}

    def make_tool(self, document):
        def invoke(call):
            question = call["args"]["question"]
            position = [item["query"] for item in self.plan["dimensions"]].index(question)
            self.events.append(("search", document["document_id"], position))
            hits = []
            # 保留 B 在实验结果维度上的空候选，不能丢弃整项任务。
            if not (document["document_id"] == "b" and position == 1):
                chunk = {"project_id": "project", **document, "pdf_page": position + 1,
                         "chunk_id": f"project:{document['document_id']}:{position}", "text": "Simulated source text"}
                if self.wrong_source:
                    chunk["document_id"] = "foreign"
                hits = [{"rank": 1, "chunk": chunk}]
            return ToolMessage(content=build_reference_text(hits) or EMPTY_REFERENCE_TEXT,
                               tool_call_id=call["id"], name=call["name"], artifact={
                                   "question": question, "retrieval_results": hits,
                                   "retrieval_trace": {"stages": [{"stage": "simulation", "results": deepcopy(hits)}]},
                               })
        return SimpleNamespace(name="search_papers", response_format="content_and_artifact", invoke=Mock(side_effect=invoke))

    def run_flow(self, **kwargs):
        return run_comparison(self.question, "project", self.documents, self.model, tools_loader=self.loader, **kwargs)

    def test_complete_pipeline_keeps_queries_sources_empty_tasks_and_inputs(self):
        history = [{"role": "assistant", "content": "Which dimensions?"}]
        before = deepcopy((self.documents, history))
        artifact = self.run_flow(history=history)
        self.assertEqual(self.events, ["plan", "load_tools", ("search", "a", 0), ("search", "a", 1),
                                      ("search", "b", 0), ("search", "b", 1), "generate"])
        self.assertEqual(artifact["response_kind"], "compare")
        self.assertEqual(artifact["status"], "completed")
        self.assertFalse(artifact["plan_reused"])
        self.assertEqual(artifact["plan_record"]["input"]["history"], history)
        self.loader.assert_called_once_with()
        self.assertEqual(self.model.bind.return_value.invoke.call_count, 2)
        self.assertEqual([record["task"] for record in artifact["records"]], artifact["tasks"])
        self.assertEqual([task["question"] for task in artifact["tasks"]], [item["query"] for item in self.plan["dimensions"]] * 2)
        self.assertEqual([len(r["retrieval_artifact"]["retrieval_results"]) for r in artifact["records"]], [1, 1, 1, 0])
        resolved = artifact["generation_record"]["resolved_comparison"]["comparisons"]
        self.assertEqual({s["document_id"] for s in resolved[0]["comparison"]["sources"]}, {"a", "b"})
        self.assertIsNone(resolved[1]["comparison"])
        self.assertEqual([s["stage"] for s in artifact["stages"]], ["plan", "tasks", "load_tools", "retrieve", "material", "generate"])
        self.assertEqual((self.documents, history), before)
        artifact["documents"][0]["source"] = "Changed"
        self.assertEqual(self.documents[0]["source"], "a.pdf")

    def test_clarification_never_loads_tools_or_generates_comparison(self):
        self.plan = {"dimensions": [], "clarification_question": "比较哪些方面？"}
        artifact = self.run_flow()
        self.assertEqual(self.events, ["plan"])
        self.loader.assert_not_called()
        self.assertEqual(artifact["response_kind"], "clarify")
        self.assertEqual(artifact["reply"], self.plan["clarification_question"])
        self.assertEqual(artifact["tasks"], [])
        self.assertEqual(artifact["records"], [])
        self.assertIsNone(artifact["generation_record"])
        self.model.bind.return_value.invoke.assert_called_once()

    def test_saved_plan_reuses_only_matching_plan_and_still_runs_search(self):
        record = self.saved_plan()
        artifact = self.run_flow(plan_record=record)
        self.assertTrue(artifact["plan_reused"])
        self.assertEqual(artifact["plan_record"], record)
        self.assertNotIn("plan", self.events)
        self.assertEqual(len(artifact["records"]), 4)
        self.model.bind.return_value.invoke.assert_called_once()
        self.model.reset_mock()
        self.loader.reset_mock()
        self.question = "Different comparison request"
        with self.assertRaises(ComparisonRunError) as caught:
            self.run_flow(plan_record=record)
        self.assertEqual(caught.exception.artifact["failed_stage"], "plan")
        self.model.bind.assert_not_called()
        self.loader.assert_not_called()

    def test_foreign_retrieval_blocks_generation_and_retains_plan(self):
        self.wrong_source = True
        with self.assertRaises(ComparisonRunError) as caught:
            self.run_flow()
        artifact = caught.exception.artifact
        self.assertEqual(artifact["failed_stage"], "retrieve")
        self.assertEqual(artifact["status"], "failed")
        self.assertEqual(len(artifact["tasks"]), 4)
        self.assertIsNotNone(artifact["plan_record"])
        self.assertIsNone(artifact["generation_record"])
        self.assertNotIn("generate", self.events)
        self.model.bind.return_value.invoke.assert_called_once()

    def test_invalid_generation_preserves_response_and_retrieval_without_retry(self):
        self.model.bind.return_value.invoke.side_effect = None
        self.model.bind.return_value.invoke.return_value = AIMessage(content="{", response_metadata={"finish_reason": "stop"})
        with self.assertRaises(ComparisonRunError) as caught:
            self.run_flow(plan_record=self.saved_plan())
        artifact = caught.exception.artifact
        self.assertEqual(artifact["failed_stage"], "generate")
        self.assertEqual(artifact["generation_record"]["raw_response"], "{")
        self.assertIsNone(artifact["generation_record"]["comparison"])
        self.assertEqual(len(artifact["records"]), 4)
        self.assertIsNone(artifact["response_kind"])
        self.model.bind.return_value.invoke.assert_called_once()
        self.loader.assert_called_once_with()

    def test_invalid_scope_loader_and_tool_map_stop_before_search(self):
        with self.assertRaises(ValueError):
            run_comparison(self.question, "project", self.documents[:1], self.model, tools_loader=self.loader)
        with self.assertRaises(ValueError):
            run_comparison(self.question, "project", self.documents, self.model, tools_loader={})
        self.model.bind.assert_not_called()
        self.loader.assert_not_called()
        self.loader.side_effect = None
        self.loader.return_value = {"foreign": object()}
        with self.assertRaises(ComparisonRunError) as caught:
            self.run_flow()
        self.assertEqual(caught.exception.artifact["failed_stage"], "load_tools")
        self.assertEqual(self.events, ["plan"])
        self.model.bind.return_value.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
