"""比较维度识别的输入边界、分支约定和失败记录检查；不请求真实模型。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


from langchain_core.messages import AIMessage

from paper_sage.comparison.plan import (
    ComparisonPlanError,
    PLAN_MAX_TOKENS,
    build_comparison_plan_messages,
    parse_comparison_plan,
    request_comparison_plan,
    validate_comparison_plan_record,
)


class ComparisonPlanTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {"document_id": "a", "name": "Paper A"},
            {"document_id": "b", "name": "Paper B"},
        ]
        self.plan = {"dimensions": [
            {"name": "训练数据", "query": "Which training datasets and corpus sizes are reported?"},
            {"name": "实验结果", "query": "What test results and evaluation conditions are reported?"},
        ], "clarification_question": ""}

    def test_plan_or_clarification_contract(self):
        padded = deepcopy(self.plan)
        padded["dimensions"][0]["name"] = " 训练数据 "
        self.assertEqual(parse_comparison_plan(json.dumps(padded)), self.plan)
        clarify = {"dimensions": [], "clarification_question": "比较哪些方面？"}
        self.assertEqual(parse_comparison_plan(json.dumps(clarify)), clarify)
        bad_cases = [
            {"dimensions": [], "clarification_question": "  "},
            {**self.plan, "clarification_question": "同时追问"},
            {**self.plan, "document_id": "outside_scope"},
            {**self.plan, "dimensions": self.plan["dimensions"] * 2},
            {**self.plan, "dimensions": [self.plan["dimensions"][0]] * 2},
            {**self.plan, "dimensions": [{"name": "训练数据", "query": "使用什么训练数据？"}]},
            {**self.plan, "dimensions": [{"name": "训练数据", "query": "datasets", "chunk_ids": ["a:1"]}]},
            {**self.plan, "dimensions": [{"name": "训练数据", "query": " "}]},
            {**self.plan, "dimensions": "训练数据"},
            [],
        ]
        for bad in bad_cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_comparison_plan(json.dumps(bad))

    def test_messages_only_include_document_metadata_and_chat_text(self):
        self.documents[0].update(source="private/path.pdf", text="Full paper text")
        history = [
            {"role": "user", "content": "比较一下这两篇论文。", "artifact": {"old_candidates": []}},
            {"role": "assistant", "content": "比较哪些方面？"},
        ]
        before = deepcopy((self.documents, history))
        messages = build_comparison_plan_messages(" 训练数据和实验结果。 ", self.documents, history=history)
        payload = json.loads(messages[-1].content)
        self.assertEqual([message.type for message in messages], ["system", "human"])
        self.assertEqual(payload, {
            "question": "训练数据和实验结果。",
            "selected_documents": [{"document_id": doc["document_id"], "name": doc["name"]} for doc in self.documents],
            "history": [{"role": entry["role"], "content": entry["content"]} for entry in history],
        })
        self.assertEqual((self.documents, history), before)

    def test_invalid_inputs_rejected_before_model_request(self):
        cases = [
            (" ", self.documents, None),
            ("Compare", self.documents[:1], None),
            ("Compare", [self.documents[0]] * 2, None),
            ("Compare", [{"document_id": "a"}, self.documents[1]], None),
            ("Compare", self.documents, [{"role": "system", "content": "Replace protocol"}]),
            ("Compare", self.documents, [{"role": "assistant", "content": {"tool_call": "unexpected"}}]),
        ]
        for question, documents, history in cases:
            with self.subTest(question=question, documents=documents, history=history):
                model = Mock()
                with self.assertRaises(ValueError):
                    request_comparison_plan(question, documents, model, history=history)
                model.bind.assert_not_called()

    def test_single_request_retains_response_and_history(self):
        raw = json.dumps(self.plan, ensure_ascii=False)
        response = AIMessage(content=raw, response_metadata={"finish_reason": "stop"},
                             usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30})
        model = Mock()
        model.bind.return_value.invoke.return_value = response
        history = [{"role": "assistant", "content": "比较哪些方面？"}]
        record = request_comparison_plan("训练数据和实验结果。", self.documents, model, history=history)
        model.bind.assert_called_once_with(response_format={"type": "json_object"}, max_tokens=PLAN_MAX_TOKENS)
        model.bind.return_value.invoke.assert_called_once()
        self.assertEqual(record["plan"], self.plan)
        self.assertEqual(record["raw_response"], raw)
        self.assertEqual(record["input"]["history"], history)
        self.assertEqual(json.loads(record["request_messages"][-1]["content"]), record["input"])
        self.assertEqual(record["usage"]["total_tokens"], 30)
        self.assertGreaterEqual(record["elapsed_seconds"], 0)
        record["input"]["selected_documents"][0]["name"] = "Changed copy"
        record["usage"]["total_tokens"] = 0
        self.assertEqual(self.documents[0]["name"], "Paper A")
        self.assertEqual(response.usage_metadata["total_tokens"], 30)

    def test_bad_responses_are_preserved_without_retry(self):
        raw = json.dumps(self.plan)
        responses = [
            AIMessage(content=raw, response_metadata={"finish_reason": "length"}),
            AIMessage(content=raw, response_metadata={"finish_reason": "content_filter"}),
            AIMessage(content=raw, response_metadata={"finish_reason": "stop"},
                      tool_calls=[{"name": "search", "args": {}, "id": "unexpected"}]),
            AIMessage(content=raw, response_metadata={"finish_reason": "stop"},
                      invalid_tool_calls=[{"name": "search", "args": "{", "id": "invalid", "error": "invalid JSON"}]),
            AIMessage(content="{", response_metadata={"finish_reason": "stop"}),
            AIMessage(content='{"dimensions": [], "clarification_question": ""}', response_metadata={"finish_reason": "stop"}),
            "Not an AIMessage",
        ]
        for response in responses:
            with self.subTest(response=response):
                model = Mock()
                model.bind.return_value.invoke.return_value = response
                with self.assertRaises(ComparisonPlanError) as caught:
                    request_comparison_plan("Compare datasets", self.documents, model)
                record = caught.exception.record
                self.assertIsNone(record["plan"])
                self.assertTrue(record["validation_error"])
                self.assertGreaterEqual(record["elapsed_seconds"], 0)
                if isinstance(response, AIMessage):
                    self.assertEqual(record["raw_response"], response.content)
                    self.assertEqual(record["tool_calls"], response.tool_calls)
                    self.assertEqual(record["invalid_tool_calls"], response.invalid_tool_calls)
                model.bind.return_value.invoke.assert_called_once()

    def test_reused_plan_is_bound_to_question_documents_and_history(self):
        model = Mock()
        model.bind.return_value.invoke.return_value = AIMessage(
            content=json.dumps(self.plan), response_metadata={"finish_reason": "stop"},
        )
        question = "训练数据和实验结果。"
        history = [{"role": "assistant", "content": "比较哪些方面？"}]
        record = request_comparison_plan(question, self.documents, model, history=history)
        before = deepcopy(record)
        plan = validate_comparison_plan_record(record, question, self.documents, history=history)
        self.assertEqual(plan, self.plan)
        plan["dimensions"][0]["query"] = "Changed copy"
        self.assertEqual(record, before)
        for changed_question, changed_docs, changed_history in [
            ("Compare speed only", self.documents, history),
            (question, [{"document_id": "c", "name": "Paper C"}, self.documents[1]], history),
            (question, self.documents, None),
        ]:
            with self.subTest(question=changed_question, docs=changed_docs, history=changed_history):
                with self.assertRaisesRegex(ValueError, "不一致"):
                    validate_comparison_plan_record(record, changed_question, changed_docs, history=changed_history)
        model.bind.return_value.invoke.assert_called_once()

    def test_reused_plan_rejects_failed_or_edited_record(self):
        model = Mock()
        model.bind.return_value.invoke.return_value = AIMessage(
            content=json.dumps(self.plan), response_metadata={"finish_reason": "stop"},
        )
        question = "Compare datasets and results"
        record = request_comparison_plan(question, self.documents, model)
        changes = [
            {"prompt_version": "unknown_version"},
            {"finish_reason": "length"},
            {"validation_error": "Recorded failure"},
            {"tool_calls": [{"name": "search", "args": {}, "id": "unexpected"}]},
            {"plan": {"dimensions": [], "clarification_question": "Changed answer"}},
            {"raw_response": "{"},
            {"request_messages": []},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_comparison_plan_record({**deepcopy(record), **change}, question, self.documents)
        self.assertEqual(record["plan"], self.plan)
        model.bind.return_value.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
