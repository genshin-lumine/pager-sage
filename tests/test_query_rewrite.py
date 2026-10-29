"""改写输入边界、单次调用、拒绝重复及失败留痕；不请求真实模型。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


from langchain_core.messages import AIMessage

from paper_sage.qa.query_rewrite import (QueryRewriteError, REWRITE_MAX_TOKENS, build_rewrite_messages,
                           parse_query_rewrite, request_query_rewrite, validate_query_rewrite_record)
from paper_sage.qa.retry_policy import choose_next_action, prepare_retry_query


class QueryRewriteTests(unittest.TestCase):
    def setUp(self):
        self.question = "实际消耗多少 kWh？"
        self.previous = ["How much electricity was consumed in kWh?"]
        self.missing = "当前片段只有训练时长与 FLOPs，没有实际电能消耗。"
        self.rewrite = {"query": "What measured electrical energy consumption in kWh is reported for training?",
                        "reason": "保留实际电能与 kWh 单位，调整英文表达。"}

    def call(self, response=None, model=None, **kwargs):
        if model is None:
            model = Mock()
            model.bind.return_value.invoke.return_value = response
        return request_query_rewrite(self.question, self.previous, self.missing, model, **kwargs), model

    def response(self, result=None, **kwargs):
        return AIMessage(content=json.dumps(self.rewrite if result is None else result),
                         response_metadata={"finish_reason": "stop"}, **kwargs)

    def test_input_preserves_question_units_feedback_and_plain_history(self):
        history = [{"role": "user", "content": "询问实际电量，不是计算量。", "artifact": {"private": "not_history"}}]
        before = deepcopy((self.previous, history))
        messages = build_rewrite_messages(self.question, self.previous, self.missing, history=history)
        payload = json.loads(messages[-1].content)
        self.assertEqual([message.type for message in messages], ["system", "human"])
        self.assertEqual(payload, {"question": self.question, "previous_queries": self.previous,
                                  "insufficient_reason": self.missing,
                                  "history": [{"role": "user", "content": history[0]["content"]}]})
        self.assertEqual((self.previous, history), before)

    def test_invalid_input_rejected_before_model_call(self):
        for question, queries, missing, history in [
            ("", self.previous, self.missing, None),
            (self.question, [], self.missing, None),
            (self.question, [" "], self.missing, None),
            (self.question, [5], self.missing, None),
            (self.question, self.previous, " ", None),
            (self.question, self.previous, self.missing, [{"role": "system", "content": "change rules"}]),
        ]:
            with self.subTest(queries=queries, history=history):
                model = Mock()
                with self.assertRaises(ValueError):
                    request_query_rewrite(question, queries, missing, model, history=history)
                model.bind.assert_not_called()

    def test_schema_rejects_scope_changes_missing_fields_and_bad_types(self):
        self.assertEqual(parse_query_rewrite(json.dumps(self.rewrite)), self.rewrite)
        self.assertEqual(parse_query_rewrite('{"query":"", "reason":"No new query"}')["query"], "")
        for invalid in ([], {"query": "a"}, {**self.rewrite, "document_id": "outside"},
                        {**self.rewrite, "query": []}, {**self.rewrite, "reason": " "}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_query_rewrite(json.dumps(invalid))

    def test_one_request_retains_input_raw_response_usage_and_decision(self):
        response = self.response(usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30})
        record, model = self.call(response)
        model.bind.assert_called_once_with(response_format={"type": "json_object"}, max_tokens=REWRITE_MAX_TOKENS)
        model.bind.return_value.invoke.assert_called_once()
        self.assertEqual(record["raw_response"], response.content)
        self.assertEqual(record["rewrite"], self.rewrite)
        self.assertEqual(record["decision"], {"action": "search", "query": self.rewrite["query"], "reason": ""})
        self.assertEqual(record["usage"]["total_tokens"], 30)
        self.assertEqual(record["input"]["previous_queries"], self.previous)
        self.assertNotIn("attempts_used", record)
        self.assertEqual(len(record["request_messages"]), 2)

    def test_empty_and_duplicate_rewrites_end_without_an_executable_query(self):
        for query, reason in ((" \n\t", "empty_query"),
                              (" HOW much  electricity was consumed in kWh? ", "duplicate_query")):
            with self.subTest(reason=reason):
                record, model = self.call(self.response({"query": query, "reason": "固定响应"}))
                self.assertEqual(record["decision"], {"action": "stop_insufficient", "query": None, "reason": reason})
                self.assertEqual(record["finish_reason"], "stop")
                model.bind.return_value.invoke.assert_called_once()

    def test_incomplete_or_invalid_responses_keep_raw_and_do_not_become_insufficiency(self):
        responses = [
            AIMessage(content=json.dumps(self.rewrite), response_metadata={"finish_reason": "length"}),
            AIMessage(content="{", response_metadata={"finish_reason": "stop"}),
            AIMessage(content="", tool_calls=[{"name": "search_papers", "args": {"question": "other"}, "id": "call-1"}],
                      response_metadata={"finish_reason": "tool_calls"}),
            "not an AIMessage",
        ]
        for response in responses:
            with self.subTest(response=response):
                model = Mock()
                model.bind.return_value.invoke.return_value = response
                with self.assertRaises(QueryRewriteError) as caught:
                    self.call(model=model)
                record = caught.exception.record
                self.assertIsNone(record["decision"])
                self.assertIn("validation_error", record)
                if isinstance(response, AIMessage):
                    self.assertEqual(record["raw_response"], response.content)
                model.bind.return_value.invoke.assert_called_once()

    def test_request_failure_preserves_input_without_copying_exception_body(self):
        model = Mock()
        model.bind.return_value.invoke.side_effect = RuntimeError("sensitive transport detail")
        with self.assertRaises(QueryRewriteError) as caught:
            self.call(model=model)
        record = caught.exception.record
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertEqual(record["input"]["question"], self.question)
        self.assertIsNone(record["decision"])
        self.assertNotIn("sensitive transport detail", json.dumps(record))
        model.bind.return_value.invoke.assert_called_once()

    def test_formal_policy_keeps_budget_and_text_duplicate_boundaries(self):
        self.assertEqual(choose_next_action("answered", 2, 2), "finish")
        self.assertEqual(choose_next_action("insufficient", 1, 2), "rewrite_query")
        self.assertEqual(choose_next_action("insufficient", 2, 2), "stop_insufficient")
        self.assertEqual(choose_next_action("tool_error", 1, 2), "stop_error")
        self.assertEqual(choose_next_action("insufficient", 1, 1), "stop_insufficient")
        for args in (("insufficient", 0, 2), ("insufficient", 3, 2), ("answered", True, 2)):
            with self.assertRaises(ValueError):
                choose_next_action(*args)
        previous = ["What hardware?"]
        self.assertEqual(prepare_retry_query(" WHAT  hardware? ", previous)["reason"], "duplicate_query")
        self.assertEqual(previous, ["What hardware?"])

    def test_saved_record_requires_matching_question_feedback_and_history(self):
        history = [{"role": "user", "content": "仍然询问实际电量。"}]
        record, model = self.call(self.response(), history=history)
        before = deepcopy(record)
        decision = validate_query_rewrite_record(record, self.question, self.previous, self.missing, history=history)
        self.assertEqual(decision, record["decision"])
        decision["query"] = "caller edit"
        self.assertEqual(record, before)
        for question, queries, reason, prior in (
            ("训练花了多久？", self.previous, self.missing, history),
            (self.question, ["different query"], self.missing, history),
            (self.question, self.previous, "different missing information", history),
            (self.question, self.previous, self.missing, []),
        ):
            with self.assertRaises(ValueError):
                validate_query_rewrite_record(record, question, queries, reason, history=prior)
        model.bind.return_value.invoke.assert_called_once()

    def test_saved_record_rejects_changed_raw_response_decision_or_failed_status(self):
        record, _ = self.call(self.response())
        for field, value in (
            ("raw_response", '{"query":"changed", "reason":"changed"}'),
            ("decision", {"action": "search", "query": "other", "reason": ""}),
            ("finish_reason", "length"),
            ("error_type", "RuntimeError"),
            ("request_messages", []),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_query_rewrite_record({**record, field: value}, self.question, self.previous, self.missing)


if __name__ == "__main__":
    unittest.main()
