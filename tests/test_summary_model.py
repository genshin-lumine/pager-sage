"""单批总结的输入、失败处理和真实客户端序列化；所有请求均在本地模拟。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


import httpx
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from paper_sage.llm.client import create_chat_model
from paper_sage.summarization.inputs import build_summary_batches
from paper_sage.summarization.model import SUMMARY_MAX_TOKENS, BatchSummaryError, summarize_batch
from paper_sage.summarization.parser import SUMMARY_FIELDS


class SummaryModelTests(unittest.TestCase):
    def setUp(self):
        text = "The encoder is composed of a stack of N = 6 identical layers."
        self.batch = build_summary_batches([
            {"document_id": "paper", "source": "sample.pdf", "pdf_page": 1, "text": text},
        ], "project")["batches"][0]
        self.empty = {field: [] for field in SUMMARY_FIELDS}
        self.summary = deepcopy(self.empty)
        self.summary["method"] = [{"text": "编码器有六层。", "part_ids": [self.batch["parts"][0]["part_id"]]}]

    def model(self, content=None, finish_reason="stop", **kwargs):
        model = Mock()
        model.bind.return_value.invoke.return_value = AIMessage(
            content=json.dumps(self.summary) if content is None else content,
            response_metadata={"finish_reason": finish_reason}, **kwargs,
        )
        return model

    def test_actual_client_sends_only_current_batch_with_output_budget(self):
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            return httpx.Response(200, json={
                "id": "local-summary", "object": "chat.completion", "created": 0,
                "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(self.summary)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with patch('paper_sage.llm.client.ChatDeepSeek', side_effect=lambda **kwargs: ChatDeepSeek(http_client=client, **kwargs)):
                model = create_chat_model("test-only-key")
            self.assertEqual(model.max_retries, 0)
            record = summarize_batch(self.batch, model)
        self.assertEqual(len(requests), 1)
        payload = requests[0]
        self.assertEqual(payload["max_tokens"], SUMMARY_MAX_TOKENS)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", payload)
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user"])
        content = json.loads(payload["messages"][-1]["content"])
        self.assertEqual(content, {"batch_id": self.batch["batch_id"], "parts": [
            {key: part[key] for key in ("part_id", "pdf_page", "text")} for part in self.batch["parts"]
        ]})
        self.assertEqual(record["summary"], self.summary)
        self.assertEqual(record["usage"]["total_tokens"], 130)

    def test_incomplete_invalid_and_foreign_results_fail_without_retry(self):
        foreign = deepcopy(self.summary)
        foreign["method"][0]["part_ids"] = ["another-paper:p1"]
        cases = [
            self.model(finish_reason="length"), self.model(content="not-json"),
            self.model(content=json.dumps(foreign)),
            self.model(tool_calls=[{"name": "unexpected", "args": {}, "id": "call1", "type": "tool_call"}]),
        ]
        for model in cases:
            with self.subTest(model=model), self.assertRaises(ValueError):
                summarize_batch(self.batch, model)
            model.bind.return_value.invoke.assert_called_once()

    def test_mixed_documents_and_bad_offsets_fail_before_request(self):
        mixed = deepcopy(self.batch)
        part = deepcopy(mixed["parts"][0])
        part.update(document_id="foreign", part_id="foreign:p1")
        mixed["parts"].append(part)
        mixed["char_count"] *= 2
        wrong_position = deepcopy(self.batch)
        wrong_position["parts"][0]["end_char"] += 1
        for batch in (mixed, wrong_position):
            model = Mock()
            with self.assertRaises(ValueError):
                summarize_batch(batch, model)
            model.bind.assert_not_called()

    def test_empty_categories_are_accepted(self):
        record = summarize_batch(self.batch, self.model(content=json.dumps(self.empty)))
        self.assertEqual(record["summary"], self.empty)

    def test_saved_input_is_independent_and_raw_response_is_preserved(self):
        before = deepcopy(self.batch)
        model = self.model()
        record = summarize_batch(self.batch, model)
        self.assertEqual(json.loads(record["raw_response"]), record["summary"])
        record["input"]["parts"][0]["text"] = "changed"
        self.assertEqual(self.batch, before)
        self.assertGreaterEqual(record["elapsed_seconds"], 0)

    def test_failed_point_shape_preserves_raw_response_and_precise_location(self):
        bad = deepcopy(self.summary)
        bad["method"][0]["content"] = bad["method"][0].pop("text")
        raw = json.dumps(bad, ensure_ascii=False)
        model = self.model(content=raw)
        with self.assertRaises(BatchSummaryError) as raised:
            summarize_batch(self.batch, model)
        error = raised.exception
        self.assertIn("method[0]", str(error))
        self.assertIn("['text']", str(error))
        self.assertIn("['content']", str(error))
        self.assertEqual(error.record["raw_response"], raw)
        self.assertEqual(error.record["input"], self.batch)
        self.assertIsNone(error.record["summary"])
        self.assertEqual(error.record["finish_reason"], "stop")
        self.assertEqual(error.record["validation_error"], str(error))
        json.dumps(error.record, allow_nan=False)
        model.bind.return_value.invoke.assert_called_once()

    def test_extra_fields_and_string_points_are_still_rejected(self):
        extra = deepcopy(self.summary)
        extra["method"][0]["pdf_page"] = 1
        string_point = deepcopy(self.summary)
        string_point["method"] = ["six layers"]
        for invalid in (extra, string_point):
            model = self.model(content=json.dumps(invalid))
            with self.assertRaises(BatchSummaryError) as raised:
                summarize_batch(self.batch, model)
            self.assertIn("method[0]", str(raised.exception))
            self.assertIsNone(raised.exception.record["summary"])
            model.bind.return_value.invoke.assert_called_once()

    def test_placeholder_content_is_rejected_and_raw_response_is_preserved(self):
        for placeholder in ("中文要点", " 中文要点。 ", "用本批次原文支持的方法要点替换此占位文字"):
            invalid = deepcopy(self.summary)
            invalid["method"][0]["text"] = placeholder
            raw = json.dumps(invalid, ensure_ascii=False)
            model = self.model(content=raw)
            with self.subTest(placeholder=placeholder), self.assertRaises(BatchSummaryError) as raised:
                summarize_batch(self.batch, model)
            self.assertIn("method[0].text 是占位文字", str(raised.exception))
            self.assertEqual(raised.exception.record["raw_response"], raw)
            self.assertIsNone(raised.exception.record["summary"])
            model.bind.return_value.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
