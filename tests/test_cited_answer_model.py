"""结构化问答的来源范围、失败处理和客户端序列化；全部使用本地模拟。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


import httpx
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from paper_sage.qa.evidence import build_evidence_catalog, parse_cited_answer, resolve_claim_sources
from paper_sage.qa.model import CITED_ANSWER_MAX_TOKENS, CitedAnswerError, request_cited_answer
from paper_sage.llm.client import create_chat_model


class CitedAnswerTests(unittest.TestCase):
    def setUp(self):
        text = "We trained our models on one machine with 8 NVIDIA P100 GPUs."
        self.results = [{"rank": 1, "chunk": {
            "chunk_id": "project:paper_a:p7:c1", "project_id": "project", "document_id": "paper_a",
            "source": "papers/a.pdf", "pdf_page": 7, "text": text,
        }}]
        self.answer = {"claims": [{"text": "训练使用了 8 块 NVIDIA P100 GPU。",
                                   "chunk_ids": [self.results[0]["chunk"]["chunk_id"]]}],
                       "insufficient_reason": ""}

    def model(self, payload=None, *, finish_reason="stop", **kwargs):
        model = Mock()
        model.bind.return_value.invoke.return_value = AIMessage(
            content=json.dumps(self.answer if payload is None else payload, ensure_ascii=False),
            response_metadata={"finish_reason": finish_reason}, **kwargs,
        )
        return model

    def test_real_client_serializes_only_current_snippets_and_records_sources(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "local-cited-answer", "object": "chat.completion", "created": 0,
                "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(self.answer)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            })

        before = deepcopy(self.results)
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with patch('paper_sage.llm.client.ChatDeepSeek', side_effect=lambda **kw: ChatDeepSeek(http_client=client, **kw)):
                model = create_chat_model("test-only-key")
            self.assertEqual(model.max_retries, 0)
            record = request_cited_answer("训练使用什么硬件？", self.results, model)
        self.assertEqual(len(requests), 1)
        payload = requests[0]
        self.assertEqual(payload["max_tokens"], CITED_ANSWER_MAX_TOKENS)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", payload)
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user"])
        self.assertEqual(json.loads(payload["messages"][-1]["content"]), {
            "question": "训练使用什么硬件？",
            "reference": [{key: self.results[0]["chunk"][key] for key in ("chunk_id", "text")}],
        })
        self.assertEqual(record["result"], self.answer)
        self.assertEqual(record["claims_with_sources"][0]["sources"][0], self.results[0]["chunk"])
        self.assertEqual(record["usage"]["total_tokens"], 130)
        record["claims_with_sources"][0]["sources"][0]["pdf_page"] = 99
        self.assertEqual(record["evidence_by_id"][self.answer["claims"][0]["chunk_ids"][0]]["pdf_page"], 7)
        self.assertEqual(self.results, before)

    def test_non_provided_missing_duplicate_and_fabricated_sources_rejected(self):
        cases = []
        for ids in (["project:paper_a:p3:c2"], ["other_project:paper_b:p7:c1"], [],
                    self.answer["claims"][0]["chunk_ids"] * 2):
            payload = deepcopy(self.answer)
            payload["claims"][0]["chunk_ids"] = ids
            cases.append(payload)
        fabricated = deepcopy(self.answer)
        fabricated["claims"][0]["pdf_page"] = 99
        cases.append(fabricated)
        for payload in cases:
            with self.subTest(payload=payload):
                model = self.model(payload)
                with self.assertRaises(CitedAnswerError) as raised:
                    request_cited_answer("硬件？", self.results, model)
                record = raised.exception.record
                self.assertEqual(json.loads(record["raw_response"]), payload)
                self.assertIsNone(record["result"])
                self.assertFalse(record["structural_checks_passed"])
                model.bind.return_value.invoke.assert_called_once()

    def test_insufficient_is_explicit_and_cannot_also_claim_answer(self):
        insufficient = {"claims": [], "insufficient_reason": "当前片段未提供耗电量。"}
        record = request_cited_answer("耗电量？", [], self.model(insufficient))
        self.assertEqual(record["evidence_by_id"], {})
        self.assertEqual(record["claims_with_sources"], [])
        contradictory = deepcopy(self.answer)
        contradictory["insufficient_reason"] = "资料不足"
        for payload in ({"claims": [], "insufficient_reason": " "}, contradictory):
            with self.subTest(payload=payload), self.assertRaises(CitedAnswerError):
                request_cited_answer("硬件？", self.results, self.model(payload))
        with self.assertRaises(CitedAnswerError):
            request_cited_answer("硬件？", [], self.model())

    def test_truncated_malformed_and_tool_responses_never_become_answers(self):
        models = [self.model(finish_reason="length"), self.model(finish_reason="content_filter"),
                  self.model(tool_calls=[{"name": "unexpected", "args": {}, "id": "c1", "type": "tool_call"}])]
        malformed = self.model()
        malformed.bind.return_value.invoke.return_value.content = "not json"
        models.append(malformed)
        for model in models:
            with self.subTest(model=model), self.assertRaises(CitedAnswerError) as raised:
                request_cited_answer("硬件？", self.results, model)
            self.assertIsNone(raised.exception.record["result"])
            self.assertFalse(raised.exception.record["structural_checks_passed"])
            model.bind.return_value.invoke.assert_called_once()

    def test_request_failure_keeps_input_without_sdk_error_body_or_retry(self):
        model = self.model()
        model.bind.return_value.invoke.side_effect = TimeoutError("fake-sensitive-sdk-details")
        with self.assertRaises(CitedAnswerError) as raised:
            request_cited_answer("硬件？", self.results, model)
        record = raised.exception.record
        self.assertEqual(record["error_type"], "TimeoutError")
        self.assertEqual(record["final_results"], self.results)
        self.assertNotIn("fake-sensitive-sdk-details", json.dumps(record) + str(raised.exception))
        model.bind.return_value.invoke.assert_called_once()

    def test_catalog_preserves_distinct_sources_and_rejects_bad_inputs_before_call(self):
        other = deepcopy(self.results[0])
        other["chunk"].update(chunk_id="project:paper_b:p2:c1", document_id="paper_b", source="papers/b.pdf", pdf_page=2)
        catalog = build_evidence_catalog(self.results + [other])
        payload = deepcopy(self.answer)
        payload["claims"][0]["chunk_ids"] = list(catalog)
        parsed = parse_cited_answer(json.dumps(payload), catalog)
        sources = resolve_claim_sources(parsed, catalog)[0]["sources"]
        self.assertEqual([s["document_id"] for s in sources], ["paper_a", "paper_b"])
        self.assertEqual(sources[0]["text"], sources[1]["text"])
        bad_page = deepcopy(self.results)
        bad_page[0]["chunk"]["pdf_page"] = True
        for results in (self.results * 2, bad_page):
            model = self.model()
            with self.assertRaises(ValueError):
                request_cited_answer("硬件？", results, model)
            model.bind.assert_not_called()

    def test_valid_id_does_not_automatically_verify_claim_content(self):
        unsupported = deepcopy(self.answer)
        unsupported["claims"][0]["text"] = "训练使用了 64 块 GPU。"
        record = request_cited_answer("硬件？", self.results, self.model(unsupported))
        self.assertTrue(record["structural_checks_passed"])
        self.assertEqual(record["semantic_review"], "pending")


if __name__ == "__main__":
    unittest.main()
