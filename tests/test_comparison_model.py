"""比较结构、消息、来源还原及一次客户端请求；模型响应全部在本地模拟。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


import httpx
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from paper_sage.comparison.context import build_comparison_material
from paper_sage.comparison.inputs import build_comparison_tasks
from paper_sage.comparison.model import (COMPARISON_MAX_TOKENS, ComparisonGenerationError, build_comparison_messages,
                              compare_papers, resolve_comparison_sources)
from paper_sage.comparison.parser import parse_comparison
from paper_sage.llm.client import create_chat_model
from paper_sage.retrieval.tools import EMPTY_REFERENCE_TEXT
from paper_sage.retrieval.context import build_reference_text


class ComparisonModelTests(unittest.TestCase):
    def setUp(self):
        documents = [{"document_id": doc_id, "source": f"{doc_id}.pdf"} for doc_id in ("a", "b")]
        dimensions = ["data", "results"]
        records, self.valid = [], {"comparisons": []}
        for task in build_comparison_tasks("project", documents, dimensions):
            chunk = {key: task[key] for key in ("project_id", "document_id", "source")}
            chunk.update(chunk_id=f"{task['document_id']}:{task['dimension']}", pdf_page=dimensions.index(task["dimension"]) + 1,
                         text=f"Original text for {task['document_id']} / {task['dimension']}.")
            hits = [{"rank": 1, "chunk": chunk}]
            records.append({"task": task, "reference_text": build_reference_text(hits),
                            "retrieval_artifact": {"question": task["question"], "retrieval_results": hits}})
        self.material = build_comparison_material("project", documents, dimensions, records)
        for dimension in dimensions:
            self.valid["comparisons"].append({
                "dimension": dimension,
                "papers": [{"document_id": doc_id, "findings": [{"text": f"Finding {doc_id}.", "chunk_ids": [f"{doc_id}:{dimension}"]}],
                            "missing_information": ""} for doc_id in ("a", "b")],
                "comparison": {"text": "Descriptive comparison.", "chunk_ids": [f"a:{dimension}", f"b:{dimension}"]},
                "comparison_limit": "Test-only fixture; no factual quality evaluation.",
            })

    def test_messages_include_all_grouped_text_but_no_generated_sources(self):
        before = deepcopy(self.material)
        messages = build_comparison_messages(self.material)
        self.assertEqual([message.type for message in messages], ["system", "human"])
        payload = json.loads(messages[-1].content)
        for group, original in zip(payload["groups"], self.material["groups"], strict=True):
            self.assertEqual(group["dimension"], original["dimension"])
            for paper, input_paper in zip(group["papers"], original["papers"], strict=True):
                self.assertEqual(paper["chunks"], [{key: result["chunk"][key] for key in ("chunk_id", "text")}
                                                 for result in input_paper["retrieval_results"]])
                self.assertNotIn("source", paper)
                self.assertNotIn("pdf_page", paper["chunks"][0])
        self.assertEqual(self.material, before)

    def test_partial_information_and_fully_missing_side(self):
        partial = deepcopy(self.valid)
        partial["comparisons"][0]["papers"][1]["missing_information"] = "Final sample count is absent."
        self.assertEqual(parse_comparison(json.dumps(partial), self.material), partial)
        material, missing = deepcopy(self.material), deepcopy(partial)
        material["groups"][0]["papers"][1].update(retrieval_results=[], reference_text=EMPTY_REFERENCE_TEXT)
        missing["comparisons"][0]["papers"][1]["findings"] = []
        missing["comparisons"][0]["comparison"] = None
        self.assertEqual(parse_comparison(json.dumps(missing), material), missing)
        missing["comparisons"][0]["papers"][1]["missing_information"] = ""
        with self.assertRaisesRegex(ValueError, "缺失信息"):
            parse_comparison(json.dumps(missing), material)

    def test_rejects_cross_scope_and_one_sided_comparison(self):
        for chunk_id in ("b:data", "a:results", "unknown"):
            invalid = deepcopy(self.valid)
            invalid["comparisons"][0]["papers"][0]["findings"][0]["chunk_ids"] = [chunk_id]
            with self.subTest(chunk_id=chunk_id), self.assertRaisesRegex(ValueError, "允许范围"):
                parse_comparison(json.dumps(invalid), self.material)
        invalid = deepcopy(self.valid)
        invalid["comparisons"][0]["comparison"]["chunk_ids"] = ["a:data"]
        with self.assertRaisesRegex(ValueError, "引用双方"):
            parse_comparison(json.dumps(invalid), self.material)

    def test_incomplete_invalid_or_tool_responses_preserve_raw_without_retry(self):
        valid_raw = json.dumps(self.valid)
        responses = [
            AIMessage(content=valid_raw, response_metadata={"finish_reason": "length"}),
            AIMessage(content="{broken", response_metadata={"finish_reason": "stop"}),
            AIMessage(content=valid_raw + valid_raw, response_metadata={"finish_reason": "stop"}),
            AIMessage(content=valid_raw, response_metadata={"finish_reason": "stop"},
                      tool_calls=[{"name": "unexpected", "args": {}, "id": "call1", "type": "tool_call"}]),
        ]
        for response in responses:
            model = Mock()
            model.bind.return_value.invoke.return_value = response
            with self.subTest(response=response), self.assertRaises(ComparisonGenerationError) as raised:
                compare_papers(self.material, model)
            self.assertEqual(raised.exception.record["raw_response"], response.content)
            self.assertIsNone(raised.exception.record["comparison"])
            self.assertIsNone(raised.exception.record["resolved_comparison"])
            model.bind.return_value.invoke.assert_called_once()

    def test_repeated_trailing_closers_recover_without_another_model_call(self):
        raw = json.dumps(self.valid) + "]}"
        model = Mock()
        model.bind.return_value.invoke.return_value = AIMessage(content=raw, response_metadata={"finish_reason": "stop"})
        record = compare_papers(self.material, model)
        self.assertEqual(record["raw_response"], raw)
        self.assertEqual(record["comparison"], self.valid)
        self.assertEqual(record["format_repair"]["removed_suffix"], "]}")
        self.assertEqual(json.loads(record["format_repair"]["parsed_response"]), self.valid)
        model.bind.return_value.invoke.assert_called_once()

    def test_format_recovery_does_not_accept_invalid_citations_or_extra_content(self):
        invalid = deepcopy(self.valid)
        invalid["comparisons"][0]["papers"][0]["findings"][0]["chunk_ids"] = ["unknown"]
        for raw in (json.dumps(invalid) + "]}", json.dumps(self.valid) + " explanation",
                    json.dumps(self.valid) + json.dumps(self.valid)):
            with self.subTest(raw=raw):
                model = Mock()
                model.bind.return_value.invoke.return_value = AIMessage(content=raw, response_metadata={"finish_reason": "stop"})
                with self.assertRaises(ComparisonGenerationError):
                    compare_papers(self.material, model)
                model.bind.return_value.invoke.assert_called_once()

    def test_resolved_sources_are_copied_and_foreign_material_is_rejected(self):
        before = deepcopy(self.material)
        resolved = resolve_comparison_sources(self.valid, self.material)
        for row in resolved["comparisons"]:
            self.assertEqual({source["document_id"] for source in row["comparison"]["sources"]}, {"a", "b"})
        resolved["comparisons"][0]["papers"][0]["findings"][0]["sources"][0]["text"] = "caller edit"
        self.assertEqual(self.material, before)
        invalid = deepcopy(self.material)
        invalid["groups"][0]["papers"][0]["retrieval_results"][0]["chunk"]["project_id"] = "foreign"
        model = Mock()
        with self.assertRaisesRegex(ValueError, "混入"):
            compare_papers(invalid, model)
        model.bind.assert_not_called()

    def test_actual_client_serialization_uses_one_json_request_and_resolves_sources(self):
        requests, raw = [], json.dumps(self.valid)

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "local-comparison", "object": "chat.completion", "created": 0, "model": "deepseek-flash",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": raw}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with patch('paper_sage.llm.client.ChatDeepSeek', side_effect=lambda **kwargs: ChatDeepSeek(http_client=client, **kwargs)):
                model = create_chat_model("test-only-key")
            self.assertEqual(model.max_retries, 0)
            record = compare_papers(self.material, model)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        self.assertEqual(requests[0]["max_tokens"], COMPARISON_MAX_TOKENS)
        self.assertNotIn("tools", requests[0])
        self.assertEqual(record["comparison"], self.valid)
        self.assertEqual(record["raw_response"], raw)
        self.assertEqual(record["usage"]["total_tokens"], 140)
        self.assertEqual(record["resolved_comparison"]["comparisons"][0]["papers"][1]["findings"][0]["sources"][0]["source"], "b.pdf")


if __name__ == "__main__":
    unittest.main()
