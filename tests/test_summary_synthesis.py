"""全文汇总输入、输出、来源传递与客户端序列化；所有请求均在本地模拟。"""

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
from paper_sage.summarization.parser import SUMMARY_FIELDS
from paper_sage.summarization.synthesis import (PaperSummaryError, SYNTHESIS_MAX_TOKENS, build_synthesis_messages,
                               parse_paper_summary, resolve_summary_sources, summarize_paper)


class SummarySynthesisTests(unittest.TestCase):
    def setUp(self):
        text = "The model uses attention."
        self.material = {
            "project_id": "project", "document_id": "doc", "source": "paper.pdf",
            "parts": [{"project_id": "project", "document_id": "doc", "source": "paper.pdf",
                       "part_id": "part1", "pdf_page": 1, "start_char": 0, "end_char": len(text), "text": text}],
            "notes": [{"note_id": f"note{i}", "batch_id": "batch1", "category": "method",
                       "text": "模型使用注意力。", "part_ids": ["part1"]} for i in (1, 2)],
            "review_corrections": [],
        }
        self.summary = {field: [] for field in SUMMARY_FIELDS}
        self.summary["method"] = [{"text": "模型使用注意力。", "note_ids": ["note1", "note2"]}]

    def test_messages_include_all_notes_but_leave_source_records_in_program(self):
        before = deepcopy(self.material)
        messages = build_synthesis_messages(self.material)
        self.assertEqual([m.type for m in messages], ["system", "human"])
        payload = json.loads(messages[-1].content)
        self.assertEqual(set(payload), {"document_name", "notes"})
        self.assertEqual(payload["notes"], [{key: note[key] for key in ("note_id", "category", "text")}
                                           for note in self.material["notes"]])
        self.assertEqual(self.material, before)

    def test_valid_references_parse_and_absent_limitations_remain_empty(self):
        parsed = parse_paper_summary(json.dumps(self.summary), self.material)
        self.assertEqual(parsed, self.summary)
        self.assertEqual(parsed["limitations"], [])

    def test_fabricated_empty_duplicate_references_and_invented_limitations_fail(self):
        variants = []
        for note_ids in (["unknown"], [], ["note1", "note1"], "note1", [True]):
            bad = deepcopy(self.summary)
            bad["method"][0]["note_ids"] = note_ids
            variants.append(bad)
        wrong_shape = deepcopy(self.summary)
        wrong_shape["method"][0]["part_ids"] = wrong_shape["method"][0].pop("note_ids")
        variants.append(wrong_shape)
        unsupported = deepcopy(self.summary)
        unsupported["limitations"] = [{"text": "虚构的局限。", "note_ids": ["note1"]}]
        variants.append(unsupported)
        placeholder = deepcopy(self.summary)
        placeholder["method"][0]["text"] = "中文要点"
        variants.append(placeholder)
        variants.append({field: [] for field in SUMMARY_FIELDS})
        for invalid in variants:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_paper_summary(json.dumps(invalid), self.material)

    def test_bad_material_fails_before_model_request(self):
        variants = []
        foreign = deepcopy(self.material)
        foreign["parts"][0]["project_id"] = "foreign"
        variants.append(foreign)
        missing_source = deepcopy(self.material)
        missing_source["notes"][0]["part_ids"] = ["foreign:p1"]
        variants.append(missing_source)
        duplicate = deepcopy(self.material)
        duplicate["notes"][1]["note_id"] = "note1"
        variants.append(duplicate)
        empty = deepcopy(self.material)
        empty["notes"] = []
        variants.append(empty)
        for material in variants:
            model = Mock()
            with self.assertRaises(ValueError):
                summarize_paper(material, model)
            model.bind.assert_not_called()

    def test_received_invalid_response_is_preserved_without_retry(self):
        for raw, finish in (("not-json", "stop"), (json.dumps(self.summary), "length")):
            model = Mock()
            model.bind.return_value.invoke.return_value = AIMessage(content=raw, response_metadata={"finish_reason": finish})
            with self.assertRaises(PaperSummaryError) as raised:
                summarize_paper(self.material, model)
            self.assertEqual(raised.exception.record["raw_response"], raw)
            self.assertEqual(raised.exception.record["input"], self.material)
            self.assertIsNone(raised.exception.record["summary"])
            self.assertIsNone(raised.exception.record["resolved_summary"])
            model.bind.assert_called_once_with(response_format={"type": "json_object"}, max_tokens=SYNTHESIS_MAX_TOKENS)
            model.bind.return_value.invoke.assert_called_once()

    def test_actual_client_roundtrip_preserves_raw_result_and_resolves_sources(self):
        requests = []
        raw = json.dumps(self.summary, ensure_ascii=False)

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "local-synthesis", "object": "chat.completion", "created": 0,
                "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": raw}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            })

        before = deepcopy(self.material)
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with patch('paper_sage.llm.client.ChatDeepSeek', side_effect=lambda **kwargs: ChatDeepSeek(http_client=client, **kwargs)):
                model = create_chat_model("test-only-key")
            self.assertEqual(model.max_retries, 0)
            record = summarize_paper(self.material, model)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["max_tokens"], SYNTHESIS_MAX_TOKENS)
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        self.assertEqual([message["role"] for message in requests[0]["messages"]], ["system", "user"])
        self.assertNotIn("tools", requests[0])
        self.assertEqual(len(json.loads(requests[0]["messages"][-1]["content"])["notes"]), 2)
        self.assertEqual(record["raw_response"], raw)
        self.assertEqual(record["summary"], self.summary)
        self.assertEqual(record["usage"]["total_tokens"], 140)
        sources = record["resolved_summary"]["method"][0]["sources"]
        self.assertEqual(sources, self.material["parts"])
        sources[0]["text"] = "changed"
        self.assertEqual(record["input"], before)
        self.assertEqual(self.material, before)

    def test_source_order_and_deduplication_preserve_summary_and_input(self):
        second_part = deepcopy(self.material["parts"][0])
        second_part.update(part_id="part2", pdf_page=2)
        self.material["parts"].append(second_part)
        self.material["notes"][0]["part_ids"] = ["part2", "part1"]
        original_summary, original_material = deepcopy(self.summary), deepcopy(self.material)
        result = resolve_summary_sources(self.summary, self.material)
        self.assertEqual([source["part_id"] for source in result["method"][0]["sources"]], ["part2", "part1"])
        self.assertEqual(self.summary, original_summary)
        self.assertEqual(self.material, original_material)
        invalid = deepcopy(self.summary)
        invalid["method"][0]["note_ids"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "不存在的要点"):
            resolve_summary_sources(invalid, self.material)


if __name__ == "__main__":
    unittest.main()
