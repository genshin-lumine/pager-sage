"""总结记录复用校验；只使用本地固定响应，不调用外部模型。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock


from langchain_core.messages import AIMessage
from paper_sage.summarization.inputs import build_summary_batches
from paper_sage.summarization.model import SUMMARY_MAX_TOKENS, BatchSummaryError, build_batch_messages, summarize_batch
from paper_sage.summarization.parser import SUMMARY_FIELDS
from paper_sage.summarization.runner import exclude_redo_batches, load_completed_summaries, summarize_batches, validate_completed_records


class SummaryResumeTests(unittest.TestCase):
    def setUp(self):
        self.prepared = build_summary_batches([
            {"document_id": "doc", "source": "paper.pdf", "pdf_page": i, "text": text}
            for i, text in enumerate(["ABCDEF", "GHIJKL"], start=1)
        ], "project", max_chars=6)
        model = Mock()
        model.bind.return_value.invoke.return_value = AIMessage(
            content=json.dumps({field: [] for field in SUMMARY_FIELDS}),
            response_metadata={"finish_reason": "stop"},
        )
        self.records = [summarize_batch(batch, model) for batch in self.prepared["batches"]]

    def test_partial_records_are_ordered_copied_and_not_duplicated(self):
        records = validate_completed_records(self.prepared, list(reversed(self.records)))
        self.assertEqual(records, self.records)
        records[0]["input"]["parts"][0]["text"] = "changed"
        self.assertEqual(self.records[0]["input"]["parts"][0]["text"], "ABCDEF")
        with self.assertRaises(ValueError):
            validate_completed_records(self.prepared, [self.records[0], self.records[0]])

    def test_changed_source_prompt_and_response_are_rejected(self):
        variants = []
        foreign = deepcopy(self.records[0])
        foreign["input"]["parts"][0]["source"] = "foreign.pdf"
        variants.append(foreign)
        stale = deepcopy(self.records[0])
        stale["prompt_version"] = "old-version"
        variants.append(stale)
        inconsistent = deepcopy(self.records[0])
        inconsistent["summary"]["method"] = [{"text": "changed", "part_ids": [self.prepared["batches"][0]["parts"][0]["part_id"]]}]
        variants.append(inconsistent)
        failed = deepcopy(self.records[0])
        failed["finish_reason"] = "length"
        variants.append(failed)
        for record in variants:
            with self.assertRaises(ValueError):
                validate_completed_records(self.prepared, [record])

    def report(self):
        return {
            "schema_version": 1, "purpose": "day33_single_batch_summary",
            "mode": "real", "model": "deepseek-flash", "max_tokens": SUMMARY_MAX_TOKENS,
            "status": "completed_single_batch", "project_id": "project", "document_id": "doc",
            "source": "paper.pdf", "result": self.records[0],
        }

    def load_fixture(self, report, redo_batch_numbers=None):
        with tempfile.TemporaryDirectory(prefix="paper-sage-summary-resume-") as directory:
            root = Path(directory).resolve()
            self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
            self.assertTrue(root.name.startswith("paper-sage-summary-resume-"))
            path = root / "synthetic-report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            return load_completed_summaries(path, self.prepared, redo_batch_numbers=redo_batch_numbers)

    def test_only_successful_real_configuration_is_accepted(self):
        self.assertEqual(self.load_fixture(self.report()), self.records[:1])
        for field, value in (("status", "validation_failed"), ("mode", "simulation"),
                             ("project_id", "another-project"), ("max_tokens", 1)):
            report = self.report()
            report[field] = value
            with self.assertRaises(ValueError):
                self.load_fixture(report)

    def test_partial_progress_reuses_only_completed_records(self):
        report = self.report()
        report.update(purpose="day33_batch_summaries", status="partial_failed",
                      prepared=self.prepared, records=self.records[:1], failed_record={"summary": None})
        self.assertEqual(self.load_fixture(report), self.records[:1])
        report["prepared"] = deepcopy(self.prepared)
        report["prepared"]["max_chars"] += 1
        with self.assertRaises(ValueError):
            self.load_fixture(report)

    def test_all_completed_records_need_no_model_or_progress_callback(self):
        model, callback = Mock(), Mock()
        records = summarize_batches(self.prepared, model, self.records, callback)
        self.assertEqual(records, self.records)
        model.bind.assert_not_called()
        callback.assert_not_called()

    def test_failure_stops_later_batches_and_resume_skips_saved_records(self):
        prepared = build_summary_batches([
            {"document_id": "doc", "source": "paper.pdf", "pdf_page": i, "text": text}
            for i, text in enumerate(["ABCDEF", "GHIJKL", "MNOPQR", "STUVWX"], start=1)
        ], "project", max_chars=6)
        good = AIMessage(content=json.dumps({field: [] for field in SUMMARY_FIELDS}),
                         response_metadata={"finish_reason": "stop"})
        bad = AIMessage(content="not-json", response_metadata={"finish_reason": "stop"})
        first_model = Mock()
        first_model.bind.return_value.invoke.side_effect = [good, bad]
        saved = []
        with self.assertRaises(BatchSummaryError):
            summarize_batches(prepared, first_model, self.records[:1], saved.append)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["batch_id"], prepared["batches"][1]["batch_id"])
        attempted = [json.loads(call.args[0][-1].content)["batch_id"]
                     for call in first_model.bind.return_value.invoke.call_args_list]
        self.assertEqual(attempted, [batch["batch_id"] for batch in prepared["batches"][1:3]])

        second_model = Mock()
        second_model.bind.return_value.invoke.return_value = good
        completed = self.records[:1] + saved
        resumed = summarize_batches(prepared, second_model, completed)
        self.assertEqual(len(resumed), 4)
        self.assertEqual(resumed[:2], completed)
        attempted = [json.loads(call.args[0][-1].content)["batch_id"]
                     for call in second_model.bind.return_value.invoke.call_args_list]
        self.assertEqual(attempted, [batch["batch_id"] for batch in prepared["batches"][2:]])

    def test_progress_callback_cannot_mutate_returned_records(self):
        model = Mock()
        model.bind.return_value.invoke.return_value = AIMessage(
            content=json.dumps({field: [] for field in SUMMARY_FIELDS}),
            response_metadata={"finish_reason": "stop"},
        )

        def mutate(record):
            record["input"]["parts"][0]["text"] = "changed"

        records = summarize_batches(self.prepared, model, self.records[:1], mutate)
        self.assertEqual(records[1]["input"], self.prepared["batches"][1])
        self.assertEqual(validate_completed_records(self.prepared, records), records)

    def test_known_old_prompt_requires_exact_original_messages(self):
        old = deepcopy(self.records[0])
        old["prompt_version"] = "day33_batch_v2_shape_example"
        old["request_messages"] = [{"role": m.type, "content": m.content}
                                   for m in build_batch_messages(old["input"], prompt_version=old["prompt_version"])]
        self.assertEqual(validate_completed_records(self.prepared, [old]), [old])
        old["request_messages"][0]["content"] += " changed"
        with self.assertRaisesRegex(ValueError, "请求消息"):
            validate_completed_records(self.prepared, [old])

    def test_redo_replaces_only_selected_invalid_record_and_keeps_original(self):
        report = self.report()
        bad = deepcopy(self.records[1])
        bad["summary"]["method"] = [{"text": "中文要点", "part_ids": [bad["input"]["parts"][0]["part_id"]]}]
        bad["raw_response"] = json.dumps(bad["summary"], ensure_ascii=False)
        report.update(purpose="day33_batch_summaries", status="completed_batches",
                      prepared=self.prepared, records=[self.records[0], bad])
        before = deepcopy(report)
        with self.assertRaisesRegex(ValueError, "占位文字"):
            self.load_fixture(report)
        with self.assertRaisesRegex(ValueError, "占位文字"):
            self.load_fixture(report, redo_batch_numbers=[1])
        retained = self.load_fixture(report, redo_batch_numbers=[2])
        self.assertEqual(retained, self.records[:1])
        model = Mock()
        model.bind.return_value.invoke.return_value = AIMessage(
            content=json.dumps({field: [] for field in SUMMARY_FIELDS}),
            response_metadata={"finish_reason": "stop"},
        )
        result = summarize_batches(self.prepared, model, retained)
        model.bind.return_value.invoke.assert_called_once()
        sent_batch = json.loads(model.bind.return_value.invoke.call_args.args[0][-1].content)["batch_id"]
        self.assertEqual(sent_batch, bad["batch_id"])
        self.assertEqual(result[0], self.records[0])
        self.assertEqual(len(result), 2)
        self.assertEqual(report, before)
        report["project_id"] = "foreign"
        with self.assertRaisesRegex(ValueError, "不属于当前项目"):
            self.load_fixture(report, redo_batch_numbers=[2])

    def test_redo_rejects_invalid_numbers_and_ambiguous_records(self):
        for numbers in ([0], [3], [-1], [True], [1.0], ["1"], [1, 1]):
            with self.subTest(numbers=numbers), self.assertRaises(ValueError):
                exclude_redo_batches(self.prepared, self.records, numbers)
        for records in ([self.records[0], self.records[0]], [None], [{"batch_id": "foreign"}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                exclude_redo_batches(self.prepared, records, [1])
        with self.assertRaisesRegex(ValueError, "已有记录"):
            exclude_redo_batches(self.prepared, self.records[:1], [2])


if __name__ == "__main__":
    unittest.main()
