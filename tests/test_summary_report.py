"""已保存总结的范围和原文一致性检查；临时报告与固定响应，不调用外部模型。"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


from langchain_core.messages import AIMessage
from paper_sage.summarization.inputs import build_summary_batches
from paper_sage.summarization.model import SUMMARY_MAX_TOKENS, summarize_batch
from paper_sage.summarization.parser import SUMMARY_FIELDS
from paper_sage.summarization.report import load_saved_summary
from paper_sage.summarization.synthesis import SYNTHESIS_MAX_TOKENS, prepare_synthesis_material, summarize_paper


class SavedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="paper-sage-summary-report-")
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith("paper-sage-summary-report-"))
        self.addCleanup(self.temporary.cleanup)
        (self.root / "data/evaluations").mkdir(parents=True)
        self.document = {"document_id": "doc", "source": "data/papers/doc.pdf", "index_version": "v1"}
        self.pages = [{"document_id": "doc", "source": self.document["source"], "pdf_page": 1,
                       "text": "The model uses attention."}]
        prepared = build_summary_batches(self.pages, "project")
        batch = prepared["batches"][0]
        summary = {field: [] for field in SUMMARY_FIELDS}
        summary["method"] = [{"text": "模型使用注意力。", "part_ids": [batch["parts"][0]["part_id"]]}]
        model = Mock()
        model.bind.return_value.invoke.return_value = AIMessage(content=json.dumps(summary), response_metadata={"finish_reason": "stop"})
        records = [summarize_batch(batch, model)]
        source = {
            "schema_version": 1, "purpose": "day33_batch_summaries", "mode": "real", "status": "completed_batches",
            "project_id": "project", "document_id": "doc", "source": self.document["source"],
            "model": "deepseek-flash", "max_tokens": SUMMARY_MAX_TOKENS, "prepared": prepared, "records": records,
        }
        self.source_path = self.root / "data/evaluations/map.json"
        self.source_path.write_text(json.dumps(source), encoding="utf-8")
        source_hash = sha256(self.source_path.read_bytes()).hexdigest()
        material = prepare_synthesis_material(prepared, records)
        final = {field: [] for field in SUMMARY_FIELDS}
        final["method"] = [{"text": "模型使用注意力。", "note_ids": [material["notes"][0]["note_id"]]}]
        model.bind.return_value.invoke.return_value = AIMessage(content=json.dumps(final), response_metadata={"finish_reason": "stop"})
        result = summarize_paper(material, model)
        self.report = {
            "schema_version": 1, "purpose": "day33_paper_summary", "mode": "real", "status": "completed_summary",
            "model": "deepseek-flash", "max_tokens": SYNTHESIS_MAX_TOKENS, "accepted_as_summary": True,
            "validation_error": None, "recorded_at": "2026-10-28T00:00:00Z",
            "source_report": "data/evaluations/map.json", "source_report_sha256": source_hash,
            "review": {"schema_version": 1, "purpose": "manual_summary_review", "source_report": "data/evaluations/map.json",
                       "source_report_sha256": source_hash, "corrections": []},
            "result": result,
        }
        self.path = self.root / "data/evaluations/reduce.json"

    def load(self, report=None, project_id="project", document=None, pages=None):
        self.path.write_text(json.dumps(self.report if report is None else report), encoding="utf-8")
        with patch('paper_sage.summarization.report.load_pdf_pages', return_value=deepcopy(self.pages if pages is None else pages)):
            return load_saved_summary(self.path, project_id, self.document if document is None else document,
                                      workspace_root=self.root)

    def test_successful_report_rebuilds_same_sources_without_changing_originals(self):
        before = deepcopy(self.report)
        source_before = self.source_path.read_bytes()
        artifact = self.load()
        self.assertTrue(artifact["reused"])
        self.assertEqual(artifact["resolved_summary"], self.report["result"]["resolved_summary"])
        self.assertEqual(artifact["coverage"]["batch_count"], 1)
        artifact["resolved_summary"]["method"][0]["sources"][0]["text"] = "changed"
        self.assertEqual(self.report, before)
        self.assertEqual(self.source_path.read_bytes(), source_before)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), before)

    def test_other_project_or_document_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不属于当前项目"):
            self.load(project_id="another")
        wrong = deepcopy(self.document)
        wrong["document_id"] = "another"
        with self.assertRaisesRegex(ValueError, "不属于当前项目"):
            self.load(document=wrong)

    def test_changed_pdf_or_batch_report_requires_new_summary(self):
        changed = deepcopy(self.pages)
        changed[0]["text"] += " New text."
        with self.assertRaises(ValueError):
            self.load(pages=changed)
        self.source_path.write_text(self.source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "批次报告已变化"):
            self.load()

    def test_modified_summary_sources_request_or_review_cannot_be_reused(self):
        variants = []
        changed = deepcopy(self.report)
        changed["result"]["summary"]["method"][0]["text"] = "changed"
        variants.append(changed)
        changed = deepcopy(self.report)
        changed["result"]["resolved_summary"]["method"][0]["sources"][0]["pdf_page"] = 99
        variants.append(changed)
        changed = deepcopy(self.report)
        changed["result"]["request_messages"][0]["content"] += " changed"
        variants.append(changed)
        changed = deepcopy(self.report)
        changed["review"]["source_report_sha256"] = "changed"
        variants.append(changed)
        for report in variants:
            with self.subTest(report=report), self.assertRaises(ValueError):
                self.load(report)

    def test_failed_report_and_external_report_path_are_rejected(self):
        for key, value in (("status", "validation_failed"), ("mode", "simulation"), ("accepted_as_summary", False)):
            report = deepcopy(self.report)
            report[key] = value
            with self.assertRaises(ValueError):
                self.load(report)
        with self.assertRaisesRegex(ValueError, "当前工作区的评估目录"):
            load_saved_summary(self.root.parent / "outside.json", "project", self.document, workspace_root=self.root)


if __name__ == "__main__":
    unittest.main()
