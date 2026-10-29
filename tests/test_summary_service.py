"""总结业务的生成、复用与失败恢复；使用临时目录、固定模型及受控 PDF 提取。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


from langchain_core.messages import AIMessage
from paper_sage.summarization.service import create_summary_runner
from paper_sage.summarization.parser import SUMMARY_FIELDS


class SummaryServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="paper-sage-summary-service-")
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith("paper-sage-summary-service-"))
        self.addCleanup(self.temporary.cleanup)
        self.document = {"document_id": "doc", "source": "data/papers/doc.pdf", "index_version": "v1"}
        self.pages = [{"document_id": "doc", "source": self.document["source"], "pdf_page": i,
                       "text": letter * 4000} for i, letter in enumerate("ABC", 1)]
        self.requests = []
        self.fail_map = None
        self.fail_reduce = False
        self.model = Mock()
        self.model.bind.return_value.invoke.side_effect = self.respond
        for target in ('paper_sage.summarization.service.load_pdf_pages', 'paper_sage.summarization.report.load_pdf_pages'):
            mock = patch(target, side_effect=lambda *args: deepcopy(self.pages))
            mock.start()
            self.addCleanup(mock.stop)

    def respond(self, messages):
        payload = json.loads(messages[-1].content)
        summary = {field: [] for field in SUMMARY_FIELDS}
        finish = "stop"
        if "parts" in payload:
            page = payload["parts"][0]["pdf_page"]
            self.requests.append(("map", page))
            if self.fail_map == page:
                return AIMessage(content="not-json", response_metadata={"finish_reason": "stop"})
            summary["method"] = [{"text": f"模拟第{page}页的要点。", "part_ids": [payload["parts"][0]["part_id"]]}]
        else:
            self.requests.append(("reduce", None))
            summary["method"] = [{"text": "模拟汇总内容。", "note_ids": [note["note_id"] for note in payload["notes"]]}]
            finish = "length" if self.fail_reduce else "stop"
        return AIMessage(content=json.dumps(summary, ensure_ascii=False), response_metadata={"finish_reason": finish})

    def runner(self, project="project", documents=None):
        return create_summary_runner(project, [self.document] if documents is None else documents,
                                     self.model, workspace_root=self.root)

    def reports(self, purpose):
        return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted((self.root / "data/evaluations").glob("*.json"))
                if json.loads(p.read_text(encoding="utf-8")).get("purpose") == purpose]

    def test_generate_once_then_reuse_but_never_cross_project(self):
        run = self.runner()
        first = run("doc")
        self.assertFalse(first["reused"])
        self.assertEqual(self.requests, [("map", 1), ("map", 2), ("map", 3), ("reduce", None)])
        before = {p: p.read_bytes() for p in (self.root / "data/evaluations").glob("*.json")}
        second = run("doc")
        self.assertTrue(second["reused"])
        self.assertEqual(second["resolved_summary"], first["resolved_summary"])
        self.assertEqual(len(self.requests), 4)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        foreign = self.runner(project="other_project")("doc")
        self.assertFalse(foreign["reused"])
        self.assertEqual(foreign["project_id"], "other_project")
        self.assertEqual(len(self.requests), 8)

    def test_map_failure_keeps_completed_parts_and_resume_skips_them(self):
        self.fail_map = 2
        with self.assertRaises(ValueError):
            self.runner()("doc")
        path, report = self.reports("day33_batch_summaries")[0]
        old_bytes = path.read_bytes()
        self.assertEqual(report["status"], "partial_failed")
        self.assertEqual(len(report["records"]), 1)
        self.assertEqual(report["failed_record"]["raw_response"], "not-json")
        self.assertEqual(self.requests, [("map", 1), ("map", 2)])
        self.fail_map = None
        self.runner()("doc")
        self.assertEqual(self.requests[2:], [("map", 2), ("map", 3), ("reduce", None)])
        self.assertEqual(path.read_bytes(), old_bytes)
        self.assertEqual(self.reports("day33_batch_summaries")[-1][1]["reused_count"], 1)

    def test_reduce_failure_reuses_all_map_parts_on_next_submit(self):
        self.fail_reduce = True
        with self.assertRaises(ValueError):
            self.runner()("doc")
        _, report = self.reports("day33_paper_summary")[0]
        self.assertEqual(report["status"], "validation_failed")
        self.assertFalse(report["accepted_as_summary"])
        self.assertEqual(report["result"]["finish_reason"], "length")
        self.assertIsNone(report["result"]["summary"])
        self.fail_reduce = False
        self.runner()("doc")
        self.assertEqual(self.requests, [("map", 1), ("map", 2), ("map", 3), ("reduce", None), ("reduce", None)])

    def test_changed_pdf_and_wrong_selection_cannot_reuse_old_summary(self):
        self.runner()("doc")
        self.pages[0]["text"] = "Z" * 4000
        result = self.runner()("doc")
        self.assertFalse(result["reused"])
        self.assertEqual(len(self.requests), 8)
        with self.assertRaises(ValueError):
            self.runner()("another")
        with self.assertRaises(ValueError):
            self.runner(documents=[self.document, {**self.document, "document_id": "second"}])("doc")
        self.assertEqual(len(self.requests), 8)


if __name__ == "__main__":
    unittest.main()
