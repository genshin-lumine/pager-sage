"""工作台 PDF 查看：真实页面和临时 PDF，问答与检索模型均为固定替身。"""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfReader

import paper_sage.ui.pdf as pdf_evidence_view
import test_project_scope as scope_tests


class PdfEvidencePageTests(scope_tests.ProjectScopeTests):
    def counts(self):
        return (len(self.classifications), len(self.tool_requests), len(self.requests),
                len(self.rewrite_requests), self.encode_calls, len(self.rerank_calls),
                self.model_instances, self.rerank_model_instances)

    def submit(self, at):
        next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(self.question)
        return self.checked(next(w for w in at.button if w.label == "提交问题").click().run())

    def assert_no_pdf_controls(self, at):
        self.assertFalse(at.toggle)
        self.assertFalse(at.get("bidi_component"))
        self.assertFalse(at.get("download_button"))

    def test_only_opening_reads_pdf_and_does_not_repeat_model_or_modify_chat(self):
        with patch('paper_sage.ui.pdf.load_pdf_page_preview', wraps=pdf_evidence_view.load_pdf_page_preview) as loader:
            at = self.app(self.project_a, [self.aid])
            self.ask(at, self.fixture_a["expected_code"])
            self.assertEqual(loader.call_count, 0)
            self.assertFalse(at.toggle[0].value)
            before, counts = deepcopy(at.session_state["chat_session"]), self.counts()
            with patch('paper_sage.ui.pdf.render_pdf_page_preview', wraps=pdf_evidence_view.render_pdf_page_preview) as renderer:
                self.checked(at.toggle[0].set_value(True).run())
                self.assertEqual(loader.call_count, 1)
                preview = renderer.call_args.args[0]
                pdf = PdfReader(BytesIO(preview["pdf_bytes"]))
                self.assertEqual(len(pdf.pages), 1)
                self.assertIn(self.fixture_a["expected_code"], pdf.pages[0].extract_text())
                self.assertEqual(preview["source"], self.doc_a["source"])
                self.assertEqual((preview["pdf_page"], preview["page_index"]), (1, 0))
            self.assertEqual(len(at.get("bidi_component")), 1)
            self.assertEqual(len(at.get("download_button")), 1)
            self.assertEqual(at.session_state["chat_session"], before)
            self.assertEqual(self.counts(), counts)
            self.checked(at.toggle[0].set_value(False).run())
            self.assertEqual(loader.call_count, 1)
            self.assertFalse(at.get("bidi_component"))
            self.assertFalse(at.get("download_button"))
            self.assertEqual(at.session_state["chat_session"], before)

    def test_same_source_in_multiple_turns_has_independent_stable_controls(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        first_key = at.toggle[0].key
        self.checked(at.toggle[0].set_value(True).run())
        saved = deepcopy(at.session_state["chat_session"]["messages"])
        self.submit(at)
        self.assertEqual(len(at.toggle), 2)
        self.assertEqual(at.toggle[0].key, first_key)
        self.assertNotEqual(at.toggle[0].key, at.toggle[1].key)
        self.assertTrue(at.toggle[0].value)
        self.assertFalse(at.toggle[1].value)
        self.assertEqual(at.session_state["chat_session"]["messages"][:2], saved)
        counts = self.counts()
        self.checked(at.toggle[1].set_value(True).run())
        self.assertEqual(len(at.get("bidi_component")), 2)
        self.assertEqual(self.counts(), counts)

    def test_clear_and_project_change_remove_viewer_and_use_new_control_identity(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        old_key = at.toggle[0].key
        self.checked(at.toggle[0].set_value(True).run())
        counts = self.counts()
        self.checked(next(w for w in at.button if w.label == "清空会话").click().run())
        self.assert_empty_chat(at)
        self.assert_no_pdf_controls(at)
        self.assertEqual(self.counts(), counts)
        self.ask(at, self.fixture_a["expected_code"])
        self.assertNotEqual(at.toggle[0].key, old_key)
        self.assertFalse(at.toggle[0].value)
        self.checked(at.toggle[0].set_value(True).run())
        self.switch(at, self.project_b)
        self.assert_empty_chat(at)
        self.assert_no_pdf_controls(at)
        self.ask(at, self.fixture_b["expected_code"])
        with patch('paper_sage.ui.pdf.render_pdf_page_preview', wraps=pdf_evidence_view.render_pdf_page_preview) as renderer:
            self.checked(at.toggle[0].set_value(True).run())
            preview = renderer.call_args.args[0]
            self.assertEqual(preview["source"], self.doc_b["source"])
            self.assertEqual(preview["project_id"], self.project_b)
            self.assertIn(self.fixture_b["expected_code"], preview["page_text"])

    def test_document_scope_change_and_deletion_remove_old_viewer(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        self.checked(at.toggle[0].set_value(True).run())
        self.select(at, [self.bid])
        self.assert_empty_chat(at)
        self.assert_no_pdf_controls(at)
        self.select(at, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        self.checked(at.toggle[0].set_value(True).run())
        self.checked(self.widget(at.selectbox, f"delete_document_{self.project_a}").set_value(self.aid).run())
        self.checked(self.widget(at.button, f"delete_submit_{self.project_a}").click().run())
        self.assert_empty_chat(at)
        self.assert_no_pdf_controls(at)

    def test_missing_or_changed_pdf_shows_local_warning_and_keeps_answer(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before, counts = deepcopy(at.session_state["chat_session"]), self.counts()
        path = (self.root / self.doc_a["source"]).resolve()
        self.assertTrue(path.is_relative_to(self.root))
        original = path.read_bytes()
        path.unlink()
        self.checked(at.toggle[0].set_value(True).run())
        self.assertTrue(any("PDF 文件不存在" in w.value for w in at.warning))
        self.assertFalse(at.get("bidi_component"))
        self.assertFalse(at.get("download_button"))
        self.assertEqual(at.session_state["chat_session"], before)
        self.assertEqual(self.counts(), counts)
        # 用同项目 B 的文件替换临时 A PDF，证据文字不匹配时也不能展示为原引用页。
        path.write_bytes((self.root / self.doc_b_in_a["source"]).read_bytes())
        self.checked(at.run())
        self.assertTrue(any("证据原文未在当前 PDF 页" in w.value for w in at.warning))
        self.assertFalse(at.get("bidi_component"))
        path.write_bytes(original)
        self.checked(at.run())
        self.assertEqual(len(at.get("bidi_component")), 1)
        self.assertEqual(at.session_state["chat_session"], before)
        self.assertEqual(self.counts(), counts)
