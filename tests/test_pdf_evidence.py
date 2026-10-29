"""单页 PDF 的范围、文件及页码检查；只使用临时 PDF，不调用模型。"""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError


from paper_sage.qa.pdf import load_pdf_page_preview


class PdfEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="paper-sage-pdf-preview-")
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith("paper-sage-pdf-preview-"))
        self.addCleanup(self.temporary.cleanup)
        self.doc = {"document_id": "paper_a", "source": "data/papers/project/paper_a/paper.pdf"}
        self.path = self.root / self.doc["source"]
        self.path.parent.mkdir(parents=True)
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=300)
        writer.add_blank_page(width=400, height=500)
        writer.write(self.path)
        self.source = {"project_id": "project", "document_id": "paper_a", "source": self.doc["source"], "pdf_page": 2}

    def preview(self, source=None, documents=None):
        return load_pdf_page_preview(self.source if source is None else source, "project",
                                     [self.doc] if documents is None else documents, workspace_root=self.root)

    def test_single_page_is_the_requested_page_and_no_files_are_written(self):
        before = self.path.read_bytes()
        file_names = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*"))
        source_before = deepcopy(self.source)
        result = self.preview()
        single = PdfReader(BytesIO(result["pdf_bytes"]))
        self.assertEqual(len(single.pages), 1)
        self.assertEqual(tuple(single.pages[0].mediabox), (0, 0, 400, 500))
        self.assertEqual((result["pdf_page"], result["page_index"], result["total_pages"]), (2, 1, 2))
        self.assertEqual(result["download_name"], "paper_p2.pdf")
        self.assertEqual(result["source_sha256"], sha256(before).hexdigest())
        self.assertIsNone(result["source_text_found"])
        self.assertEqual(self.source, source_before)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*")), file_names)

    def test_invalid_scope_and_page_numbers_are_rejected(self):
        for field, value in (("project_id", "other"), ("document_id", "other"), ("source", "other.pdf"),
                             ("pdf_page", 0), ("pdf_page", True), ("pdf_page", 3)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.preview(dict(self.source, **{field: value}))
        with self.assertRaises(ValueError):
            self.preview(documents=[])

    def test_missing_file_and_files_outside_papers_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.preview(dict(self.source, source="data/papers/missing.pdf"), [dict(self.doc, source="data/papers/missing.pdf")])
        for path in ("elsewhere.pdf", "data/papers/../outside.pdf", (self.root.parent / "outside.pdf").as_posix()):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.preview(dict(self.source, source=path), [dict(self.doc, source=path)])
        with patch('paper_sage.qa.pdf.Path.is_symlink', return_value=True), self.assertRaisesRegex(ValueError, "符号链接"):
            self.preview()
        with patch('paper_sage.qa.pdf.Path.is_junction', return_value=True), self.assertRaisesRegex(ValueError, "目录联接"):
            self.preview()

    def test_encrypted_and_unreadable_pdfs_do_not_return_preview_bytes(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.encrypt("test-password")
        writer.write(self.path)
        with self.assertRaisesRegex(ValueError, "加密"):
            self.preview()
        with patch('paper_sage.qa.pdf.PdfReader', side_effect=PdfReadError("bad fixture")), self.assertRaisesRegex(ValueError, "无法读取"):
            self.preview()

    def test_text_missing_from_current_page_cannot_be_presented_as_matching_evidence(self):
        with self.assertRaisesRegex(ValueError, "证据原文未在当前 PDF 页"):
            self.preview(dict(self.source, text="An old quotation that is absent from this page."))
        with self.assertRaisesRegex(ValueError, "非空字符串"):
            self.preview(dict(self.source, text=""))


if __name__ == "__main__":
    unittest.main()
