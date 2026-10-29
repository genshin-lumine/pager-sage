"""文档导入回归：真实页面、PDF 解析与索引，固定上传内容和向量。"""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pypdf import PdfWriter
from streamlit.delta_generator import DeltaGenerator

import test_project_scope as scope
from paper_sage.projects.store import get_index_dir, load_project, save_project
from paper_sage.retrieval.scope import load_scope_index


class DocumentImportPageTests(scope.ProjectScopeTests):
    def activity(self):
        return (
            len(self.classifications), len(self.tool_requests), len(self.requests),
            len(self.rewrite_requests), len(self.summary_requests), self.encode_calls,
            len(self.rerank_calls), self.model_instances, self.rerank_model_instances,
        )

    def test_empty_project_blocks_question_and_clears_old_chat(self):
        empty_id = "scope_empty_project"
        save_project({"project_id": empty_id, "name": "Empty", "documents": []})
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = self.activity()
        files_before = scope.data_snapshot(Path("data"))

        self.switch(at, empty_id)

        self.assert_empty_chat(at)
        self.assertEqual(at.session_state["chat_session"]["scope_key"], (empty_id, ()))
        self.assertTrue(any("当前项目还没有论文" in item.value for item in at.info))
        self.assertFalse(any(widget.label == "提交问题" for widget in at.button))
        self.assertFalse(any(str(widget.key).startswith("question_") for widget in at.text_input))
        self.assertEqual(self.activity(), before)
        self.assertEqual(scope.data_snapshot(Path("data")), files_before)

    def test_blank_pdf_is_rejected_and_import_copy_is_cleaned(self):
        # 有效的空白 PDF，与损坏文件或扩展名伪装的文件是不同场景。
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        output = BytesIO()
        writer.write(output)
        raw = output.getvalue()
        fixture = Path("data/fixtures/blank.pdf")
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_bytes(raw)
        at = self.app(self.project_a, [self.aid])
        import paper_sage.documents.importer as importer

        before = self.activity()
        project_before = load_project(self.project_a)
        files_before = scope.data_snapshot(Path("data"))
        document_id = "doc_" + importer.get_file_hash(raw)
        uploaded = SimpleNamespace(name="blank.pdf", getvalue=lambda: raw)
        # 只替换上传控件的返回值，导入按钮、解析、登记和清理仍执行应用代码。
        with patch.object(DeltaGenerator, "file_uploader", return_value=uploaded), \
             patch("paper_sage.ui.app.build_document_index", wraps=importer.build_document_index) as build_index, \
             patch.object(importer, "SentenceTransformer", side_effect=AssertionError("blank PDF must not load embeddings")) as model:
            self.checked(at.run())
            self.widget(at.button, f"parse_pdf_{self.project_a}").click().run()
            self.assertFalse(at.exception, [item.message for item in at.exception])
            self.assertTrue(any("构建索引" in item.value and "未提取到可用文字" in item.value for item in at.error))
            self.assertEqual(build_index.call_count, 1)
            model.assert_not_called()

        self.assertEqual(load_project(self.project_a), project_before)
        self.assertNotIn("pending_cleanup", load_project(self.project_a))
        self.assertFalse((Path("data/papers") / self.project_a / document_id / "blank.pdf").exists())
        self.assertFalse(get_index_dir(self.project_a, document_id, "v1").exists())
        self.assertEqual(scope.data_snapshot(Path("data")), files_before)
        self.assertEqual(self.activity(), before)

    def test_duplicate_content_skips_import_but_other_project_can_import(self):
        at = self.app(self.project_a, [self.aid])
        import paper_sage.documents.importer as importer

        raw = Path(self.doc_a["source"]).read_bytes()
        uploaded = SimpleNamespace(name="renamed_paper.pdf", getvalue=lambda: raw)
        project_before = load_project(self.project_a)
        files_before = scope.data_snapshot(Path("data"))
        before_calls = (len(self.classifications), len(self.requests), self.encode_calls)

        def encode_passages(texts, **kwargs):
            self.assertTrue(texts)
            self.assertTrue(all(text.startswith("passage: ") for text in texts))
            vectors = np.zeros((len(texts), 384), dtype="float32")
            vectors[:, 0] = 1.0
            return vectors

        embedding = Mock()
        embedding.encode.side_effect = encode_passages
        with patch.object(DeltaGenerator, "file_uploader", return_value=uploaded), \
             patch("paper_sage.ui.app.build_document_index", wraps=importer.build_document_index) as build_index, \
             patch.object(importer, "SentenceTransformer", return_value=embedding) as model:
            self.checked(at.run())
            self.assertTrue(any("已存在于当前项目" in item.value for item in at.info))
            self.assertFalse(any(widget.key == f"parse_pdf_{self.project_a}" for widget in at.button))
            project_after = load_project(self.project_a)
            # 数量保持不变还不够：下面也检查完整记录、文件和建索引调用。
            self.assertEqual(len(project_after["documents"]), len(project_before["documents"]))
            self.assertEqual(project_after, project_before)
            self.assertEqual(scope.data_snapshot(Path("data")), files_before)
            build_index.assert_not_called()
            model.assert_not_called()

            # 相同字节在另一项目可以独立导入，不能使用全局去重阻止它。
            other_before = load_project(self.project_b)
            self.switch(at, self.project_b)
            self.checked(self.widget(at.button, f"parse_pdf_{self.project_b}").click().run())
            self.assertTrue(any("已导入 renamed_paper.pdf" in item.value for item in at.success))
            other_after = load_project(self.project_b)
            self.assertEqual(len(other_after["documents"]), len(other_before["documents"]) + 1)
            imported = next(doc for doc in other_after["documents"] if doc["document_id"] == self.aid)
            self.assertEqual(Path(imported["source"]).read_bytes(), raw)
            self.assertEqual(imported["source"], f"data/papers/{self.project_b}/{self.aid}/renamed_paper.pdf")
            self.assertNotIn("pending_cleanup", other_after)
            index, metadata = load_scope_index(other_after, [self.aid])
            self.assertGreater(index.ntotal, 0)
            self.assertEqual(index.ntotal, len(metadata["chunks"]))
            for chunk in metadata["chunks"]:
                self.assertEqual((chunk["project_id"], chunk["document_id"], chunk["source"]),
                                 (self.project_b, self.aid, imported["source"]))
            self.assertEqual(build_index.call_count, 1)
            model.assert_called_once()
            embedding.encode.assert_called_once()

            # 新项目完成导入后的再次上传也应进入已有分支。
            imported_files = scope.data_snapshot(Path("data"))
            self.checked(at.run())
            self.assertTrue(any("已存在于当前项目" in item.value for item in at.info))
            self.assertEqual(build_index.call_count, 1)
            self.assertEqual(scope.data_snapshot(Path("data")), imported_files)

        self.assertEqual(load_project(self.project_a), project_before)
        for name, digest in files_before.items():
            if name != f"projects/{self.project_b}/project.json":
                self.assertEqual(imported_files[name], digest, name)
        self.assertEqual((len(self.classifications), len(self.requests), self.encode_calls), before_calls)


def load_tests(loader, tests, pattern):
    # 单独发现这个模块时也仅运行新增三项，不重复运行继承的四个范围场景。
    return unittest.TestSuite(DocumentImportPageTests(name) for name in (
        "test_empty_project_blocks_question_and_clears_old_chat",
        "test_blank_pdf_is_rejected_and_import_copy_is_cleaned",
        "test_duplicate_content_skips_import_but_other_project_can_import",
    ))


if __name__ == "__main__":
    unittest.main()
