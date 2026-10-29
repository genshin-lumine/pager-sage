"""证据和独立索引、页面记录的交叉检查，不调用模型或网络。"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest


from paper_sage.qa.source_checks import validate_source_record


class AnswerSourceChecksTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "chunk_id": "test:paper:p7:c1", "project_id": "test", "document_id": "paper",
            "source": "paper.pdf", "pdf_page": 7, "text": "We used 8 NVIDIA P100 GPUs.",
        }
        self.indexed = dict(self.source, chunk_size=800, chunk_overlap=100)
        self.page = {"document_id": "paper", "source": "paper.pdf", "pdf_page": 7,
                     "text": "Hardware\nWe used 8 NVIDIA P100 GPUs.\nTraining schedule"}

    def test_valid_source_and_whitespace_variation_keep_inputs_unchanged(self):
        self.page["text"] = self.page["text"].replace("NVIDIA P100", "NVIDIA\nP100")
        before = deepcopy((self.source, self.indexed, self.page))
        result = validate_source_record(self.source, self.indexed, self.page)
        self.assertTrue(result["index_fields_match"] and result["page_text_found"])
        result["pdf_page"] = 20
        self.assertEqual((self.source, self.indexed, self.page), before)

    def test_all_source_fields_must_match_the_independent_index(self):
        for field, value in (
            ("chunk_id", "other-id"), ("project_id", "other-project"), ("document_id", "other-paper"),
            ("source", "other.pdf"), ("pdf_page", 8), ("text", "We used 16 NVIDIA P100 GPUs."),
        ):
            changed = dict(self.source, **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_source_record(changed, self.indexed, self.page)
        with self.assertRaisesRegex(ValueError, "不在当前索引"):
            validate_source_record(self.source, None, self.page)
        with self.assertRaises(ValueError):
            validate_source_record(dict(self.source, pdf_page=True), self.indexed, self.page)

    def test_equal_text_in_another_document_or_page_is_not_the_expected_source(self):
        for field, value in (("document_id", "other"), ("source", "other.pdf"), ("pdf_page", 8), ("pdf_page", True)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "PDF 页面"):
                validate_source_record(self.source, self.indexed, dict(self.page, **{field: value}))
        with self.assertRaisesRegex(ValueError, "没有对应"):
            validate_source_record(self.source, self.indexed, None)

    def test_missing_page_text_requires_review_instead_of_claiming_the_pdf_has_no_information(self):
        for value in (None, "", "   "):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "未提取到可用文字"):
                validate_source_record(self.source, self.indexed, dict(self.page, text=value))

    def test_consistent_but_wrong_index_and_report_still_fail_against_page(self):
        changed = dict(self.source, text="We used 16 NVIDIA P100 GPUs.")
        with self.assertRaisesRegex(ValueError, "未在对应 PDF 页"):
            validate_source_record(changed, deepcopy(changed), self.page)
        # 不允许将词内空白直接删掉后误认为命中。
        with self.assertRaisesRegex(ValueError, "未在对应 PDF 页"):
            validate_source_record(self.source, self.indexed, dict(self.page, text="We used 8 NVIDIAP100 GPUs."))


if __name__ == "__main__":
    unittest.main()
