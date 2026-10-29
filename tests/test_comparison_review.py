"""人工修订的对象绑定、字段限制、证据归属和原记录不变检查。"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest


from paper_sage.comparison.review import apply_comparison_review, comparison_digest
from paper_sage.retrieval.context import build_reference_text


class ComparisonReviewTests(unittest.TestCase):
    def setUp(self):
        papers, outputs = [], []
        for doc_id in ("a", "b"):
            chunk = {"project_id": "project", "document_id": doc_id, "source": f"{doc_id}.pdf",
                     "chunk_id": f"{doc_id}:data", "text": "Training corpus described here.", "pdf_page": 1}
            results = [{"rank": 1, "chunk": chunk}]
            papers.append({"document_id": doc_id, "name": f"{doc_id}.pdf", "source": f"{doc_id}.pdf",
                           "question": "training data", "retrieval_results": results, "reference_text": build_reference_text(results)})
            outputs.append({"document_id": doc_id, "findings": [{"text": "Original finding.", "chunk_ids": [chunk["chunk_id"]]}],
                            "missing_information": ""})
        self.material = {"project_id": "project", "dimensions": ["data"], "groups": [{"dimension": "data", "papers": papers}]}
        self.result = {"comparisons": [{"dimension": "data", "papers": outputs,
                                       "comparison": {"text": "Original comparison.", "chunk_ids": ["a:data", "b:data"]},
                                       "comparison_limit": "Original limit."}]}
        self.review = {"schema_version": 1, "purpose": "manual_comparison_review",
                       "source_comparison_sha256": comparison_digest(self.result), "corrections": [{
                           "path": ["comparisons", 0, "papers", 0, "findings", 0, "text"],
                           "original_text": "Original finding.", "replacement_text": "Revised finding with conditions.",
                           "reason": "Preserve a condition from the cited text.", "evidence_chunk_ids": ["a:data"],
                       }]}

    def test_changes_only_derived_text_and_preserves_sources(self):
        before = deepcopy((self.result, self.material, self.review))
        revised = apply_comparison_review(self.result, self.material, self.review)
        finding = revised["comparisons"][0]["papers"][0]["findings"][0]
        self.assertEqual(finding["text"], "Revised finding with conditions.")
        self.assertEqual(finding["chunk_ids"], ["a:data"])
        self.assertEqual((self.result, self.material, self.review), before)

    def test_rejects_different_original_or_stale_correction(self):
        changed = deepcopy(self.result)
        changed["comparisons"][0]["comparison_limit"] = "Changed"
        with self.assertRaisesRegex(ValueError, "不对应"):
            apply_comparison_review(changed, self.material, self.review)
        review = deepcopy(self.review)
        review["corrections"][0]["original_text"] = "Wrong original"
        with self.assertRaisesRegex(ValueError, "原文字"):
            apply_comparison_review(self.result, self.material, review)

    def test_rejects_identity_or_citation_changes_and_foreign_evidence(self):
        for path in (["comparisons", 0, "papers", 0, "document_id"],
                     ["comparisons", 0, "papers", 0, "findings", 0, "chunk_ids"]):
            review = deepcopy(self.review)
            review["corrections"][0]["path"] = path
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "只能修改文字"):
                apply_comparison_review(self.result, self.material, review)
        review = deepcopy(self.review)
        review["corrections"][0]["evidence_chunk_ids"] = ["b:data"]
        with self.assertRaisesRegex(ValueError, "范围之外"):
            apply_comparison_review(self.result, self.material, review)

    def test_rejects_duplicate_paths_and_invalid_revised_structure(self):
        review = deepcopy(self.review)
        review["corrections"].append(deepcopy(review["corrections"][0]))
        with self.assertRaisesRegex(ValueError, "重复修订"):
            apply_comparison_review(self.result, self.material, review)
        review = deepcopy(self.review)
        review["corrections"][0]["replacement_text"] = ""
        with self.assertRaisesRegex(ValueError, "新文字"):
            apply_comparison_review(self.result, self.material, review)


if __name__ == "__main__":
    unittest.main()
