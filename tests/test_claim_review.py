"""人工判读与原回答关联的检查；固定模型响应，不执行网络请求。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


from langchain_core.messages import AIMessage

from paper_sage.qa.model import request_cited_answer
from paper_sage.qa.review import answer_record_digest, apply_claim_review


class ClaimReviewTests(unittest.TestCase):
    def setUp(self):
        self.results = [{"rank": 1, "chunk": {
            "project_id": "test", "document_id": "paper", "source": "paper.pdf", "pdf_page": 7,
            "chunk_id": "test:paper:p7:c1", "text": "Training used 8 NVIDIA P100 GPUs.",
        }}]
        self.result = {"claims": [
            {"text": text, "chunk_ids": ["test:paper:p7:c1"]} for text in
            ("Training used 8 GPUs.", "Training used 16 GPUs.", "Training consumed 1200 kWh.")
        ], "insufficient_reason": ""}
        self.artifact = self.make_artifact(self.result)
        self.review = {
            "schema_version": 1, "purpose": "manual_claim_review", "reviewer": "fixed-human-review",
            "source_answer_sha256": answer_record_digest(self.artifact),
            "entries": [
                {"claim_number": number, "original_text": claim["text"], "chunk_ids": list(claim["chunk_ids"]),
                 "verdict": verdict, "reason": reason}
                for number, (claim, verdict, reason) in enumerate(zip(
                    self.result["claims"], ["supported", "contradicted", "insufficient"],
                    ["The source says 8.", "The source says 8, not 16.", "No energy consumption was provided."],
                ), start=1)
            ],
        }

    def make_artifact(self, payload):
        model = Mock()
        model.bind.return_value.invoke.return_value = AIMessage(
            content=json.dumps(payload), response_metadata={"finish_reason": "stop"},
        )
        return request_cited_answer("Review the statements.", self.results, model)

    def test_same_source_keeps_three_separate_judgments_and_independent_copies(self):
        before, review_before = deepcopy(self.artifact), deepcopy(self.review)
        reviewed = apply_claim_review(self.artifact, self.review)
        self.assertEqual((reviewed["claim_count"], reviewed["reviewed_count"]), (3, 3))
        self.assertEqual([c["verdict"] for c in reviewed["claims"]], ["supported", "contradicted", "insufficient"])
        self.assertEqual([c["claim_number"] for c in reviewed["claims"]], [1, 2, 3])
        reviewed["claims"][0]["sources"][0]["text"] = "Changed display copy"
        reviewed["review_record"]["entries"][0]["reason"] = "Changed reason copy"
        self.assertEqual(self.artifact, before)
        self.assertEqual(self.review, review_before)
        self.assertEqual(self.artifact["semantic_review"], "pending")

    def test_partial_empty_and_no_claim_reviews_never_mark_unreviewed_claims_supported(self):
        for count in (0, 1):
            review = deepcopy(self.review)
            review["entries"] = review["entries"][:count]
            reviewed = apply_claim_review(self.artifact, review)
            self.assertEqual(reviewed["reviewed_count"], count)
            self.assertEqual([c["verdict"] for c in reviewed["claims"]][count:], ["pending"] * (3 - count))
        insufficient = self.make_artifact({"claims": [], "insufficient_reason": "No evidence."})
        review = deepcopy(self.review)
        review.update(source_answer_sha256=answer_record_digest(insufficient), entries=[])
        reviewed = apply_claim_review(insufficient, review)
        self.assertEqual((reviewed["claim_count"], reviewed["reviewed_count"], reviewed["claims"]), (0, 0, []))

    def test_changed_source_snapshot_cannot_reuse_old_review(self):
        changed = deepcopy(self.artifact)
        changed["final_results"][0]["chunk"]["pdf_page"] = 9
        changed["evidence_by_id"]["test:paper:p7:c1"]["pdf_page"] = 9
        for claim in changed["claims_with_sources"]:
            claim["sources"][0]["pdf_page"] = 9
        # 修改后的记录仍能自洽地通过来源关联检查，但它不是当时人工查看的那份记录。
        self.assertNotEqual(answer_record_digest(changed), self.review["source_answer_sha256"])
        with self.assertRaisesRegex(ValueError, "快照"):
            apply_claim_review(changed, self.review)

    def test_invalid_or_misdirected_review_entries_are_rejected(self):
        for field, value in (
            ("claim_number", 0), ("claim_number", True), ("claim_number", 1.0), ("claim_number", 4),
            ("original_text", self.result["claims"][1]["text"]), ("chunk_ids", ["other-source"]),
            ("verdict", "automatic_success"), ("reason", " "),
        ):
            review = deepcopy(self.review)
            review["entries"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                apply_claim_review(self.artifact, review)
        duplicate = deepcopy(self.review)
        duplicate["entries"].append(deepcopy(duplicate["entries"][0]))
        with self.assertRaisesRegex(ValueError, "重复"):
            apply_claim_review(self.artifact, duplicate)
        for field, value in (("reviewer", ""), ("purpose", "automatic_review"), ("schema_version", True),
                             ("source_answer_sha256", "0" * 64)):
            review = deepcopy(self.review)
            review[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                apply_claim_review(self.artifact, review)

    def test_binding_does_not_claim_to_check_the_human_judgment(self):
        review = deepcopy(self.review)
        review["entries"][0].update(verdict="contradicted", reason="An intentionally incorrect human judgment.")
        reviewed = apply_claim_review(self.artifact, review)
        self.assertEqual(reviewed["method"], "manual")
        self.assertEqual(reviewed["claims"][0]["verdict"], "contradicted")
        self.assertEqual(self.artifact["semantic_review"], "pending")


if __name__ == "__main__":
    unittest.main()
