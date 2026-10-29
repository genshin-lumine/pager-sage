"""汇总输入的完整性与人工校正来源检查；不调用外部模型。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


from langchain_core.messages import AIMessage
from paper_sage.summarization.inputs import build_summary_batches
from paper_sage.summarization.model import summarize_batch
from paper_sage.summarization.parser import SUMMARY_FIELDS
from paper_sage.summarization.reduce import build_summary_note_batches


class SummaryReduceInputTests(unittest.TestCase):
    def setUp(self):
        self.prepared = build_summary_batches([
            {"document_id": "doc", "source": "paper.pdf", "pdf_page": 1, "text": "A is better than unscaled B."},
            {"document_id": "doc", "source": "paper.pdf", "pdf_page": 2, "text": "References only."},
        ], "project", max_chars=30)
        self.part_id = self.prepared["batches"][0]["parts"][0]["part_id"]
        self.original = {"text": "Unscaled A is better than B.", "part_ids": [self.part_id]}
        summary = {field: [] for field in SUMMARY_FIELDS}
        summary["results"] = [self.original]
        model = Mock()
        model.bind.return_value.invoke.side_effect = [
            AIMessage(content=json.dumps(summary), response_metadata={"finish_reason": "stop"}),
            AIMessage(content=json.dumps({field: [] for field in SUMMARY_FIELDS}), response_metadata={"finish_reason": "stop"}),
        ]
        self.records = [summarize_batch(batch, model) for batch in self.prepared["batches"]]
        self.correction = {
            "note_id": f"{self.records[0]['batch_id']}:results:1", "original": deepcopy(self.original),
            "replacement": {"text": "A outperforms B when B is not scaled.", "part_ids": [self.part_id]},
            "reason": "The modifier applies to B.",
            "evidence": [{"part_id": self.part_id, "quote": "A is better\nthan unscaled B."}],
        }

    def test_review_changes_derived_note_only_and_preserves_empty_batch(self):
        before = deepcopy(self.records)
        notes = build_summary_note_batches(self.prepared, self.records, [self.correction])
        self.assertEqual([len(batch) for batch in notes], [1, 0])
        self.assertEqual(notes[0][0]["text"], self.correction["replacement"]["text"])
        self.assertEqual(notes[0][0]["part_ids"], [self.part_id])
        self.assertEqual(notes[0][0]["note_id"], self.correction["note_id"])
        self.assertEqual(self.records, before)
        notes[0][0]["part_ids"].clear()
        self.assertEqual(self.records, before)

    def test_missing_batch_is_not_treated_as_full_paper(self):
        with self.assertRaisesRegex(ValueError, "仍有未完成批次"):
            build_summary_note_batches(self.prepared, self.records[:1])

    def test_stale_foreign_or_unquoted_correction_is_rejected(self):
        variants = []
        stale = deepcopy(self.correction)
        stale["original"]["text"] = "different output"
        variants.append(stale)
        foreign = deepcopy(self.correction)
        foreign["replacement"]["part_ids"] = ["foreign:p1"]
        variants.append(foreign)
        fabricated = deepcopy(self.correction)
        fabricated["evidence"][0]["quote"] = "This sentence does not exist."
        variants.append(fabricated)
        no_evidence = deepcopy(self.correction)
        no_evidence["evidence"] = []
        variants.append(no_evidence)
        wrong_evidence = deepcopy(self.correction)
        wrong_evidence["evidence"][0]["part_id"] = self.prepared["batches"][1]["parts"][0]["part_id"]
        variants.append(wrong_evidence)
        for correction in variants:
            with self.subTest(correction=correction), self.assertRaises(ValueError):
                build_summary_note_batches(self.prepared, self.records, [correction])

    def test_duplicate_review_and_placeholder_replacement_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "不存在或重复"):
            build_summary_note_batches(self.prepared, self.records, [self.correction, self.correction])
        bad = deepcopy(self.correction)
        bad["replacement"]["text"] = "中文要点"
        with self.assertRaisesRegex(ValueError, "占位文字"):
            build_summary_note_batches(self.prepared, self.records, [bad])


if __name__ == "__main__":
    unittest.main()
