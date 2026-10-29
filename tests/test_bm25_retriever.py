import copy
from pathlib import Path
import sys
import unittest



from paper_sage.retrieval.bm25 import retrieve_bm25


def make_chunk(number, text):
    return {
        "project_id": "test_project",
        "document_id": "test_document",
        "chunk_id": f"test_project:test_document:p{number}:c1",
        "source": "test_source.pdf",
        "pdf_page": number,
        "text": text,
    }


class BM25RetrievalTests(unittest.TestCase):
    def test_ranking_preserves_sources_and_excludes_unmatched_chunks(self):
        chunks = [
            make_chunk(1, "P100"),
            make_chunk(2, "NVIDIA P100"),
            make_chunk(3, "sine cosine"),
        ]
        before = copy.deepcopy(chunks)
        results = retrieve_bm25("NVIDIA P100", chunks, top_k=10)
        self.assertEqual([item["rank"] for item in results], [1, 2])
        self.assertEqual([item["chunk"] for item in results], [chunks[1], chunks[0]])
        self.assertGreater(results[0]["score"], results[1]["score"])
        self.assertEqual(retrieve_bm25("NVIDIA P100", chunks, top_k=1), results[:1])
        self.assertEqual(chunks, before)

    def test_empty_or_unmatched_terms_return_no_candidates(self):
        chunks = [make_chunk(1, "NVIDIA P100")]
        for question in ["编码器有几层", "!!!", "unseenword"]:
            with self.subTest(question=question):
                self.assertEqual(retrieve_bm25(question, chunks), [])
        self.assertEqual(retrieve_bm25("P100", []), [])
        self.assertEqual(retrieve_bm25("P100", [make_chunk(1, "   ")]), [])

    def test_invalid_question_or_limit_is_rejected(self):
        chunks = [make_chunk(1, "P100")]
        with self.assertRaises(ValueError):
            retrieve_bm25("   ", chunks)
        for top_k in [0, -1, True, 1.5, "3"]:
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                retrieve_bm25("P100", chunks, top_k=top_k)

    def test_zero_and_negative_scores_can_still_have_term_matches(self):
        target = make_chunk(1, "P100")
        results = retrieve_bm25("P100", [target, make_chunk(2, "encoder")])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk"], target)
        self.assertEqual(results[0]["score"], 0.0)

        results = retrieve_bm25("P100", [target])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk"], target)
        self.assertLess(results[0]["score"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
