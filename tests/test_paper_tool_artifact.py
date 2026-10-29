"""检索工具的数据传递、范围与调用隔离；真实 FAISS/BM25 配合控制模型。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import faiss
import numpy as np
from langchain_core.messages import ToolMessage

WORKSPACE = Path(__file__).resolve().parents[1]

from paper_sage.retrieval.tools import create_search_tool, EMPTY_REFERENCE_TEXT
from paper_sage.retrieval.context import build_reference_text


class PaperToolArtifactTests(unittest.TestCase):
    def setUp(self):
        texts = [
            "We trained our models on one machine with 8 NVIDIA P100 GPUs.",
            "The encoder is composed of a stack of N = 6 identical layers.",
            "We use sine and cosine functions for positions.",
            "References and background research.",
        ]
        chunks = [{
            "project_id": "test_project", "document_id": "test_document",
            "chunk_id": f"test_project:test_document:p{number}:c1",
            "source": "fixture.pdf", "pdf_page": number, "text": text,
        } for number, text in enumerate(texts, start=1)]
        self.metadata = {
            "embedding_config": {"query_prefix": "query: ", "normalize_embeddings": True},
            "chunks": chunks,
        }
        self.original_metadata = deepcopy(self.metadata)
        self.index = faiss.IndexFlatIP(2)
        self.index.add(np.asarray([[1, 0], [0.8, 0.6], [0.6, 0.8], [0, 1]], dtype="float32"))
        self.embedding = Mock()
        self.embedding.encode.return_value = np.asarray([[1, 0]], dtype="float32")
        self.reranker = Mock()
        score_by_text = dict(zip(texts, [4, 3, 2, 1], strict=True))
        self.reranker.predict.side_effect = lambda pairs, **kwargs: np.asarray([score_by_text[text] for _, text in pairs])
        self.tool = create_search_tool(self.embedding, self.index, self.metadata, self.reranker)

    def invoke(self, tool=None, question="NVIDIA P100", call_id="test_call"):
        return (tool or self.tool).invoke({
            "name": "search_papers", "args": {"question": question},
            "id": call_id, "type": "tool_call",
        })

    def test_full_tool_call_preserves_reference_and_six_stages(self):
        message = self.invoke()
        self.assertIsInstance(message, ToolMessage)
        self.assertEqual(message.tool_call_id, "test_call")
        artifact = message.artifact
        self.assertEqual(message.content, build_reference_text(artifact["retrieval_results"]))
        self.assertEqual(len(artifact["retrieval_results"]), 3)
        self.assertEqual(len(artifact["rerank_results"]), 4)
        stages = artifact["retrieval_trace"]["stages"]
        self.assertEqual([s["stage"] for s in stages], ["dense", "bm25", "merge", "rrf", "rerank", "select"])
        self.assertEqual([s["input_count"] for s in stages], [4, 4, 5, 4, 4, 4])
        self.assertEqual([s["output_count"] for s in stages], [4, 1, 4, 4, 4, 3])
        self.assertEqual(stages[-1]["results"], artifact["retrieval_results"])
        canonical = {chunk["chunk_id"]: chunk for chunk in self.metadata["chunks"]}
        for stage in stages:
            self.assertGreaterEqual(stage["elapsed_seconds"], 0)
            for result in stage["results"]:
                self.assertEqual(result["chunk"], canonical[result["chunk"]["chunk_id"]])
        paths = artifact["retrieval_trace"]["candidate_paths"]
        self.assertEqual(sum(row["final_selected"] for row in paths), 3)
        self.assertTrue(any(row["selection_status"] == "excluded_by_final_top_k" for row in paths))
        self.embedding.encode.assert_called_once()
        self.reranker.predict.assert_called_once()
        self.assertEqual(len(self.reranker.predict.call_args.args[0]), 4)
        self.assertEqual(set(self.tool.args), {"question"})
        self.assertEqual(self.tool.response_format, "content_and_artifact")
        self.assertEqual(json.loads(json.dumps(artifact, allow_nan=False)), artifact)

    def test_arguments_only_call_keeps_text_interface(self):
        text = self.tool.invoke({"question": "NVIDIA P100"})
        message = self.invoke()
        self.assertIsInstance(text, str)
        self.assertEqual(text, message.content)

    def test_call_records_do_not_mutate_index_or_leak_to_next_call(self):
        first = self.invoke()
        first.artifact["retrieval_channels"]["dense"][0]["chunk"]["text"] = "caller edit"
        first.artifact["retrieval_trace"]["stages"].clear()
        before_second = deepcopy(first.artifact)
        self.assertEqual(self.metadata, self.original_metadata)
        second = self.invoke(question="encoder", call_id="second_call")
        self.assertEqual(second.tool_call_id, "second_call")
        self.assertEqual(second.artifact["question"], "encoder")
        self.assertEqual(len(second.artifact["retrieval_trace"]["stages"]), 6)
        self.assertEqual(first.artifact, before_second)
        self.assertEqual(second.content, build_reference_text(second.artifact["retrieval_results"]))
        self.assertEqual(self.metadata, self.original_metadata)

    def test_empty_index_returns_empty_stages_without_model_inference(self):
        empty_tool = create_search_tool(self.embedding, faiss.IndexFlatIP(2), {"chunks": []}, self.reranker)
        message = self.invoke(empty_tool)
        self.assertEqual(message.content, EMPTY_REFERENCE_TEXT)
        self.assertEqual(message.artifact["retrieval_results"], [])
        self.assertEqual(len(message.artifact["retrieval_trace"]["stages"]), 6)
        self.assertTrue(all(stage["input_count"] == stage["output_count"] == 0 for stage in message.artifact["retrieval_trace"]["stages"]))
        self.embedding.encode.assert_not_called()
        self.reranker.predict.assert_not_called()

    def test_foreign_source_fails_before_reranking(self):
        foreign = {**self.metadata["chunks"][0], "source": "outside.pdf"}
        with patch('paper_sage.retrieval.dual.retrieve_dense', return_value=[{"rank": 1, "score": 1.0, "chunk": foreign}]):
            with self.assertRaises(ValueError):
                self.invoke()
        self.reranker.predict.assert_not_called()

    def test_rerank_failure_propagates_and_next_call_has_fresh_trace(self):
        prediction = self.reranker.predict.side_effect
        self.reranker.predict.side_effect = RuntimeError("controlled failure")
        with self.assertRaisesRegex(RuntimeError, "controlled failure"):
            self.invoke()
        self.reranker.predict.side_effect = prediction
        recovered = self.invoke(call_id="recovered")
        self.assertEqual(len(recovered.artifact["retrieval_trace"]["stages"]), 6)
        self.assertEqual(recovered.tool_call_id, "recovered")
        self.assertEqual(self.metadata, self.original_metadata)

    def test_factory_rejects_invalid_configuration_before_inference(self):
        for name, value in (("recall_top_k", True), ("rrf_top_k", 0), ("final_top_k", -1), ("rank_constant", 1.5)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                create_search_tool(self.embedding, self.index, self.metadata, self.reranker, **{name: value})
        self.embedding.encode.assert_not_called()
        self.reranker.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
