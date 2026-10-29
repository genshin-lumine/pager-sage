"""对比工具的单篇资源绑定与记录隔离；真实 FAISS 配合控制模型，无外部请求。"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import faiss
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]

from paper_sage.comparison.inputs import build_comparison_tasks
from paper_sage.comparison.retrieval import collect_comparison_evidence, validate_comparison_result
from paper_sage.comparison.tools import create_comparison_tools


class ComparisonToolsTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {"document_id": doc_id, "source": f"{doc_id}.pdf", "index_version": "v1"}
            for doc_id in ("paper_a", "paper_b")
        ]
        self.resources = {}
        for document in self.documents:
            doc_id = document["document_id"]
            chunks = [{
                "project_id": "test_project", "document_id": doc_id,
                "source": document["source"], "pdf_page": number,
                "chunk_id": f"test_project:{doc_id}:p{number}:c1", "text": text,
            } for number, text in enumerate([f"Training dataset for {doc_id}.", f"Experimental results for {doc_id}."], 1)]
            index = faiss.IndexFlatIP(2)
            index.add(np.asarray([[1.0, 0.0], [0.8, 0.6]], dtype="float32"))
            self.resources[doc_id] = {
                "project_id": "test_project", "document": deepcopy(document), "index": index,
                "metadata": {"dimension": 2, "chunks": chunks, "embedding_config": {
                    "model_name": "controlled-test-model", "query_prefix": "query: ",
                    "passage_prefix": "passage: ", "normalize_embeddings": True,
                }},
            }
        self.embedding = Mock()
        self.embedding.encode.return_value = np.asarray([[1.0, 0.0]], dtype="float32")
        self.reranker = Mock()
        self.reranker.predict.side_effect = lambda pairs, **kwargs: np.asarray(
            [2.0 if "Training" in text else 1.0 for _, text in pairs]
        )

    def test_shared_models_keep_four_tasks_and_six_stages_in_each_document(self):
        before = {key: deepcopy(value["metadata"]) for key, value in self.resources.items()}
        tools = create_comparison_tools(self.resources, self.embedding, self.reranker)
        self.embedding.encode.assert_not_called()
        self.reranker.predict.assert_not_called()
        self.assertIsNot(tools["paper_a"], tools["paper_b"])
        tasks = build_comparison_tasks("test_project", self.documents, ["训练数据", "实验结果"])
        records = collect_comparison_evidence(tasks, tools)
        self.assertEqual([record["task"] for record in records], tasks)
        self.assertEqual(self.embedding.encode.call_count, 4)
        self.assertEqual(self.reranker.predict.call_count, 4)
        for record in records:
            artifact = record["retrieval_artifact"]
            self.assertEqual(len(artifact["retrieval_results"]), 2)
            stages = artifact["retrieval_trace"]["stages"]
            self.assertEqual([stage["stage"] for stage in stages], ["dense", "bm25", "merge", "rrf", "rerank", "select"])
            for stage in stages:
                for result in stage["results"]:
                    self.assertEqual(result["chunk"]["document_id"], record["task"]["document_id"])
        records[0]["retrieval_artifact"]["retrieval_results"][0]["chunk"]["text"] = "changed by caller"
        self.assertNotEqual(records[1]["retrieval_artifact"]["retrieval_results"][0]["chunk"]["text"], "changed by caller")
        self.assertEqual({key: value["metadata"] for key, value in self.resources.items()}, before)

    def test_rejects_foreign_document_or_project_before_using_models(self):
        for field in ("document_id", "project_id", "source"):
            with self.subTest(field=field):
                resources = deepcopy(self.resources)
                resources["paper_b"]["metadata"]["chunks"][0][field] = "foreign"
                with self.assertRaisesRegex(ValueError, "混入"):
                    create_comparison_tools(resources, self.embedding, self.reranker)
        resources = deepcopy(self.resources)
        resources["paper_b"]["project_id"] = "other_project"
        for chunk in resources["paper_b"]["metadata"]["chunks"]:
            chunk["project_id"] = "other_project"
        with self.assertRaisesRegex(ValueError, "同一项目"):
            create_comparison_tools(resources, self.embedding, self.reranker)
        self.embedding.encode.assert_not_called()

    def test_rejects_encoding_mismatch_and_unpaired_index(self):
        resources = deepcopy(self.resources)
        resources["paper_b"]["metadata"]["embedding_config"]["model_name"] = "other-model"
        with self.assertRaisesRegex(ValueError, "编码配置一致"):
            create_comparison_tools(resources, self.embedding, self.reranker)
        resources = deepcopy(self.resources)
        resources["paper_b"]["index"] = faiss.IndexFlatIP(2)
        with self.assertRaisesRegex(ValueError, "不配套"):
            create_comparison_tools(resources, self.embedding, self.reranker)
        self.embedding.encode.assert_not_called()

    def test_rejects_foreign_candidate_in_intermediate_stage(self):
        tools = create_comparison_tools(self.resources, self.embedding, self.reranker)
        task = build_comparison_tasks("test_project", self.documents, ["训练数据"])[0]
        tool_call = {"name": tools["paper_a"].name, "args": {"question": task["question"]},
                     "id": "test_call", "type": "tool_call"}
        message = tools["paper_a"].invoke(tool_call)
        message.artifact["retrieval_trace"]["stages"][0]["results"][0]["chunk"]["project_id"] = "foreign"
        with self.assertRaisesRegex(ValueError, "不属于当前任务"):
            validate_comparison_result(task, message, tool_call)


if __name__ == "__main__":
    unittest.main()
