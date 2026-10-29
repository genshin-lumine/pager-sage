"""真实磁盘索引与受控模型验证服务绑定、延后加载、报告和输入一致性。"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


import faiss
import numpy as np
from langchain_core.messages import AIMessage

from paper_sage.comparison.report import format_paper_comparison, validate_comparison_artifact
from paper_sage.comparison.model import LEGACY_COMPARISON_PROMPT_VERSION, build_comparison_messages
from paper_sage.comparison.review import comparison_digest
from paper_sage.comparison.service import ComparisonServiceError, create_comparison_runner, load_saved_comparison
from paper_sage.retrieval.index_store import save_index
import test_comparison_runner as runner_fixtures


class ComparisonServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="paper-sage-comparison-service-")
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith("paper-sage-comparison-service-"))
        self.addCleanup(self.temporary.cleanup)
        self.fixture = runner_fixtures.ComparisonRunnerTests()
        self.fixture.setUp()
        self.model = self.fixture.model
        self.documents = [{"document_id": doc_id, "source": f"data/papers/{doc_id}.pdf", "index_version": "v1"}
                          for doc_id in ("a", "b")]
        self.project = {"project_id": "project", "name": "Test", "documents": deepcopy(self.documents)}
        self.project_path = self.root / "data/projects/project/project.json"
        self.project_path.parent.mkdir(parents=True)
        self.project_path.write_text(json.dumps(self.project), encoding="utf-8")
        for doc in self.documents:
            path = self.root / doc["source"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"Synthetic PDF bytes for hashing only")
            chunks = [{"project_id": "project", "document_id": doc["document_id"], "source": doc["source"],
                       "chunk_id": f"project:{doc['document_id']}:p{number}", "pdf_page": number, "text": text}
                      for number, text in enumerate(("Training datasets and corpus sizes.", "Test scores and evaluation results."), 1)]
            index = faiss.IndexFlatIP(2)
            index.add(np.asarray([[1.0, 0.0], [0.8, 0.6]], dtype="float32"))
            save_index(index, chunks, self.root / f"data/indexes/project/{doc['document_id']}/v1", {
                "model_name": "controlled-embedding", "normalize_embeddings": True,
                "query_prefix": "query: ", "passage_prefix": "passage: ",
            })
        self.embedding = Mock()
        self.embedding.encode.side_effect = lambda texts, **kwargs: np.tile(np.asarray([[1.0, 0.0]], dtype="float32"), (len(texts), 1))
        self.reranker = Mock()
        self.reranker.predict.side_effect = lambda pairs, **kwargs: np.asarray([2.0 - i for i in range(len(pairs))])
        self.embedding_loader = Mock(return_value=self.embedding)
        self.rerank_loader = Mock(return_value=self.reranker)

    def factory(self):
        return create_comparison_runner(self.project, self.documents, self.model,
                                        embedding_model_loader=self.embedding_loader, rerank_model_loader=self.rerank_loader,
                                        workspace_root=self.root)

    def reports(self):
        return [(path, json.loads(path.read_text(encoding="utf-8")))
                for path in (self.root / "data/evaluations").glob("day34_comparison_run_*.json")]

    def test_factory_is_lazy_and_real_indexes_stay_scoped_with_one_shared_model_load(self):
        runner = self.factory()
        self.model.bind.assert_not_called()
        self.embedding_loader.assert_not_called()
        self.assertEqual(self.reports(), [])
        artifact = runner(self.fixture.question)
        self.embedding_loader.assert_called_once_with("controlled-embedding")
        self.rerank_loader.assert_called_once_with()
        self.assertEqual(self.embedding.encode.call_count, 4)
        self.assertEqual(self.reranker.predict.call_count, 4)
        self.assertEqual(self.fixture.events, ["plan", "generate"])
        self.assertEqual(len(artifact["records"]), 4)
        for record in artifact["records"]:
            stages = record["retrieval_artifact"]["retrieval_trace"]["stages"]
            self.assertEqual([stage["stage"] for stage in stages], ["dense", "bm25", "merge", "rrf", "rerank", "select"])
            for stage in stages:
                for hit in stage["results"]:
                    self.assertEqual(hit["chunk"]["document_id"], record["task"]["document_id"])
        path, report = self.reports()[0]
        self.assertEqual(report["status"], "completed_comparison")
        self.assertTrue(report["accepted"])
        self.assertEqual(report["semantic_review"], "pending")
        self.assertEqual(report["result"], artifact)
        self.assertEqual(len(report["input_files"]), 2)
        self.assertEqual(artifact["report_path"], path.relative_to(self.root).as_posix())

    def test_repaired_response_can_be_saved_reloaded_and_repair_tampering_is_rejected(self):
        original_respond = self.fixture.respond

        def respond(messages):
            response = original_respond(messages)
            if "groups" in json.loads(messages[-1].content):
                response.content += "]}"
            return response

        self.model.bind.return_value.invoke.side_effect = respond
        artifact = self.factory()(self.fixture.question)
        validate_comparison_artifact(artifact)
        record = artifact["generation_record"]
        self.assertTrue(record["raw_response"].endswith("]}"))
        self.assertEqual(record["format_repair"]["removed_suffix"], "]}")
        calls = self.model.bind.return_value.invoke.call_count
        self.assertEqual(calls, 2)  # 计划一次，生成一次；修复不请求模型。
        loaded = load_saved_comparison(artifact["report_path"], self.project, self.documents, workspace_root=self.root)
        self.assertEqual(loaded["generation_record"], record)
        self.assertEqual(self.model.bind.return_value.invoke.call_count, calls)
        altered = deepcopy(artifact)
        altered["generation_record"]["format_repair"]["parsed_response"] = "{}"
        with self.assertRaisesRegex(ValueError, "修复记录"):
            validate_comparison_artifact(altered)

    def test_legacy_prompt_keeps_saved_report_readable_and_rejects_mismatched_messages(self):
        artifact = self.factory()("Compare training data and results")
        path = self.root / artifact["report_path"]
        report = json.loads(path.read_text(encoding="utf-8"))
        record = report["result"]["generation_record"]
        current_messages = deepcopy(record["request_messages"])
        record["prompt_version"] = LEGACY_COMPARISON_PROMPT_VERSION
        record["request_messages"] = [
            {"role": message.type, "content": message.content}
            for message in build_comparison_messages(record["input"], prompt_version=LEGACY_COMPARISON_PROMPT_VERSION)
        ]
        path.write_text(json.dumps(report), encoding="utf-8")
        before = path.read_bytes()
        call_count = self.model.bind.return_value.invoke.call_count
        loaded = load_saved_comparison(artifact["report_path"], self.project, self.documents, workspace_root=self.root)
        self.assertEqual(loaded["generation_record"]["prompt_version"], LEGACY_COMPARISON_PROMPT_VERSION)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.model.bind.return_value.invoke.call_count, call_count)
        record["request_messages"] = current_messages
        with self.assertRaisesRegex(ValueError, "请求消息"):
            validate_comparison_artifact(report["result"])
        record["prompt_version"] = "unknown"
        with self.assertRaisesRegex(ValueError, "生成记录"):
            validate_comparison_artifact(report["result"])

    def test_clarification_skips_indexes_and_models(self):
        self.fixture.plan = {"dimensions": [], "clarification_question": "比较哪些方面？"}
        with patch('paper_sage.comparison.service.load_comparison_indexes', side_effect=AssertionError("Must not load indexes")) as load:
            artifact = self.factory()(self.fixture.question)
        load.assert_not_called()
        self.embedding_loader.assert_not_called()
        self.rerank_loader.assert_not_called()
        self.assertEqual(self.fixture.events, ["plan"])
        self.assertEqual(artifact["response_kind"], "clarify")
        self.assertEqual(artifact["input_files"], [])
        self.assertEqual(self.reports()[0][1]["status"], "completed_clarification")

    def test_plan_reuse_still_retrieves_and_does_not_overwrite_old_report(self):
        runner = self.factory()
        first = runner(self.fixture.question)
        path = self.root / first["report_path"]
        before = path.read_bytes()
        self.fixture.events.clear()
        second = runner(self.fixture.question, plan_record=first["plan_record"])
        self.assertEqual(self.fixture.events, ["generate"])
        self.assertTrue(second["plan_reused"])
        self.assertEqual(self.embedding.encode.call_count, 8)
        self.assertNotEqual(first["report_path"], second["report_path"])
        self.assertEqual(path.read_bytes(), before)

    def test_generation_failure_preserves_raw_response_and_retrieval(self):
        original_respond = self.fixture.respond
        def respond(messages):
            if "groups" in json.loads(messages[-1].content):
                return AIMessage(content="{", response_metadata={"finish_reason": "stop"})
            return original_respond(messages)
        self.model.bind.return_value.invoke.side_effect = respond
        with self.assertRaises(ComparisonServiceError) as caught:
            self.factory()(self.fixture.question)
        report = json.loads((self.root / caught.exception.report_path).read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["accepted"])
        self.assertEqual(report["result"]["generation_record"]["raw_response"], "{")
        self.assertEqual(len(report["result"]["records"]), 4)
        self.assertEqual(self.model.bind.return_value.invoke.call_count, 2)

    def test_files_changed_during_generation_are_not_accepted(self):
        original_respond = self.fixture.respond
        def respond(messages):
            if "groups" in json.loads(messages[-1].content):
                (self.root / self.documents[0]["source"]).write_bytes(b"Changed during generation")
            return original_respond(messages)
        self.model.bind.return_value.invoke.side_effect = respond
        with self.assertRaises(ComparisonServiceError):
            self.factory()(self.fixture.question)
        report = self.reports()[0][1]
        self.assertFalse(report["accepted"])
        self.assertEqual(report["result"]["status"], "failed")
        self.assertEqual(report["result"]["failed_stage"], "verify_inputs")

    def test_changed_project_and_mismatched_encoding_stop_before_model_loading(self):
        runner = self.factory()
        changed = deepcopy(self.project)
        changed["documents"][0]["index_version"] = "v2"
        self.project_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(ValueError):
            runner(self.fixture.question)
        self.model.bind.assert_not_called()
        self.assertEqual(self.reports(), [])
        self.project_path.write_text(json.dumps(self.project), encoding="utf-8")
        metadata_path = self.root / "data/indexes/project/b/v1/metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["embedding_config"]["model_name"] = "another-model"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ComparisonServiceError):
            runner(self.fixture.question)
        self.embedding_loader.assert_not_called()
        self.rerank_loader.assert_not_called()

    def write_review(self, artifact):
        result = artifact["generation_record"]["comparison"]
        finding = result["comparisons"][0]["papers"][0]["findings"][0]
        report_path = self.root / artifact["report_path"]
        review = {
            "schema_version": 1, "purpose": "manual_comparison_review", "reviewer": "test",
            "source_report": artifact["report_path"],
            "source_report_sha256": sha256(report_path.read_bytes()).hexdigest(),
            "source_comparison_sha256": comparison_digest(result),
            "corrections": [{"path": ["comparisons", 0, "papers", 0, "findings", 0, "text"],
                             "original_text": finding["text"], "replacement_text": "Reviewed test finding",
                             "reason": "Exercise separate review storage", "evidence_chunk_ids": finding["chunk_ids"]}],
        }
        path = report_path.with_name("review.json")
        path.write_text(json.dumps(review), encoding="utf-8")
        return path, review

    def test_readonly_reload_and_review_keep_original_without_model_calls_or_writes(self):
        original = self.factory()(self.fixture.question)
        review_path, _ = self.write_review(original)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        calls = (self.model.bind.return_value.invoke.call_count, self.embedding.encode.call_count,
                 self.reranker.predict.call_count)
        raw = load_saved_comparison(original["report_path"], self.project, self.documents, workspace_root=self.root)
        reviewed = load_saved_comparison(original["report_path"], self.project, self.documents,
                                         workspace_root=self.root, review_path=review_path)
        self.assertNotIn("content_review", raw)
        self.assertEqual(raw["generation_record"], original["generation_record"])
        self.assertEqual(reviewed["generation_record"], original["generation_record"])
        self.assertIn("Reviewed test finding", format_paper_comparison(reviewed))
        self.assertNotIn("Reviewed test finding", format_paper_comparison(raw))
        self.assertEqual(calls, (self.model.bind.return_value.invoke.call_count, self.embedding.encode.call_count,
                                self.reranker.predict.call_count))
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_review_rejects_wrong_report_binding_and_modified_derived_content(self):
        original = self.factory()(self.fixture.question)
        path, review = self.write_review(original)
        for field, value in (("source_report", "another.json"), ("source_report_sha256", "0" * 64)):
            with self.subTest(field=field):
                wrong = {**review, field: value}
                path.write_text(json.dumps(wrong), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "修订记录不对应"):
                    load_saved_comparison(original["report_path"], self.project, self.documents,
                                          workspace_root=self.root, review_path=path)
        path.write_text(json.dumps(review), encoding="utf-8")
        reviewed = load_saved_comparison(original["report_path"], self.project, self.documents,
                                         workspace_root=self.root, review_path=path)
        for field in ("comparison", "resolved_comparison"):
            with self.subTest(field=field):
                changed = deepcopy(reviewed)
                changed["content_review"][field]["comparisons"][0]["comparison_limit"] = "Unrecorded change"
                with self.assertRaisesRegex(ValueError, "修订正文或来源"):
                    format_paper_comparison(changed)

    def test_reload_rejects_stale_files_wrong_scope_and_unfinished_report(self):
        artifact = self.factory()(self.fixture.question)
        report_path = self.root / artifact["report_path"]
        original_bytes = report_path.read_bytes()
        report = json.loads(original_bytes)
        for field, value in (("accepted", False), ("status", "failed"), ("project_id", "other")):
            with self.subTest(field=field):
                report_path.write_text(json.dumps({**report, field: value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_saved_comparison(artifact["report_path"], self.project, self.documents, workspace_root=self.root)
        report_path.write_bytes(original_bytes)
        pdf_path = self.root / self.documents[0]["source"]
        pdf_path.write_bytes(b"Changed after the report was completed")
        with self.assertRaisesRegex(ValueError, "当前 PDF 或索引"):
            load_saved_comparison(artifact["report_path"], self.project, self.documents, workspace_root=self.root)

    def test_reload_checks_intermediate_chunks_against_current_index(self):
        artifact = self.factory()(self.fixture.question)
        path = self.root / artifact["report_path"]
        report = json.loads(path.read_bytes())
        # 中间候选未进入生成输入；仅校验最终正文及文档归属无法发现此处篡改。
        retrieval = report["result"]["records"][0]["retrieval_artifact"]
        retrieval["retrieval_channels"]["dense"][0]["chunk"]["text"] = "Changed source text"
        validate_comparison_artifact(report["result"])
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "报告片段与当前索引"):
            load_saved_comparison(artifact["report_path"], self.project, self.documents, workspace_root=self.root)


if __name__ == "__main__":
    unittest.main()
