"""单轮输入快照：临时文件真实摘要、完整历史及范围检查，不运行图或模型。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest


from paper_sage.sessions.chat import create_chat_session
from paper_sage.workflows.context import create_run_context, validate_run_context


class WorkflowContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="paper_sage_run_context_")
        self.root = Path(self.temporary.name).resolve()
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        assert self.root.name.startswith("paper_sage_run_context_")
        self.addCleanup(self.temporary.cleanup)
        self.project_id = "practice_project"
        self.documents = []
        self.paths = []
        for document_id in ("paper_a", "paper_b"):
            source = f"data/papers/{self.project_id}/{document_id}/paper.pdf"
            document = {"document_id": document_id, "source": source, "index_version": "v1"}
            self.documents.append(document)
            index_dir = Path("data/indexes") / self.project_id / document_id / "v1"
            # 仅用于内容摘要的固定字节，不声称是可解析的 PDF 或 FAISS 索引。
            for relative, data in ((source, b"fixed-pdf-bytes"),
                                   (index_dir / "metadata.json", b'{"chunks":[]}'),
                                   (index_dir / "index.faiss", b"fixed-index-bytes")):
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                self.paths.append(path)
        self.project_path = self.root / "data/projects" / self.project_id / "project.json"
        self.project_path.parent.mkdir(parents=True, exist_ok=True)
        self.project = {"project_id": self.project_id, "name": "Practice", "documents": self.documents}
        self.project_path.write_text(json.dumps(self.project), encoding="utf-8")
        self.scope = (self.project_id, tuple((d["document_id"], d["index_version"], d["source"])
                                             for d in self.documents))
        self.session = create_chat_session(self.scope)
        self.session["messages"] = [
            {"role": "user", "content": "Prior question"},
            {"role": "assistant", "content": "Prior answer", "answer_artifact": {
                "sources": [{"project_id": self.project_id, **self.documents[0],
                             "pdf_page": 1, "text": "A teaching quote."}],
            }},
        ]
        self.question = "What hardware was used?"
        self.context = create_run_context(self.question, self.session, self.project_id,
                                          self.documents, workspace_root=self.root)

    def check(self, record=None, session=None, documents=None, project_id=None):
        return validate_run_context(
            self.context if record is None else record,
            self.session if session is None else session,
            self.project_id if project_id is None else project_id,
            self.documents if documents is None else documents,
            workspace_root=self.root,
        )

    def test_json_roundtrip_detached_copies_and_order_independent_scope(self):
        original = deepcopy((self.context, self.session, self.documents))
        files = {path: path.read_bytes() for path in [*self.paths, self.project_path]}
        reloaded = json.loads(json.dumps(self.context))
        reordered = deepcopy(self.session)
        reordered["scope_key"] = (self.project_id, tuple(reversed(self.scope[1])))
        checked = self.check(record=reloaded, session=reordered, documents=list(reversed(self.documents)))
        self.assertEqual(checked, self.context)
        checked["session_snapshot"]["session"]["messages"][0]["content"] = "Modified copy"
        checked["input_files"][0]["files"][0]["sha256"] = "0" * 64
        self.assertEqual((self.context, self.session, self.documents), original)
        next_run = create_run_context(self.question, self.session, self.project_id,
                                     self.documents, workspace_root=self.root)
        self.assertNotEqual(next_run["run_id"], self.context["run_id"])
        self.assertEqual(next_run["input_files"], self.context["input_files"])
        self.assertEqual({path: path.read_bytes() for path in files}, files)
        self.assertFalse((self.root / "data/checkpoints").exists())
        self.assertFalse((self.root / "data/sessions").exists())

    def test_same_count_history_or_evidence_changes_and_new_session_are_rejected(self):
        for field in ("text", "evidence", "append", "session_id"):
            changed = deepcopy(self.session)
            if field == "text":
                changed["messages"][0]["content"] = "A different question"
            elif field == "evidence":
                changed["messages"][1]["answer_artifact"]["sources"][0]["text"] = "Changed evidence"
            elif field == "append":
                changed["messages"].extend([{"role": "user", "content": "Next question"},
                                            {"role": "assistant", "content": "Next answer"}])
            else:
                changed["session_id"] = create_chat_session(self.scope)["session_id"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(session=changed)
        self.assertEqual(self.check(), self.context)

    def test_file_changes_at_same_path_and_missing_file_are_rejected(self):
        context_before = deepcopy(self.context)
        for path in self.paths[:3]:
            original = path.read_bytes()
            with self.subTest(file=path.name):
                # 相同文件名、路径和长度；只改变内容。
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                with self.assertRaisesRegex(ValueError, "PDF、索引或元数据"):
                    self.check()
                path.write_bytes(original)
        target = self.paths[0]
        assert target.resolve().is_relative_to(self.root)
        original = target.read_bytes()
        target.unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()
        target.write_bytes(original)
        self.assertEqual(self.context, context_before)
        self.assertEqual(self.check(), context_before)

    def test_scope_and_project_registration_changes_are_rejected(self):
        variants = [
            ("other_project", self.documents),
            (self.project_id, self.documents[:1]),
            (self.project_id, [{**self.documents[0], "index_version": "v2"}, self.documents[1]]),
            (self.project_id, [{**self.documents[0], "source": "data/papers/other.pdf"}, self.documents[1]]),
        ]
        for project_id, documents in variants:
            with self.subTest(project=project_id, documents=documents), self.assertRaises(ValueError):
                self.check(project_id=project_id, documents=documents)
        original = self.project_path.read_bytes()
        changed = deepcopy(self.project)
        changed["documents"][0]["file_hash"] = "0" * 64
        self.project_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.check()
        self.project_path.write_bytes(original)
        self.assertEqual(self.check(), self.context)

    def test_malformed_context_and_file_snapshot_cannot_pass(self):
        for field, value in (("schema_version", True), ("schema_version", 2), ("purpose", "other"),
                             ("run_id", "../invalid"), ("question", " "),
                             ("session_snapshot", None), ("documents", []), ("input_files", [])):
            changed = deepcopy(self.context)
            changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.check(record=changed)
        changed = deepcopy(self.context)
        changed["extra"] = "unexpected"
        with self.assertRaises(ValueError):
            self.check(record=changed)


if __name__ == "__main__":
    unittest.main()
