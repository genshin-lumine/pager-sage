"""聊天存档的完整往返、范围隔离、损坏文件与原子写入；无模型调用。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


from paper_sage.sessions.chat import build_history_messages, create_chat_session
from paper_sage.sessions.store import (load_chat_session, make_session_snapshot, restore_session_snapshot,
                           save_chat_session, list_chat_sessions)


def make_fixture():
    scope = ("practice_project", (("paper_a", "v1", "practice/paper_a.pdf"),))
    session = create_chat_session(scope)
    session["messages"] = [
        {"role": "user", "content": "教学问题"},
        {
            "role": "assistant", "content": "教学回答", "response_kind": "answer",
            "answer_artifact": {
                "teaching_fixture": True,
                "sources": [{"project_id": "practice_project", "document_id": "paper_a",
                             "source": "practice/paper_a.pdf", "pdf_page": 7,
                             "text": "Teaching evidence."}],
            },
            "workflow_trace": {"scope_key": scope, "path": ["start", "generate_answer"]},
            "retrieval_trace": {"scope_key": scope, "stages": []},
            "qa_retry_artifact": {"scope_key": scope, "attempt_records": []},
        },
    ]
    return scope, session


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="paper-sage-session-")
        self.root = Path(self.temporary.name).resolve()
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        assert self.root.name.startswith("paper-sage-session-")
        self.addCleanup(self.temporary.cleanup)
        self.scope, self.session = make_fixture()

    def save(self, session=None):
        return save_chat_session(self.session if session is None else session, workspace_root=self.root)

    def load(self, scope=None):
        return load_chat_session(self.scope[0], self.session["session_id"],
                                 self.scope if scope is None else scope, workspace_root=self.root)

    def test_complete_roundtrip_preserves_records_and_detaches_mutable_data(self):
        before = deepcopy(self.session)
        path = self.save()
        file_before = path.read_bytes()
        self.assertIsInstance(json.loads(file_before)["session"]["scope_key"], list)
        restored = self.load()
        self.assertEqual(restored, before)
        self.assertIsInstance(restored["scope_key"], tuple)
        for field in ("workflow_trace", "retrieval_trace", "qa_retry_artifact"):
            self.assertIsInstance(restored["messages"][-1][field]["scope_key"], tuple)
        history = build_history_messages(restored, self.scope)
        self.assertTrue(all(set(item) == {"role", "content"} for item in history))
        restored["messages"][-1]["answer_artifact"]["sources"][0]["text"] = "Changed copy"
        self.assertEqual(self.session, before)
        self.assertEqual(self.load(), before)
        self.assertEqual(path.read_bytes(), file_before)

    def test_scope_change_rejected_and_document_order_is_not_a_new_scope(self):
        self.save()
        for scope in [
            ("another_project", self.scope[1]),
            (self.scope[0], ()),
            (self.scope[0], (("paper_b", "v1", "practice/paper_b.pdf"),)),
            (self.scope[0], (("paper_a", "v2", "practice/paper_a.pdf"),)),
            (self.scope[0], (("paper_a", "v1", "practice/changed.pdf"),)),
        ]:
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "不匹配"):
                self.load(scope)
        scope = (self.scope[0], self.scope[1] + (("paper_b", "v1", "practice/paper_b.pdf"),))
        session = create_chat_session(scope)
        snapshot = make_session_snapshot(session)
        self.assertEqual(restore_session_snapshot(snapshot, (scope[0], tuple(reversed(scope[1])))), session)

    def test_invalid_or_incomplete_messages_never_create_a_file(self):
        variants = []
        incomplete = deepcopy(self.session)
        incomplete["messages"].pop()
        variants.append(incomplete)
        for field, value in [("role", "system"), ("content", " "), ("response_kind", "error")]:
            changed = deepcopy(self.session)
            changed["messages"][-1][field] = value
            variants.append(changed)
        for value in (b"%PDF-test", float("nan"), object()):
            changed = deepcopy(self.session)
            changed["messages"][-1]["extra"] = value
            variants.append(changed)
        for session in variants:
            with self.subTest(session=session), self.assertRaises(ValueError):
                self.save(session)
        self.assertFalse((self.root / "data").exists())

    def test_nested_scope_and_source_mismatch_rejected_before_restore(self):
        snapshot = make_session_snapshot(self.session)
        for field, value in [("project_id", "other"), ("document_id", "paper_b"),
                             ("source", "wrong.pdf"), ("index_version", "v2")]:
            changed = deepcopy(snapshot)
            changed["session"]["messages"][-1]["answer_artifact"]["sources"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                restore_session_snapshot(changed, self.scope)
        changed = deepcopy(snapshot)
        changed["session"]["messages"][-1]["workflow_trace"]["scope_key"][0] = "other"
        with self.assertRaisesRegex(ValueError, "范围"):
            restore_session_snapshot(changed, self.scope)
        changed = deepcopy(snapshot)
        changed["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "版本"):
            restore_session_snapshot(changed, self.scope)

    def test_missing_corrupt_and_misidentified_files_fail_without_writing(self):
        with self.assertRaises(FileNotFoundError):
            self.load()
        self.assertFalse((self.root / "data").exists())
        path = self.save()
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        snapshot["session"]["session_id"] = create_chat_session(self.scope)["session_id"]
        corrupt_inputs = ["{", '{"schema_version":1,"schema_version":1}', json.dumps(snapshot)]
        for text in corrupt_inputs:
            path.write_text(text, encoding="utf-8")
            before = path.read_bytes()
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load()
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_paths_and_linked_storage_rejected(self):
        with self.assertRaisesRegex(ValueError, "编号"):
            load_chat_session(self.scope[0], "../outside", self.scope, workspace_root=self.root)
        for project_id in ("../outside", "CON", "project."):
            scope = (project_id, ())
            with self.subTest(project_id=project_id), self.assertRaisesRegex(ValueError, "目录"):
                self.save(create_chat_session(scope))
        linked = self.root / "data" / "sessions"
        original = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda item: item == linked or original(item)):
            with self.assertRaisesRegex(ValueError, "链接"):
                self.save()
        self.assertFalse((self.root / "data").exists())

    def test_failed_replacement_and_scope_rebinding_keep_previous_snapshot(self):
        path = self.save()
        before = path.read_bytes()
        changed = deepcopy(self.session)
        changed["messages"].extend([{"role": "user", "content": "下一问"},
                                    {"role": "assistant", "content": "下一答"}])
        with patch('paper_sage.sessions.store.os.replace', side_effect=OSError("simulated write failure")):
            with self.assertRaisesRegex(OSError, "simulated"):
                self.save(changed)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])
        rebound = deepcopy(self.session)
        rebound["messages"] = []
        rebound["scope_key"] = (self.scope[0], ())
        with self.assertRaisesRegex(ValueError, "不匹配"):
            self.save(rebound)
        self.assertEqual(path.read_bytes(), before)
        self.save(changed)
        self.assertEqual(self.load(), changed)

    def test_listing_does_not_create_directories_or_expose_other_scopes(self):
        self.assertEqual(list_chat_sessions(self.scope, workspace_root=self.root), [])
        self.assertFalse((self.root / "data").exists())
        path = self.save()
        for scope in [("other_project", self.scope[1]), (self.scope[0], ())]:
            other = create_chat_session(scope)
            other["messages"] = [{"role": "user", "content": "Other question"},
                                 {"role": "assistant", "content": "Other reply"}]
            self.save(other)
        corrupt = path.with_name("session_" + "1" * 32 + ".json")
        corrupt.write_text("{", encoding="utf-8")
        before = {p: p.read_bytes() for p in self.root.rglob("*.json")}
        records = list_chat_sessions(self.scope, workspace_root=self.root)
        self.assertEqual([r["session_id"] for r in records], [self.session["session_id"]])
        self.assertEqual(records[0]["turn_count"], 1)
        self.assertEqual({p: p.read_bytes() for p in self.root.rglob("*.json")}, before)


if __name__ == "__main__":
    unittest.main()
