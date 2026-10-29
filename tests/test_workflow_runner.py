"""持久化后端、只读恢复和按运行编号写回一次；仅使用临时数据与固定模型。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_workflow_context as context_tests
from paper_sage.sessions.chat import create_chat_session
from paper_sage.workflows.context import create_run_context
from paper_sage.workflows.runner import start_workflow_run, resume_workflow_run, read_workflow_run, WorkflowRunError
from paper_sage.workflows.store import create_run_storage, load_run_record, list_workflow_runs, lock_workflow_run, run_paths
from workflow_run_fixtures import make_fixed_resources


class WorkflowRunnerTests(unittest.TestCase):
    def setUp(self):
        context_tests.WorkflowContextTests.setUp(self)
        # 本测试仅使用一篇；先前 fixture 的两条历史不是完整工作台响应，所以从空会话开始。
        self.documents = self.documents[:1]
        self.scope = (self.project_id, self.scope[1][:1])
        self.session = create_chat_session(self.scope)
        self.chunk = {"project_id": self.project_id, "document_id": "paper_a",
                      "source": self.documents[0]["source"], "pdf_page": 1,
                      "chunk_id": "practice_project:paper_a:p1:c1", "text": "Fixed teaching evidence."}
        self.paths[1].write_text(json.dumps({"chunks": [self.chunk]}), encoding="utf-8")

    def start(self, session=None, *, pause=True, route="qa", error=None):
        model, tool, calls = make_fixed_resources(self.question, self.chunk, route=route, answer_error=error)
        outcome = start_workflow_run(self.question, self.session if session is None else session,
            self.project_id, self.documents, model, tool, pause_before_answer=pause, workspace_root=self.root)
        return outcome, calls

    def read(self, run_id, session=None):
        return read_workflow_run(self.project_id, run_id, self.session if session is None else session,
                                 self.documents, workspace_root=self.root)

    def resume(self, run_id, session=None, **kwargs):
        model, tool, calls = make_fixed_resources(self.question, self.chunk)
        outcome = resume_workflow_run(self.project_id, run_id, self.session if session is None else session,
                                      self.documents, model, tool, workspace_root=self.root, **kwargs)
        return outcome, calls

    def test_pause_resume_read_and_duplicate_application(self):
        before = deepcopy(self.session)
        paused, calls = self.start()
        self.assertEqual((paused["status"], paused["next"]), ("paused", ("generate_answer",)))
        self.assertEqual(calls, {"classify": 1, "request_tool": 1, "search": 1, "answer": 0})
        run_id = paused["run_id"]
        self.assertEqual(self.read(run_id)["status"], "paused")
        complete, resumed_calls = self.resume(run_id)
        self.assertEqual(resumed_calls, {"classify": 0, "request_tool": 0, "search": 0, "answer": 1})
        self.assertTrue(complete["added"])
        self.assertEqual(len(complete["session"]["messages"]), 2)
        message = complete["session"]["messages"][-1]
        self.assertEqual(message["workflow_trace"]["path"], ["start", "classify_request", "request_tool",
            "begin_attempt", "execute_tool", "generate_answer", "decide_next", "finish_run"])
        self.assertEqual(message["workflow_run_id"], run_id)
        again = self.read(run_id, complete["session"])
        self.assertFalse(again["added"])
        self.assertEqual(again["session"], complete["session"])
        replay, counts = self.resume(run_id, complete["session"])
        self.assertEqual(counts, {"classify": 0, "request_tool": 0, "search": 0, "answer": 0})
        self.assertFalse(replay["added"])
        self.assertEqual(self.session, before)

    def test_two_turns_keep_history_and_old_completed_run_is_not_appended_again(self):
        first, _ = self.start(pause=False)
        second, _ = self.start(first["session"], pause=False)
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(len(second["session"]["messages"]), 4)
        again = self.read(first["run_id"], second["session"])
        self.assertFalse(again["added"])
        self.assertEqual(again["session"], second["session"])
        changed = deepcopy(first["session"])
        changed["messages"][-1]["content"] += " changed"
        with self.assertRaises(ValueError):
            self.read(first["run_id"], changed)

    def test_changed_session_or_files_block_before_new_model_calls(self):
        paused, _ = self.start()
        model, tool, calls = make_fixed_resources(self.question, self.chunk)
        for session in (create_chat_session(self.scope),
                        {**self.session, "messages": [{"role": "user", "content": "New"},
                                                     {"role": "assistant", "content": "Different"}]}):
            with self.assertRaises(ValueError):
                resume_workflow_run(self.project_id, paused["run_id"], session, self.documents,
                                    model, tool, workspace_root=self.root)
        self.paths[0].write_bytes(b"changed-pdf-content")
        with self.assertRaises(ValueError):
            resume_workflow_run(self.project_id, paused["run_id"], self.session, self.documents,
                                model, tool, workspace_root=self.root)
        self.assertEqual(calls, {"classify": 0, "request_tool": 0, "search": 0, "answer": 0})

    def test_failed_answer_does_not_append_and_requires_explicit_retry(self):
        with self.assertRaises(WorkflowRunError) as caught:
            self.start(pause=False, error=TimeoutError("fixed failure"))
        run_id = caught.exception.run_id
        failed = self.read(run_id)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["session"], self.session)
        with self.assertRaisesRegex(ValueError, "明确"):
            self.resume(run_id)
        complete, counts = self.resume(run_id, retry_failed=True)
        self.assertEqual(counts, {"classify": 0, "request_tool": 0, "search": 0, "answer": 1})
        self.assertEqual(complete["status"], "completed")

    def test_clarification_complete_and_not_started_run_can_be_read_without_models(self):
        completed, counts = self.start(pause=False, route="clarify")
        self.assertEqual(counts, {"classify": 1, "request_tool": 0, "search": 0, "answer": 0})
        self.assertEqual(self.read(completed["run_id"])["session"], completed["session"])
        context = create_run_context(self.question, self.session, self.project_id, self.documents, workspace_root=self.root)
        paths = create_run_storage(context, workspace_root=self.root)
        self.assertEqual(self.read(context["run_id"])["status"], "not_started")
        self.assertFalse(paths["database"].exists())
        resumed, counts = self.resume(context["run_id"])
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(counts, {"classify": 1, "request_tool": 1, "search": 1, "answer": 1})

    def test_listing_no_overwrite_corrupt_input_lock_and_path_checks(self):
        self.assertEqual(list_workflow_runs(self.scope, workspace_root=self.root), [])
        self.assertFalse((self.root / "data/checkpoints").exists())
        paused, _ = self.start()
        run_id = paused["run_id"]
        record = load_run_record(self.project_id, run_id, workspace_root=self.root)
        paths = run_paths(self.project_id, run_id, workspace_root=self.root)
        before = paths["context"].read_bytes()
        with self.assertRaises(FileExistsError):
            create_run_storage(record["context"], workspace_root=self.root)
        self.assertEqual(paths["context"].read_bytes(), before)
        self.assertEqual(len(list_workflow_runs(self.scope, workspace_root=self.root)), 1)
        with lock_workflow_run(self.project_id, run_id, workspace_root=self.root):
            with self.assertRaisesRegex(RuntimeError, "另一个进程"):
                self.read(run_id)
        self.assertEqual(self.read(run_id)["status"], "paused")
        changed = deepcopy(record)
        changed["context"]["question"] = "Another question?"
        paths["context"].write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.read(run_id)
        paths["context"].write_bytes(b'{"broken":')
        self.assertEqual(list_workflow_runs(self.scope, workspace_root=self.root), [])
        with self.assertRaises(ValueError):
            run_paths("../outside", run_id, workspace_root=self.root)

    def test_input_write_failure_never_calls_model(self):
        model, tool, calls = make_fixed_resources(self.question, self.chunk)
        with patch('paper_sage.workflows.store.os.replace', side_effect=OSError("fixed disk failure")):
            with self.assertRaises(OSError):
                start_workflow_run(self.question, self.session, self.project_id, self.documents,
                                   model, tool, workspace_root=self.root)
        self.assertEqual(calls, {"classify": 0, "request_tool": 0, "search": 0, "answer": 0})
        self.assertEqual(list_workflow_runs(self.scope, workspace_root=self.root), [])

    def test_legacy_history_is_rejected_before_creating_run_or_calling_model(self):
        session = deepcopy(self.session)
        session["messages"] = [{"role": "user", "content": "Old question"},
                               {"role": "assistant", "content": "Old answer"}]
        model, tool, calls = make_fixed_resources(self.question, self.chunk)
        with self.assertRaisesRegex(ValueError, "旧记录"):
            start_workflow_run(self.question, session, self.project_id, self.documents,
                               model, tool, workspace_root=self.root)
        self.assertEqual(calls, {"classify": 0, "request_tool": 0, "search": 0, "answer": 0})
        self.assertFalse((self.root / "data/checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
