"""正式图的诊断日志：固定模型、临时运行，核对恢复、异常和只读行为。"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_workflow_runner as runner_tests
from paper_sage.workflows.events import read_events, summarize_calls
from paper_sage.workflows.runner import WorkflowRunError
from paper_sage.workflows.store import run_paths


class WorkflowObservationTests(unittest.TestCase):
    def setUp(self):
        runner_tests.WorkflowRunnerTests.setUp(self)

    start = runner_tests.WorkflowRunnerTests.start
    read = runner_tests.WorkflowRunnerTests.read
    resume = runner_tests.WorkflowRunnerTests.resume

    def path(self, run_id):
        return run_paths(self.project_id, run_id, workspace_root=self.root)["events"]

    def rows(self, run_id):
        events, issues = read_events(self.path(run_id), run_id)
        self.assertEqual(issues, [])
        return summarize_calls(events)

    def test_pause_read_resume_and_completed_read_do_not_fabricate_calls(self):
        paused, _ = self.start()
        run_id = paused["run_id"]
        rows = self.rows(run_id)
        self.assertEqual([r["step"] for r in rows if r["step"].startswith("model_")],
                         ["model_classify_request", "model_request_tool"])
        original = self.path(run_id).read_bytes()
        self.read(run_id)
        self.assertEqual(self.path(run_id).read_bytes(), original)
        complete, counts = self.resume(run_id)
        self.assertEqual(counts["answer"], 1)
        rows = self.rows(run_id)
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        self.assertEqual(sum(r["step"] == "execute_tool" for r in rows), 1)
        self.assertEqual(sum(r["step"].startswith("model_") for r in rows), 3)
        self.assertTrue(self.path(run_id).read_bytes().startswith(original))
        completed_bytes = self.path(run_id).read_bytes()
        self.read(run_id, complete["session"])
        self.resume(run_id, complete["session"])
        self.assertEqual(self.path(run_id).read_bytes(), completed_bytes)
        self.assertNotIn(self.question, completed_bytes.decode("utf-8"))
        self.assertNotIn(self.chunk["text"], completed_bytes.decode("utf-8"))

    def test_failed_request_then_retry_keeps_both_model_calls(self):
        before = deepcopy(self.session)
        with self.assertRaises(WorkflowRunError) as caught:
            self.start(pause=False, error=TimeoutError("fixed failure text"))
        run_id = caught.exception.run_id
        self.assertEqual(self.session, before)
        failed = [r for r in self.rows(run_id) if r["step"] == "model_generate_answer"]
        self.assertEqual([(r["status"], r["error_type"]) for r in failed], [("failed", "TimeoutError")])
        original = self.path(run_id).read_bytes()
        self.resume(run_id, retry_failed=True)
        calls = [r for r in self.rows(run_id) if r["step"] == "model_generate_answer"]
        self.assertEqual([r["status"] for r in calls], ["failed", "ok"])
        self.assertNotEqual(calls[0]["call_id"], calls[1]["call_id"])
        self.assertTrue(self.path(run_id).read_bytes().startswith(original))
        self.assertNotIn("fixed failure text", self.path(run_id).read_text(encoding="utf-8"))

    def test_log_disk_failure_keeps_complete_result_and_reports_missing_events(self):
        with patch('paper_sage.workflows.events.append_event', side_effect=OSError("controlled log failure")):
            result, calls = self.start(pause=False)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["session"]["messages"]), 2)
        self.assertEqual(calls["answer"], 1)
        self.assertTrue(result["log_write_errors"])
        self.assertEqual(self.read(result["run_id"])["session"], result["session"])
        self.assertFalse(self.path(result["run_id"]).exists())

    def test_loading_old_run_without_logs_does_not_write_missing_history(self):
        result, _ = self.start(pause=False)
        path = self.path(result["run_id"])
        self.assertTrue(path.resolve().is_relative_to(self.root))
        path.unlink()  # 仅删除临时测试副本，模拟启用日志之前的运行。
        self.read(result["run_id"])
        self.assertFalse(path.exists())
