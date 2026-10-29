"""诊断日志的追加、失败传播与坏行报告；只使用临时目录与固定操作。"""

from collections import defaultdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

from paper_sage.workflows.events import RunEventLog, append_event, read_events, summarize_calls


class WorkflowEventTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="paper-sage-events-")
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "events.jsonl"
        self.run_id = "run_" + "a" * 32
        self.addCleanup(self.clean)

    def clean(self):
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        assert self.root.name.startswith("paper-sage-events-")
        self.temporary.cleanup()

    def test_success_failure_retry_and_same_exception_are_paired(self):
        ticks = iter([0.0, 0.25, 1.0, 1.75, 2.0, 2.5])
        log = RunEventLog(self.run_id, self.path, clock=lambda: next(ticks))
        result = {"teaching": "result"}
        self.assertIs(log.call("classify_request", lambda: result), result)
        error = TimeoutError("private exception text")

        def fail():
            raise error

        with self.assertRaises(TimeoutError) as caught:
            log.call("execute_tool", fail)
        self.assertIs(caught.exception, error)
        self.assertIsNone(log.call("execute_tool", lambda: None))
        events, issues = read_events(self.path, self.run_id)
        self.assertEqual(issues, [])
        self.assertEqual(len(events), 6)
        grouped = defaultdict(list)
        for event in events:
            grouped[event["call_id"]].append(event)
        self.assertEqual(len(grouped), 3)
        for pair in grouped.values():
            self.assertEqual([event["phase"] for event in pair], ["start", "end"])
            self.assertEqual(pair[0]["step"], pair[1]["step"])
        endings = events[1::2]
        self.assertEqual([event["status"] for event in endings], ["ok", "failed", "ok"])
        self.assertEqual([event["elapsed_seconds"] for event in endings], [0.25, 0.75, 0.5])
        self.assertNotIn("private exception text", self.path.read_text(encoding="utf-8"))
        self.assertFalse(log.write_errors)

    def test_start_without_end_and_new_logger_keep_existing_events(self):
        log = RunEventLog(self.run_id, self.path)
        start = log.begin("generate_answer")
        original = self.path.read_bytes()
        events, issues = read_events(self.path, self.run_id)
        self.assertEqual(events, [start])
        self.assertEqual(issues, [])
        later = RunEventLog(self.run_id, self.path)
        later.call("generate_answer", lambda: None)
        self.assertTrue(self.path.read_bytes().startswith(original))
        events, issues = read_events(self.path, self.run_id)
        self.assertEqual(len(events), 3)
        self.assertNotEqual(events[0]["call_id"], events[1]["call_id"])
        self.assertEqual([row["status"] for row in summarize_calls(events)], ["unfinished", "ok"])
        self.assertEqual(summarize_calls(events[2:])[0]["status"], "missing_start")
        self.assertEqual(summarize_calls([events[1], events[1], events[2]])[0]["status"], "invalid_pair")

    def test_write_failure_never_replaces_business_result_or_exception(self):
        log = RunEventLog(self.run_id, self.path)
        original = ValueError("business error")

        def fail():
            raise original

        with patch('paper_sage.workflows.events.append_event', side_effect=OSError("disk unavailable")):
            self.assertEqual(log.call("classify_request", lambda: 42), 42)
            with self.assertRaises(ValueError) as caught:
                log.call("execute_tool", fail)
        self.assertIs(caught.exception, original)
        self.assertEqual(len(log.write_errors), 4)
        self.assertEqual([e["phase"] for e in log.write_errors], ["start", "end", "start", "end"])

    def test_keyboard_interrupt_is_recorded_and_propagated(self):
        log = RunEventLog(self.run_id, self.path)
        original = KeyboardInterrupt()

        def cancel():
            raise original

        with self.assertRaises(KeyboardInterrupt) as caught:
            log.call("generate_answer", cancel)
        self.assertIs(caught.exception, original)
        events, issues = read_events(self.path, self.run_id)
        self.assertEqual(issues, [])
        self.assertEqual(events[-1]["status"], "cancelled")
        self.assertEqual(events[-1]["error_type"], "KeyboardInterrupt")

    def test_incomplete_or_foreign_lines_are_reported_without_rewriting(self):
        self.assertEqual(read_events(self.path, self.run_id), ([], []))
        log = RunEventLog(self.run_id, self.path)
        log.call("classify_request", lambda: None)
        original = self.path.read_bytes() + b'{"unfinished": "\xe4'
        self.path.write_bytes(original)
        events, issues = read_events(self.path, self.run_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(issues[0]["line"], 3)
        self.assertEqual(self.path.read_bytes(), original)
        log.call("execute_tool", lambda: None)
        self.assertTrue(self.path.read_bytes().startswith(original + b"\n"))
        events, issues = read_events(self.path, self.run_id)
        self.assertEqual(len(events), 4)
        self.assertEqual([item["line"] for item in issues], [3])
        # 外来运行及不合法耗时不进入返回的有效事件。
        foreign = {**events[0], "run_id": "run_" + "b" * 32}
        append_event(self.path, foreign)
        bad_elapsed = {**events[1], "elapsed_seconds": float("nan")}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(bad_elapsed) + "\n")
        events, issues = read_events(self.path, self.run_id)
        self.assertEqual(len(events), 4)
        self.assertEqual([item["line"] for item in issues], [3, 6, 7])


if __name__ == "__main__":
    unittest.main()
