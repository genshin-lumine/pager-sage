"""真实页面的运行载入/续跑；临时项目、SQLite 与固定模型，无外部请求。"""

from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

import streamlit as st
import pyarrow as pa

import test_project_scope as scope_tests
from paper_sage.workflows.events import RunEventLog, read_events, summarize_calls
from paper_sage.workflows.store import run_paths


class WorkflowRecoveryPageTests(scope_tests.ProjectScopeTests):
    def counters(self):
        return (len(self.classifications), len(self.tool_requests), self.encode_calls,
                len(self.requests), len(self.rewrite_requests))

    def log_rows(self, run_id):
        events, issues = read_events(run_paths(self.project_a, run_id)["events"], run_id)
        self.assertEqual(issues, [])
        return summarize_calls(events)

    def displayed_log_rows(self, at):
        # 直接核对发给浏览器的 Arrow；绕开当前 Pandas/Arrow 的列索引回读兼容问题。
        rows = []
        for widget in at.dataframe:
            data = widget.proto.arrow_data.data
            if not data and widget.proto.HasField("lazy_data"):
                data = widget.proto.lazy_data.initial_chunk.data
            table = pa.ipc.open_stream(pa.BufferReader(data)).read_all()
            if "调用编号" in table.column_names:
                rows.extend(table.to_pylist())
        return rows

    def button(self, at, key):
        return next(w for w in at.button if w.key == key)

    def choose_run(self, at, run_id):
        widget = next(w for w in at.selectbox if str(w.key).startswith("workflow_run_"))
        return self.checked(widget.set_value(run_id).run())

    def submit(self, at):
        next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(self.question)
        return next(w for w in at.button if w.label == "提交问题").click().run()

    def paused(self):
        at = self.app(self.project_a, [self.aid])
        self.checked(self.widget(at.checkbox, "pause_before_answer").set_value(True).run())
        self.checked(self.submit(at))
        run = deepcopy(at.session_state["workflow_loaded"])
        self.assertEqual(run["status"], "paused")
        self.assertEqual(at.session_state["chat_session"]["messages"], [])
        self.assertEqual(self.counters(), (1, 1, 1, 0, 0))
        return at, run

    def test_pause_fresh_page_read_resume_once_and_followup(self):
        at, run = self.paused()
        fresh = self.app(self.project_a, [self.aid])
        self.assert_empty_chat(fresh)
        self.assertNotEqual(fresh.session_state["chat_session"]["session_id"], run["session_id"])
        self.choose_run(fresh, run["run_id"])
        self.assert_empty_chat(fresh)
        # 载入不需要密钥，也不能初始化聊天模型。
        with patch.object(st, "secrets", {}), patch("paper_sage.ui.app.create_chat_model",
                                                         side_effect=AssertionError("read invoked model")):
            self.checked(self.button(fresh, "load_workflow_run").click().run())
        self.assert_empty_chat(fresh)
        self.assertEqual(fresh.session_state["chat_session"]["session_id"], run["session_id"])
        self.assertEqual(self.counters(), (1, 1, 1, 0, 0))
        self.checked(self.button(fresh, "resume_workflow_run").click().run())
        saved = deepcopy(fresh.session_state["chat_session"])
        self.assertEqual(len(saved["messages"]), 2)
        self.assertIn(self.fixture_a["expected_code"], saved["messages"][-1]["content"])
        self.assertEqual(self.counters(), (1, 1, 1, 1, 0))
        self.assertTrue(any(e.label == "查看运行日志" for e in fresh.expander))
        self.assertEqual(sum(row["step"].startswith("model_") for row in self.log_rows(run["run_id"])), 3)
        self.assertEqual(saved["messages"][-1]["workflow_run_id"], run["run_id"])
        self.assertEqual(saved["messages"][-1]["workflow_trace"]["path"], [
            "start", "classify_request", "request_tool", "begin_attempt", "execute_tool",
            "generate_answer", "decide_next", "finish_run",
        ])
        self.assertTrue(any(e.label == "查看结论原文依据" for e in fresh.expander))
        self.checked(self.button(fresh, "load_workflow_run").click().run())
        self.assertEqual(fresh.session_state["chat_session"], saved)
        self.assertEqual(self.counters(), (1, 1, 1, 1, 0))
        self.checked(self.submit(fresh))
        later = deepcopy(fresh.session_state["chat_session"])
        self.assertEqual(len(later["messages"]), 4)
        self.assertEqual(self.classifications[-1]["history"], [
            {"role": m["role"], "content": m["content"]} for m in saved["messages"]
        ])
        self.choose_run(fresh, run["run_id"])
        self.checked(self.button(fresh, "load_workflow_run").click().run())
        self.assertEqual(fresh.session_state["chat_session"], later)

    def test_complete_run_loads_without_model_then_clear_stays_empty(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        saved = deepcopy(at.session_state["chat_session"])
        run_id = saved["messages"][-1]["workflow_run_id"]
        calls = self.counters()
        with patch.object(st, "secrets", {}), patch("paper_sage.ui.app.create_chat_model",
                                                         side_effect=AssertionError("read invoked model")):
            fresh = self.app(self.project_a, [self.aid])
            self.choose_run(fresh, run_id)
            self.checked(self.button(fresh, "load_workflow_run").click().run())
            self.assertEqual(fresh.session_state["chat_session"], saved)
            self.assertFalse(any(w.key == "resume_workflow_run" for w in fresh.button))
            self.checked(next(w for w in fresh.button if w.label == "清空会话").click().run())
            self.assert_empty_chat(fresh)
            self.checked(fresh.run())
            self.assert_empty_chat(fresh)
            self.assertNotIn("workflow_loaded", fresh.session_state)
        self.assertEqual(self.counters(), calls)

    def test_pending_scope_switch_never_runs_and_requires_explicit_load(self):
        at, run = self.paused()
        self.select(at, [self.bid])
        self.assert_empty_chat(at)
        self.assertNotIn("workflow_loaded", at.session_state)
        self.assertFalse(any(str(w.key).startswith("workflow_run_") for w in at.selectbox))
        self.switch(at, self.project_b)
        self.assert_empty_chat(at)
        self.switch(at, self.project_a)
        self.select(at, [self.aid])
        self.assert_empty_chat(at)
        self.assertFalse(any(w.key == "resume_workflow_run" for w in at.button))
        self.choose_run(at, run["run_id"])
        self.checked(self.button(at, "load_workflow_run").click().run())
        self.assertEqual(at.session_state["chat_session"]["session_id"], run["session_id"])
        self.assertEqual(self.counters(), (1, 1, 1, 0, 0))

    def test_failed_answer_is_explicitly_retried_without_repeating_search(self):
        at = self.app(self.project_a, [self.aid])
        self.answer_finish_reason = "length"
        self.submit(at)
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertEqual(at.session_state["chat_session"]["messages"], [])
        run = deepcopy(at.session_state["workflow_loaded"])
        self.assertEqual(run["status"], "failed")
        self.assertEqual(self.counters(), (1, 1, 1, 1, 0))
        self.answer_finish_reason = "stop"
        fresh = self.app(self.project_a, [self.aid])
        self.choose_run(fresh, run["run_id"])
        self.checked(self.button(fresh, "load_workflow_run").click().run())
        self.assertEqual(self.button(fresh, "resume_workflow_run").label, "重试失败步骤")
        self.assertTrue(any(row["step"] == "generate_answer" and row["status"] == "failed"
                            for row in self.log_rows(run["run_id"])))
        self.assertTrue(any(row["步骤"] == "generate_answer" and row["状态"] == "失败"
                            for row in self.displayed_log_rows(fresh)))
        self.assertEqual(self.counters(), (1, 1, 1, 1, 0))
        self.checked(self.button(fresh, "resume_workflow_run").click().run())
        self.assertEqual(len(fresh.session_state["chat_session"]["messages"]), 2)
        self.assertEqual(self.counters(), (1, 1, 1, 2, 0))

    def test_changed_files_block_resume_without_model_and_keep_current_chat(self):
        at, run = self.paused()
        before = deepcopy(at.session_state["chat_session"])
        metadata = Path("data/indexes") / self.project_a / self.aid / "v1/metadata.json"
        metadata.write_bytes(metadata.read_bytes() + b"\n")
        with patch("paper_sage.ui.app.create_chat_model", side_effect=AssertionError("invalid resume")):
            self.button(at, "resume_workflow_run").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("不一致" in item.value for item in at.error))
        self.assertEqual(at.session_state["chat_session"], before)
        self.assertEqual(self.counters(), (1, 1, 1, 0, 0))

    def test_log_view_reports_incomplete_records_without_executing_or_changing_chat(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        saved = deepcopy(at.session_state["chat_session"])
        run_id = saved["messages"][-1]["workflow_run_id"]
        path = run_paths(self.project_a, run_id)["events"]
        RunEventLog(run_id, path).begin("teaching_unfinished")
        with path.open("ab") as file:
            file.write(b'{"incomplete"')
        original = path.read_bytes()
        counts = self.counters()
        with patch("paper_sage.ui.app.create_chat_model", side_effect=AssertionError("log view invoked model")):
            self.checked(at.run())
        self.assertEqual(at.session_state["chat_session"], saved)
        self.assertEqual(self.counters(), counts)
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(any("日志行无法使用" in w.value for w in at.warning))
        self.assertTrue(any("未配对" in w.value for w in at.warning))
        self.assertTrue(any(row["步骤"] == "teaching_unfinished" and row["状态"] == "未记录结束"
                            for row in self.displayed_log_rows(at)))

    def test_log_write_failure_warns_but_successful_answer_is_kept(self):
        at = self.app(self.project_a, [self.aid])
        with patch('paper_sage.workflows.events.append_event', side_effect=OSError("controlled log error")):
            self.checked(self.submit(at))
        self.assertEqual(len(at.session_state["chat_session"]["messages"]), 2)
        self.assertTrue(any("事件未成功写入" in w.value for w in at.warning))
        self.assertEqual(self.counters(), (1, 1, 1, 1, 0))
