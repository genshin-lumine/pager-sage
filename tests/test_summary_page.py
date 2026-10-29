"""真实 Streamlit + LangGraph 总结流程；使用固定模型、临时项目和本地 PDF。"""

from copy import deepcopy
from unittest.mock import patch

import test_project_scope as scope_tests
from paper_sage.workflows.events import read_events, summarize_calls
from paper_sage.workflows.store import run_paths


class SummaryPageTests(scope_tests.ProjectScopeTests):
    def test_saved_summary_restores_sources_in_a_fresh_page_without_generation(self):
        at = self.app(self.project_a, [self.aid])
        self.checked(self.submit_summary(at))
        saved = deepcopy(at.session_state["chat_session"])
        self.checked(next(w for w in at.button if w.key == "save_chat_session").click().run())
        counts = (len(self.classifications), len(self.summary_requests), self.encode_calls, len(self.requests))
        fresh = self.app(self.project_a, [self.aid])
        self.assert_empty_chat(fresh)
        widget = next(w for w in fresh.selectbox if str(w.key).startswith("saved_session_"))
        self.checked(widget.set_value(saved["session_id"]).run())
        self.checked(next(w for w in fresh.button if w.key == "restore_chat_session").click().run())
        self.assertEqual(fresh.session_state["chat_session"], saved)
        self.assertEqual((len(self.classifications), len(self.summary_requests), self.encode_calls, len(self.requests)), counts)
        self.assertTrue(any(e.label == "查看总结原文依据" for e in fresh.expander))
        self.assertTrue(fresh.toggle)

        # 完成的图也能独立恢复总结，不依赖手动聊天存档，不重新调用 Map/Reduce。
        run_page = self.app(self.project_a, [self.aid])
        widget = next(w for w in run_page.selectbox if str(w.key).startswith("workflow_run_"))
        self.checked(widget.set_value(saved["messages"][-1]["workflow_run_id"]).run())
        self.checked(next(w for w in run_page.button if w.key == "load_workflow_run").click().run())
        self.assertEqual(run_page.session_state["chat_session"], saved)
        self.assertEqual((len(self.classifications), len(self.summary_requests), self.encode_calls, len(self.requests)), counts)
        self.assertTrue(any(e.label == "查看总结原文依据" for e in run_page.expander))

    def submit_summary(self, at):
        self.route_override = {"intent": "summary", "reason": "固定总结分类", "clarification_question": ""}
        next(w for w in at.text_input if str(w.key).startswith("question_")).set_value("请概括这篇论文。")
        return next(w for w in at.button if w.label == "提交问题").click().run()

    def test_generation_sources_reuse_and_followup_qa(self):
        at = self.app(self.project_a, [self.aid])
        self.checked(self.submit_summary(at))
        message = at.session_state["chat_session"]["messages"][-1]
        self.assertEqual(message["response_kind"], "summary")
        self.assertEqual(message["workflow_trace"]["path"], ["start", "classify_request", "summary_handler"])
        self.assertNotIn("retrieval_trace", message)
        artifact = message["summary_artifact"]
        self.assertFalse(artifact["reused"])
        self.assertEqual((artifact["project_id"], artifact["document_id"]), (self.project_a, self.aid))
        self.assertIn(self.fixture_a["expected_code"], message["content"])
        self.assertTrue(any(exp.label == "查看总结原文依据" for exp in at.expander))
        self.assertTrue(any(self.fixture_a["expected_code"] in block.value for block in at.code))
        sources = artifact["resolved_summary"]["method"][0]["sources"]
        self.assertEqual([(s["source"], s["pdf_page"]) for s in sources], [(self.doc_a["source"], 1)])
        self.assertEqual((len(self.summary_requests), self.encode_calls, len(self.requests), self.rerank_model_instances), (2, 0, 0, 0))
        run_id = message["workflow_run_id"]
        events, issues = read_events(run_paths(self.project_a, run_id)["events"], run_id)
        self.assertEqual(issues, [])
        self.assertEqual(sum(r["step"] == "model_summary_handler" for r in summarize_calls(events)), 2)
        saved = deepcopy(at.session_state["chat_session"]["messages"])
        self.assertTrue(at.toggle)
        self.checked(at.toggle[0].set_value(True).run())
        self.assertEqual(len(at.get("bidi_component")), 1)
        self.assertEqual(len(at.get("download_button")), 1)
        self.assertEqual(at.session_state["chat_session"]["messages"], saved)
        self.assertEqual((len(self.summary_requests), self.encode_calls, len(self.requests)), (2, 0, 0))
        self.checked(at.toggle[0].set_value(False).run())
        self.checked(self.submit_summary(at))
        self.assertTrue(at.session_state["chat_session"]["messages"][-1]["summary_artifact"]["reused"])
        self.assertEqual(len(self.summary_requests), 2)
        self.assertEqual(len(self.classifications), 2)
        reused_id = at.session_state["chat_session"]["messages"][-1]["workflow_run_id"]
        events, issues = read_events(run_paths(self.project_a, reused_id)["events"], reused_id)
        self.assertEqual(issues, [])
        self.assertFalse(any(r["step"] == "model_summary_handler" for r in summarize_calls(events)))

        self.route_override = None
        next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(self.question)
        self.checked(next(w for w in at.button if w.label == "提交问题").click().run())
        self.assertEqual(at.session_state["chat_session"]["messages"][-1]["response_kind"], "answer")
        self.assertEqual(len(self.classifications[-1]["history"]), 4)
        self.assertTrue(all(set(record) == {"role", "content"} for record in self.classifications[-1]["history"]))
        self.assertEqual(len(self.summary_requests), 2)
        self.assertEqual((self.encode_calls, self.rerank_model_instances), (1, 1))

    def test_same_document_in_other_project_is_not_reused_and_scope_clears_chat(self):
        at = self.app(self.project_a, [self.bid])
        self.checked(self.submit_summary(at))
        first = deepcopy(at.session_state["chat_session"]["messages"][-1]["summary_artifact"])
        self.switch(at, self.project_b)
        self.assert_empty_chat(at)
        self.checked(self.submit_summary(at))
        second = at.session_state["chat_session"]["messages"][-1]["summary_artifact"]
        self.assertEqual(first["document_id"], second["document_id"])
        self.assertNotEqual(first["project_id"], second["project_id"])
        self.assertFalse(second["reused"])
        self.assertEqual(len(self.summary_requests), 4)
        self.checked(next(w for w in at.button if w.label == "清空会话").click().run())
        self.assert_empty_chat(at)
        self.checked(self.submit_summary(at))
        self.assertTrue(at.session_state["chat_session"]["messages"][-1]["summary_artifact"]["reused"])
        self.assertEqual(len(self.summary_requests), 4)

    def test_summary_failure_and_foreign_sources_do_not_append_partial_turn(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = deepcopy(at.session_state["chat_session"])
        self.summary_failure = ValueError("controlled summary failure")
        self.submit_summary(at)
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertEqual(at.session_state["chat_session"], before)
        self.summary_failure = None
        self.checked(self.submit_summary(at))
        before = deepcopy(at.session_state["chat_session"])
        foreign = deepcopy(before["messages"][-1]["summary_artifact"])
        foreign["resolved_summary"]["method"][0]["sources"][0]["project_id"] = self.project_b
        with patch('paper_sage.ui.app.create_summary_runner', return_value=lambda document_id: foreign):
            self.submit_summary(at)
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertEqual(at.session_state["chat_session"], before)
