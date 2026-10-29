"""工作台有界重试：真实页面/图/索引，使用受控模型，验证记录和会话边界。"""

from copy import deepcopy
import json
import sys
from unittest.mock import Mock, patch

import test_project_scope as scope_tests
from paper_sage.workflows.events import read_events, summarize_calls
from paper_sage.workflows.store import run_paths


class QARetryPageTests(scope_tests.ProjectScopeTests):
    def submit(self, at, question=None):
        next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(question or self.question)
        return next(w for w in at.button if w.label == "提交问题").click().run()

    def answer_sequence(self, outcomes):
        original = sys.modules['paper_sage.llm.client'].answer_response
        outcomes = iter(outcomes)

        def respond(messages, api_key):
            result = original(messages, api_key)
            outcome = next(outcomes)
            if outcome == "insufficient":
                return {"answer": "模拟资料不足", "evidence": [], "insufficient_reason": "当前候选缺少所需细节。"}
            if outcome == "invalid":
                return {"answer": "missing fields"}
            return result
        return patch.object(sys.modules['paper_sage.llm.client'], "answer_response", side_effect=respond)

    def test_first_answer_saved_with_retry_detail_and_rerun_does_not_call_models(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        message = at.session_state["chat_session"]["messages"][-1]
        audit = message["qa_retry_artifact"]
        self.assertEqual((audit["attempts_used"], audit["max_attempts"], audit["stop_reason"]), (1, 2, "answered"))
        self.assertEqual(audit["scope_key"], at.session_state["chat_session"]["scope_key"])
        self.assertEqual(audit["rewrite_records"], [])
        self.assertEqual(audit["attempt_records"][0]["retrieval_artifact"]["retrieval_results"], message["retrieval_results"])
        self.assertTrue(any(item.label == "查看本轮检索尝试" for item in at.expander))
        before = deepcopy(at.session_state["chat_session"])
        counts = (self.encode_calls, len(self.requests), len(self.rewrite_requests))
        self.checked(at.run())
        self.assertEqual(at.session_state["chat_session"], before)
        self.assertEqual((self.encode_calls, len(self.requests), len(self.rewrite_requests)), counts)
        json.dumps(message, allow_nan=False)

    def test_second_answer_keeps_both_attempts_but_only_final_reply_in_history(self):
        from paper_sage.sessions.chat import build_history_messages

        at = self.app(self.project_a, [self.aid])
        with self.answer_sequence(["insufficient", "answered"]):
            self.checked(self.submit(at))
        session = at.session_state["chat_session"]
        self.assertEqual(len(session["messages"]), 2)
        message = session["messages"][-1]
        audit = message["qa_retry_artifact"]
        self.assertEqual((self.encode_calls, len(self.classifications), len(self.tool_requests), len(self.requests), len(self.rewrite_requests)),
                         (2, 1, 1, 2, 1))
        self.assertEqual(audit["attempts_used"], 2)
        self.assertEqual([r["status"] for r in audit["attempt_records"]], ["insufficient", "answered"])
        self.assertEqual(audit["stop_reason"], "answered")
        self.assertEqual(message["retrieval_query"], self.rewrite_override["query"])
        self.assertEqual(message["workflow_trace"]["path"].count("begin_attempt"), 2)
        self.assertIn("finish_run", message["workflow_trace"]["path"])
        for record in audit["attempt_records"]:
            self.assertEqual(len(record["retrieval_artifact"]["retrieval_trace"]["stages"]), 6)
            for item in record["retrieval_artifact"]["retrieval_results"]:
                self.assertEqual(item["chunk"]["project_id"], self.project_a)
                self.assertEqual(item["chunk"]["document_id"], self.aid)
        self.assertEqual([m.type for m in self.answer_protocols[-1]], ["system", "human"])
        self.assertEqual(json.loads(self.answer_protocols[-1][1].content)["question"], self.question)
        history = build_history_messages(session, session["scope_key"])
        self.assertEqual(history, [{"role": m["role"], "content": m["content"]} for m in session["messages"]])
        self.assertNotIn("模拟资料不足", history[-1]["content"])
        self.assertTrue(any("尝试 1 · 资料不足" in item.value for item in at.markdown))
        self.assertTrue(any("尝试 2 · 已得到回答" in item.value for item in at.markdown))
        audit["attempt_records"][-1]["retrieval_artifact"]["retrieval_results"][0]["chunk"]["text"] = "edited copy"
        self.assertNotEqual(message["retrieval_results"][0]["chunk"]["text"], "edited copy")

    def test_two_insufficient_attempts_are_saved_as_one_complete_turn(self):
        at = self.app(self.project_a, [self.aid])
        with self.answer_sequence(["insufficient", "insufficient"]):
            self.checked(self.submit(at))
        session = at.session_state["chat_session"]
        audit = session["messages"][-1]["qa_retry_artifact"]
        self.assertEqual(len(session["messages"]), 2)
        self.assertEqual((audit["attempts_used"], audit["status"], audit["stop_reason"]), (2, "insufficient", "attempt_limit"))
        self.assertEqual([r["result"]["claims"] for r in audit["attempt_records"]], [[], []])
        self.assertEqual(len(self.rewrite_requests), 1)
        run_id = session["messages"][-1]["workflow_run_id"]
        events, issues = read_events(run_paths(self.project_a, run_id)["events"], run_id)
        self.assertEqual(issues, [])
        rows = summarize_calls(events)
        self.assertEqual(sum(r["step"] == "model_rewrite_query" for r in rows), 1)
        self.assertEqual(sum(r["step"] == "model_generate_answer" for r in rows), 2)
        self.assertTrue(any("已达到尝试上限" in item.value for item in at.markdown))

    def test_empty_or_duplicate_rewrite_stops_without_second_search(self):
        for candidate, reason in ((" ", "empty_query"), ("  " + self.question + "  ", "duplicate_query")):
            with self.subTest(reason=reason):
                at = self.app(self.project_a, [self.aid])
                previous_count = self.encode_calls
                self.rewrite_override = {"query": candidate, "reason": "模拟停止"}
                with self.answer_sequence(["insufficient"]):
                    self.checked(self.submit(at))
                audit = at.session_state["chat_session"]["messages"][-1]["qa_retry_artifact"]
                self.assertEqual(self.encode_calls, previous_count + 1)
                self.assertEqual((audit["attempts_used"], audit["stop_reason"]), (1, reason))
                self.assertEqual(len(audit["rewrite_records"]), 1)

    def test_rewrite_failure_and_invalid_second_answer_leave_existing_session_unchanged(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = deepcopy(at.session_state["chat_session"])
        for failure in ("rewrite", "second_answer"):
            with self.subTest(failure=failure):
                self.rewrite_failure = TimeoutError("PRIVATE_ERROR_BODY") if failure == "rewrite" else None
                outcomes = ["insufficient"] if failure == "rewrite" else ["insufficient", "invalid"]
                with self.answer_sequence(outcomes):
                    self.submit(at)
                self.assertFalse(at.exception)
                self.assertTrue(at.error)
                self.assertEqual(at.session_state["chat_session"], before)
                self.assertNotIn("PRIVATE_ERROR_BODY", " ".join(item.value for item in at.error))

    def test_second_retrieval_failure_preserves_session_and_skips_second_answer(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = deepcopy(at.session_state["chat_session"])
        count = len(self.rerank_calls)
        answers = len(self.requests)

        def fail_second(pairs):
            if len(self.rerank_calls) == count + 2:
                raise RuntimeError("controlled second retrieval failure")
            return [-1.0] * len(pairs)
        self.rerank_score_fn = fail_second
        with self.answer_sequence(["insufficient"]):
            self.submit(at)
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertEqual(len(self.requests), answers + 1)
        self.assertEqual(at.session_state["chat_session"], before)

    def test_scope_switch_and_clear_remove_attempt_details(self):
        at = self.app(self.project_a, [self.aid])
        with self.answer_sequence(["insufficient", "answered"]):
            self.checked(self.submit(at))
        self.switch(at, self.project_b)
        self.assert_empty_chat(at)
        self.assertFalse(any(item.label == "查看本轮检索尝试" for item in at.expander))
        self.ask(at, self.fixture_b["expected_code"])
        self.checked(next(w for w in at.button if w.label == "清空会话").click().run())
        self.assert_empty_chat(at)
        self.assertFalse(any(item.label == "查看本轮检索尝试" for item in at.expander))

    def test_legacy_answer_without_retry_record_renders_without_new_calls(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        session = deepcopy(at.session_state["chat_session"])
        session["messages"][-1].pop("qa_retry_artifact")
        session["messages"][-1].pop("answer_artifact")
        at.session_state["chat_session"] = session
        before = (self.encode_calls, len(self.requests), len(self.rewrite_requests))
        self.checked(at.run())
        self.assertEqual((self.encode_calls, len(self.requests), len(self.rewrite_requests)), before)
        self.assertFalse(any(item.label == "查看本轮检索尝试" for item in at.expander))

    def test_first_attempt_source_tampering_is_rejected_even_when_last_attempt_is_valid(self):
        from paper_sage.qa.retry_records import build_qa_retry_artifact

        at = self.app(self.project_a, [self.aid])
        with self.answer_sequence(["insufficient", "answered"]):
            self.checked(self.submit(at))
        session = at.session_state["chat_session"]
        audit = deepcopy(session["messages"][-1]["qa_retry_artifact"])
        attempts = audit["attempt_records"]
        attempts[0]["retrieval_artifact"]["retrieval_channels"]["dense"][0]["chunk"]["project_id"] = "foreign"
        state = {"response_kind": "answer", "retry_error": None, "attempt_records": attempts,
                 "attempts_used": 2, "previous_queries": [r["query"] for r in attempts],
                 "rewrite_records": audit["rewrite_records"], "question": audit["question"], "history": [],
                 "result": attempts[-1]["result"], "retrieval_artifact": attempts[-1]["retrieval_artifact"],
                 "answer_artifact": attempts[-1]["answer_artifact"],
                 "reference_text": attempts[-1]["reference_text"], "retry_status": "answered", "stop_reason": "answered"}
        with self.assertRaisesRegex(ValueError, "当前会话范围"):
            build_qa_retry_artifact(state, session["scope_key"], [self.doc_a], 2)

    def test_foreign_tool_result_is_rejected_before_answer_generation(self):
        from paper_sage.agents.workflow import run_paper_turn
        from paper_sage.retrieval.context import build_reference_text

        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        session = deepcopy(at.session_state["chat_session"])
        before = deepcopy(session)
        artifact = deepcopy(session["messages"][-1]["qa_retry_artifact"]["attempt_records"][0]["retrieval_artifact"])
        artifact["retrieval_results"][0]["chunk"]["project_id"] = "foreign"
        continued = []

        def stream(*args, **kwargs):
            yield {"last_node": "execute_tool", "retrieval_artifact": artifact,
                   "reference_text": build_reference_text(artifact["retrieval_results"])}
            continued.append(True)
        graph = Mock()
        graph.stream.side_effect = stream
        with patch('paper_sage.agents.workflow.build_paper_graph', return_value=graph), self.assertRaises(ValueError):
            run_paper_turn(self.question, session, session["scope_key"], [self.doc_a], Mock(), Mock(), max_attempts=2)
        self.assertEqual(continued, [])
        self.assertEqual(session, before)
