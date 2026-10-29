"""真实 Streamlit + LangGraph + 范围索引；分类、生成与向量均受控，无 API 请求。"""

from copy import deepcopy
import json
import sys
from unittest.mock import patch

from langchain_core.messages import ToolMessage
import test_project_scope as scope_tests


class AgentWorkflowPageTests(scope_tests.ProjectScopeTests):
    def submit(self, at, question):
        next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(question)
        return next(w for w in at.button if w.label == "提交问题").click().run()

    def test_qa_graph_retains_protocol_scope_and_stage_records(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        answer = at.session_state["chat_session"]["messages"][-1]
        self.assertEqual(answer["workflow_trace"]["path"], [
            "start", "classify_request", "request_tool", "begin_attempt", "execute_tool",
            "generate_answer", "decide_next", "finish_run",
        ])
        self.assertEqual((len(self.classifications), len(self.tool_requests), len(self.requests)), (1, 1, 1))
        messages = self.answer_protocols[-1]
        self.assertEqual([m.type for m in messages], ["system", "human"])
        payload = json.loads(messages[-1].content)
        self.assertEqual(payload["question"], self.question)
        self.assertEqual(payload["reference"], [
            {key: result["chunk"][key] for key in ("chunk_id", "text")} for result in answer["retrieval_results"]
        ])
        self.assertEqual(answer["answer_artifact"]["final_results"], answer["retrieval_results"])
        self.assertEqual([s["stage"] for s in answer["retrieval_trace"]["stages"]],
                         ["dense", "bm25", "merge", "rrf", "rerank", "select"])
        self.assertEqual(answer["retrieval_trace"]["scope_key"], at.session_state["chat_session"]["scope_key"])
        self.assertTrue(any(e.label == "查看本轮工作流程" for e in at.expander))
        json.dumps(answer, allow_nan=False)

    def test_clarification_skips_retrieval_and_reply_reuses_history(self):
        at = self.app(self.project_a, [self.aid])
        self.route_override = {"intent": "clarify", "reason": "模拟缺少具体问题", "clarification_question": "您想了解什么？"}
        self.checked(self.submit(at, "帮我分析一下"))
        saved = deepcopy(at.session_state["chat_session"]["messages"])
        self.assertEqual(saved[-1]["response_kind"], "clarify")
        self.assertEqual(saved[-1]["workflow_trace"]["path"], ["start", "classify_request", "clarify_handler"])
        self.assertNotIn("retrieval_trace", saved[-1])
        self.assertEqual((self.encode_calls, self.rerank_model_instances, len(self.tool_requests), len(self.requests)), (0, 0, 0, 0))
        self.route_override = None
        self.checked(self.submit(at, self.question))
        self.assertEqual(self.classifications[-1]["history"], [{"role": m["role"], "content": m["content"]} for m in saved])
        self.assertEqual(len(at.session_state["chat_session"]["messages"]), 4)
        self.assertEqual(len(self.requests[-1]), 4)
        self.assertEqual(self.rerank_model_instances, 1)

    def test_followup_preserves_original_question_and_records_rewritten_query(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = deepcopy(at.session_state["chat_session"]["messages"])
        self.search_query_fn = lambda question: "verification code"
        self.checked(self.submit(at, "校验码呢？"))
        saved = at.session_state["chat_session"]["messages"]
        self.assertEqual(saved[:2], before)
        self.assertEqual(saved[2]["content"], "校验码呢？")
        self.assertEqual(saved[-1]["retrieval_query"], "verification code")
        self.assertEqual(saved[-1]["retrieval_trace"]["question"], "verification code")
        self.assertEqual(saved[-1]["workflow_trace"]["question"], "校验码呢？")
        self.assertEqual([m.type for m in self.tool_requests[-1]], ["system", "human", "ai", "human"])
        self.assertFalse(any(isinstance(m, ToolMessage) for m in self.tool_requests[-1]))
        self.assertEqual(self.classifications[-1]["history"], [{"role": m["role"], "content": m["content"]} for m in before])

    def test_multi_document_summary_and_single_document_comparison_clarify(self):
        at = self.app(self.project_a, [self.aid, self.bid])
        for intent in ("summary", "compare"):
            if intent == "compare":
                self.select(at, [self.aid])
            self.route_override = {"intent": intent, "reason": "固定模拟分类", "clarification_question": ""}
            self.checked(self.submit(at, intent))
            message = at.session_state["chat_session"]["messages"][-1]
            self.assertEqual(message["response_kind"], "clarify")
            self.assertIn("只勾选" if intent == "summary" else "恰好勾选", message["content"])
            self.assertEqual(message["workflow_trace"]["path"][-1], intent + "_handler")
            self.assertNotIn("retrieval_trace", message)
        self.assertEqual((self.encode_calls, self.rerank_model_instances, len(self.requests)), (0, 0, 0))
        self.assertEqual(self.summary_requests, [])

    def test_router_and_tool_request_failures_keep_previous_session(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = deepcopy(at.session_state["chat_session"])
        for stage in ("classification", "invalid_route", "request"):
            with self.subTest(stage=stage):
                self.classify_failure = ValueError("controlled routing failure") if stage == "classification" else None
                self.route_override = {"intent": "invalid", "reason": "test", "clarification_question": ""} if stage == "invalid_route" else None
                self.request_failure = ValueError("controlled request failure") if stage == "request" else None
                self.submit(at, self.question)
                self.assertFalse(at.exception)
                self.assertTrue(at.error)
                self.assertEqual(at.session_state["chat_session"], before)
                self.assertEqual((self.encode_calls, len(self.requests)), (1, 1))

    def test_truncated_and_invalid_answer_keep_previous_session(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = deepcopy(at.session_state["chat_session"])
        self.answer_finish_reason = "length"
        self.submit(at, self.question)
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertEqual(at.session_state["chat_session"], before)
        self.answer_finish_reason = "stop"
        with patch.object(sys.modules['paper_sage.llm.client'], "answer_response", return_value={"answer": "missing evidence"}):
            self.submit(at, self.question)
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertEqual(at.session_state["chat_session"], before)

    def test_scope_mismatch_blocks_graph_before_model_call(self):
        from paper_sage.agents.workflow import run_paper_turn
        from paper_sage.sessions.chat import create_chat_session
        from unittest.mock import Mock
        scope = (self.project_a, ((self.aid, "v1", self.doc_a["source"]),))
        model, tool = Mock(), Mock()
        session = create_chat_session(scope)
        for scope_key, docs in ((scope, [self.doc_b]), ((self.project_b, scope[1]), [self.doc_a])):
            with self.assertRaises(ValueError):
                run_paper_turn("question", session, scope_key, docs, model, tool)
        model.bind_tools.assert_not_called()
        self.assertEqual(session["messages"], [])
