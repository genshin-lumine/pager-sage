"""真实图配合可控模型/工具，验证重试次数、消息隔离和失败记录；不联网。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from paper_sage.agents.graph import build_paper_graph, make_initial_state
from paper_sage.qa.retry import QARetryError
from paper_sage.retrieval.context import build_reference_text


QUESTION = "训练模型使用了什么硬件？"
QUERY = "What hardware was used to train the model?"
DOCUMENTS = [{"document_id": "paper_a", "name": "paper_a.pdf"}]


def response(value, finish="stop"):
    return AIMessage(content=json.dumps(value, ensure_ascii=False), response_metadata={"finish_reason": finish})


def result(outcome):
    return {"answer": "模拟回答" if outcome == "answered" else "资料不足。", "evidence": [],
            "insufficient_reason": "" if outcome == "answered" else "本次片段缺少硬件信息。"}


class Resources:
    def __init__(self, outcomes, *, query=QUERY, fail_tool_at=None, rewrite_response=None, route="qa"):
        self.outcomes = list(outcomes)
        self.searches = []
        self.answer_messages = []
        self.artifacts = []
        self.classifications = 0
        self.model = Mock()
        self.request = Mock()
        self.rewrite = Mock()
        self.request.invoke.return_value = AIMessage(content="", response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": "search_papers", "args": {"question": QUESTION}, "id": "initial-call"}])
        self.rewrite.invoke.return_value = rewrite_response if rewrite_response is not None else response(
            {"query": query, "reason": "固定教学改写"})
        self.model.bind_tools.return_value = self.request

        def invoke(messages):
            if not isinstance(messages[-1], ToolMessage):
                self.classifications += 1
                return response({"intent": route, "reason": "固定分类",
                                 "clarification_question": "想了解什么？" if route == "clarify" else ""})
            self.answer_messages.append(deepcopy(messages))
            value = self.outcomes[len(self.answer_messages) - 1]
            if isinstance(value, Exception):
                raise value
            return value if isinstance(value, AIMessage) else response(result(value))

        self.answer = Mock()
        self.answer.invoke.side_effect = invoke
        self.model.bind.side_effect = lambda **kwargs: self.rewrite if kwargs.get("max_tokens") == 512 else self.answer

        @tool(response_format="content_and_artifact")
        def search_papers(question: str) -> tuple[str, dict]:
            """只返回当前教学文档中的固定片段，不执行真实检索。"""
            self.searches.append(question)
            number = len(self.searches)
            if fail_tool_at == number:
                raise TimeoutError("private SDK body must not enter the record")
            chunk = {"project_id": "practice", "document_id": "paper_a", "source": "practice/paper_a.pdf",
                     "pdf_page": number, "chunk_id": f"practice:paper_a:p{number}:c1",
                     "text": f"UNIQUE_REFERENCE_{number}"}
            artifact = {"question": question.strip(), "retrieval_results": [{"rank": 1, "chunk": chunk}],
                        "retrieval_trace": {"stages": [{"stage": "fixture", "marker": number}]}}
            self.artifacts.append(deepcopy(artifact))
            return build_reference_text(artifact["retrieval_results"]), artifact

        self.tool = search_papers

    def run(self, *, max_attempts=2, documents=DOCUMENTS, history=None):
        initial = make_initial_state(QUESTION, history=history)
        graph = build_paper_graph(self.model, self.tool, documents=documents, max_attempts=max_attempts)
        return list(graph.stream(initial, config={"recursion_limit": 24}, stream_mode="values"))


class QARetryGraphTests(unittest.TestCase):
    def test_answer_on_first_attempt_stops_without_rewrite(self):
        resources = Resources(["answered"])
        states = resources.run()
        final = states[-1]
        self.assertEqual([s["last_node"] for s in states], ["start", "classify_request", "request_tool",
                         "begin_attempt", "execute_tool", "generate_answer", "decide_next", "finish_run"])
        self.assertEqual((final["attempts_used"], final["retry_status"], final["stop_reason"]), (1, "answered", "answered"))
        self.assertEqual(resources.searches, [QUESTION])
        self.assertEqual(resources.classifications, 1)
        resources.request.invoke.assert_called_once()
        resources.rewrite.invoke.assert_not_called()
        self.assertEqual(final["attempt_records"][0]["retrieval_artifact"], resources.artifacts[0])

    def test_retry_uses_original_question_new_tool_result_and_independent_records(self):
        history = [{"role": "user", "content": "先前问题"}, {"role": "assistant", "content": "先前回答"}]
        before = deepcopy(history)
        resources = Resources(["insufficient", "answered"])
        final = resources.run(history=history)[-1]
        self.assertEqual((final["attempts_used"], final["stop_reason"]), (2, "answered"))
        self.assertEqual(resources.searches, [QUESTION, QUERY])
        self.assertEqual(final["previous_queries"], resources.searches)
        self.assertEqual(final["question"], QUESTION)
        self.assertEqual(history, before)
        self.assertEqual(final["history"], before)
        self.assertEqual(resources.classifications, 1)
        resources.request.invoke.assert_called_once()
        resources.rewrite.invoke.assert_called_once()
        rewrite_input = json.loads(resources.rewrite.invoke.call_args.args[0][-1].content)
        self.assertEqual(rewrite_input["question"], QUESTION)
        self.assertEqual(rewrite_input["previous_queries"], [QUESTION])
        self.assertEqual(rewrite_input["history"], before)
        for number, messages in enumerate(resources.answer_messages, start=1):
            self.assertEqual([m.type for m in messages], ["system", "human", "ai", "human", "ai", "tool"])
            self.assertEqual(messages[-3].content, QUESTION)
            self.assertEqual(messages[-1].tool_call_id, messages[-2].tool_calls[0]["id"])
            self.assertIn(f"UNIQUE_REFERENCE_{number}", messages[-1].content)
        second = resources.answer_messages[1]
        self.assertNotIn("UNIQUE_REFERENCE_1", " ".join(str(m.content) for m in second))
        self.assertNotEqual(resources.answer_messages[0][-1].tool_call_id, second[-1].tool_call_id)
        records = final["attempt_records"]
        self.assertEqual([r["status"] for r in records], ["insufficient", "answered"])
        self.assertEqual([r["tool_request_origin"] for r in records], ["model", "query_rewrite"])
        self.assertEqual([r["retrieval_artifact"] for r in records], resources.artifacts)
        final["retrieval_artifact"]["retrieval_trace"]["stages"][0]["marker"] = 99
        self.assertEqual(records[1]["retrieval_artifact"]["retrieval_trace"]["stages"][0]["marker"], 2)

    def test_two_insufficient_attempts_stop_and_do_not_request_third(self):
        resources = Resources(["insufficient", "insufficient"])
        final = resources.run()[-1]
        self.assertEqual((final["attempts_used"], final["retry_status"], final["stop_reason"]),
                         (2, "insufficient", "attempt_limit"))
        self.assertEqual(final["response_kind"], "answer")
        self.assertTrue(final["result"]["insufficient_reason"])
        self.assertEqual(len(resources.searches), 2)
        resources.rewrite.invoke.assert_called_once()

    def test_empty_and_duplicate_queries_do_not_consume_another_attempt(self):
        for query, reason in (("  ", "empty_query"), ("  " + QUESTION + "  ", "duplicate_query")):
            with self.subTest(reason=reason):
                resources = Resources(["insufficient"], query=query)
                final = resources.run()[-1]
                self.assertEqual((final["attempts_used"], final["stop_reason"]), (1, reason))
                self.assertEqual(final["previous_queries"], [QUESTION])
                self.assertEqual(len(final["rewrite_records"]), 1)
                self.assertEqual(len(resources.searches), 1)
                self.assertTrue(final["result"]["insufficient_reason"])

    def test_failed_tool_counts_attempt_stops_and_preserves_prior_records(self):
        for failure_at in (1, 2):
            with self.subTest(failure_at=failure_at):
                resources = Resources(["insufficient"], fail_tool_at=failure_at)
                with self.assertRaises(QARetryError) as caught:
                    resources.run()
                state = caught.exception.state
                self.assertEqual((state["attempts_used"], state["retry_status"]), (failure_at, "tool_error"))
                self.assertEqual(state["response_kind"], "error")
                self.assertIsNone(state["result"])
                self.assertEqual(state["attempt_records"][-1]["status"], "tool_error")
                self.assertNotIn("private SDK body", str(state))
                self.assertEqual(len(resources.answer_messages), failure_at - 1)
                self.assertEqual(resources.rewrite.invoke.call_count, failure_at - 1)
                if failure_at == 2:
                    self.assertEqual(state["attempt_records"][0]["status"], "insufficient")

    def test_bad_answer_is_error_not_a_reason_to_rewrite(self):
        invalid = [response(result("answered"), "length"), response({"answer": "missing fields"}),
                   AIMessage(content="{", response_metadata={"finish_reason": "stop"}),
                   AIMessage(content=[{"type": "text", "text": "not JSON string"}], response_metadata={"finish_reason": "stop"}),
                   TimeoutError("hidden transport text")]
        for value in invalid:
            with self.subTest(value=type(value).__name__):
                resources = Resources([value])
                with self.assertRaises(QARetryError) as caught:
                    resources.run()
                state = caught.exception.state
                self.assertEqual(state["retry_status"], "answer_error")
                self.assertEqual(state["attempts_used"], 1)
                self.assertIsNone(state["result"])
                self.assertEqual(state["attempt_records"][0]["retrieval_artifact"], resources.artifacts[0])
                if isinstance(value, AIMessage):
                    self.assertEqual(state["attempt_records"][0]["answer_response"]["raw_response"], value.content)
                resources.rewrite.invoke.assert_not_called()

    def test_failed_rewrite_keeps_error_record_and_does_not_start_search(self):
        for failure in (response({"query": QUERY, "reason": "test"}, "length"), TimeoutError("hidden rewrite text")):
            with self.subTest(failure=type(failure).__name__):
                resources = Resources(["insufficient"], rewrite_response=failure)
                if isinstance(failure, Exception):
                    resources.rewrite.invoke.side_effect = failure
                with self.assertRaises(QARetryError) as caught:
                    resources.run()
                state = caught.exception.state
                self.assertEqual((state["attempts_used"], state["retry_status"]), (1, "rewrite_error"))
                self.assertIsNone(state["result"])
                self.assertEqual(len(state["rewrite_records"]), 1)
                self.assertEqual(state["attempt_records"][0]["status"], "insufficient")
                self.assertEqual(resources.searches, [QUESTION])
                self.assertNotIn("hidden rewrite text", str(state))

    def test_request_failure_has_no_search_attempt(self):
        resources = Resources([])
        resources.request.invoke.side_effect = TimeoutError("private request body")
        with self.assertRaises(QARetryError) as caught:
            resources.run()
        state = caught.exception.state
        self.assertEqual((state["attempts_used"], state["retry_status"]), (0, "request_error"))
        self.assertEqual(state["attempt_records"], [])
        self.assertEqual(resources.searches, [])

    def test_disabled_default_path_and_single_attempt_limit_remain_bounded(self):
        resources = Resources(["insufficient"])
        states = resources.run(max_attempts=None)
        self.assertEqual([s["last_node"] for s in states],
                         ["start", "classify_request", "request_tool", "execute_tool", "generate_answer"])
        self.assertEqual(states[-1]["attempt_records"], [])
        resources.rewrite.invoke.assert_not_called()
        resources = Resources(["insufficient"])
        final = resources.run(max_attempts=1)[-1]
        self.assertEqual((final["attempts_used"], final["stop_reason"]), (1, "attempt_limit"))
        resources.rewrite.invoke.assert_not_called()

    def test_invalid_limits_rejected_before_binding_model(self):
        for limit in (0, -1, 3, True, 1.0, "2"):
            with self.subTest(limit=limit):
                model = Mock()
                with self.assertRaises(ValueError):
                    build_paper_graph(model, Mock(), max_attempts=limit)
                model.bind.assert_not_called()
                model.bind_tools.assert_not_called()

    def test_other_routes_do_not_enter_retry_loop(self):
        for route in ("clarify", "summary", "compare"):
            with self.subTest(route=route):
                resources = Resources([], route=route)
                final = resources.run()[-1]
                self.assertEqual(final["attempts_used"], 0)
                self.assertEqual(final["attempt_records"], [])
                self.assertEqual(resources.searches, [])
                resources.request.invoke.assert_not_called()
                resources.rewrite.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
