"""比较结果的会话范围、完整消息返回、历史过滤和失败时不保存。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from paper_sage.agents.workflow import run_paper_turn
from paper_sage.sessions.chat import build_history_messages, create_chat_session
import test_comparison_runner as runner_fixtures


class ComparisonSessionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runner_fixtures.ComparisonRunnerTests()
        self.fixture.setUp()
        for doc in self.fixture.documents:
            doc["index_version"] = "v1"
        self.documents = deepcopy(self.fixture.documents)
        self.scope = self.make_scope()
        self.session = create_chat_session(self.scope)
        self.model = Mock()
        self.model.bind.return_value.invoke.return_value = AIMessage(
            content=json.dumps({"intent": "compare", "reason": "Fixed route", "clarification_question": ""}),
            response_metadata={"finish_reason": "stop"},
        )
        self.model.bind_tools.return_value.invoke.side_effect = AssertionError("QA request must not run")

        @tool
        def search_papers(question: str) -> str:
            """Unused QA tool for the comparison path."""
            raise AssertionError("QA search must not run")
        self.search_tool = search_papers

    def make_scope(self, project_id="project", documents=None):
        return (project_id, tuple(sorted((d["document_id"], d["index_version"], d["source"])
                                        for d in (self.documents if documents is None else documents))))

    def turn(self, runner, *, session=None, scope=None, documents=None):
        return run_paper_turn(
            self.fixture.question, self.session if session is None else session,
            self.scope if scope is None else scope, self.documents if documents is None else documents,
            self.model, self.search_tool, comparison_runner=runner,
        )

    def test_success_returns_two_serializable_messages_without_modifying_session(self):
        artifact = self.fixture.run_flow()
        before = deepcopy(self.session)
        callback = Mock(return_value=artifact)
        messages = self.turn(callback)
        self.assertEqual(self.session, before)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        answer = messages[-1]
        self.assertEqual(answer["response_kind"], "compare")
        self.assertEqual(answer["comparison_artifact"], artifact)
        self.assertEqual(len(answer["comparison_artifact"]["records"]), 4)
        self.assertEqual(answer["workflow_trace"]["path"], ["start", "classify_request", "compare_handler"])
        self.assertEqual(answer["workflow_trace"]["scope_key"], self.scope)
        self.assertIn("Test comparison", answer["content"])
        self.assertNotIn("retrieval_trace", answer)
        json.dumps(messages, allow_nan=False)
        self.session["messages"].extend(messages)
        self.assertEqual(len(self.session["messages"]), 2)
        answer["comparison_artifact"]["records"][0]["task"]["question"] = "Caller change"
        self.assertNotEqual(artifact["records"][0]["task"]["question"], "Caller change")

    def test_plan_clarification_keeps_plan_and_no_retrieval(self):
        self.fixture.plan = {"dimensions": [], "clarification_question": "比较哪些方面？"}
        artifact = self.fixture.run_flow()
        messages = self.turn(Mock(return_value=artifact))
        self.assertEqual(self.session["messages"], [])
        answer = messages[-1]
        self.assertEqual(answer["response_kind"], "clarify")
        self.assertEqual(answer["content"], self.fixture.plan["clarification_question"])
        self.assertEqual(answer["comparison_artifact"]["plan_record"], artifact["plan_record"])
        self.assertEqual(answer["comparison_artifact"]["records"], [])
        self.assertIsNone(answer["comparison_artifact"]["generation_record"])
        self.fixture.loader.assert_not_called()

    def test_document_count_clarification_has_no_comparison_artifact(self):
        documents = self.documents[:1]
        scope = self.make_scope(documents=documents)
        session = create_chat_session(scope)
        callback = Mock(side_effect=AssertionError("Must not call comparison"))
        messages = self.turn(callback, session=session, scope=scope, documents=documents)
        self.assertEqual(messages[-1]["response_kind"], "clarify")
        self.assertNotIn("comparison_artifact", messages[-1])
        self.assertEqual(session["messages"], [])
        callback.assert_not_called()

    def test_followup_history_contains_only_chat_text(self):
        initial_artifact = self.fixture.run_flow()
        self.session["messages"].extend(self.turn(Mock(return_value=initial_artifact)))
        before = deepcopy(self.session)
        history = build_history_messages(self.session, self.scope)
        self.fixture.question = "Compare the test results again"
        artifact = self.fixture.run_flow(history=history)
        callback = Mock(return_value=artifact)
        messages = self.turn(callback)
        callback.assert_called_once_with(self.fixture.question, history=history)
        self.assertTrue(all(set(m) == {"role", "content"} for m in callback.call_args.kwargs["history"]))
        router_messages = self.model.bind.return_value.invoke.call_args.args[0]
        self.assertEqual(json.loads(router_messages[-1].content)["history"], history)
        self.assertEqual(self.session, before)
        self.session["messages"].extend(messages)
        self.assertEqual(len(self.session["messages"]), 4)
        self.assertEqual(self.session["messages"][:2], before["messages"])

    def test_failure_does_not_return_partial_messages_or_change_history(self):
        self.session["messages"] = [{"role": "user", "content": "Previous question"},
                                    {"role": "assistant", "content": "Previous answer", "comparison_artifact": {"old": True}}]
        before = deepcopy(self.session)
        callback = Mock(side_effect=ValueError("Controlled generation failure"))
        with self.assertRaisesRegex(ValueError, "Controlled generation failure"):
            self.turn(callback)
        self.assertEqual(self.session, before)
        self.assertEqual(len(self.session["messages"]), 2)
        callback.assert_called_once_with(self.fixture.question, history=build_history_messages(self.session, self.scope))

    def test_other_project_source_or_index_version_rejected_before_saving(self):
        for changed in ("project", "source", "index_version"):
            with self.subTest(changed=changed):
                fixture = runner_fixtures.ComparisonRunnerTests()
                fixture.setUp()
                for doc in fixture.documents:
                    doc["index_version"] = "v1"
                scope = self.make_scope("other_project" if changed == "project" else "project")
                if changed == "source":
                    fixture.documents[0]["source"] = "other_source.pdf"
                elif changed == "index_version":
                    fixture.documents[0]["index_version"] = "v2"
                artifact = fixture.run_flow()
                session = create_chat_session(scope)
                before = deepcopy(session)
                with self.assertRaisesRegex(ValueError, "当前项目|来源或索引版本"):
                    self.turn(Mock(return_value=artifact), session=session, scope=scope)
                self.assertEqual(session, before)


if __name__ == "__main__":
    unittest.main()
