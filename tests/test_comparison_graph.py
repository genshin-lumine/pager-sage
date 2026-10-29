"""真实 LangGraph 配合模拟比较产物；验证路由边界、历史与来源记录。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from paper_sage.agents.graph import build_paper_graph, make_initial_state
from paper_sage.comparison.report import format_paper_comparison, validate_comparison_artifact
import test_comparison_runner as runner_fixtures


class ComparisonGraphTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runner_fixtures.ComparisonRunnerTests()
        self.fixture.setUp()
        self.documents = [{"document_id": d["document_id"], "name": d["source"]} for d in self.fixture.documents]
        self.model = Mock()
        self.model.bind.return_value.invoke.return_value = AIMessage(
            content=json.dumps({"intent": "compare", "reason": "Fixed route", "clarification_question": ""}),
            response_metadata={"finish_reason": "stop"},
        )
        self.model.bind_tools.return_value.invoke.side_effect = AssertionError("QA request must not run")

        @tool
        def search_papers(question: str) -> str:
            """Never called in this comparison test."""
            raise AssertionError("QA tool must not run")
        self.search_tool = search_papers

    def run_graph(self, runner=None, *, documents=None, history=None, initial=None):
        graph = build_paper_graph(self.model, self.search_tool,
                                 documents=self.documents if documents is None else documents, comparison_runner=runner)
        state = initial if initial is not None else make_initial_state(self.fixture.question, history=history)
        return list(graph.stream(state, config={"recursion_limit": 12}, stream_mode="values"))

    def test_comparison_path_keeps_full_artifact_and_clears_other_response_data(self):
        artifact = self.fixture.run_flow()
        runner = Mock(return_value=artifact)
        initial = make_initial_state(self.fixture.question)
        initial.update(retrieval_artifact={"stale": True}, summary_artifact={"stale": True}, result={"stale": True})
        states = self.run_graph(runner, initial=initial)
        final = states[-1]
        self.assertEqual([s["last_node"] for s in states], ["start", "classify_request", "compare_handler"])
        runner.assert_called_once_with(self.fixture.question, history=[])
        self.assertEqual(final["response_kind"], "compare")
        self.assertEqual(final["comparison_artifact"], artifact)
        self.assertEqual([m.type for m in final["messages"]], ["system", "human", "ai"])
        self.assertEqual(final["reply"], format_paper_comparison(artifact))
        self.assertIn("Test comparison", final["reply"])
        self.assertIn("Missing in supplied text", final["reply"])
        self.assertIn("PDF 第", final["reply"])
        for key in ("retrieval_artifact", "summary_artifact", "result"):
            self.assertIsNone(final[key])
        final["comparison_artifact"]["records"][0]["task"]["question"] = "Caller edit"
        self.assertNotEqual(artifact["records"][0]["task"]["question"], "Caller edit")
        self.model.bind_tools.return_value.invoke.assert_not_called()
        self.model.bind.return_value.invoke.assert_called_once()

    def test_compare_route_can_end_with_plan_clarification(self):
        self.fixture.plan = {"dimensions": [], "clarification_question": "比较哪些方面？"}
        artifact = self.fixture.run_flow()
        runner = Mock(return_value=artifact)
        final = self.run_graph(runner)[-1]
        self.assertEqual(final["route"]["intent"], "compare")
        self.assertEqual(final["response_kind"], "clarify")
        self.assertEqual(final["reply"], self.fixture.plan["clarification_question"])
        self.assertEqual(final["comparison_artifact"], artifact)
        self.assertEqual(final["comparison_artifact"]["records"], [])
        self.fixture.loader.assert_not_called()
        runner.assert_called_once()

    def test_wrong_document_count_and_router_clarification_skip_runner(self):
        runner = Mock(side_effect=AssertionError("Runner must not run"))
        for documents in ([], self.documents[:1], self.documents + [{"document_id": "c", "name": "c.pdf"}]):
            with self.subTest(documents=documents):
                final = self.run_graph(runner, documents=documents)[-1]
                self.assertEqual(final["response_kind"], "clarify")
                self.assertIsNone(final["comparison_artifact"])
                self.assertIn("两篇", final["reply"])
        self.model.bind.return_value.invoke.return_value = AIMessage(
            content=json.dumps({"intent": "clarify", "reason": "Fixed route", "clarification_question": "Which papers?"}),
            response_metadata={"finish_reason": "stop"},
        )
        states = self.run_graph(runner)
        self.assertEqual(states[-1]["last_node"], "clarify_handler")
        self.assertIsNone(states[-1]["comparison_artifact"])
        runner.assert_not_called()

    def test_history_is_forwarded_to_callback_and_router_without_mutating_state(self):
        history = [{"role": "assistant", "content": "Which dimensions?"}]
        artifact = self.fixture.run_flow(history=history)
        runner = Mock(return_value=artifact)
        final = self.run_graph(runner, history=history)[-1]
        runner.assert_called_once_with(self.fixture.question, history=history)
        router_messages = self.model.bind.return_value.invoke.call_args.args[0]
        self.assertEqual(json.loads(router_messages[-1].content)["history"], history)
        self.assertEqual(final["history"], history)
        runner.call_args.kwargs["history"][0]["content"] = "Callback changed its copy"
        self.assertEqual(final["history"], history)

    def test_mismatched_or_failed_callback_result_cannot_become_answer(self):
        artifact = self.fixture.run_flow()
        for changes in ({"question": "Other question"}, {"history": [{"role": "user", "content": "Other"}]},
                        {"documents": [{"document_id": "foreign"}]}, {"status": "failed"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_graph(Mock(return_value={**deepcopy(artifact), **changes}))
        with self.assertRaisesRegex(RuntimeError, "Controlled failure"):
            self.run_graph(Mock(side_effect=RuntimeError("Controlled failure")))

    def test_inconsistent_queries_material_or_resolved_sources_are_rejected(self):
        artifact = self.fixture.run_flow()
        for field in ("tasks", "records", "material", "generation_record"):
            changed = deepcopy(artifact)
            if field == "tasks":
                changed[field][0]["question"] = "Old query"
            elif field == "records":
                changed[field][0]["retrieval_artifact"]["retrieval_results"][0]["chunk"]["project_id"] = "other_project"
            elif field == "material":
                changed[field]["groups"][0]["papers"][0]["reference_text"] = "Changed source"
            else:
                changed[field]["resolved_comparison"]["comparisons"][0]["comparison"]["sources"][0]["pdf_page"] = 99
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_comparison_artifact(changed)
        self.assertEqual(format_paper_comparison(artifact), format_paper_comparison(deepcopy(artifact)))

    def test_no_callback_preserves_unavailable_behavior(self):
        final = self.run_graph()[-1]
        self.assertEqual(final["response_kind"], "unavailable")
        self.assertIsNone(final["comparison_artifact"])
        self.assertIn("尚未接入", final["reply"])


if __name__ == "__main__":
    unittest.main()
