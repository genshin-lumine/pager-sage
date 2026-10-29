"""输入上下文与正式图检查点的关联；固定分类、临时 SQLite，无模型请求。"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from paper_sage.agents.graph import build_paper_graph
import test_workflow_context as context_tests
from test_cited_answer_graph import CitedResources
from paper_sage.workflows.context import validate_run_context
from paper_sage.workflows.checkpoint import prepare_bound_run, validate_run_checkpoint


class CheckpointBindingTests(unittest.TestCase):
    def setUp(self):
        # 复用临时项目与摘要字节准备；不继承其测试用例。
        context_tests.WorkflowContextTests.setUp(self)
        self.initial, self.config = prepare_bound_run(self.context)

    def graph(self, saver, resources):
        return build_paper_graph(resources.model, resources.tool,
            documents=[{"document_id": doc["document_id"], "name": Path(doc["source"]).name}
                       for doc in self.documents],
            max_attempts=2, cited_answers=True, checkpointer=saver)

    def paused(self, saver):
        resources = CitedResources([], route="clarify")
        graph = self.graph(saver, resources)
        graph.invoke(self.initial, self.config, interrupt_before=["clarify_handler"])
        resources.classify.invoke.assert_called_once()
        resources.request.invoke.assert_not_called()
        self.assertEqual(resources.searches, [])
        return graph, graph.get_state(self.config)

    def test_binding_survives_database_reopen_and_graph_completion(self):
        database = str(self.root / "checkpoints.sqlite")
        with SqliteSaver.from_conn_string(database) as saver:
            _, paused = self.paused(saver)
            self.assertEqual(paused.next, ("clarify_handler",))
            self.assertEqual(validate_run_checkpoint(paused, self.context), paused.values)
        resources = CitedResources([], route="clarify")
        with SqliteSaver.from_conn_string(database) as saver:
            graph = self.graph(saver, resources)
            checked = validate_run_context(self.context, self.session, self.project_id,
                                           self.documents, workspace_root=self.root)
            self.assertEqual(validate_run_checkpoint(graph.get_state(self.config), checked), paused.values)
            result = graph.invoke(None, self.config)
            final = graph.get_state(self.config)
            self.assertEqual(final.next, ())
            self.assertEqual(validate_run_checkpoint(final, checked), result)
            self.assertEqual(result["run_binding"], self.initial["run_binding"])
            resources.classify.invoke.assert_not_called()
            resources.request.invoke.assert_not_called()
            resources.answer.invoke.assert_not_called()
            self.assertEqual(resources.searches, [])

    def test_self_consistent_context_changes_cannot_reuse_old_checkpoint(self):
        with SqliteSaver.from_conn_string(":memory:") as saver:
            _, snapshot = self.paused(saver)
            changed = deepcopy(self.context)
            changed["question"] = "A different question?"
            checked = validate_run_context(changed, self.session, self.project_id,
                                           self.documents, workspace_root=self.root)
            with self.assertRaisesRegex(ValueError, "摘要"):
                validate_run_checkpoint(snapshot, checked)
            changed = deepcopy(self.context)
            current = deepcopy(self.session)
            current["messages"][0]["content"] = "Changed history"
            changed["session_snapshot"]["session"]["messages"][0]["content"] = "Changed history"
            checked = validate_run_context(changed, current, self.project_id,
                                           self.documents, workspace_root=self.root)
            with self.assertRaisesRegex(ValueError, "摘要"):
                validate_run_checkpoint(snapshot, checked)

    def test_thread_binding_version_and_graph_options_must_match(self):
        with SqliteSaver.from_conn_string(":memory:") as saver:
            _, snapshot = self.paused(saver)
            wrong_config = deepcopy(snapshot.config)
            wrong_config["configurable"]["thread_id"] = "run_" + "0" * 32
            with self.assertRaisesRegex(ValueError, "线程"):
                validate_run_checkpoint(snapshot._replace(config=wrong_config), self.context)
            for key, value in (("workflow_version", "other"), ("run_id", "run_" + "0" * 32),
                               ("context_sha256", "0" * 64),
                               ("options", {"max_attempts": 2, "cited_answers": 1})):
                altered = deepcopy(snapshot.values)
                altered["run_binding"][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_run_checkpoint(snapshot._replace(values=altered), self.context)
            for options in ({"max_attempts": 1}, {"cited_answers": False}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    validate_run_checkpoint(snapshot, self.context, **options)

    def test_state_fields_and_input_messages_are_checked_independently(self):
        with SqliteSaver.from_conn_string(":memory:") as saver:
            graph, snapshot = self.paused(saver)
            for field in ("question", "history", "system", "role", "missing_messages"):
                altered = deepcopy(snapshot.values)
                if field == "question":
                    altered["question"] = "Another question"
                elif field == "history":
                    altered["history"][0]["content"] = "Another prior question"
                elif field == "system":
                    altered["messages"][0].content = "Changed system instructions"
                elif field == "role":
                    altered["messages"][1] = AIMessage(content=altered["messages"][1].content)
                else:
                    altered["messages"] = []
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_run_checkpoint(snapshot._replace(values=altered), self.context)
            checked = validate_run_checkpoint(snapshot, self.context)
            checked["messages"][0].content = "Only change the returned copy"
            checked["run_binding"]["context_sha256"] = "changed"
            self.assertEqual(graph.get_state(self.config).values, snapshot.values)

    def test_old_or_empty_state_rejected_and_json_key_order_does_not_matter(self):
        with SqliteSaver.from_conn_string(":memory:") as saver:
            _, snapshot = self.paused(saver)
            for kind in ("missing", "none", "empty"):
                altered = deepcopy(snapshot.values)
                if kind == "missing":
                    altered.pop("run_binding")
                elif kind == "none":
                    altered["run_binding"] = None
                else:
                    altered = {}
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    validate_run_checkpoint(snapshot._replace(values=altered), self.context)
            reordered = dict(reversed(list(self.context.items())))
            self.assertEqual(validate_run_checkpoint(snapshot, reordered), snapshot.values)


if __name__ == "__main__":
    unittest.main()
