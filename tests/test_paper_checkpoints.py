"""正式图可选检查点：真实 SQLite，固定模型，不访问工作区论文或 API。"""

from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory, gettempdir
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from paper_sage.agents.graph import build_paper_graph, make_initial_state
from paper_sage.qa.model import validate_cited_answer_record
from paper_sage.qa.retry import QARetryError
from test_cited_answer_graph import CitedResources, QUESTION


def build_graph(resources, saver):
    return build_paper_graph(
        resources.model, resources.tool,
        documents=[{"document_id": "paper_a", "name": "paper_a.pdf"}],
        max_attempts=2, cited_answers=True, checkpointer=saver,
    )


def check_temporary_directory(directory):
    target = Path(directory).resolve()
    assert target.parent == Path(gettempdir()).resolve()
    assert target.name.startswith("paper_sage_checkpoint_")


class PaperCheckpointTests(unittest.TestCase):
    def test_reopened_database_resumes_answer_without_repeating_retrieval(self):
        config = {"configurable": {"thread_id": "run_first"}, "recursion_limit": 24}
        history = [{"role": "user", "content": "Prior question"},
                   {"role": "assistant", "content": "Prior answer"}]
        with TemporaryDirectory(prefix="paper_sage_checkpoint_") as directory:
            check_temporary_directory(directory)
            database = str(Path(directory) / "checkpoints.sqlite")
            first = CitedResources(["answered"])
            with SqliteSaver.from_conn_string(database) as saver:
                graph = build_graph(first, saver)
                graph.invoke(make_initial_state(QUESTION, history=history), config,
                             interrupt_before=["generate_answer"])
                paused = graph.get_state(config)
                self.assertEqual(paused.next, ("generate_answer",))
                first.classify.invoke.assert_called_once()
                first.request.invoke.assert_called_once()
                self.assertEqual(first.searches, [QUESTION])
                first.answer.invoke.assert_not_called()
                self.assertEqual(paused.values["attempt_records"][0]["status"], "running")
                self.assertIsNone(paused.values["result"])
                prior_values = deepcopy(paused.values)

            # 新连接、新图、新固定模型；已执行的外部函数全部重新计数。
            resumed = CitedResources(["answered"])
            with SqliteSaver.from_conn_string(database) as saver:
                graph = build_graph(resumed, saver)
                before = graph.get_state(config)
                self.assertEqual(before.values, prior_values)
                result = graph.invoke(None, config)
                self.assertEqual(graph.get_state(config).next, ())
                self.assertEqual(result["retrieval_artifact"], prior_values["retrieval_artifact"])
                self.assertEqual(result["history"], history)
                self.assertEqual(len(result["messages"]), 7)
                self.assertEqual(result["messages"][:-1], prior_values["messages"])
                self.assertEqual(len(result["attempt_records"]), 1)
                self.assertEqual(result["attempt_records"][0]["status"], "answered")
                tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
                self.assertEqual(tool_message.artifact, result["retrieval_artifact"])
                self.assertEqual(tool_message.tool_call_id, result["attempt_records"][0]["tool_call"]["id"])
                validate_cited_answer_record(result["answer_artifact"], QUESTION,
                    result["retrieval_artifact"]["retrieval_results"], history=history)
                self.assertEqual(result["attempt_records"][0]["answer_artifact"], result["answer_artifact"])
                resumed.classify.invoke.assert_not_called()
                resumed.request.invoke.assert_not_called()
                resumed.rewrite.invoke.assert_not_called()
                self.assertEqual(resumed.searches, [])
                resumed.answer.invoke.assert_called_once()

    def test_run_threads_do_not_merge_full_history_or_results(self):
        with SqliteSaver.from_conn_string(":memory:") as saver:
            history = [{"role": "user", "content": "Prior question"},
                       {"role": "assistant", "content": "Prior answer"}]
            first = CitedResources(["answered"])
            graph = build_graph(first, saver)
            a = {"configurable": {"thread_id": "run_a"}, "recursion_limit": 24}
            b = {"configurable": {"thread_id": "run_b"}, "recursion_limit": 24}
            result_a = graph.invoke(make_initial_state(QUESTION, history=history), a)
            self.assertEqual(graph.get_state(b).values, {})
            second = CitedResources(["answered"])
            rebuilt = build_graph(second, saver)
            result_b = rebuilt.invoke(make_initial_state(QUESTION, history=history), b)
            self.assertEqual(rebuilt.get_state(a).values, result_a)
            self.assertEqual(len(result_a["messages"]), 7)
            self.assertEqual(len(result_b["messages"]), 7)
            self.assertEqual(result_a["history"], result_b["history"])
            self.assertNotEqual(result_a["messages"][0].id, result_b["messages"][0].id)

    def test_completed_clarification_is_read_without_model_calls(self):
        config = {"configurable": {"thread_id": "run_clarify"}}
        with TemporaryDirectory(prefix="paper_sage_checkpoint_") as directory:
            check_temporary_directory(directory)
            database = str(Path(directory) / "checkpoints.sqlite")
            with SqliteSaver.from_conn_string(database) as saver:
                first = CitedResources([], route="clarify")
                result = build_graph(first, saver).invoke(make_initial_state(QUESTION), config)
                self.assertEqual(result["response_kind"], "clarify")
            with SqliteSaver.from_conn_string(database) as saver:
                second = CitedResources([], route="clarify")
                snapshot = build_graph(second, saver).get_state(config)
                self.assertEqual(snapshot.values, result)
                self.assertEqual(snapshot.next, ())
                second.classify.invoke.assert_not_called()
                second.request.invoke.assert_not_called()
                second.answer.invoke.assert_not_called()
                self.assertEqual(second.searches, [])

    def test_answer_failure_remains_an_exception_with_pending_node(self):
        config = {"configurable": {"thread_id": "run_failure"}, "recursion_limit": 24}
        with SqliteSaver.from_conn_string(":memory:") as saver:
            resources = CitedResources([TimeoutError("fixed test timeout")])
            graph = build_graph(resources, saver)
            with self.assertRaises(QARetryError) as caught:
                graph.invoke(make_initial_state(QUESTION), config)
            self.assertEqual(caught.exception.state["response_kind"], "error")
            snapshot = graph.get_state(config)
            self.assertEqual(snapshot.next, ("generate_answer",))
            self.assertIsNone(snapshot.values["result"])
            self.assertNotEqual(snapshot.values["response_kind"], "answer")
            self.assertTrue(snapshot.tasks[0].error)


if __name__ == "__main__":
    unittest.main()
