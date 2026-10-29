"""持久化运行后端：只执行用户明确启动/继续的运行，完整结果按编号写回一次。"""

from copy import deepcopy
from pathlib import Path

from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from paper_sage.agents.graph import build_paper_graph
from paper_sage.agents.workflow import completed_state_messages
from paper_sage.qa.retry_records import validate_retrieval_scope
from paper_sage.sessions.recovery import validate_workbench_session
from paper_sage.sessions.store import make_session_snapshot, restore_session_snapshot
from paper_sage.workflows.checkpoint import prepare_bound_run, validate_run_checkpoint
from paper_sage.workflows.context import create_run_context, validate_run_context
from paper_sage.workflows.store import create_run_storage, load_run_record, lock_workflow_run
from paper_sage.workflows.events import RunEventLog
from paper_sage.workflows.observation import make_node_observer


MAX_ATTEMPTS = 2
CITED_ANSWERS = True


class WorkflowRunError(RuntimeError):
    def __init__(self, run_id, cause, *, log_write_errors=None):
        self.run_id = run_id
        self.cause = cause
        self.log_write_errors = deepcopy(log_write_errors or [])
        super().__init__(f"本轮运行未完成（{run_id}）；输入与已写入的检查点保留")


class _ReadOnlyModel:
    def bind_tools(self, *args, **kwargs):
        return self

    def bind(self, **kwargs):
        return self

    def invoke(self, *args, **kwargs):
        raise RuntimeError("只读恢复不能调用模型")


@tool
def _read_only_search(question: str) -> str:
    """只读检查点使用的占位工具，不允许执行检索。"""
    raise RuntimeError("只读恢复不能执行检索")


def apply_run_result(session, context, messages):
    """对已通过业务校验的完整消息对写回一次；返回副本和是否新增的标记。"""
    saved = context["session_snapshot"]
    scope = saved["session"]["scope_key"]
    base = restore_session_snapshot(saved, scope)
    current = restore_session_snapshot(make_session_snapshot(session), scope)
    if current["session_id"] != base["session_id"]:
        raise ValueError("完成结果不属于当前聊天会话")
    if (not isinstance(messages, list) or len(messages) != 2
            or messages[0] != {"role": "user", "content": context["question"]}
            or not isinstance(messages[1], dict)
            or messages[1].get("role") != "assistant"
            or messages[1].get("workflow_run_id") != context["run_id"]):
        raise ValueError("完成结果缺少正确的消息对或运行编号")
    expected = deepcopy(base)
    expected["messages"].extend(deepcopy(messages))
    expected = restore_session_snapshot(make_session_snapshot(expected), scope)
    positions = [i for i, message in enumerate(current["messages"])
                 if message.get("workflow_run_id") == context["run_id"]]
    if positions:
        if (positions != [len(base["messages"]) + 1]
                or current["messages"][:len(expected["messages"])] != expected["messages"]):
            raise ValueError("当前会话中的同编号结果与原运行不一致")
        return current, False
    if current != base:
        raise ValueError("聊天记录已变化，不能追加旧运行结果")
    return expected, True


def _graph_path(graph, config):
    path = []
    for snapshot in reversed(list(graph.get_state_history(config))):
        node = snapshot.values.get("last_node")
        if node and (not path or path[-1] != node):
            path.append(node)
    return path


def _check_retrieval(state, scope, documents):
    artifact = state.get("retrieval_artifact")
    if artifact is not None:
        validate_retrieval_scope(artifact, scope, documents, query=artifact["question"],
                                 reference_text=state["reference_text"])


def _process_run(project_id, run_id, session, documents, *, execute, model=None, search_tool=None,
                 summary_runner=None, comparison_runner=None, pause_before_answer=False,
                 retry_failed=False, workspace_root=None):
    with lock_workflow_run(project_id, run_id, workspace_root=workspace_root) as paths:
        context = load_run_record(project_id, run_id, workspace_root=workspace_root)["context"]
        base = restore_session_snapshot(context["session_snapshot"], session["scope_key"])
        checked = validate_run_context(context, base, project_id, documents, workspace_root=workspace_root)
        if base["messages"]:
            validate_workbench_session(base, project_id, checked["documents"], workspace_root=workspace_root)
        scope = base["scope_key"]
        initial, config = prepare_bound_run(checked, max_attempts=MAX_ATTEMPTS, cited_answers=CITED_ANSWERS)
        log = RunEventLog(run_id, paths["events"]) if execute else None
        if not execute and not paths["database"].is_file():
            validate_run_context(checked, session, project_id, documents, workspace_root=workspace_root)
            return {"run_id": run_id, "status": "not_started", "next": (), "session": deepcopy(session), "added": False}
        with SqliteSaver.from_conn_string(str(paths["database"])) as saver:
            graph = build_paper_graph(
                model if execute and model is not None else _ReadOnlyModel(),
                search_tool if execute and search_tool is not None else _read_only_search,
                documents=[{"document_id": doc["document_id"], "name": Path(doc["source"]).name}
                           for doc in checked["documents"]],
                summary_runner=summary_runner, comparison_runner=comparison_runner,
                max_attempts=MAX_ATTEMPTS, cited_answers=CITED_ANSWERS, checkpointer=saver,
                node_observer=make_node_observer(log) if log is not None else None,
            )
            snapshot = graph.get_state(config)
            state = validate_run_checkpoint(snapshot, checked) if snapshot.values else None
            if state is not None:
                _check_retrieval(state, scope, checked["documents"])
            completed = state is not None and not snapshot.next
            if not completed:
                validate_run_context(checked, session, project_id, documents, workspace_root=workspace_root)
                failed = any(task.error for task in snapshot.tasks)
                if any(task.interrupts for task in snapshot.tasks):
                    raise ValueError("当前恢复入口不处理动态人工中断")
                if execute and failed and not retry_failed:
                    raise ValueError("上次节点执行失败；请明确选择重试失败步骤后再继续")
                if execute:
                    if model is None or search_tool is None:
                        raise ValueError("继续执行需要提供当前范围的模型和工具")
                    try:
                        for update in graph.stream(
                            None if state is not None else initial, config=config, stream_mode="values",
                            interrupt_before=["generate_answer"] if pause_before_answer else None,
                        ):
                            _check_retrieval(update, scope, checked["documents"])
                    except Exception as error:
                        raise WorkflowRunError(run_id, error, log_write_errors=log.write_errors) from error
                    snapshot = graph.get_state(config)
                    state = validate_run_checkpoint(snapshot, checked)
                # 运行过程中文件也可能变化；暂停或完成都重新核对。
                validate_run_context(checked, session, project_id, documents, workspace_root=workspace_root)
                if state is None or snapshot.next:
                    status = "not_started" if state is None else "failed" if failed and not execute else "paused"
                    return {"run_id": run_id, "status": status, "next": snapshot.next,
                            "session": deepcopy(session), "added": False,
                            "log_write_errors": deepcopy(log.write_errors) if log is not None else []}

            messages = completed_state_messages(state, _graph_path(graph, config), base, scope,
                checked["documents"], max_attempts=MAX_ATTEMPTS, cited_answers=CITED_ANSWERS)
            messages[-1]["input_files"] = deepcopy(checked["input_files"])
            messages[-1]["workflow_run_id"] = run_id
            candidate, _ = apply_run_result(base, checked, messages)
            validate_workbench_session(candidate, project_id, checked["documents"], workspace_root=workspace_root)
            validate_run_context(checked, base, project_id, documents, workspace_root=workspace_root)
            updated, added = apply_run_result(session, checked, messages)
            return {"run_id": run_id, "status": "completed", "next": (), "session": updated, "added": added,
                    "log_write_errors": deepcopy(log.write_errors) if log is not None else []}


def start_workflow_run(question, session, project_id, documents, model, search_tool, *,
                       summary_runner=None, comparison_runner=None, pause_before_answer=False, workspace_root=None):
    if session.get("messages"):
        validate_workbench_session(session, project_id, documents, workspace_root=workspace_root)
    context = create_run_context(question, session, project_id, documents, workspace_root=workspace_root)
    checked = validate_run_context(context, session, project_id, documents, workspace_root=workspace_root)
    create_run_storage(checked, workspace_root=workspace_root)
    return _process_run(project_id, checked["run_id"], session, documents, execute=True,
                        model=model, search_tool=search_tool, summary_runner=summary_runner,
                        comparison_runner=comparison_runner, pause_before_answer=pause_before_answer,
                        workspace_root=workspace_root)


def read_workflow_run(project_id, run_id, session, documents, *, workspace_root=None):
    """读取待办或已完成结果；已完成时返回可写回的会话副本，不调用模型。"""
    return _process_run(project_id, run_id, session, documents, execute=False, workspace_root=workspace_root)


def resume_workflow_run(project_id, run_id, session, documents, model, search_tool, *,
                        summary_runner=None, comparison_runner=None, retry_failed=False, workspace_root=None):
    return _process_run(project_id, run_id, session, documents, execute=True, model=model, search_tool=search_tool,
                        summary_runner=summary_runner, comparison_runner=comparison_runner,
                        retry_failed=retry_failed, workspace_root=workspace_root)
