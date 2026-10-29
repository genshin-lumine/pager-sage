"""关联已校验的运行输入与图检查点；不续跑、不写入聊天消息。"""

from copy import deepcopy
from hashlib import sha256
import json
import re

from paper_sage.agents.graph import make_initial_state
from paper_sage.sessions.chat import build_history_messages
from paper_sage.qa.retry import validate_attempt_limit
from paper_sage.sessions.store import restore_session_snapshot
from paper_sage.workflows.context import CONTEXT_FIELDS, CONTEXT_PURPOSE


# 工作流状态协议变化时需显式升级；不是自动生成的代码版本指纹。
WORKFLOW_VERSION = "paper_sage_bound_graph_v1"


def _canonical_json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("运行输入或关联信息无法编码为规范 JSON") from error


def _expected_input(context, max_attempts, cited_answers):
    """只检查记录自身；调用方须先 validate_run_context 核对当前会话和文件。"""
    validate_attempt_limit(max_attempts)
    if type(cited_answers) is not bool:
        raise ValueError("cited_answers 必须是布尔值")
    if not isinstance(context, dict) or set(context) != CONTEXT_FIELDS:
        raise ValueError("运行上下文结构不完整")
    if (type(context["schema_version"]) is not int or context["schema_version"] != 1
            or context["purpose"] != CONTEXT_PURPOSE):
        raise ValueError("运行上下文版本不支持")
    run_id, question = context["run_id"], context["question"]
    if not isinstance(run_id, str) or not re.fullmatch(r"run_[0-9a-f]{32}", run_id):
        raise ValueError("运行编号格式无效")
    if not isinstance(question, str) or not question.strip() or question != question.strip():
        raise ValueError("原问题格式无效")
    try:
        snapshot = context["session_snapshot"]
        session = restore_session_snapshot(snapshot, snapshot["session"]["scope_key"])
    except (KeyError, TypeError) as error:
        raise ValueError("运行上下文缺少完整会话快照") from error
    history = build_history_messages(session, session["scope_key"])
    binding = {
        "workflow_version": WORKFLOW_VERSION,
        "run_id": run_id,
        "context_sha256": sha256(_canonical_json(context).encode("utf-8")).hexdigest(),
        "options": {"max_attempts": max_attempts, "cited_answers": cited_answers},
    }
    return question, history, binding


def prepare_bound_run(checked_context, *, max_attempts=2, cited_answers=True):
    """新运行使用返回的初始状态；同一份状态不要再次提交到已有检查点线程。

    checked_context 应来自 validate_run_context；调用方建图时须使用相同选项。
    """
    question, history, binding = _expected_input(checked_context, max_attempts, cited_answers)
    state = make_initial_state(question, history=history)
    state["run_binding"] = binding
    config = {"configurable": {"thread_id": binding["run_id"]},
              "recursion_limit": 24 if max_attempts is not None else 12}
    return state, config


def validate_run_checkpoint(snapshot, checked_context, *, max_attempts=2, cited_answers=True):
    """核对检查点对应的输入，返回状态副本；不判断任务是否允许续跑。

    调用前仍需重新 validate_run_context；任务错误、暂停位置和完成状态由恢复入口处理。
    摘要用于发现记录错配，不是数字签名，也不验证回答的事实正确性。
    """
    question, history, binding = _expected_input(checked_context, max_attempts, cited_answers)
    config = getattr(snapshot, "config", None)
    if (not isinstance(config, dict) or not isinstance(config.get("configurable"), dict)
            or config["configurable"].get("thread_id") != binding["run_id"]):
        raise ValueError("检查点线程与本轮运行编号不一致")
    state = getattr(snapshot, "values", None)
    if not isinstance(state, dict) or not state:
        raise ValueError("该运行没有可读取的图状态")
    if state.get("run_binding") is None:
        raise ValueError("旧检查点缺少运行输入关联，不能直接恢复")
    if _canonical_json(state["run_binding"]) != _canonical_json(binding):
        raise ValueError("检查点与运行输入摘要或工作流约定不一致")
    if state.get("question") != question or state.get("history") != history:
        raise ValueError("检查点中的原问题或历史与运行输入不一致")
    expected_messages = make_initial_state(question, history=history)["messages"]
    messages = state.get("messages")
    if not isinstance(messages, list) or len(messages) < len(expected_messages):
        raise ValueError("检查点缺少原始输入消息")
    for actual, expected in zip(messages, expected_messages):
        if (getattr(actual, "type", None) != expected.type
                or getattr(actual, "content", None) != expected.content):
            raise ValueError("检查点中的输入消息与原问题、历史或提示词不一致")
    return deepcopy(state)
