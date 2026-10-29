"""绑定单轮工作流的输入；恢复前检查会话、文档登记及当前文件，不执行图。"""

from copy import deepcopy
import json
import re
from uuid import uuid4

from paper_sage.sessions.recovery import capture_scope_inputs
from paper_sage.sessions.store import make_session_snapshot, normalize_session_scope, restore_session_snapshot


CONTEXT_PURPOSE = "paper_sage_workflow_context"
CONTEXT_FIELDS = {"schema_version", "purpose", "run_id", "question", "session_snapshot",
                  "documents", "input_files"}


def _prepare_current(session, project_id, documents):
    if not isinstance(documents, list) or not documents:
        raise ValueError("运行上下文需要至少一篇所选文档")
    for document in documents:
        if not isinstance(document, dict) or any(
            not isinstance(document.get(field), str) or not document[field].strip()
            for field in ("document_id", "index_version", "source")
        ):
            raise ValueError("所选文档缺少文档标识、索引版本或来源")
    scope = normalize_session_scope((project_id, [
        (doc["document_id"], doc["index_version"], doc["source"]) for doc in documents
    ]))
    snapshot = make_session_snapshot(session)
    # 恢复为规范范围再生成 JSON 快照，文档勾选顺序不改变范围。
    snapshot = make_session_snapshot(restore_session_snapshot(snapshot, scope))
    try:
        ordered = json.loads(json.dumps(sorted(documents, key=lambda doc: doc["document_id"]),
                                        ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("文档记录必须能保存为 JSON") from error
    return snapshot, scope, ordered


def create_run_context(question, session, project_id, documents, *, workspace_root=None):
    """新提交创建独立运行编号和输入副本；本函数不保存磁盘文件。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("本轮问题不能为空")
    snapshot, _, ordered = _prepare_current(session, project_id, documents)
    inputs = capture_scope_inputs(project_id, ordered, workspace_root=workspace_root)
    return {
        "schema_version": 1, "purpose": CONTEXT_PURPOSE,
        "run_id": f"run_{uuid4().hex}", "question": question.strip(),
        "session_snapshot": snapshot, "documents": ordered, "input_files": deepcopy(inputs),
    }


def validate_run_context(record, session, project_id, documents, *, workspace_root=None):
    """按当前输入复核旧运行，返回独立副本；不生成新编号、不打开检查点库。

    此处校验输入关联，不校验图节点状态或回答事实。调用方还须将此记录与
    对应检查点关联；执行前、结果写回前都应重新检查，不能只在列表展示时检查。
    """
    if not isinstance(record, dict) or set(record) != CONTEXT_FIELDS:
        raise ValueError("运行上下文缺少字段或包含未约定字段")
    if (type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["purpose"] != CONTEXT_PURPOSE):
        raise ValueError("运行上下文格式版本不支持")
    if not isinstance(record["run_id"], str) or not re.fullmatch(r"run_[0-9a-f]{32}", record["run_id"]):
        raise ValueError("运行编号格式无效")
    question = record["question"]
    if not isinstance(question, str) or not question.strip() or question != question.strip():
        raise ValueError("运行上下文的原问题格式无效")
    current_snapshot, scope, ordered = _prepare_current(session, project_id, documents)
    saved_snapshot = make_session_snapshot(restore_session_snapshot(record["session_snapshot"], scope))
    if saved_snapshot["session"]["session_id"] != current_snapshot["session"]["session_id"]:
        raise ValueError("运行记录不属于当前聊天会话")
    # 比较完整快照，消息条数相同也可能有正文或证据变化。
    if saved_snapshot != current_snapshot:
        raise ValueError("聊天记录已变化，不能继续原运行")
    if record["documents"] != ordered:
        raise ValueError("所选文档登记信息与运行开始时不一致")
    current_inputs = capture_scope_inputs(project_id, ordered, workspace_root=workspace_root)
    if record["input_files"] != current_inputs:
        raise ValueError("当前 PDF、索引或元数据与运行开始时不一致")
    checked = deepcopy(record)
    checked["session_snapshot"] = saved_snapshot
    return checked
