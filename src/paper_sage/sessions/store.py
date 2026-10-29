"""保存完整聊天消息及范围；不调用模型，不保存页面资源或运行中的图。

本层检查存档基础结构和来源归属；具体问答、总结、比较报告仍需各自的业务校验器。
"""

import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile


SNAPSHOT_PURPOSE = "paper_sage_chat_session"
RESPONSE_KINDS = {"answer", "summary", "compare", "clarify", "unavailable"}


def normalize_session_scope(scope_key):
    """将 JSON 数组或内存元组恢复为项目中使用的排序范围元组。"""
    if not isinstance(scope_key, (list, tuple)) or len(scope_key) != 2:
        raise ValueError("会话范围必须包含项目和文档列表")
    project_id, documents = scope_key
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("会话范围缺少项目标识")
    if not isinstance(documents, (list, tuple)):
        raise ValueError("会话范围中的文档必须是列表或元组")
    rows = []
    seen = set()
    for document in documents:
        if (not isinstance(document, (list, tuple)) or len(document) != 3
                or any(not isinstance(value, str) or not value.strip() for value in document)):
            raise ValueError("范围文档必须包含文档标识、索引版本和来源")
        if document[0] in seen:
            raise ValueError("会话范围包含重复文档标识")
        seen.add(document[0])
        rows.append(tuple(document))
    return project_id, tuple(sorted(rows))


def _copy_record(value, scope, *, to_json):
    """复制 JSON 数据；scope_key 是保留字段，写入数组、读取时恢复元组。"""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("会话记录的字典键必须是字符串")
        if "project_id" in value and value["project_id"] != scope[0]:
            raise ValueError("消息附加记录来自其他项目")
        if "document_id" in value:
            by_id = {row[0]: row for row in scope[1]}
            document_id = value["document_id"]
            if not isinstance(document_id, str) or document_id not in by_id:
                raise ValueError("消息附加记录来自未选中的文档")
            _, index_version, source = by_id[document_id]
            if "source" in value and value["source"] != source:
                raise ValueError("消息附加记录的文档来源不匹配")
            if "index_version" in value and value["index_version"] != index_version:
                raise ValueError("消息附加记录的索引版本不匹配")
        copied = {}
        for key, item in value.items():
            if key == "scope_key":
                item_scope = normalize_session_scope(item)
                if item_scope != scope:
                    raise ValueError("消息附加记录的范围与当前会话不一致")
                copied[key] = [scope[0], [list(row) for row in scope[1]]] if to_json else scope
            else:
                copied[key] = _copy_record(item, scope, to_json=to_json)
        return copied
    if isinstance(value, list):
        return [_copy_record(item, scope, to_json=to_json) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("会话附加记录只能包含 JSON 数据；不能保存 PDF 字节、模型对象或非有限数值")


def _check_session_id(session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r"session_[0-9a-f]{32}", session_id):
        raise ValueError("会话编号格式无效")


def _checked_session(session, current_scope=None):
    if not isinstance(session, dict) or set(session) != {"session_id", "scope_key", "messages"}:
        raise ValueError("会话必须包含 session_id、scope_key 和 messages")
    _check_session_id(session["session_id"])
    scope = normalize_session_scope(session["scope_key"])
    if current_scope is not None and scope != normalize_session_scope(current_scope):
        raise ValueError("已保存会话与当前项目、文档或索引版本不匹配")
    messages = session["messages"]
    if not isinstance(messages, list) or len(messages) % 2:
        raise ValueError("会话消息必须按完整的用户与助手消息对保存")
    for index, message in enumerate(messages):
        role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError("会话消息必须按 user、assistant 顺序成对保存")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("聊天正文必须是非空字符串")
        if "response_kind" in message:
            kind = message["response_kind"]
            if not isinstance(kind, str) or kind not in RESPONSE_KINDS:
                raise ValueError("不支持的聊天响应类型")
    try:
        return _copy_record(session, scope, to_json=False)
    except RecursionError as error:
        raise ValueError("会话记录存在循环引用或嵌套过深") from error


def make_session_snapshot(session):
    """完整保留消息附加数据；返回可写入 JSON 的独立副本。"""
    checked = _checked_session(session)
    return {
        "schema_version": 1,
        "purpose": SNAPSHOT_PURPOSE,
        "session": _copy_record(checked, checked["scope_key"], to_json=True),
    }


def restore_session_snapshot(snapshot, current_scope):
    """恢复前核对范围；检查结构和来源标识，不自动核实回答事实。"""
    if not isinstance(snapshot, dict) or set(snapshot) != {"schema_version", "purpose", "session"}:
        raise ValueError("会话存档结构不完整")
    if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1:
        raise ValueError("不支持的会话存档版本")
    if snapshot["purpose"] != SNAPSHOT_PURPOSE:
        raise ValueError("不是 Paper Sage 聊天会话存档")
    return _checked_session(snapshot["session"], current_scope)


def _session_path(project_id, session_id, workspace_root):
    _check_session_id(session_id)
    if (not isinstance(project_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", project_id)
            or re.fullmatch(r"(?i:con|prn|aux|nul|com[0-9]|lpt[0-9])", project_id)):
        raise ValueError("项目标识不能作为会话目录名")
    root = Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    path = root / "data" / "sessions" / project_id / f"{session_id}.json"
    # 逐级拒绝链接或 Windows 目录联接，避免写到指定工作区之外。
    candidate = root
    for part in path.relative_to(root).parts:
        candidate = candidate / part
        if candidate.is_symlink() or candidate.is_junction():
            raise ValueError("会话存储路径不能包含链接或目录联接")
    if not path.resolve().is_relative_to(root):
        raise ValueError("会话存储路径超出工作区")
    return path


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("会话存档包含重复的 JSON 字段")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"会话存档包含非有限数值：{value}")


def _read_snapshot(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)


def save_chat_session(session, *, workspace_root=None):
    """先准备完整存档，再原子替换；同一编号不能重新绑定另一文档范围。"""
    snapshot = make_session_snapshot(session)
    checked = restore_session_snapshot(snapshot, session["scope_key"])
    path = _session_path(checked["scope_key"][0], checked["session_id"], workspace_root)
    if path.exists():
        previous = restore_session_snapshot(_read_snapshot(path), checked["scope_key"])
        if previous["session_id"] != checked["session_id"]:
            raise ValueError("已有存档的会话编号与文件名不一致")
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".session-", suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def load_chat_session(project_id, session_id, current_scope, *, workspace_root=None):
    scope = normalize_session_scope(current_scope)
    if project_id != scope[0]:
        raise ValueError("待恢复会话的项目与当前项目不匹配")
    path = _session_path(project_id, session_id, workspace_root)
    restored = restore_session_snapshot(_read_snapshot(path), scope)
    if restored["session_id"] != session_id:
        raise ValueError("存档会话编号与请求的会话编号不一致")
    return restored


def list_chat_sessions(current_scope, *, workspace_root=None):
    """只列出同项目、同文档范围的有效非空存档；不创建目录或修改文件。"""
    scope = normalize_session_scope(current_scope)
    directory = _session_path(scope[0], "session_" + "0" * 32, workspace_root).parent
    records = []
    for path in directory.glob("session_*.json"):
        try:
            session = load_chat_session(scope[0], path.stem, scope, workspace_root=workspace_root)
            if not session["messages"]:
                continue
            records.append({
                "session_id": session["session_id"],
                "title": session["messages"][0]["content"],
                "turn_count": len(session["messages"]) // 2,
                "modified_at": path.stat().st_mtime,
            })
        except (OSError, ValueError, TypeError):
            # 其他范围及损坏存档不进入可选项；不删除原文件。
            continue
    return sorted(records, key=lambda item: (item["modified_at"], item["session_id"]), reverse=True)
