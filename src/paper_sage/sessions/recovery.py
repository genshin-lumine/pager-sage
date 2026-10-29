"""工作台会话保存与恢复：校验当前文件、来源及回答记录，不调用模型。"""

from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath

from paper_sage.qa.model import validate_cited_answer_record
from paper_sage.qa.report import format_cited_answer
from paper_sage.comparison.report import format_paper_comparison, validate_comparison_artifact
from paper_sage.comparison.service import _check_current_project
from paper_sage.agents.router import parse_route
from paper_sage.projects.store import get_index_dir
from paper_sage.qa.retry_records import build_qa_retry_artifact, validate_retrieval_scope
from paper_sage.retrieval.context import build_reference_text
from paper_sage.sessions.store import (load_chat_session, make_session_snapshot, normalize_session_scope,
                           restore_session_snapshot, save_chat_session)
from paper_sage.summarization.report import format_paper_summary, load_saved_summary


RETRIEVAL_FIELDS = (
    "retrieval_results", "retrieval_channels", "retrieval_candidates", "rrf_results",
    "rerank_results", "reference_method", "rerank_elapsed_seconds", "retrieval_trace",
    "fusion_config", "rerank_config",
)


def _file_path(root, value, area):
    path = (root / value).absolute()
    if not path.is_relative_to(root / "data" / area):
        raise ValueError(f"会话来源文件必须位于 data/{area} 目录")
    current = root
    for part in path.relative_to(root).parts:
        if part == ".." or ":" in part or part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved():
            raise ValueError("会话来源路径包含无效片段")
        current /= part
        if current.is_symlink() or current.is_junction():
            raise ValueError("会话来源路径不能包含链接或目录联接")
    if not path.is_file():
        raise FileNotFoundError(f"会话来源文件不存在：{value}")
    return path


def _scope(project_id, documents):
    return normalize_session_scope((project_id, [
        (doc["document_id"], doc["index_version"], doc["source"]) for doc in documents
    ]))


def capture_scope_inputs(project_id, documents, *, workspace_root=None):
    """检查当前项目登记，再记录所选 PDF、索引与元数据的摘要。"""
    root = Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    scope = _scope(project_id, documents)
    if not scope[1]:
        raise ValueError("会话保存和恢复需要选中文档")
    ordered = sorted(documents, key=lambda doc: doc["document_id"])
    _check_current_project(root, project_id, ordered)
    records = []
    for document in ordered:
        index_dir = get_index_dir(project_id, document["document_id"], document["index_version"])
        files = []
        for value, area in [(document["source"], "papers"),
                            (index_dir / "metadata.json", "indexes"),
                            (index_dir / "index.faiss", "indexes")]:
            path = _file_path(root, value, area)
            files.append({"path": path.relative_to(root).as_posix(),
                          "sha256": sha256(path.read_bytes()).hexdigest()})
        if document.get("file_hash") and files[0]["sha256"] != document["file_hash"]:
            raise ValueError("PDF 内容与当前项目登记不一致")
        records.append({"document_id": document["document_id"],
                        "index_version": document["index_version"], "files": files})
    return records


def _current_chunks(root, scope):
    chunks = {}
    for document_id, version, source in scope[1]:
        value = get_index_dir(scope[0], document_id, version) / "metadata.json"
        metadata = json.loads(_file_path(root, value, "indexes").read_text(encoding="utf-8"))
        for chunk in metadata["chunks"]:
            if (chunk["project_id"] != scope[0] or chunk["document_id"] != document_id
                    or chunk["source"] != source or chunk["chunk_id"] in chunks):
                raise ValueError("当前索引的片段归属或标识不一致")
            chunks[chunk["chunk_id"]] = chunk
    return chunks


def _check_chunks(value, canonical):
    if isinstance(value, dict):
        if "chunk_id" in value:
            chunk = canonical.get(value["chunk_id"])
            if chunk is None:
                raise ValueError("会话中的片段不在当前索引中")
            for field in ("project_id", "document_id", "source", "pdf_page", "text", "chunk_size", "chunk_overlap"):
                if field in value and value[field] != chunk.get(field):
                    raise ValueError("会话片段的原文或来源与当前索引不一致")
        for item in value.values():
            _check_chunks(item, canonical)
    elif isinstance(value, list):
        for item in value:
            _check_chunks(item, canonical)


def _check_qa(message, question, history, scope, documents):
    artifact = message["answer_artifact"]
    validate_cited_answer_record(artifact, question, message["retrieval_results"], history=history)
    retrieval = {field: message[field] for field in RETRIEVAL_FIELDS}
    retrieval["question"] = message["retrieval_query"]
    validate_retrieval_scope(retrieval, scope, documents, query=retrieval["question"],
                             reference_text=build_reference_text(message["retrieval_results"]))
    retry = message["qa_retry_artifact"]
    last = retry["attempt_records"][-1]
    state = {
        "question": question, "history": history, "response_kind": "answer", "retry_error": None,
        "attempt_records": retry["attempt_records"], "attempts_used": retry["attempts_used"],
        "previous_queries": [record["query"] for record in retry["attempt_records"]],
        "rewrite_records": retry["rewrite_records"], "result": last["result"],
        "retrieval_artifact": last["retrieval_artifact"], "reference_text": last["reference_text"],
        "answer_artifact": artifact, "retry_status": retry["status"], "stop_reason": retry["stop_reason"],
    }
    if build_qa_retry_artifact(state, scope, documents, retry["max_attempts"]) != retry:
        raise ValueError("会话中的检索尝试记录不一致")
    for field in RETRIEVAL_FIELDS:
        if field != "retrieval_trace" and message[field] != last["retrieval_artifact"][field]:
            raise ValueError("会话候选展示与最终检索尝试不一致")
    if message["retrieval_query"] != last["query"]:
        raise ValueError("会话最终检索查询不一致")
    return format_cited_answer(artifact)


def validate_workbench_session(session, project_id, documents, *, workspace_root=None):
    root = Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    scope = _scope(project_id, documents)
    checked = restore_session_snapshot(make_session_snapshot(session), scope)
    if not checked["messages"]:
        raise ValueError("当前会话为空，没有可保存或恢复的消息")
    inputs = capture_scope_inputs(project_id, documents, workspace_root=root)
    try:
        canonical = _current_chunks(root, scope)
        history = []
        for index in range(0, len(checked["messages"]), 2):
            user, message = checked["messages"][index:index + 2]
            if message.get("input_files") is None:
                raise ValueError("这份旧记录缺少文件快照，请新建会话后保存")
            if message["input_files"] != inputs:
                raise ValueError("当前 PDF 或索引与会话保存的版本不一致")
            workflow = message["workflow_trace"]
            if workflow["question"] != user["content"] or workflow["scope_key"] != scope:
                raise ValueError("工作流记录与会话问题或范围不一致")
            route = parse_route(json.dumps(workflow["route"]))
            path = workflow["path"]
            if not isinstance(path, list) or not path or path[0] != "start":
                raise ValueError("会话工作流路径无效")
            _check_chunks(message, canonical)
            kind = message["response_kind"]
            if kind == "answer":
                if route["intent"] != "qa" or path[-1] not in {"finish_run", "generate_answer"}:
                    raise ValueError("问答记录与工作流路由不一致")
                expected = _check_qa(message, user["content"], history, scope, documents)
            elif kind == "summary":
                if route["intent"] != "summary" or path[-1] != "summary_handler" or len(documents) != 1:
                    raise ValueError("总结记录与工作流路由不一致")
                artifact = message["summary_artifact"]
                current = load_saved_summary(artifact["report_path"], project_id, documents[0], workspace_root=root)
                if type(artifact["reused"]) is not bool:
                    raise ValueError("总结复用标记无效")
                current["reused"] = artifact["reused"]
                if current != artifact:
                    raise ValueError("总结记录与当前报告或原文不一致")
                expected = format_paper_summary(artifact)
            elif kind in {"compare", "clarify"} and message.get("comparison_artifact") is not None:
                artifact = message["comparison_artifact"]
                validate_comparison_artifact(artifact)
                if (route["intent"] != "compare" or path[-1] != "compare_handler"
                        or artifact["question"] != user["content"] or artifact["history"] != history
                        or artifact["response_kind"] != kind):
                    raise ValueError("比较记录与会话问题或历史不一致")
                expected = format_paper_comparison(artifact)
            elif kind == "clarify":
                if route["intent"] == "clarify" and path[-1] == "clarify_handler":
                    expected = route["clarification_question"]
                elif route["intent"] == "summary" and path[-1] == "summary_handler" and len(documents) != 1:
                    expected = "当前总结流程一次处理一篇论文，请只勾选你希望总结的那篇论文后再提交。"
                elif route["intent"] == "compare" and path[-1] == "compare_handler" and len(documents) != 2:
                    expected = "当前比较流程一次处理两篇论文，请恰好勾选希望比较的两篇论文后再提交。"
                else:
                    raise ValueError("澄清记录与当前流程不一致")
            else:
                raise ValueError("这份会话不是当前工作台支持恢复的完整响应")
            if message["content"] != expected:
                raise ValueError("聊天正文与保存的回答依据或响应不一致")
            history.extend({"role": item["role"], "content": item["content"]} for item in (user, message))
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("会话的回答、来源或工作流记录不完整") from error
    if capture_scope_inputs(project_id, documents, workspace_root=root) != inputs:
        raise ValueError("检查期间论文或索引发生变化，请重新操作")
    return checked


def save_workbench_session(session, project_id, documents, *, workspace_root=None):
    checked = validate_workbench_session(session, project_id, documents, workspace_root=workspace_root)
    return save_chat_session(checked, workspace_root=workspace_root)


def restore_workbench_session(session_id, project_id, documents, *, workspace_root=None):
    scope = _scope(project_id, documents)
    restored = load_chat_session(project_id, session_id, scope, workspace_root=workspace_root)
    return validate_workbench_session(restored, project_id, documents, workspace_root=workspace_root)
