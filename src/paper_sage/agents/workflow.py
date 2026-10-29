"""工作台的一轮处理：验证会话范围，运行路由图，再返回可保存的完整消息。"""

from copy import deepcopy
from pathlib import Path

from paper_sage.agents.graph import build_paper_graph, make_initial_state
from paper_sage.sessions.chat import build_history_messages
from paper_sage.qa.model import validate_cited_answer_record
from paper_sage.qa.report import format_cited_answer
from paper_sage.comparison.report import format_paper_comparison
from paper_sage.qa.retry_records import build_qa_retry_artifact, validate_retrieval_scope
from paper_sage.summarization.report import format_paper_summary


def format_answer(result):
    answer = f"**结论**\n\n{result['answer']}\n\n**原文依据**"
    if result["evidence"]:
        for quote in result["evidence"]:
            answer += f"\n\n> {quote}"
    else:
        answer += "\n\n无。"
    if result["insufficient_reason"]:
        answer += f"\n\n**资料不足说明**\n\n{result['insufficient_reason']}"
    return answer


def run_paper_turn(question, session, scope_key, documents, model, search_tool, *, summary_runner=None,
                   comparison_runner=None, max_attempts=None, cited_answers=False):
    """不修改 session；只有完整成功才返回一对可保存的消息。"""
    history = build_history_messages(session, scope_key)
    document_scope = tuple(sorted(
        (doc["document_id"], doc["index_version"], doc["source"]) for doc in documents
    ))
    if not documents or document_scope != scope_key[1]:
        raise ValueError("本轮文档与当前会话范围不一致")
    graph = build_paper_graph(model, search_tool, documents=[
        {"document_id": doc["document_id"], "name": Path(doc["source"]).name}
        for doc in documents
    ], summary_runner=summary_runner, comparison_runner=comparison_runner, max_attempts=max_attempts,
        cited_answers=cited_answers)
    path = []
    final = None
    # 节点执行步数与检索尝试次数不同；两次尝试需经过完整的循环路径。
    for state in graph.stream(
        make_initial_state(question, history=history),
        config={"recursion_limit": 24 if max_attempts is not None else 12}, stream_mode="values",
    ):
        path.append(state["last_node"])
        final = state
        if state["last_node"] == "execute_tool":
            # 每次检索结束都检查范围；检查通过后才继续生成本次回答。
            validate_retrieval_scope(state["retrieval_artifact"], scope_key, documents,
                                     query=state["retrieval_artifact"]["question"],
                                     reference_text=state["reference_text"])

    return completed_state_messages(final, path, session, scope_key, documents,
                                    max_attempts=max_attempts, cited_answers=cited_answers)


def completed_state_messages(final, path, session, scope_key, documents, *, max_attempts=None,
                             cited_answers=False):
    """将已完成状态转换为消息对；供普通运行和检查点恢复共用业务校验。"""
    history = build_history_messages(session, scope_key)
    document_scope = tuple(sorted((d["document_id"], d["index_version"], d["source"]) for d in documents))
    if not documents or document_scope != scope_key[1]:
        raise ValueError("完成结果的文档与当前范围不一致")
    if not isinstance(final, dict) or final.get("history") != history:
        raise ValueError("完成状态与当前聊天历史不一致")
    terminals = {"finish_run", "generate_answer", "clarify_handler", "summary_handler", "compare_handler"}
    if (not isinstance(path, list) or not path or path[0] != "start"
            or path[-1] not in terminals or path[-1] != final.get("last_node")):
        raise ValueError("工作流尚未形成完整执行路径")
    message = {"role": "assistant", "response_kind": final["response_kind"]}
    comparison_artifact = final.get("comparison_artifact")
    if comparison_artifact is not None:
        artifact = deepcopy(comparison_artifact)
        if (not isinstance(artifact, dict) or artifact.get("project_id") != scope_key[0]
                or artifact.get("question") != final["question"] or artifact.get("history") != history
                or artifact.get("response_kind") != final["response_kind"]):
            raise ValueError("比较记录不属于当前项目、问题或会话历史")
        artifact_documents = artifact.get("documents")
        if (not isinstance(artifact_documents, list) or len(artifact_documents) != 2
                or any(not isinstance(doc, dict) for doc in artifact_documents)):
            raise ValueError("比较记录必须保留两篇文档的来源和索引版本")
        by_id = {doc.get("document_id"): doc for doc in artifact_documents}
        if len(by_id) != 2 or set(by_id) != {doc["document_id"] for doc in documents}:
            raise ValueError("比较记录中的文档不属于当前会话范围")
        for document in documents:
            stored = by_id[document["document_id"]]
            if any(stored.get(field) != document[field] for field in ("source", "index_version")):
                raise ValueError("比较记录中的文档来源或索引版本与当前范围不一致")
        message["comparison_artifact"] = artifact
    if final["response_kind"] == "answer":
        if max_attempts is not None:
            message["qa_retry_artifact"] = build_qa_retry_artifact(final, scope_key, documents, max_attempts)
        artifact = deepcopy(final["retrieval_artifact"])
        if not isinstance(artifact, dict):
            raise ValueError("问答流程缺少检索记录")
        validate_retrieval_scope(artifact, scope_key, documents,
                                 query=artifact["question"], reference_text=final["reference_text"])
        for field in (
            "retrieval_results", "retrieval_channels", "retrieval_candidates",
            "rrf_results", "rerank_results", "reference_method", "rerank_elapsed_seconds",
            "retrieval_trace", "fusion_config", "rerank_config",
        ):
            message[field] = artifact[field]
        message["retrieval_query"] = artifact["question"]
        message["retrieval_trace"]["scope_key"] = deepcopy(scope_key)
        timings = message["retrieval_trace"].setdefault("other_elapsed_seconds", {})
        timings["answer"] = final["answer_elapsed_seconds"]
        answer_artifact = final.get("answer_artifact")
        if cited_answers:
            parsed = validate_cited_answer_record(
                answer_artifact, final["question"], artifact["retrieval_results"], history=history,
            )
            if parsed != final["result"]:
                raise ValueError("最终结论与结构化回答记录不一致")
            message["answer_artifact"] = deepcopy(answer_artifact)
            message["content"] = format_cited_answer(message["answer_artifact"])
        else:
            if answer_artifact is not None:
                raise ValueError("回答协议与本轮工作流配置不一致")
            message["content"] = format_answer(final["result"])
    elif final["response_kind"] == "summary":
        artifact = deepcopy(final["summary_artifact"])
        if not isinstance(artifact, dict) or len(documents) != 1:
            raise ValueError("总结缺少记录或所选文档不是一篇")
        document = documents[0]
        identity = {"project_id": scope_key[0], "document_id": document["document_id"], "source": document["source"]}
        if any(artifact.get(key) != value for key, value in identity.items()):
            raise ValueError("总结记录不属于当前会话范围")
        for points in artifact["resolved_summary"].values():
            for point in points:
                for source in point["sources"]:
                    if any(source.get(key) != value for key, value in identity.items()):
                        raise ValueError("总结原文来源不属于当前会话范围")
        message["summary_artifact"] = artifact
        message["content"] = format_paper_summary(artifact)
    elif final["response_kind"] == "compare":
        if "comparison_artifact" not in message:
            raise ValueError("比较流程缺少可保存的完整记录")
        message["content"] = format_paper_comparison(message["comparison_artifact"])
    elif final["response_kind"] in {"clarify", "unavailable"}:
        message["content"] = final["reply"]
    else:
        raise ValueError("工作流未生成可保存的响应")
    message["workflow_trace"] = {
        "route": deepcopy(final["route"]), "path": path,
        "question": final["question"], "scope_key": deepcopy(scope_key),
    }
    return [{"role": "user", "content": final["question"]}, message]
