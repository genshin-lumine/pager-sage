"""准备可保存的问答尝试记录；检查来源、关联和终止条件，不判断事实真伪。"""

from copy import deepcopy

from paper_sage.qa.legacy_parser import parse_answer
from paper_sage.qa.model import validate_cited_answer_record
from paper_sage.retrieval.tools import EMPTY_REFERENCE_TEXT
from paper_sage.qa.retry import validate_attempt_limit
from paper_sage.qa.query_rewrite import validate_query_rewrite_record
from paper_sage.retrieval.context import build_reference_text


def validate_retrieval_scope(artifact, scope_key, documents, *, query, reference_text):
    if not isinstance(artifact, dict) or artifact.get("question") != query:
        raise ValueError("检索记录与本次查询不一致")
    sources = {doc["document_id"]: doc["source"] for doc in documents}
    lists = [artifact[field] for field in ("retrieval_results", "retrieval_candidates", "rrf_results", "rerank_results")]
    lists.extend(artifact["retrieval_channels"].values())
    trace = artifact["retrieval_trace"]
    if trace["question"] != query:
        raise ValueError("检索流程记录与查询不一致")
    for stage in trace["stages"]:
        lists.append(stage["results"])
    chunks_by_id = {}
    for items in lists:
        if not isinstance(items, list):
            raise ValueError("检索候选必须是列表")
        for item in items:
            chunk = item["chunk"]
            if (chunk["project_id"] != scope_key[0] or chunk["document_id"] not in sources
                    or chunk["source"] != sources[chunk["document_id"]]):
                raise ValueError("检索记录不属于当前会话范围")
            previous = chunks_by_id.setdefault(chunk["chunk_id"], chunk)
            if previous != chunk:
                raise ValueError("同一片段在不同阶段的原文或来源不一致")
    for row in trace["candidate_paths"]:
        chunk = chunks_by_id.get(row["chunk_id"])
        if chunk is None or any(row[field] != chunk[field] for field in ("source", "pdf_page")):
            raise ValueError("片段排名路径与原文来源不一致")
    if reference_text != (build_reference_text(artifact["retrieval_results"]) or EMPTY_REFERENCE_TEXT):
        raise ValueError("参考资料与本次最终候选不一致")


def build_qa_retry_artifact(state, scope_key, documents, max_attempts):
    validate_attempt_limit(max_attempts)
    if max_attempts is None or state["response_kind"] != "answer" or state["retry_error"] is not None:
        raise ValueError("只有正常完成的问答重试流程可以保存")
    attempts = state["attempt_records"]
    used = state["attempts_used"]
    if type(used) is not int or not 1 <= used <= max_attempts or len(attempts) != used:
        raise ValueError("尝试次数与保存记录不一致")
    if state["previous_queries"] != [record["query"] for record in attempts]:
        raise ValueError("查询历史与尝试记录不一致")
    cited = state.get("answer_artifact") is not None
    for number, record in enumerate(attempts, start=1):
        call = record["tool_call"]
        if (record["attempt"] != number or record["query"] != call["args"]["question"].strip()
                or record["tool_request_origin"] != ("model" if number == 1 else "query_rewrite")):
            raise ValueError("尝试编号、查询或请求来源不一致")
        validate_retrieval_scope(record["retrieval_artifact"], scope_key, documents,
                                 query=record["query"], reference_text=record["reference_text"])
        response = record["answer_response"]
        if response["finish_reason"] != "stop" or response["tool_calls"] or response["invalid_tool_calls"]:
            raise ValueError("不能保存未完整结束的回答")
        answer_artifact = record.get("answer_artifact")
        if (answer_artifact is not None) != cited:
            raise ValueError("各次尝试的回答协议不一致")
        if cited:
            parsed = validate_cited_answer_record(
                answer_artifact, state["question"], record["retrieval_artifact"]["retrieval_results"],
                history=state["history"],
            )
            if response != {key: deepcopy(answer_artifact.get(key)) for key in
                            ("raw_response", "finish_reason", "tool_calls", "invalid_tool_calls", "usage")}:
                raise ValueError("尝试响应与结构化回答原始响应不一致")
        else:
            parsed = parse_answer(response["raw_response"])
        if parsed != record["result"]:
            raise ValueError("解析结果与本次原始回答不一致")
        status = "insufficient" if record["result"]["insufficient_reason"].strip() else "answered"
        if record["status"] != status or (number < used and status != "insufficient"):
            raise ValueError("尝试结果不支持当前执行路径")
    rewrites = state["rewrite_records"]
    if len(rewrites) not in (used - 1, used) or len(rewrites) > max_attempts - 1:
        raise ValueError("改写次数与尝试次数不一致")
    decisions = []
    for number, record in enumerate(rewrites, start=1):
        decision = validate_query_rewrite_record(
            record, state["question"], state["previous_queries"][:number],
            attempts[number - 1]["result"]["insufficient_reason"], history=state["history"],
        )
        decisions.append(decision)
        if number < used and (decision["action"] != "search" or decision["query"] != attempts[number]["query"]):
            raise ValueError("重试查询与改写决定不一致")
    last = attempts[-1]
    if (state["result"] != last["result"] or state["retrieval_artifact"] != last["retrieval_artifact"]
            or state["reference_text"] != last["reference_text"] or state["retry_status"] != last["status"]):
        raise ValueError("最终回答或来源与最后一次尝试不一致")
    if state.get("answer_artifact") != last.get("answer_artifact"):
        raise ValueError("最终结构化证据与最后一次尝试不一致")
    if last["status"] == "answered":
        expected_reason = "answered"
        if len(rewrites) != used - 1:
            raise ValueError("资料充分后不应继续改写")
    elif used == max_attempts:
        expected_reason = "attempt_limit"
    else:
        if len(decisions) != used or decisions[-1]["action"] != "stop_insufficient":
            raise ValueError("未达到上限时缺少提前停止的决定")
        expected_reason = decisions[-1]["reason"]
    if state["stop_reason"] != expected_reason:
        raise ValueError("终止原因与实际尝试记录不一致")
    return {
        "schema_version": 2 if cited else 1, "question": state["question"], "scope_key": deepcopy(scope_key),
        "max_attempts": max_attempts, "attempts_used": used, "status": last["status"],
        "stop_reason": state["stop_reason"], "attempt_records": deepcopy(attempts),
        "rewrite_records": deepcopy(rewrites),
    }
