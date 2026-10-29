"""问答图的有界重试：控制次数、保存尝试快照，并让故障作为异常向外传播。"""

from copy import deepcopy
from typing import TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.graph import END

from paper_sage.qa.model import CitedAnswerError
from paper_sage.qa.query_rewrite import QueryRewriteError, request_query_rewrite
from paper_sage.qa.retry_policy import choose_next_action, prepare_retry_query


class RetryState(TypedDict):
    attempts_used: int
    previous_queries: list[str]
    attempt_records: list[dict]
    rewrite_records: list[dict]
    retry_action: str
    retry_status: str
    stop_reason: str
    retry_error: dict | None


def initial_retry_state():
    return {
        "attempts_used": 0, "previous_queries": [], "attempt_records": [],
        "rewrite_records": [], "retry_action": "", "retry_status": "",
        "stop_reason": "", "retry_error": None,
    }


def validate_attempt_limit(max_attempts):
    # None 保留原单次图；显式传入 1 或 2 才启用尝试记录与控制节点。
    if max_attempts is not None and (type(max_attempts) is not int or max_attempts not in (1, 2)):
        raise ValueError("当前问答图的总尝试上限只能为 1 或 2，None 表示不启用重试流程")


def answer_response_record(response):
    if not isinstance(response, AIMessage):
        return None
    return {
        "raw_response": deepcopy(response.content),
        "finish_reason": response.response_metadata.get("finish_reason"),
        "tool_calls": deepcopy(response.tool_calls),
        "invalid_tool_calls": deepcopy(response.invalid_tool_calls),
        "usage": deepcopy(response.usage_metadata),
    }


class QARetryError(RuntimeError):
    """包含失败时的独立状态快照；调用方不能把它当成成功回答写入会话。"""

    def __init__(self, state):
        self.state = deepcopy(state)
        super().__init__(f"问答流程在 {state['retry_error']['stage']} 失败；本轮未完成")


def retry_failure(state, stage, error, *, response=None):
    snapshot = deepcopy(state)
    status = {
        "request_tool": "request_error", "execute_tool": "tool_error",
        "generate_answer": "answer_error", "rewrite_query": "rewrite_error",
    }.get(stage, "workflow_error")
    snapshot.update(
        result=None, answer_artifact=None, response_kind="error", reply="", retry_status=status,
        stop_reason=status, last_node=stage,
        retry_error={"stage": stage, "error_type": type(error).__name__},
    )
    records = snapshot["attempt_records"]
    if records and records[-1]["status"] == "running":
        records[-1].update(status=status, error=deepcopy(snapshot["retry_error"]))
        if response is not None:
            records[-1]["answer_response"] = answer_response_record(response)
    if isinstance(error, QueryRewriteError):
        snapshot["rewrite_records"].append(deepcopy(error.record))
    if isinstance(error, CitedAnswerError):
        # 失败记录保留本次资料与原始响应，不能当作成功答案写入会话。
        snapshot["answer_artifact"] = deepcopy(error.record)
        if records and records[-1]["status"] == status:
            records[-1]["answer_artifact"] = deepcopy(error.record)
            if error.record.get("raw_response") is not None:
                records[-1]["answer_response"] = {
                    key: deepcopy(error.record.get(key))
                    for key in ("raw_response", "finish_reason", "tool_calls", "invalid_tool_calls", "usage")
                }
    # 只保存错误类型与阶段，不把 SDK 异常正文写入运行记录。
    return QARetryError(snapshot)


def track_qa_node(stage, node):
    def tracked(state):
        try:
            update = node(state)
        except QARetryError:
            raise
        except Exception as error:
            raise retry_failure(state, stage, error) from error
        if stage == "request_tool":
            return update
        records = deepcopy(state["attempt_records"])
        record = records[-1]
        if stage == "execute_tool":
            record.update(reference_text=update["reference_text"],
                          retrieval_artifact=deepcopy(update["retrieval_artifact"]))
        elif stage == "generate_answer":
            result = update["result"]
            record.update(
                result=deepcopy(result),
                status="insufficient" if result["insufficient_reason"].strip() else "answered",
                answer_response=answer_response_record(update["messages"][-1]),
                answer_elapsed_seconds=update["answer_elapsed_seconds"],
                answer_artifact=deepcopy(update.get("answer_artifact")),
            )
        return {**update, "attempt_records": records}
    return tracked


def connect_qa_retry(builder, model, search_tool, max_attempts, validate_search_call, *, node_observer=None):
    """只连接 QA 分支；工具由调用方按范围绑定，改写不能换工具或选择文档。"""

    def begin_attempt(state):
        if state["attempts_used"] >= max_attempts:
            raise ValueError("已经达到问答总尝试上限")
        call = validate_search_call(state["messages"][-1], search_tool.name)
        query = call["args"]["question"].strip()
        if state["previous_queries"]:
            if prepare_retry_query(query, state["previous_queries"])["action"] != "search":
                raise ValueError("不能执行空或重复的重试查询")
        attempts_used = state["attempts_used"] + 1
        record = {
            "attempt": attempts_used, "query": query, "tool_call": deepcopy(call),
            "tool_request_origin": "model" if attempts_used == 1 else "query_rewrite",
            "status": "running", "reference_text": "", "retrieval_artifact": None,
            "result": None, "answer_response": None, "answer_elapsed_seconds": 0.0,
            "answer_artifact": None,
        }
        # 在执行工具前计数；工具报错也已占用本次尝试。清除上次供回答使用的数据。
        return {
            "attempts_used": attempts_used,
            "previous_queries": [*state["previous_queries"], query],
            "attempt_records": [*deepcopy(state["attempt_records"]), record],
            "reference_text": "", "retrieval_artifact": None, "result": None,
            "answer_artifact": None,
            "response_kind": "", "reply": "", "last_node": "begin_attempt",
        }

    def decide_next(state):
        outcome = state["attempt_records"][-1]["status"]
        action = choose_next_action(outcome, state["attempts_used"], max_attempts)
        reason = "answered" if action == "finish" else "attempt_limit" if action == "stop_insufficient" else ""
        return {"retry_action": action, "retry_status": outcome,
                "stop_reason": reason, "last_node": "decide_next"}

    def rewrite_query(state):
        if (state["retry_action"] != "rewrite_query" or state["attempts_used"] >= max_attempts
                or not state["result"]["insufficient_reason"].strip()):
            raise ValueError("只有尚有次数的合法资料不足结果才能触发改写")
        try:
            record = request_query_rewrite(
                state["question"], state["previous_queries"],
                state["result"]["insufficient_reason"], model, history=state["history"],
            )
        except Exception as error:
            raise retry_failure(state, "rewrite_query", error) from error
        decision = record["decision"]
        update = {
            "rewrite_records": [*deepcopy(state["rewrite_records"]), deepcopy(record)],
            "retry_action": decision["action"], "stop_reason": decision["reason"],
            "last_node": "rewrite_query",
        }
        if decision["action"] == "search":
            # 程序直接把已校验查询交给原工具，不再请模型重新选择工具或改动参数。
            # 来源记为 query_rewrite，不能把这条协议消息误认为新的模型工具请求。
            call = {"name": search_tool.name, "args": {"question": decision["query"]},
                    "id": f"retry_{uuid4().hex}", "type": "tool_call"}
            update["messages"] = [AIMessage(content="", tool_calls=[call],
                                             response_metadata={"finish_reason": "tool_calls"})]
        return update

    def finish_run(state):
        return {"last_node": "finish_run"}

    for name, node in (("begin_attempt", begin_attempt), ("decide_next", decide_next),
                       ("rewrite_query", rewrite_query), ("finish_run", finish_run)):
        builder.add_node(name, node_observer(name, node) if node_observer is not None else node)
    builder.add_edge("request_tool", "begin_attempt")
    builder.add_edge("begin_attempt", "execute_tool")
    builder.add_edge("execute_tool", "generate_answer")
    builder.add_edge("generate_answer", "decide_next")
    builder.add_conditional_edges("decide_next", lambda state: state["retry_action"], {
        "finish": "finish_run", "stop_insufficient": "finish_run", "rewrite_query": "rewrite_query",
    })
    builder.add_conditional_edges("rewrite_query", lambda state: state["retry_action"], {
        "search": "begin_attempt", "stop_insufficient": "finish_run",
    })
    builder.add_edge("finish_run", END)
