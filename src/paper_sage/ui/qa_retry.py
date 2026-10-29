"""仅展示已保存的问答尝试；页面重运行不执行模型、检索或磁盘读取。"""

import streamlit as st


STOP_LABELS = {
    "answered": "已得到回答", "attempt_limit": "已达到尝试上限，资料仍不足",
    "duplicate_query": "改写查询重复，停止检索", "empty_query": "没有新的查询，停止检索",
}


def render_qa_retry_details(artifact):
    with st.expander("查看本轮检索尝试"):
        st.write(f"检索尝试：{artifact['attempts_used']} / {artifact['max_attempts']}")
        st.write(f"停止原因：{STOP_LABELS[artifact['stop_reason']]}")
        st.caption("以下是各次尝试的候选资料和模型判断，可对照原文核对；得到回答不等于事实已自动验证。")
        for record in artifact["attempt_records"]:
            status = "已得到回答" if record["status"] == "answered" else "资料不足"
            st.markdown(f"**尝试 {record['attempt']} · {status}**")
            st.write("检索查询：", record["query"])
            if record["result"]["insufficient_reason"]:
                st.write("资料不足说明：", record["result"]["insufficient_reason"])
            answer = record.get("answer_artifact")
            if answer is not None:
                for number, claim in enumerate(answer["result"]["claims"], start=1):
                    st.write(f"结论 {number}：", claim["text"])
                    st.caption("引用片段：" + "、".join(claim["chunk_ids"]))
            for item in record["retrieval_artifact"]["retrieval_results"]:
                chunk = item["chunk"]
                st.caption(f"候选 {item['rank']} · PDF 第 {chunk['pdf_page']} 页 · 来源：{chunk['source']}")
                st.caption(f"片段标识：{chunk['chunk_id']}")
                st.code(chunk["text"], language=None, wrap_lines=True)
            number = record["attempt"]
            if number <= len(artifact["rewrite_records"]):
                rewrite = artifact["rewrite_records"][number - 1]
                st.write("候选改写：", rewrite["rewrite"]["query"] or "未提供新查询")
                st.write("改写理由：", rewrite["rewrite"]["reason"])
                if rewrite["decision"]["action"] == "stop_insufficient":
                    st.caption(STOP_LABELS[rewrite["decision"]["reason"]])


def render_qa_retry_error(error):
    from openai import AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError
    from paper_sage.retrieval.tools import EmptyRetrievalError

    state = error.state
    cause = error.__cause__
    causes = []
    while cause is not None and len(causes) < 5:
        causes.append(cause)
        cause = cause.__cause__
    if any(isinstance(item, EmptyRetrievalError) for item in causes):
        st.warning("当前检索没有可用的论文片段，本轮未保存。")
        return
    message = {
        "request_error": "工具请求失败", "tool_error": "检索工具执行失败",
        "answer_error": "回答调用失败或未通过完整性、结构与引用检查", "rewrite_error": "查询改写失败",
    }.get(state["retry_status"], "问答流程失败")
    for kind, label in (
        (AuthenticationError, "身份验证失败，请检查 DeepSeek API Key"),
        (RateLimitError, "请求受到限流，请稍后重试"),
        (APITimeoutError, "请求超时，请稍后重试"),
        (APIConnectionError, "无法连接 DeepSeek，请检查网络连接"),
    ):
        if any(isinstance(item, kind) for item in causes):
            message = label
            break
    if state["retry_status"] == "tool_error":
        for prefix in ("精排资源加载失败", "论文精排失败"):
            if any(str(item).startswith(prefix) for item in causes):
                message = prefix
    st.error(message + "。本轮未加入会话，原会话保持不变。")
    st.caption(f"本轮已占用 {state['attempts_used']} 次检索尝试；错误不会被当作正常的资料不足回答。")
