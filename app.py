import streamlit as st
import requests
from llm import ask_deepseek

st.set_page_config(page_title="Paper Sage", page_icon="📚")

st.title("Paper Sage")
st.caption("我的论文问答工作台 · Day 1")

# 第一次打开会话时，创建聊天记录列表
if "messages" not in st.session_state:
    st.session_state.messages = []

# 清除回话记录
if st.button("清空会话"):
    st.session_state.messages = []

# 每次页面重新运行时，重新显示已有记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        

question = st.text_input(
    "你想了解论文的什么内容？",
    placeholder="例如：这篇论文的主要贡献是什么？",
)


if st.button("提交问题"):
    if not question.strip():
        st.warning("请先输入一个问题")
    else:
        user_message = {
            "role": "user",
            "content": question.strip(),
        }

        # 页面负责组装本次请求的完整消息
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位论文学习助手，请用中文清楚地回答。"
                    "没有提供论文原文时，不要声称已经阅读论文。"
                ),
            },
        ] + st.session_state.messages + [user_message]

        try:
            with st.spinner("正在等待 DeepSeek 回答……"):
                answer = ask_deepseek(
                    messages,
                    st.secrets["DEEPSEEK_API_KEY"],
                )

            assistant_message = {
                "role": "assistant",
                "content": answer,
            }

            # 页面负责保存成功完成的一轮对话
            st.session_state.messages.extend(
                [user_message, assistant_message]
            )

            # 页面负责展示新消息
            for message in [user_message, assistant_message]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        except requests.Timeout:
            st.error("请求超时，请稍后重试。")
        except requests.RequestException as error:
            st.error(f"调用失败：{error}")
