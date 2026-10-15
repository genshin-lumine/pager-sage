import streamlit as st
from openai import APIError, APITimeoutError
from llm import ask_deepseek
from json import JSONDecodeError

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
        

reference_text = st.text_area(
    "参考资料",
    placeholder="在这里粘贴一小段论文原文",
    height=180,
)

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
            "content": (
              f"<reference>\n"
              f"{reference_text.strip()}\n"
              f"</reference>\n\n"
              f"问题：{question.strip()}"  
            ),
        }

        # 页面负责组装本次请求的完整消息
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位论文阅读助手，请用中文回答。"
                    "论文事实只能依据本次用户消息中 <reference> 内的资料。"
                    "聊天历史仅用于理解追问，不作为事实依据。"
                    "只输出一个 JSON 对象，不要输出代码块或额外文字。"
                    "必须包含 answer、evidence、insufficient_reason 三个字段。"
                    "answer 是字符串，存放回答正文。"
                    "evidence 是字符串数组，每条依据必须逐字摘录参考资料原文。"
                    "insufficient_reason 是字符串，存放资料不足的原因。"
                    "资料足够时，evidence 提供支持回答的原文，"
                    "insufficient_reason 为空字符串。"
                    "资料为空或不足时，不要猜测；answer 明确说明无法依据资料回答，"
                    "evidence 使用空数组，insufficient_reason 说明原因。"
                    'JSON 格式示例：{"answer": "资料不足，无法回答。", '
                    '"evidence": [], '
                    '"insufficient_reason": "没有提供参考资料。"}'
                ),
            },
        ] + st.session_state.messages + [user_message]

        try:

            with st.spinner("正在等待 DeepSeek 回答……"):
                result = ask_deepseek(
                    messages,
                    st.secrets["DEEPSEEK_API_KEY"],
                )

            answer = f"**结论**\n\n{result['answer']}"
            answer += "\n\n**原文依据**"

            if result["evidence"]:
                for evidence in result["evidence"]:
                    answer += f"\n\n> {evidence}"
            else:
                answer += "\n\n无。"

            if result["insufficient_reason"]:
                answer += (
                    "\n\n**资料不足说明**\n\n"
                    f"{result['insufficient_reason']}"
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

        except APITimeoutError:
            st.error("请求超时，请稍后重试。")
        except APIError as error:
            st.error(f"调用失败：{error}")
        except JSONDecodeError:
            st.error("模型返回的内容不是有效的 JSON，请重试。")
        except ValueError as error:
            st.error(f"回答结构不符合要求：{error}")