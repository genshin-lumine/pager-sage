from langchain_deepseek import ChatDeepSeek
from paper_sage.qa.legacy_parser import parse_answer
from openai import LengthFinishReasonError, ContentFilterFinishReasonError


class IncompleteAnswerError(Exception):
    pass

def create_chat_model(api_key):
    """问答与 Agent 共用模型配置；只创建客户端，不发出请求。"""
    return ChatDeepSeek(
        model="deepseek-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        max_tokens=1024,
        timeout=60,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )


def ask_deepseek(messages, api_key):
    model = create_chat_model(api_key)
    json_model = model.bind(
        response_format = {"type": "json_object"}
    )
    
    try:
        response = json_model.invoke(messages)

    except LengthFinishReasonError as error:
        raise IncompleteAnswerError(
            "回答达到长度上限，可能不完整。请缩小问题范围后重试。"
        ) from error

    except ContentFilterFinishReasonError as error:
        raise IncompleteAnswerError(
            "服务因内容过滤未返回完整回答，请调整问题后重试。"
        ) from error

    finish_reason = response.response_metadata.get("finish_reason")

    if finish_reason == "length":
        raise IncompleteAnswerError(
            "回答达到长度上限，可能不完整。请缩小问题范围后重试。"
        )

    if finish_reason != "stop":
        raise IncompleteAnswerError(
            "回答未正常结束，请稍后重试。"
        )

    return parse_answer(response.content)
