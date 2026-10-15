from langchain_deepseek import ChatDeepSeek
from answer_parser import parse_answer


def ask_deepseek(messages, api_key):
    model = ChatDeepSeek(
        model="deepseek-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        max_tokens=1024,
        timeout=60,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )


    json_model = model.bind(
        response_format = {"type": "json_object"}
    )
    
    response = json_model.invoke(messages)

    return parse_answer(response.content)