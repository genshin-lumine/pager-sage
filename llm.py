from langchain_deepseek import ChatDeepSeek


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

    response = model.invoke(messages)

    return response.content