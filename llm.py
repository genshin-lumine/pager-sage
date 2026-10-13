import requests

def ask_deepseek(messages, api_key):
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": "deepseek-flash",
            "messages": messages,
            "thinking": {"type": "disabled"},
            "max_tokens": 1024,
        },
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    return answer