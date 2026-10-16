history = [
    {"role": "user", "content": "之前的问题"},
    {"role": "assistant", "content": "之前的回答"},
]


def mock_ask_deepseek():
    print("进入模拟调用函数")
    return "模拟回答"


try:
    print("准备调用")
    answer = mock_ask_deepseek()

    history.extend([
        {"role": "user", "content": "新的问题"},
        {"role": "assistant", "content": answer},
    ])

    print("本轮对话已保存")

except TimeoutError as error:
    print("捕获到错误：", error)

print("历史消息数量：", len(history))