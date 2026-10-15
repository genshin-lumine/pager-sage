from answer_parser import parse_answer

# 模拟模型返回的 JSON 文本
raw_response = """
{
    "answer": "模型在 WMT 2014 英语到德语翻译任务上取得了 28.4 BLEU。",
    "evidence": [
        "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task"
    ],
    "insufficient_reason": ""
}
"""

print("解析前的类型：", type(raw_response))

result = parse_answer(raw_response)

print("解析后的类型：", type(result))
print("回答：", result["answer"])
print("evidence", result["evidence"][0])