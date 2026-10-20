import json


def parse_answer(raw_response):
    result = json.loads(raw_response)

    if not isinstance(result, dict):
        raise ValueError("回答必须是一个 JSON 对象")

    for field in ("answer", "evidence", "insufficient_reason"):
        if field not in result:
            raise ValueError(f"回答缺少字段：{field}")

    if not isinstance(result["answer"], str):
        raise ValueError("answer 必须是字符串")

    if not isinstance(result["insufficient_reason"], str):
        raise ValueError("insufficient_reason 必须是字符串")

    if not isinstance(result["evidence"], list):
        raise ValueError("evidence 必须是列表")

    for evidence in result["evidence"]:
        if not isinstance(evidence, str):
            raise ValueError("evidence 中的每条依据必须是字符串")

    if result["insufficient_reason"].strip() and result["evidence"]:
        raise ValueError("资料不足时，evidence 必须为空列表")
    return result