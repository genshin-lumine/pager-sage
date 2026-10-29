"""从 Day 35 已完成练习整理的重试判断；不调用模型、检索或修改状态。"""


def choose_next_action(outcome, attempts_used, max_attempts=2):
    if outcome not in {"answered", "insufficient", "tool_error"}:
        raise ValueError("不支持的尝试结果")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("总尝试上限必须是正整数")
    if type(attempts_used) is not int or not 1 <= attempts_used <= max_attempts:
        raise ValueError("已使用次数必须在 1 到总尝试上限之间")
    if outcome == "tool_error":
        return "stop_error"
    if outcome == "answered":
        return "finish"
    can_retry = attempts_used < max_attempts
    return "rewrite_query" if can_retry else "stop_insufficient"


def normalize_query(query):
    if not isinstance(query, str):
        raise ValueError("检索查询必须是字符串")
    return " ".join(query.split()).casefold()


def prepare_retry_query(candidate_query, previous_queries):
    """候选通过检查不代表已执行检索；次数与实际尝试记录由调用方更新。"""
    if not isinstance(candidate_query, str):
        raise ValueError("候选查询必须是字符串")
    if not isinstance(previous_queries, list) or not previous_queries:
        raise ValueError("需要至少一条已尝试的查询")
    previous_keys = set()
    for previous_query in previous_queries:
        key = normalize_query(previous_query)
        if not key:
            raise ValueError("已尝试的查询不能为空")
        previous_keys.add(key)
    query = " ".join(candidate_query.split())
    if not query:
        return {"action": "stop_insufficient", "query": None, "reason": "empty_query"}
    query_key = normalize_query(query)
    is_duplicate = query_key in previous_keys
    if is_duplicate:
        return {"action": "stop_insufficient", "query": None, "reason": "duplicate_query"}
    return {"action": "search", "query": query, "reason": ""}
