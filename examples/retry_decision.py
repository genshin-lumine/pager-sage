"""Day 35：用固定结果判断下一步，不执行检索或模型调用。"""


def choose_next_action(outcome, attempts_used, max_attempts=2):
    """在一次检索尝试结束后决定下一步。

    attempts_used 包含刚才的尝试，从 1 开始计数，失败的工具尝试也计入。
    max_attempts 是总尝试上限，而不是允许额外重试的次数。
    outcome 在本练习中由固定场景提供，不是程序对事实正确性的判断：
      answered：已拿到正常结束、通过结构校验且未报告资料不足的回答。
      insufficient：已拿到合法的资料不足结果。
      tool_error：检索工具执行失败。
    本函数只返回动作名称，不执行该动作，也不改变计数。
    """
    if outcome not in {"answered", "insufficient", "tool_error"}:
        raise ValueError("不支持的尝试结果")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("总尝试上限必须是正整数")
    if type(attempts_used) is not int or not 1 <= attempts_used <= max_attempts:
        raise ValueError("已使用次数必须在 1 到总尝试上限之间")

    # 工具故障与资料不足分别处理；今天不自动重试工具故障。
    if outcome == "tool_error":
        return "stop_error"

    # 若最后一次允许的尝试已经得到回答，仍正常完成。
    if outcome == "answered":
        return "finish"

    # 用 attempts_used 和 max_attempts 判断是否还允许再检索一次。
    can_retry = attempts_used < max_attempts
    if can_retry is None:
        raise NotImplementedError("请先填写 can_retry 的布尔表达式")
    if type(can_retry) is not bool:
        raise ValueError("can_retry 应当是布尔值")

    if can_retry:
        return "rewrite_query"
    return "stop_insufficient"


def main():
    cases = [
        ("最后一次尝试已经得到回答", "answered", 2, 2, "finish"),
        ("首次检索后资料不足", "insufficient", 1, 2, "rewrite_query"),
        ("两次检索后仍资料不足", "insufficient", 2, 2, "stop_insufficient"),
        ("首次检索时工具报错", "tool_error", 1, 2, "stop_error"),
    ]
    print("模式：固定场景，只检查下一步动作，不运行模型或检索。")
    for title, outcome, attempts_used, max_attempts, expected in cases:
        action = choose_next_action(outcome, attempts_used, max_attempts)
        print(f"场景：{title} · 已尝试：{attempts_used}/{max_attempts} · 下一步：{action}")
        if action != expected:
            raise AssertionError(f"“{title}”的动作不符合规则，请检查次数边界")
    print("四组判断通过；本步未执行查询改写、循环或任何外部调用。")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, NotImplementedError, AssertionError) as error:
        print("练习尚未通过：", error)
        raise SystemExit(1)
