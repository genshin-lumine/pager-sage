
from langchain_core.tools import tool


@tool
def search_papers(question: str) -> dict:
    """教学用模拟检索工具，接收论文问题并返回调用信息，不执行真实检索。"""
    if not question.strip():
        raise ValueError("问题不能为空")

    print("进入工具函数，收到问题：", question)
    return {
        "question": question,
        "mode": "simulation",
        "message": "已执行模拟工具函数，尚未执行论文检索。",
    }


def main():
    print("工具名称：", search_papers.name)
    print("工具描述：", search_papers.description)
    print("输入参数定义：", search_papers.args)

    # 手工构造的教学请求，尚未调用 DeepSeek。
    tool_call = {
        "name": "search_papers",
        "args": {"question": "训练模型使用了什么硬件？"},
        "id": "call_demo_001",
        "type": "tool_call",
    }

    # 从 tool_call 中取出完整的参数字典（包含参数名和参数值）。
    tool_args = tool_call.get("args", {})

    if not isinstance(tool_args, dict):
        raise ValueError("工具调用参数必须是字典")

    print("准备执行工具")
    result = search_papers.invoke(tool_args)
    print("返回结果类型：", type(result))
    print("返回结果：", result)


if __name__ == "__main__":
    main()
