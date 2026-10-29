
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class PracticeState(TypedDict):
    question: str
    clean_question: str
    question_length: int


def normalize_question(state: PracticeState):
    print("进入节点：normalize_question")
    clean_question = state["question"].strip()

    # 返回更新 clean_question 字段的字典。
    return {"clean_question": clean_question}


def measure_question(state: PracticeState):
    print("进入节点：measure_question")
    clean_question = state["clean_question"]
    print("第二个节点读到的问题：", repr(clean_question))
    return {"question_length": len(clean_question)}


def build_graph():
    builder = StateGraph(PracticeState)

    # 注册节点：第一个参数是节点名，第二个参数是函数对象。
    builder.add_node("normalize_question", normalize_question)
    builder.add_node("measure_question", measure_question)

    # 连接执行顺序：开始 → 清理首尾空白 → 统计字符数 → 结束。
    builder.add_edge(START, "normalize_question")
    builder.add_edge("normalize_question", "measure_question")
    builder.add_edge("measure_question", END)

    return builder.compile()


def main():
    graph = build_graph()
    initial_state = {
        "question": "  训练模型使用了什么硬件？  ",
        "clean_question": "",
        "question_length": 0,
    }

    print("初始状态：", initial_state)
    result = graph.invoke(initial_state)
    print("最终状态：", result)
    print("原问题字段保留：", result["question"] == initial_state["question"])
    print("本次只运行两个普通 Python 节点，未调用检索模型或 DeepSeek。")


if __name__ == "__main__":
    main()
