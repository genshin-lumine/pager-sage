"""Day 39：观察内存检查点和会话标识；只运行普通 Python 节点。"""

from copy import deepcopy
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from paper_sage.sessions.chat import create_chat_session


class PracticeState(TypedDict):
    question: str
    clean_question: str
    question_length: int


def make_config(session, current_scope):
    # thread_id 负责查找检查点；项目和文档范围仍由程序单独核对。
    if session["scope_key"] != current_scope:
        raise ValueError("会话范围与当前范围不匹配")

    # 取出传入 session 中已有的会话编号，作为图的 thread_id。
    # 不要在这里生成新编号，否则下一次就不能按原编号查回状态。
    thread_id = session.get("session_id")
    if thread_id is None:
        raise NotImplementedError("请先完成 make_config() 中的 thread_id 填空")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("thread_id 必须是非空字符串")
    if thread_id != session["session_id"]:
        raise ValueError("本练习约定 thread_id 使用当前会话编号")
    return {"configurable": {"thread_id": thread_id}}


def build_graph(checkpointer, calls):
    def normalize_question(state):
        calls["normalize"] += 1
        return {"clean_question": state["question"].strip()}

    def measure_question(state):
        calls["measure"] += 1
        return {"question_length": len(state["clean_question"])}

    builder = StateGraph(PracticeState)
    builder.add_node("normalize_question", normalize_question)
    builder.add_node("measure_question", measure_question)
    builder.add_edge(START, "normalize_question")
    builder.add_edge("normalize_question", "measure_question")
    builder.add_edge("measure_question", END)
    # 编译时提供保存检查点的对象。
    return builder.compile(checkpointer=checkpointer)


def main():
    print("模式：内存检查点，只运行清理问题和统计长度两个节点。")
    scope = ("practice_project", (("paper_a", "v1", "practice/paper_a.pdf"),))
    session_a = create_chat_session(scope)
    session_b = create_chat_session(scope)
    sessions_before = deepcopy((session_a, session_b))
    config_a = make_config(session_a, scope)
    config_b = make_config(session_b, scope)
    assert config_a != config_b
    assert make_config(session_a, scope) == config_a
    print("同一会话的 thread_id 稳定：", True)
    print("同范围的新会话使用不同 thread_id：", True)

    saver = InMemorySaver()
    calls = {"normalize": 0, "measure": 0}
    graph = build_graph(saver, calls)
    initial_state = {
        "question": "  训练模型使用了什么硬件？  ",
        "clean_question": "",
        "question_length": 0,
    }
    result = graph.invoke(initial_state, config=config_a)
    calls_after_run = dict(calls)
    # get_state() 读取快照；values 是快照里的状态字段。
    snapshot_a = graph.get_state(config_a)
    restored = snapshot_a.values
    assert restored == result
    assert calls_after_run == calls == {"normalize": 1, "measure": 1}
    assert snapshot_a.next == ()
    print("恢复的状态类型：", type(restored))
    print("恢复的问题：", restored["clean_question"])
    print("恢复的字符数：", restored["question_length"])
    print("读取快照后节点调用次数：", calls)
    print("待执行节点：", snapshot_a.next)

    # 新的图对象仍连接原来的存储对象，并使用原会话配置。
    rebuilt_graph = build_graph(saver, calls)
    assert rebuilt_graph.get_state(config_a).values == result
    print("重建图、复用原存储后仍能查回：", True)

    # 同一存储中，另一个会话在首次运行前没有自己的检查点。
    before_b = rebuilt_graph.get_state(config_b).values
    assert before_b == {}
    print("会话 B 首次运行前的状态：", before_b)
    result_b = rebuilt_graph.invoke({
        "question": "  编码器有多少层？  ",
        "clean_question": "",
        "question_length": 0,
    }, config=config_b)
    assert rebuilt_graph.get_state(config_a).values == result
    assert rebuilt_graph.get_state(config_b).values == result_b
    print("B 运行后，A 的状态保持不变：", True)

    # 换成全新的内存存储；这里没有真的重启 Python 进程。
    empty_graph = build_graph(InMemorySaver(), calls)
    missing = empty_graph.get_state(config_a).values
    assert missing == {}
    print("更换全新内存存储后，原编号查到的状态：", missing)

    changed_scopes = [
        ("切换项目", ("other_project", scope[1])),
        ("切换所选文档", ("practice_project", (("paper_b", "v1", "practice/paper_b.pdf"),))),
        ("切换索引版本", ("practice_project", (("paper_a", "v2", "practice/paper_a.pdf"),))),
    ]
    for label, changed_scope in changed_scopes:
        try:
            make_config(session_a, changed_scope)
        except ValueError as error:
            assert str(error) == "会话范围与当前范围不匹配"
            print(label, "：拦截，", error)
        else:
            raise AssertionError(label + " 未拦截旧会话")

    assert (session_a, session_b) == sessions_before
    assert calls == {"normalize": 2, "measure": 2}
    print("原聊天会话未修改，A 的聊天消息数量：", len(session_a["messages"]))
    print("内存检查点、会话区分和范围检查通过。")
    print("本次未保存磁盘文件、重启进程、请求模型或修改工作台。")
    print("图状态可以查回，不代表聊天列表已恢复或流程已从中断处续跑。")


if __name__ == "__main__":
    main()
