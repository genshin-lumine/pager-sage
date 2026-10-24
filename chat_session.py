from uuid import uuid4


def create_chat_session(scope_key):
    return {
        "session_id": f"session_{uuid4().hex}",
        "scope_key": scope_key,  # 保存传入的范围标识
        "messages": [],   # 为这次会话创建空消息列表
    }

def ensure_chat_session(session, scope_key):
    if session is None:
        return create_chat_session(scope_key)

    if session["scope_key"] != scope_key:  # 判断原会话的范围是否与当前范围不同
        return create_chat_session(scope_key)

    return session


def build_history_messages(session, scope_key):
    if session["scope_key"] != scope_key:
        raise ValueError("会话范围与当前范围不匹配")

    messages = []

    for message in session["messages"]:
        message = {"role": message["role"], "content": message["content"]}
        messages.append(message)

    return messages
