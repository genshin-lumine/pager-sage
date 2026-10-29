"""页面的手动会话保存与恢复；只有点击恢复才替换当前聊天记录。"""

from datetime import datetime
from hashlib import sha256
import json

from pypdf.errors import PyPdfError
import streamlit as st

from paper_sage.sessions.recovery import restore_workbench_session, save_workbench_session
from paper_sage.sessions.store import list_chat_sessions


def render_session_controls(session, project_id, documents, scope_key, *, container=None):
    panel = st.sidebar if container is None else container
    if container is None:
        panel.markdown("**聊天会话**")
    panel.caption("保存后可在重启后手动恢复。清空当前会话不会删除已保存的存档。")
    if panel.button("保存当前会话", key="save_chat_session"):
        try:
            save_workbench_session(session, project_id, documents)
        except (OSError, ValueError, RuntimeError, PyPdfError) as error:
            panel.error(f"会话未保存：{error}")
        else:
            panel.success("当前会话已保存。")

    try:
        records = list_chat_sessions(scope_key)
    except (OSError, ValueError) as error:
        panel.warning(f"暂时无法读取会话列表：{error}")
        return session
    if not records:
        panel.caption("当前所选论文范围还没有已保存的会话。")
        return session
    by_id = {record["session_id"]: record for record in records}

    def label(session_id):
        record = by_id[session_id]
        timestamp = datetime.fromtimestamp(record["modified_at"]).strftime("%m-%d %H:%M:%S")
        return f"{timestamp} · {record['title'][:40]} · {record['turn_count']} 轮"

    scope_token = sha256(json.dumps(scope_key, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    widget_key = f"saved_session_{scope_token}"
    if st.session_state.get(widget_key) not in by_id:
        st.session_state[widget_key] = None
    selected_id = panel.selectbox(
        "当前范围的已保存会话", options=list(by_id), index=None, format_func=label,
        key=widget_key, placeholder="选择一份会话存档",
    )
    if panel.button("恢复所选会话", key="restore_chat_session", disabled=selected_id is None):
        try:
            restored = restore_workbench_session(selected_id, project_id, documents)
        except (OSError, ValueError, RuntimeError, PyPdfError) as error:
            panel.error(f"会话未恢复：{error}")
        else:
            # 问题输入框稍后才创建，恢复时清除该会话以前留下的输入值。
            st.session_state.pop(f"question_{restored['session_id']}", None)
            panel.success("会话已恢复；未重新调用模型。")
            return restored
    return session
