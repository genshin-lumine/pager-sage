"""工作台运行记录的手动载入与继续；页面重运行不自动执行图。"""

from datetime import datetime
from hashlib import sha256
import json
import sqlite3

import streamlit as st

from paper_sage.qa.retry import QARetryError
from paper_sage.ui.qa_retry import render_qa_retry_error
from paper_sage.sessions.store import restore_session_snapshot
from paper_sage.workflows.runner import read_workflow_run
from paper_sage.workflows.store import list_workflow_runs, load_run_record
from paper_sage.ui.workflow_log import render_run_log


def clear_loaded_workflow():
    for key in ("workflow_loaded", "workflow_select_next", "workflow_notice"):
        st.session_state.pop(key, None)


def remember_workflow_result(outcome, scope_key, question, *, select=False):
    st.session_state["workflow_loaded"] = {
        "run_id": outcome["run_id"], "session_id": outcome["session"]["session_id"],
        "scope_key": scope_key, "status": outcome["status"], "question": question,
        "log_write_errors": outcome.get("log_write_errors", []),
    }
    if select:
        # 下拉框已在本次页面创建；留到下一次渲染前再选择新运行。
        st.session_state["workflow_select_next"] = {"scope_key": scope_key, "run_id": outcome["run_id"]}


def render_workflow_controls(session, project_id, documents, scope_key, *, on_resume, container=None):
    panel = st.sidebar if container is None else container
    loaded = st.session_state.get("workflow_loaded")
    if loaded and (loaded["scope_key"] != scope_key or loaded["session_id"] != session["session_id"]):
        clear_loaded_workflow()
        loaded = None
    if container is None:
        panel.markdown("**运行记录**")
    panel.caption("仅显示当前论文范围。载入会切换到该运行所属的会话；清空会话保留磁盘记录。")
    panel.button("刷新运行记录", key="refresh_workflow_runs")
    try:
        records = list_workflow_runs(scope_key)
    except (OSError, ValueError, sqlite3.Error) as error:
        panel.warning(f"运行列表暂不可用：{error}")
        return session
    if not records:
        panel.caption("当前范围还没有运行记录。")
        return session
    by_id = {item["run_id"]: item for item in records}
    token = sha256(json.dumps(scope_key, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    key = f"workflow_run_{token}"
    queued = st.session_state.pop("workflow_select_next", None)
    if queued and queued["scope_key"] == scope_key and queued["run_id"] in by_id:
        st.session_state[key] = queued["run_id"]
    if st.session_state.get(key) not in by_id:
        st.session_state[key] = None

    def label(run_id):
        record = by_id[run_id]
        try:
            created = datetime.fromisoformat(record["created_at"]).astimezone().strftime("%m-%d %H:%M:%S")
        except ValueError:
            created = "时间未知"
        return f"{created} · {record['question'][:40]} · {run_id[-6:]}"

    selected = panel.selectbox("当前范围的运行", options=list(by_id), index=None,
                                   format_func=label, placeholder="选择要查看的运行", key=key)
    notice = st.session_state.pop("workflow_notice", None)
    if notice and notice["scope_key"] == scope_key:
        panel.success(notice["text"])
    if panel.button("查看并载入运行", key="load_workflow_run", disabled=selected is None):
        try:
            context = load_run_record(project_id, selected)["context"]
            base = restore_session_snapshot(context["session_snapshot"], scope_key)
            # 同一会话继续遵守历史检查；用户显式载入其他会话时才切换到原始快照。
            target = session if session["session_id"] == base["session_id"] else base
            outcome = read_workflow_run(project_id, selected, target, documents)
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
            panel.error(f"运行未载入：{error}")
            st.session_state.pop("workflow_loaded", None)
        else:
            session = outcome["session"]
            st.session_state.pop(f"question_{session['session_id']}", None)
            remember_workflow_result(outcome, scope_key, context["question"])
            panel.success("运行已载入；未调用模型。")

    loaded = st.session_state.get("workflow_loaded")
    if selected is not None:
        render_run_log(project_id, selected, scope_key, container=panel,
                       write_errors=loaded.get("log_write_errors") if loaded and loaded["run_id"] == selected else None)
    if not loaded or loaded["run_id"] != selected:
        return session
    names = {"paused": "待继续", "failed": "上次执行失败", "not_started": "尚未开始", "completed": "已完成"}
    panel.write(f"运行状态：{names[loaded['status']]}")
    panel.caption(f"原问题：{loaded['question']}")
    if loaded["status"] == "completed":
        panel.caption("完整结果已在当前会话中；重复载入不会重复追加。")
        return session
    failed = loaded["status"] == "failed"
    if failed:
        panel.caption("重试会重新执行未完成步骤，可能再次调用模型。")
    button_label = "重试失败步骤" if failed else "继续运行"
    if panel.button(button_label, key="resume_workflow_run"):
        try:
            # 再次读取防止使用过期的页面状态；已经完成时直接返回，不获取模型。
            latest = read_workflow_run(project_id, selected, session, documents)
            if latest["status"] == "failed" and not failed:
                remember_workflow_result(latest, scope_key, loaded["question"])
                st.rerun()
            outcome = latest if latest["status"] == "completed" else on_resume(
                selected, session, retry_failed=failed,
            )
        except QARetryError as error:
            render_qa_retry_error(error)
        except Exception as error:
            panel.error(f"运行未完成：{error}")
        else:
            session = outcome["session"]
            st.session_state.pop(f"question_{session['session_id']}", None)
            remember_workflow_result(outcome, scope_key, loaded["question"])
            st.session_state["chat_session"] = session
            st.session_state["workflow_notice"] = {"scope_key": scope_key,
                "text": "运行已完成，结果已写入会话。" if outcome["status"] == "completed" else "运行进度已保存。"}
            st.rerun()
    return session
