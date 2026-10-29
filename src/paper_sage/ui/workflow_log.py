"""只读运行日志面板；只显示当前所选范围，日志不参与检查点恢复判断。"""

from pathlib import Path

import streamlit as st

from paper_sage.sessions.store import restore_session_snapshot
from paper_sage.workflows.events import read_events, summarize_calls
from paper_sage.workflows.store import load_run_record, run_paths


STATUS_NAMES = {"ok": "成功", "failed": "失败", "cancelled": "已中断",
                "unfinished": "未记录结束", "missing_start": "缺少开始记录", "invalid_pair": "事件配对异常"}


def render_run_log(project_id, run_id, scope_key, *, write_errors=None, container=None):
    panel = st.sidebar if container is None else container
    with panel.expander("查看运行日志"):
        try:
            record = load_run_record(project_id, run_id)
            restore_session_snapshot(record["context"]["session_snapshot"], scope_key)
            path = run_paths(project_id, run_id)["events"]
            events, issues = read_events(path, run_id)
        except (OSError, ValueError, RuntimeError) as error:
            st.warning(f"日志暂不可用：{type(error).__name__}")
            return
        st.caption(f"运行编号：{run_id}")
        st.caption(f"日志位置：{path.relative_to(Path.cwd().resolve()).as_posix()}")
        if write_errors:
            st.warning(f"本次有 {len(write_errors)} 个事件未成功写入，日志可能不完整。业务结果仍按原流程处理。")
        if issues:
            st.warning("以下日志行无法使用：" + "、".join(str(item["line"]) for item in issues))
        if not events:
            st.info("未找到可用日志。旧运行或日志未成功写入时可能出现这种情况；不会补造过去的调用记录。")
            return
        rows = summarize_calls(events)
        model_count = sum(row["step"].startswith("model_") for row in rows)
        st.caption(f"事件数量：{len(events)} · 调用记录：{len(rows)} · 模型调用记录：{model_count}")
        if any(row["status"] in ("unfinished", "missing_start", "invalid_pair") for row in rows):
            st.warning("存在未配对的事件，只能说明日志不完整；请结合运行状态和检查点判断。")
        table = [{"步骤": row["step"], "状态": STATUS_NAMES[row["status"]],
                  "耗时（秒）": row["elapsed_seconds"], "异常类型": row["error_type"] or "—",
                  "调用编号": row["call_id"], "开始/首条时间": row["timestamp"]} for row in rows]
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("节点耗时包含内部模型和资源获取，不能把这些行直接相加。模型调用成功仅表示返回，回答仍需通过结构检查和原文核对。")
        st.caption("resource_ 行记录资源获取，可能命中缓存，不能仅凭耗时断言命中；运行前初始化和暂停等待不计在这些节点耗时内。")
