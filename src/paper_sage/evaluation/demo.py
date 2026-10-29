"""按 Day 42 固定计划演示真实工作台；默认只核对文件，--real 才提交问题。"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import traceback
from time import perf_counter
from uuid import uuid4

ROOT = Path.cwd().resolve()
PLAN = ROOT / "data/evaluations/day42_demo_plan.json"


def source_fingerprints(root=ROOT):
    """Include package code and entry points, so a nested module edit invalidates replay."""
    paths = [root / "app.py", root / "pyproject.toml", root / "requirements.txt"]
    paths.extend((root / "src/paper_sage").rglob("*.py"))
    paths.extend((root / "scripts").glob("*.py"))
    return {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def verify_inputs(plan):
    project = ROOT / "data/projects" / plan["project_id"] / "project.json"
    if sha256(project.read_bytes()).hexdigest() != plan["project_record_sha256"]:
        raise ValueError("演示项目已变化，需重新核对计划")
    for document in plan["documents"]:
        for record in document["files"]:
            if sha256((ROOT / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("演示 PDF 或索引已变化，需重新核对计划")


def write_report(path, report):
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def page_ok(at):
    # 不复制 SDK 错误正文或客户端配置，失败详情可从本轮运行及日志检查。
    require(not at.exception and not at.error, "页面出现执行异常或错误提示")


def session_value(at, key):
    try:
        return at.session_state[key]
    except KeyError:
        return None


def event_data(project_id, run_id):
    from paper_sage.workflows.events import read_events, summarize_calls
    from paper_sage.workflows.store import run_paths
    path = run_paths(project_id, run_id)["events"]
    events, issues = read_events(path, run_id)
    require(not issues, "本次运行日志不完整")
    calls = summarize_calls(events)
    return {"event_count": len(events), "calls": calls,
            "model_call_counts": dict(Counter(row["step"] for row in calls if row["step"].startswith("model_"))),
            "sha256": sha256(path.read_bytes()).hexdigest()}


def collect_sources(message):
    """按项目、文档、页码和原文去重，保留结论之外的完整原记录。"""
    records = []

    def walk(value):
        if isinstance(value, dict):
            if {"project_id", "document_id", "source", "pdf_page", "text"} <= value.keys():
                records.append(value)
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    if message["response_kind"] == "answer":
        walk(message["answer_artifact"]["claims_with_sources"])
    elif message["response_kind"] == "summary":
        walk(message["summary_artifact"]["resolved_summary"])
    elif message["response_kind"] == "compare":
        walk(message["comparison_artifact"]["generation_record"]["resolved_comparison"])
    unique = {}
    for record in records:
        key = (record["document_id"], record["pdf_page"], record["text"])
        unique.setdefault(key, deepcopy(record))
    return list(unique.values())


def run_case(case, plan, on_progress, *, completed_run_id=None, retry_failed=False):
    from pypdf import PdfReader
    from streamlit.testing.v1 import AppTest
    from paper_sage.qa.pdf import load_pdf_page_preview
    from paper_sage.projects.store import load_project
    from paper_sage.retrieval.scope import get_selected_documents
    from paper_sage.sessions.recovery import validate_workbench_session, save_workbench_session

    record = {"case_id": case["case_id"], "question": case["question"], "document_ids": case["document_ids"],
              "status": "running", "semantic_review": "pending", "checks": {}, "error_type": None}
    on_progress(record)
    started = perf_counter()
    at = None
    try:
        verify_inputs(plan)
        project_id = plan["project_id"]
        documents = get_selected_documents(load_project(project_id), case["document_ids"])
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
        at.session_state["selected_project_id"] = project_id
        at.session_state[f"scope_mode_{project_id}"] = "指定文档"
        at.session_state[f"scope_documents_{project_id}"] = case["document_ids"]
        at.session_state["pause_before_answer"] = False
        at.run()
        page_ok(at)
        require(at.session_state["chat_session"]["messages"] == [], "初始聊天不是空白")
        if completed_run_id is None:
            record["submission"] = "new"
            next(widget for widget in at.text_input if str(widget.key).startswith("question_")).set_value(case["question"])
            next(widget for widget in at.button if widget.label == "提交问题").click().run()
        else:
            record["submission"] = "retry_failed_graph_step" if retry_failed else "reloaded_existing_completed_run"
            next(widget for widget in at.selectbox if str(widget.key).startswith("workflow_run_")).set_value(completed_run_id).run()
            page_ok(at)
            next(widget for widget in at.button if widget.key == "load_workflow_run").click().run()
            if retry_failed:
                page_ok(at)
                require(session_value(at, "workflow_loaded")["status"] == "failed", "所选运行不是失败步骤")
                record["prior_log"] = event_data(project_id, completed_run_id)
                on_progress(record)
                next(widget for widget in at.button if widget.key == "resume_workflow_run").click().run()
        # 在进一步检查前保存已有的真实结果，验收失败也不丢失原回答。
        session = deepcopy(at.session_state["chat_session"])
        record["session_id"] = session["session_id"]
        record["session"] = session
        loaded = session_value(at, "workflow_loaded")
        if loaded:
            record["run_id"] = loaded["run_id"]
        on_progress(record)
        page_ok(at)
        require(len(session["messages"]) == 2, "一项演示必须得到完整的一轮回复")
        require(session["messages"][0]["content"] == case["question"], "运行原问题不属于本项演示")
        message = session["messages"][-1]
        run_id = message["workflow_run_id"]
        record.update(run_id=run_id, response_kind=message["response_kind"],
                      path=message["workflow_trace"]["path"], reply=message["content"],
                      actual_intent=message["workflow_trace"]["route"]["intent"])
        require(record["actual_intent"] == case["expected_intent"], "实际分类与计划不一致")
        require(record["response_kind"] == case["expected_response_kind"], "响应类型与计划不一致")
        validate_workbench_session(session, project_id, documents)
        record["checks"].update(empty_initial_chat=True, route=True, complete_turn=True, saved_sources=True)
        record["log"] = event_data(project_id, run_id)
        prior_ids = {call["call_id"] for call in record.get("prior_log", {}).get("calls", [])}
        current_calls = [call for call in record["log"]["calls"] if call["call_id"] not in prior_ids]
        require(all(call["status"] == "ok" for call in current_calls), "有新调用未正常结束")
        if case["expected_intent"] == "qa":
            retry = message["qa_retry_artifact"]
            record.update(attempts_used=retry["attempts_used"], stop_reason=retry["stop_reason"],
                          result=message["answer_artifact"]["result"])
            require(1 <= retry["attempts_used"] <= 2, "检索次数不符合上限")
            if case["case_id"] == "demo_insufficient":
                require(not record["result"]["claims"] and bool(record["result"]["insufficient_reason"].strip()),
                        "电能问题没有按预期说明资料不足")
            else:
                require(bool(record["result"]["claims"]), "硬件问题未得到带引用的结论")
        elif case["expected_intent"] == "summary":
            record["summary_reused"] = message["summary_artifact"]["reused"]
        elif case["expected_intent"] == "compare":
            artifact = message["comparison_artifact"]
            record["comparison_report"] = artifact["report_path"]
            require(len(artifact["records"]) == 4, "本次两个维度应有四项逐篇检索")
        elif case["expected_intent"] == "clarify":
            require(record["path"] == ["start", "classify_request", "clarify_handler"], "澄清执行了额外分支")

        previews = []
        for source in collect_sources(message):
            preview = load_pdf_page_preview(source, project_id, documents)
            pdf = PdfReader(BytesIO(preview["pdf_bytes"]))
            require(len(pdf.pages) == 1 and preview["source_text_found"] is True, "来源单页或原文查找失败")
            previews.append({"document_id": source["document_id"], "pdf_page": source["pdf_page"],
                             "source_text_found": True, "single_page_pdf": True,
                             "preview_sha256": sha256(preview["pdf_bytes"]).hexdigest()})
        record["pdf_source_checks"] = previews
        if previews:
            require(bool(at.toggle), "页面没有 PDF 查看入口")
            at.toggle[0].set_value(True).run()
            page_ok(at)
            require(bool(at.get("bidi_component")) and bool(at.get("download_button")), "PDF 组件或下载按钮缺失")
            record["checks"]["pdf_component_and_download"] = True
        require(at.session_state["chat_session"] == session, "查看来源改变了聊天")
        require(event_data(project_id, run_id)["sha256"] == record["log"]["sha256"], "查看来源新增调用")
        # 显式载入完成运行、刷新和清空均使用真实页面，不能触发新的模型调用。
        next(widget for widget in at.button if widget.key == "load_workflow_run").click().run()
        page_ok(at)
        require(at.session_state["chat_session"] == session, "重复载入改变或重复追加聊天")
        record["session_archive"] = str(save_workbench_session(session, project_id, documents))
        next(widget for widget in at.button if widget.label == "清空会话").click().run()
        page_ok(at)
        at.run()
        page_ok(at)
        require(at.session_state["chat_session"]["messages"] == [], "清空后刷新恢复了旧聊天")
        require(event_data(project_id, run_id)["sha256"] == record["log"]["sha256"], "只读操作新增模型调用")
        record["checks"].update(read_only_no_new_calls=True, reload_no_duplicate=True, clear_then_refresh=True)
        verify_inputs(plan)
        record["status"] = "behavior_passed"
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__)
        record["error_frames"] = [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                                  for frame in traceback.extract_tb(error.__traceback__)]
        if at is not None:
            record["page_errors"] = len(at.error)
            record["page_exceptions"] = len(at.exception)
            loaded = session_value(at, "workflow_loaded")
            if loaded:
                record["run_id"] = loaded["run_id"]
    finally:
        record["elapsed_seconds"] = perf_counter() - started
        on_progress(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--case", help="只运行指定 case_id；省略时按顺序执行五项")
    parser.add_argument("--replay-from", type=Path, help="复用指定本次演示报告中已完成的运行，补做页面检查；剩余项正常执行")
    parser.add_argument("--retry-failed", action="store_true", help="显式继续原报告中失败运行的未完成步骤，保留原错误记录")
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "请在项目根目录运行")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    verify_inputs(plan)
    cases = [case for case in plan["cases"] if args.case is None or case["case_id"] == args.case]
    require(bool(cases), "没有匹配的演示场景")
    recovered = {}
    failed_runs = {}
    previous = None
    if args.replay_from is not None:
        previous = json.loads(args.replay_from.read_text(encoding="utf-8"))
        require(previous.get("purpose") == "day42_live_workbench_demo"
                and previous.get("plan_sha256") == sha256(PLAN.read_bytes()).hexdigest(), "原演示与计划不一致")
        require(previous.get("source_sha256") == source_fingerprints(),
                "应用源码或目录结构已变化，不能复用原演示；请重新执行所需场景")
        for record in previous["results"]:
            messages = record.get("session", {}).get("messages", [])
            if len(messages) == 2 and messages[-1].get("workflow_run_id"):
                recovered[record["case_id"]] = messages[-1]["workflow_run_id"]
            elif args.retry_failed and record.get("status") == "failed" and record.get("run_id"):
                failed_runs[record["case_id"]] = record["run_id"]
    if not args.real:
        print(f"演示输入检查通过：{len(cases)} 项；未加载模型、提交问题或生成报告。")
        return 0
    # 仅使用本地模型缓存；DeepSeek 网络调用不受这两个变量影响。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    now = datetime.now(timezone.utc)
    path = ROOT / "data/evaluations" / f"day42_demo_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}.json"
    report = {"schema_version": 1, "purpose": "day42_live_workbench_demo", "mode": "real_streamlit_apptest",
              "status": "running", "created_at": now.isoformat(), "python_version": platform.python_version(),
              "plan_path": PLAN.relative_to(ROOT).as_posix(), "plan_sha256": sha256(PLAN.read_bytes()).hexdigest(),
              "source_sha256": source_fingerprints(),
              "results": [], "semantic_review": "pending", "browser_visual_inspection": False}
    if args.replay_from is not None:
        report["replay_from"] = {"path": args.replay_from.as_posix(), "sha256": sha256(args.replay_from.read_bytes()).hexdigest(),
                                 "completed_run_ids": recovered, "failed_run_ids": failed_runs}
    with path.open("x", encoding="utf-8") as file:
        file.write("{}\n")
    print("本次真实演示报告：", path.as_posix(), flush=True)
    for case in cases:
        print("开始：", case["case_id"], case["question"], flush=True)
        report["results"].append({})

        def checkpoint(record):
            report["results"][-1] = deepcopy(record)
            write_report(path, report)

        record = run_case(case, plan, checkpoint,
                          completed_run_id=recovered.get(case["case_id"], failed_runs.get(case["case_id"])),
                          retry_failed=case["case_id"] in failed_runs)
        print("完成：", case["case_id"], record["status"], record.get("run_id"), flush=True)
        if record["status"] == "failed":
            report["status"] = "stopped_after_failure"
            break
    else:
        report["status"] = "behavior_checks_passed_content_review_pending"
    write_report(path, report)
    print("报告已保存：", path.as_posix(), flush=True)
    return 0 if report["status"] == "behavior_checks_passed_content_review_pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
