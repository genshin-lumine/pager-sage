"""校验比较流程产物并生成聊天正文；不请求模型或读写报告。"""

from pathlib import Path

from paper_sage.sessions.chat import validate_history_messages
from paper_sage.comparison.context import build_comparison_material
from paper_sage.comparison.inputs import prepare_comparison_tasks, validate_comparison_documents
from paper_sage.comparison.model import (COMPARISON_PROMPT_VERSIONS, build_comparison_messages,
                                        resolve_comparison_sources, parse_comparison_response)
from paper_sage.comparison.plan import validate_comparison_plan_record
from paper_sage.comparison.review import apply_comparison_review


def validate_comparison_artifact(artifact):
    """检查计划、任务、检索、原始响应和查回来源之间的一致性，不证明事实正确。"""
    if (not isinstance(artifact, dict) or artifact.get("status") != "completed"
            or artifact.get("failed_stage") or artifact.get("error_type")):
        raise ValueError("比较流程没有完整成功，不能展示为完成结果")
    project_id, documents = artifact.get("project_id"), artifact.get("documents")
    validate_comparison_documents(project_id, documents)
    history = validate_history_messages(artifact.get("history"))
    if artifact.get("history") != history:
        raise ValueError("比较历史必须只保留普通聊天文本")
    document_info = [{"document_id": doc["document_id"], "name": doc.get("title") or Path(doc["source"]).name}
                     for doc in documents]
    plan = validate_comparison_plan_record(artifact.get("plan_record"), artifact.get("question"), document_info, history=history)
    prepared = prepare_comparison_tasks(project_id, documents, plan)
    if artifact.get("tasks") != prepared["tasks"]:
        raise ValueError("比较任务与本次计划不一致")
    stages = artifact.get("stages")
    if not isinstance(stages, list) or any(not isinstance(stage, dict) for stage in stages):
        raise ValueError("比较流程缺少阶段记录")
    stage_names = [stage.get("stage") for stage in stages]
    if prepared["response_kind"] == "clarify":
        if (artifact.get("response_kind") != "clarify" or artifact.get("reply") != prepared["reply"]
                or artifact.get("records") != [] or artifact.get("material") is not None
                or artifact.get("generation_record") is not None or stage_names != ["plan", "tasks"]):
            raise ValueError("比较澄清结果与计划不一致或混入了后续结果")
        if "content_review" in artifact:
            raise ValueError("澄清记录不能附带比较内容修订")
        return

    if artifact.get("response_kind") != "compare" or artifact.get("reply") != "":
        raise ValueError("比较结果响应类型不正确")
    if stage_names != ["plan", "tasks", "load_tools", "retrieve", "material", "generate"]:
        raise ValueError("比较结果缺少完整执行阶段")
    records = artifact.get("records")
    if (not isinstance(records, list) or any(not isinstance(record, dict) for record in records)
            or [record.get("task") for record in records] != prepared["tasks"]):
        raise ValueError("检索记录与本次比较任务不一致")
    dimensions = [item["name"] for item in plan["dimensions"]]
    material = build_comparison_material(project_id, documents, dimensions, records)
    if artifact.get("material") != material:
        raise ValueError("比较输入与本次检索记录不一致")
    record = artifact.get("generation_record")
    if (not isinstance(record, dict) or record.get("input") != material
            or record.get("prompt_version") not in COMPARISON_PROMPT_VERSIONS
            or record.get("finish_reason") != "stop" or record.get("validation_error")
            or record.get("tool_calls") or record.get("invalid_tool_calls")):
        raise ValueError("比较生成记录未正常完成或与输入不一致")
    messages = [{"role": message.type, "content": message.content}
                for message in build_comparison_messages(material, prompt_version=record["prompt_version"])]
    if record.get("request_messages") != messages:
        raise ValueError("比较生成的请求消息与资料不一致")
    result, repair = parse_comparison_response(record.get("raw_response"), material)
    if record.get("format_repair") != repair:
        raise ValueError("比较格式修复记录与原始响应不一致")
    if record.get("comparison") != result or record.get("resolved_comparison") != resolve_comparison_sources(result, material):
        raise ValueError("比较正文或查回来源与原始响应不一致")
    if "content_review" in artifact:
        saved = artifact["content_review"]
        review = saved.get("record") if isinstance(saved, dict) else None
        if (not isinstance(review, dict) or not artifact.get("report_path")
                or not artifact.get("source_report_sha256")
                or review.get("source_report") != artifact["report_path"]
                or review.get("source_report_sha256") != artifact["source_report_sha256"]):
            raise ValueError("内容修订与原始比较报告不一致")
        reviewed = apply_comparison_review(result, material, review)
        if (saved.get("comparison") != reviewed
                or saved.get("resolved_comparison") != resolve_comparison_sources(reviewed, material)):
            raise ValueError("修订正文或来源与修订记录不一致")


def format_paper_comparison(artifact):
    """生成聊天正文；完整原文、片段标识和阶段信息仍保存在 artifact。"""
    validate_comparison_artifact(artifact)
    if artifact["response_kind"] == "clarify":
        return artifact["reply"]
    lines = ["**论文比较**"]
    resolved = artifact["generation_record"]["resolved_comparison"]
    if "content_review" in artifact:
        resolved = artifact["content_review"]["resolved_comparison"]
        lines.append("\n以下正文应用了单独记录的内容修订，原始模型响应保留。修订记录不等于自动事实验证。")
    for row in resolved["comparisons"]:
        lines.append(f"\n**{row['dimension']}**")
        for paper in row["papers"]:
            lines.append(f"\n**{paper['name']}**")
            for index, finding in enumerate(paper["findings"], 1):
                pages = list(dict.fromkeys(chunk["pdf_page"] for chunk in finding["sources"]))
                lines.append(f"\n{index}. {finding['text']}（PDF 第 {', '.join(map(str, pages))} 页）")
            if paper["missing_information"]:
                lines.append(f"\n缺失信息：{paper['missing_information']}")
        comparison = row["comparison"]
        lines.append(f"\n**比较**\n\n{comparison['text'] if comparison else '当前资料不足以形成双方比较。'}")
        if row["comparison_limit"]:
            lines.append(f"\n比较限制：{row['comparison_limit']}")
    return "\n".join(lines)
