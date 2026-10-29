"""读取已完成的全文总结，核对当前文档与来源；不调用模型或修改旧报告。"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from paper_sage.documents.pdf import load_pdf_pages
from paper_sage.summarization.inputs import build_summary_batches
from paper_sage.summarization.parser import SUMMARY_FIELDS, SUMMARY_LABELS
from paper_sage.summarization.runner import load_completed_summaries
from paper_sage.summarization.synthesis import (SYNTHESIS_MAX_TOKENS, SYNTHESIS_PROMPT_VERSION, build_synthesis_messages,
                               parse_paper_summary, prepare_synthesis_material, resolve_summary_sources)




def _report_path(value, root):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("缺少总结报告路径")
    path = (root / value).resolve()
    if not path.is_relative_to((root / "data/evaluations").resolve()):
        raise ValueError("总结报告必须位于当前工作区的评估目录")
    return path


def load_saved_summary(report_path, project_id, document, *, workspace_root=None):
    """仅复用与当前项目、文档、提取文字及输入要点一致的成功报告。"""
    root = Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    path = _report_path(report_path, root)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or any(report.get(key) != value for key, value in {
        "schema_version": 1, "purpose": "day33_paper_summary", "mode": "real",
        "status": "completed_summary", "model": "deepseek-flash", "max_tokens": SYNTHESIS_MAX_TOKENS,
    }.items()) or report.get("accepted_as_summary") is not True or report.get("validation_error"):
        raise ValueError("不是当前配置下成功完成的全文总结报告")
    record = report.get("result")
    if not isinstance(record, dict) or not isinstance(record.get("input"), dict):
        raise ValueError("总结报告缺少输入记录")
    identity = {"project_id": project_id, "document_id": document["document_id"], "source": document["source"]}
    if any(record["input"].get(key) != value for key, value in identity.items()):
        raise ValueError("总结报告不属于当前项目或所选文档")
    if (record.get("prompt_version") != SYNTHESIS_PROMPT_VERSION or record.get("finish_reason") != "stop"
            or record.get("validation_error") or record.get("tool_calls") or record.get("invalid_tool_calls")):
        raise ValueError("总结报告未正常完成或提示词版本已变化")
    source_path = _report_path(report.get("source_report"), root)
    source_hash = sha256(source_path.read_bytes()).hexdigest()
    if source_hash != report.get("source_report_sha256"):
        raise ValueError("原始批次报告已变化，不能直接复用总结")
    review = report.get("review")
    if (not isinstance(review, dict) or review.get("schema_version") != 1
            or review.get("purpose") != "manual_summary_review"
            or review.get("source_report_sha256") != source_hash
            or _report_path(review.get("source_report"), root) != source_path):
        raise ValueError("人工校正记录与批次报告不一致")
    if not isinstance(review.get("corrections"), list):
        raise ValueError("人工校正记录缺少 corrections 列表")

    # 从指定工作区读取 PDF，元数据继续使用项目内保存的来源字符串。
    pdf_path = (root / document["source"]).resolve()
    pages = load_pdf_pages(str(pdf_path), document["document_id"])
    for page in pages:
        page["source"] = document["source"]
    prepared = build_summary_batches(pages, project_id)
    records = load_completed_summaries(source_path, prepared)
    material = prepare_synthesis_material(prepared, records, review["corrections"])
    if material != record["input"]:
        raise ValueError("论文文字、要点或人工校正发生变化，不能复用此总结")
    expected_messages = [{"role": m.type, "content": m.content} for m in build_synthesis_messages(material)]
    if expected_messages != record.get("request_messages"):
        raise ValueError("总结请求消息与当前输入不一致")
    summary = parse_paper_summary(record.get("raw_response"), material)
    resolved = resolve_summary_sources(summary, material)
    if summary != record.get("summary") or resolved != record.get("resolved_summary"):
        raise ValueError("总结正文或来源记录与原始响应不一致")
    return {
        **identity, "summary": summary, "resolved_summary": resolved,
        "notes": deepcopy(material["notes"]), "coverage": deepcopy(material["coverage"]),
        "review_corrections": deepcopy(material["review_corrections"]),
        "report_path": path.relative_to(root).as_posix(), "generated_at": report["recorded_at"],
        "model": report["model"], "reused": True,
    }


def format_paper_summary(artifact):
    """生成可写入聊天消息的正文，原文记录另外保存在 artifact 中。"""
    lines = [f"**论文总结：{Path(artifact['source']).name}**"]
    for field in SUMMARY_FIELDS:
        lines.append(f"\n**{SUMMARY_LABELS[field]}**")
        points = artifact["resolved_summary"][field]
        if not points:
            lines.append("\n现有批次要点未提供明确的局限信息，不代表论文没有局限。"
                         if field == "limitations" else "\n现有批次要点未提供相关信息。")
        for index, point in enumerate(points, start=1):
            pages = list(dict.fromkeys(source["pdf_page"] for source in point["sources"]))
            lines.append(f"\n{index}. {point['text']}（PDF 第 {', '.join(map(str, pages))} 页）")
    return "\n".join(lines)
