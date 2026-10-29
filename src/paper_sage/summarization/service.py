"""工作台总结业务：按当前范围复用报告，或逐批生成并保存可恢复进度。"""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from openai import APIError

from paper_sage.documents.pdf import load_pdf_pages
from paper_sage.summarization.inputs import build_summary_batches
from paper_sage.summarization.model import BatchSummaryError, SUMMARY_MAX_TOKENS
from paper_sage.summarization.report import load_saved_summary
from paper_sage.summarization.runner import load_completed_summaries, summarize_batches, validate_completed_records
from paper_sage.summarization.synthesis import PaperSummaryError, SYNTHESIS_MAX_TOKENS, prepare_synthesis_material, summarize_paper


def _now():
    return datetime.now(timezone.utc).isoformat()


def _reports(root, purposes):
    directory = (root / "data/evaluations").resolve()
    for path in sorted(directory.glob("day33_summary_*.json"), key=lambda p: p.name, reverse=True):
        if not path.resolve().is_relative_to(directory):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(report, dict) and report.get("purpose") in purposes:
            yield path, report


def _same_document(record, project_id, document):
    return isinstance(record, dict) and all(record.get(key) == value for key, value in {
        "project_id": project_id, "document_id": document["document_id"], "source": document["source"],
    }.items())


def _find_saved(root, project_id, document):
    for path, report in _reports(root, {"day33_paper_summary"}):
        record = report.get("result")
        if report.get("status") != "completed_summary" or not isinstance(record, dict):
            continue
        if not _same_document(record.get("input"), project_id, document):
            continue
        try:
            return load_saved_summary(path, project_id, document, workspace_root=root)
        except (ValueError, KeyError, TypeError, FileNotFoundError):
            # 旧输入或不完整报告不进入本次结果，继续寻找当前范围可用的报告。
            continue
    return None


def _completed_batches(root, project_id, document, prepared):
    best_path, best_records = None, []
    for path, report in _reports(root, {"day33_batch_summaries", "day33_single_batch_summary"}):
        if not _same_document(report, project_id, document):
            continue
        try:
            records = load_completed_summaries(path, prepared)
        except (ValueError, KeyError, TypeError):
            continue
        if len(records) > len(best_records):
            best_path, best_records = path, records
    return best_path, best_records


def _load_review(root, source_path, source_hash):
    # 新生成或续传的批次报告可内嵌继承的校正；旧练习另存为同名 _review.json。
    source = json.loads(source_path.read_text(encoding="utf-8"))
    review_path = source_path.with_name(source_path.stem + "_review.json")
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if (not isinstance(review, dict) or review.get("schema_version") != 1
                or review.get("purpose") != "manual_summary_review"
                or review.get("source_report_sha256") != source_hash
                or (root / review.get("source_report", "")).resolve() != source_path.resolve()
                or not isinstance(review.get("corrections"), list)):
            raise ValueError("已保存的人工校正与当前批次报告不一致")
        return deepcopy(review["corrections"])
    corrections = source.get("review_corrections", [])
    if not isinstance(corrections, list):
        raise ValueError("批次报告中的人工校正格式不正确")
    return deepcopy(corrections)


def _new_report(root, kind, report):
    directory = (root / "data/evaluations").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("总结报告目录超出当前工作区")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"day33_summary_{kind}_{stamp}_{uuid4().hex[:8]}.json"
    with path.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    return path


def _save_progress(path, report):
    """只替换本次运行新建的报告，写入失败时保留上次完整进度。"""
    path = path.resolve()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.stem + "_", suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name).resolve()
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        if temporary_path.parent != path.parent:
            raise ValueError("进度临时文件不在目标目录")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.parent == path.parent:
            temporary_path.unlink(missing_ok=True)


def _error_text(error):
    if isinstance(error, APIError):
        return f"{type(error).__name__}，HTTP 状态：{getattr(error, 'status_code', None)}"
    return str(error)


def create_summary_runner(project_id, documents, model, *, on_progress=None, workspace_root=None):
    """绑定范围；只有图实际调用返回的函数时才读取 PDF 或请求总结。"""
    root = Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    documents = deepcopy(documents)
    documents_by_id = {document["document_id"]: document for document in documents}
    if not documents or len(documents_by_id) != len(documents):
        raise ValueError("总结范围不能为空或包含重复文档")

    def progress(message):
        if on_progress is not None:
            on_progress(message)

    def run(document_id):
        if len(documents) != 1 or document_id not in documents_by_id:
            raise ValueError("总结仅允许处理当前单选的论文")
        document = documents_by_id[document_id]
        progress("正在检查当前论文的已有总结……")
        artifact = _find_saved(root, project_id, document)
        if artifact is not None:
            progress("已复用与当前论文内容一致的总结。")
            return artifact

        pages = load_pdf_pages(str((root / document["source"]).resolve()), document_id)
        for page in pages:
            page["source"] = document["source"]
        prepared = build_summary_batches(pages, project_id)
        source_path, records = _completed_batches(root, project_id, document, prepared)
        corrections = []
        if source_path is not None:
            corrections = _load_review(root, source_path, sha256(source_path.read_bytes()).hexdigest())
        reused_count = len(records)
        batch_count = len(prepared["batches"])
        progress(f"全文共 {batch_count} 部分，已复用 {reused_count} 部分的要点。")
        if reused_count < batch_count:
            report = {
                "schema_version": 1, "purpose": "day33_batch_summaries", "mode": "real", "status": "in_progress",
                "model": "deepseek-flash", "max_tokens": SUMMARY_MAX_TOKENS,
                "recorded_at": _now(), "updated_at": _now(),
                "project_id": project_id, "document_id": document_id, "source": document["source"],
                "prepared": prepared, "records": deepcopy(records), "reused_count": reused_count,
                "reused_from": source_path.relative_to(root).as_posix() if source_path else None,
                "review_corrections": corrections, "error": None, "failed_record": None,
            }
            source_path = _new_report(root, "batches", report)

            def on_record(record):
                report["records"].append(record)
                report["records"] = validate_completed_records(prepared, report["records"])
                report["updated_at"] = _now()
                _save_progress(source_path, report)
                progress(f"已保存 {len(report['records'])} / {batch_count} 部分的要点。")

            try:
                records = summarize_batches(prepared, model, records, on_record=on_record)
            except Exception as error:
                report.update(status="partial_failed", updated_at=_now(), error=_error_text(error))
                if isinstance(error, BatchSummaryError):
                    report["failed_record"] = error.record
                _save_progress(source_path, report)
                raise
            report.update(records=records, status="completed_batches", updated_at=_now())
            _save_progress(source_path, report)

        material = prepare_synthesis_material(prepared, records, corrections)
        source_hash = sha256(source_path.read_bytes()).hexdigest()
        review = {
            "schema_version": 1, "purpose": "manual_summary_review",
            "source_report": source_path.relative_to(root).as_posix(), "source_report_sha256": source_hash,
            "corrections": corrections,
            "scope": "仅继承已有的逐条校正；空列表表示没有人工校正，不表示内容已全面审核。",
        }
        progress("正在根据全部要点生成全文总结……")
        report = {
            "schema_version": 1, "purpose": "day33_paper_summary", "mode": "real", "status": "in_progress",
            "recorded_at": _now(), "model": "deepseek-flash", "max_tokens": SYNTHESIS_MAX_TOKENS,
            "source_report": source_path.relative_to(root).as_posix(), "source_report_sha256": source_hash,
            "review": review, "result": None, "accepted_as_summary": False, "validation_error": None,
            "limitations": ["仅归纳提取文字的批次要点，可能遗漏图片信息。", "结构与来源标识检查不代表内容已经完整核验。"],
        }
        report_path = _new_report(root, "reduce", report)
        try:
            record = summarize_paper(material, model)
        except Exception as error:
            report.update(status="validation_failed" if isinstance(error, PaperSummaryError) else "request_failed",
                          validation_error=_error_text(error))
            if isinstance(error, PaperSummaryError):
                report["result"] = error.record
            _save_progress(report_path, report)
            raise
        report.update(result=record, status="completed_summary", accepted_as_summary=True)
        _save_progress(report_path, report)
        artifact = load_saved_summary(report_path, project_id, document, workspace_root=root)
        artifact["reused"] = False
        progress("全文总结已生成，原文来源已保存。")
        return artifact

    return run
