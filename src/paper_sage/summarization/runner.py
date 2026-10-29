"""逐批提取总结要点，校验并复用已完成记录；本模块不生成全文汇总。"""

from copy import deepcopy
import json
from pathlib import Path

from paper_sage.summarization.model import SUMMARY_MAX_TOKENS, SUPPORTED_SUMMARY_PROMPT_VERSIONS, build_batch_messages, summarize_batch
from paper_sage.summarization.parser import parse_batch_summary, reject_summary_placeholders


def validate_summary_record(record, batch):
    """复用前检查正文、来源、提示词与原始响应，不能只相信报告的成功标记。"""
    if not isinstance(record, dict):
        raise ValueError("已完成记录必须是字典")
    if record.get("batch_id") != batch["batch_id"] or record.get("input") != batch:
        raise ValueError("已完成记录与当前批次的原文或来源不一致")
    prompt_version = record.get("prompt_version")
    if not isinstance(prompt_version, str) or prompt_version not in SUPPORTED_SUMMARY_PROMPT_VERSIONS:
        raise ValueError("已完成记录使用了不受支持的总结提示词版本")
    # 可复用已知旧版本的合格记录，但必须与该版本的完整请求逐字一致。
    expected_messages = [{"role": m.type, "content": m.content}
                         for m in build_batch_messages(batch, prompt_version=prompt_version)]
    if record.get("request_messages") != expected_messages:
        raise ValueError("已完成记录的请求消息与当前批次不一致")
    if (record.get("finish_reason") != "stop" or record.get("validation_error")
            or record.get("tool_calls") or record.get("invalid_tool_calls")):
        raise ValueError("未正常完成的记录不能复用")
    parsed = parse_batch_summary(record.get("raw_response"), batch)
    if parsed != record.get("summary"):
        raise ValueError("已完成记录的总结与原始响应不一致")
    reject_summary_placeholders(parsed)


def validate_completed_records(prepared, records):
    """返回独立副本，按本文的批次顺序排列，拒绝错范围和重复记录。"""
    if not isinstance(records, list):
        raise ValueError("已完成记录必须是列表")
    batches_by_id = {batch["batch_id"]: batch for batch in prepared["batches"]}
    if len(batches_by_id) != len(prepared["batches"]):
        raise ValueError("当前准备数据包含重复批次标识")
    records_by_id = {}
    for record in records:
        batch_id = record.get("batch_id") if isinstance(record, dict) else None
        if not isinstance(batch_id, str) or batch_id not in batches_by_id:
            raise ValueError("已完成记录不属于当前论文的批次")
        if batch_id in records_by_id:
            raise ValueError("已完成记录存在重复批次")
        validate_summary_record(record, batches_by_id[batch_id])
        records_by_id[batch_id] = deepcopy(record)
    return [records_by_id[batch["batch_id"]] for batch in prepared["batches"] if batch["batch_id"] in records_by_id]


def exclude_redo_batches(prepared, records, batch_numbers):
    """仅从本次复用列表中排除指定批次，不改写旧记录或原报告。编号从 1 开始。"""
    if not isinstance(records, list) or not isinstance(batch_numbers, (list, tuple)):
        raise ValueError("记录和重做批次编号必须是列表")
    total = len(prepared["batches"])
    if any(type(number) is not int or not 1 <= number <= total for number in batch_numbers):
        raise ValueError(f"重做批次编号必须是 1 到 {total} 的整数")
    if len(set(batch_numbers)) != len(batch_numbers):
        raise ValueError("重做批次编号不能重复")
    allowed_ids = {batch["batch_id"] for batch in prepared["batches"]}
    seen = set()
    for record in records:
        batch_id = record.get("batch_id") if isinstance(record, dict) else None
        if not isinstance(batch_id, str) or batch_id not in allowed_ids or batch_id in seen:
            raise ValueError("旧报告存在不属于当前论文或重复的批次记录")
        seen.add(batch_id)
    redo_ids = {prepared["batches"][number - 1]["batch_id"] for number in batch_numbers}
    if not redo_ids <= seen:
        raise ValueError("只能指定旧报告中已有记录的批次重做")
    return [record for record in records if record["batch_id"] not in redo_ids]


def load_completed_summaries(report_path, prepared, redo_batch_numbers=None):
    """读取已有单批报告或批次进度报告；失败的批次不会计入已完成。"""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("总结报告版本不受支持")
    if (report.get("mode") != "real" or report.get("model") != "deepseek-flash"
            or report.get("max_tokens") != SUMMARY_MAX_TOKENS):
        raise ValueError("只能复用与当前配置一致的真实总结报告")
    for field in ("project_id", "document_id", "source"):
        if report.get(field) != prepared[field]:
            raise ValueError("总结报告不属于当前项目或文档")
    if report.get("purpose") == "day33_single_batch_summary":
        if report.get("status") != "completed_single_batch":
            raise ValueError("该单批报告未成功完成，不能复用")
        records = [report.get("result")]
    elif report.get("purpose") == "day33_batch_summaries":
        if report.get("status") not in {"in_progress", "partial_failed", "completed_batches"}:
            raise ValueError("批次进度报告状态不受支持")
        if report.get("prepared") != prepared:
            raise ValueError("论文原文或分批配置已经变化，不能复用这份进度")
        records = report.get("records")
    else:
        raise ValueError("不是支持的总结报告")
    # 指定重做的响应不再作为完成记录；其余记录仍经过全部复用校验。
    if redo_batch_numbers:
        records = exclude_redo_batches(prepared, records, redo_batch_numbers)
    return validate_completed_records(prepared, records)


def summarize_batches(prepared, model, completed_records=None, on_record=None):
    """顺序处理未完成批次；任一失败立即抛出，不自动重试、不生成最终总结。"""
    # 在调用模型之前检查全部输入，避免后面的无效批次导致前面白白调用。
    for batch in prepared["batches"]:
        build_batch_messages(batch)
    records = validate_completed_records(prepared, [] if completed_records is None else completed_records)
    records_by_id = {record["batch_id"]: record for record in records}
    for batch in prepared["batches"]:
        if batch["batch_id"] in records_by_id:
            continue

        # 调用已有的 summarize_batch，传入本轮 batch 和 model，取得本批次记录。
        record = summarize_batch(batch, model)

        validate_summary_record(record, batch)
        records_by_id[batch["batch_id"]] = record
        if on_record is not None:
            # 回调由练习脚本用来保存进度，不把持久化逻辑放进模型函数。
            on_record(deepcopy(record))
    return [records_by_id[batch["batch_id"]] for batch in prepared["batches"]]
