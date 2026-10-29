"""全文汇总的输入准备：保留要点来源与独立人工校正，不改写模型原始记录。"""

from copy import deepcopy
import json

from paper_sage.summarization.parser import SUMMARY_FIELDS, parse_batch_summary, reject_summary_placeholders
from paper_sage.summarization.runner import validate_completed_records


def build_summary_note_batches(prepared, records, corrections=()):
    """按批次生成带唯一标识的要点；空批次仍保留，缺少批次则拒绝汇总。"""
    records = validate_completed_records(prepared, records)
    if len(records) != len(prepared["batches"]):
        raise ValueError("仍有未完成批次，不能准备全文汇总")
    parts_by_id = {part["part_id"]: part for batch in prepared["batches"] for part in batch["parts"]}
    batches, notes_by_id = [], {}
    for record in records:
        notes = []
        for field in SUMMARY_FIELDS:
            for index, point in enumerate(record["summary"][field], start=1):
                note_id = f"{record['batch_id']}:{field}:{index}"
                note = {
                    "note_id": note_id, "batch_id": record["batch_id"], "category": field,
                    "text": point["text"], "part_ids": deepcopy(point["part_ids"]),
                }
                notes.append(note)
                notes_by_id[note_id] = note
        batches.append(notes)

    if not isinstance(corrections, (list, tuple)):
        raise ValueError("人工校正必须是列表")
    corrected_ids = set()
    for correction in corrections:
        required = {"note_id", "original", "replacement", "reason", "evidence"}
        if not isinstance(correction, dict) or set(correction) != required:
            raise ValueError("人工校正必须包含要点标识、原要点、替换要点、理由与原文依据")
        note_id = correction["note_id"]
        if not isinstance(note_id, str) or note_id not in notes_by_id or note_id in corrected_ids:
            raise ValueError("人工校正指向不存在或重复的要点")
        note = notes_by_id[note_id]
        original = {key: note[key] for key in ("text", "part_ids")}
        if correction["original"] != original:
            raise ValueError("待校正要点已变化，不能套用旧校正")
        replacement = correction["replacement"]
        if not isinstance(replacement, dict) or replacement.get("part_ids") != original["part_ids"]:
            raise ValueError("本步骤的人工校正必须保留该要点原有来源")
        candidate = {field: [] for field in SUMMARY_FIELDS}
        candidate[note["category"]] = [replacement]
        parsed = parse_batch_summary(json.dumps(candidate), {"parts": [parts_by_id[pid] for pid in original["part_ids"]]})
        reject_summary_placeholders(parsed)
        if not isinstance(correction["reason"], str) or not correction["reason"].strip():
            raise ValueError("人工校正必须说明理由")
        evidence = correction["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("人工校正必须保留原文依据")
        quoted_ids = set()
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {"part_id", "quote"}:
                raise ValueError("校正依据必须包含 part_id 与 quote")
            part_id, quote = item["part_id"], item["quote"]
            if not isinstance(part_id, str) or part_id not in original["part_ids"]:
                raise ValueError("校正依据不属于该要点的来源")
            if not isinstance(quote, str) or not quote.strip():
                raise ValueError("校正引用不能是空文字")
            # 仅忽略提取文字中的换行和连续空白；不做模糊匹配或语义判断。
            if " ".join(quote.split()) not in " ".join(parts_by_id[part_id]["text"].split()):
                raise ValueError("校正引用未出现在指定原文中")
            quoted_ids.add(part_id)
        if quoted_ids != set(original["part_ids"]):
            raise ValueError("校正依据未覆盖该要点引用的来源")
        note["text"] = replacement["text"]
        corrected_ids.add(note_id)
    return batches


def collect_summary_notes(note_batches):
    """将每批的要点列表按顺序合并为一维列表，供后续模型归纳与去重。"""
    notes = []
    for batch_notes in note_batches:
        # 把 batch_notes 内的每条要点加入 notes，保持一维列表。
        notes.extend(batch_notes)

    if len(notes) != sum(len(batch) for batch in note_batches) or any(not isinstance(note, dict) for note in notes):
        raise ValueError("合并结果必须是一维要点列表，不能嵌套、遗漏或重复")
    expected_ids = [note["note_id"] for batch in note_batches for note in batch]
    if [note["note_id"] for note in notes] != expected_ids:
        raise ValueError("合并结果改变了要点顺序或标识")
    return notes
