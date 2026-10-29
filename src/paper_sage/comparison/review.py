"""把有明确来源的人工文字修订应用到副本，保留模型原始响应与引用标识。"""

from copy import deepcopy
from hashlib import sha256
import json

from paper_sage.comparison.parser import index_comparison_material, parse_comparison


def comparison_digest(result):
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _item(items, position):
    if type(position) is not int or position < 0 or position >= len(items):
        raise ValueError("人工修订位置超出当前比较结果")
    return items[position]


def _target(result, path):
    """只允许改正文、缺失说明和比较限制，不允许改身份、引用或来源。"""
    if not isinstance(path, list) or len(path) < 3 or path[0] != "comparisons":
        raise ValueError("人工修订路径不受支持")
    row = _item(result["comparisons"], path[1])
    dimension = row["dimension"]
    if path[2:] == ["comparison_limit"]:
        return row, "comparison_limit", dimension, None, None
    if path[2:] == ["comparison", "text"] and row["comparison"] is not None:
        return row["comparison"], "text", dimension, None, set(row["comparison"]["chunk_ids"])
    if len(path) in (5, 7) and path[2] == "papers":
        paper = _item(row["papers"], path[3])
        if path[4:] == ["missing_information"]:
            return paper, "missing_information", dimension, paper["document_id"], None
        if len(path) == 7 and path[4] == "findings" and path[6] == "text":
            finding = _item(paper["findings"], path[5])
            return finding, "text", dimension, paper["document_id"], set(finding["chunk_ids"])
    raise ValueError("人工修订只能修改文字字段，不能修改文档标识或引用来源")


def apply_comparison_review(result, material, review):
    """校验修订适用对象和证据归属；此校验不证明修订文字的事实正确性。"""
    validated = parse_comparison(json.dumps(result, ensure_ascii=False), material)
    if (not isinstance(review, dict) or review.get("schema_version") != 1
            or review.get("purpose") != "manual_comparison_review"
            or review.get("source_comparison_sha256") != comparison_digest(validated)
            or not isinstance(review.get("corrections"), list)):
        raise ValueError("人工修订不对应当前原始比较结果")
    document_ids, cells = index_comparison_material(material)
    reviewed, seen = deepcopy(validated), set()
    for correction in review["corrections"]:
        if not isinstance(correction, dict):
            raise ValueError("人工修订条目必须是字典")
        target, field, dimension, document_id, cited_ids = _target(reviewed, correction.get("path"))
        key = tuple(correction["path"])
        if key in seen:
            raise ValueError("同一字段不能重复修订")
        seen.add(key)
        if correction.get("original_text") != target[field]:
            raise ValueError("人工修订的原文字与当前结果不一致")
        replacement = correction.get("replacement_text")
        if not isinstance(replacement, str) or (field == "text" and not replacement.strip()):
            raise ValueError("人工修订的新文字格式不正确")
        if not isinstance(correction.get("reason"), str) or not correction["reason"].strip():
            raise ValueError("人工修订必须说明理由")
        ids = correction.get("evidence_chunk_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(cid, str) or not cid.strip() for cid in ids):
            raise ValueError("人工修订必须提供有效的证据标识列表")
        allowed_ids = set().union(*(set(cells[(doc_id, dimension)])
                                   for doc_id in ([document_id] if document_id is not None else document_ids)))
        if cited_ids is not None:
            allowed_ids &= cited_ids
        if len(set(ids)) != len(ids) or not set(ids).issubset(allowed_ids):
            raise ValueError("人工修订引用了当前字段范围之外或重复的证据")
        target[field] = replacement
    return parse_comparison(json.dumps(reviewed, ensure_ascii=False), material)
