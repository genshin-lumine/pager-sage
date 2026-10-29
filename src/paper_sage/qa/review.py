"""将人工支持关系判断关联到原始结构化回答；不自动判断内容真伪。"""

from copy import deepcopy
from hashlib import sha256
import json

from paper_sage.qa.model import validate_cited_answer_record


REVIEW_VERDICTS = {"supported", "contradicted", "insufficient"}
VERDICT_LABELS = {"supported": "支持", "contradicted": "矛盾", "insufficient": "证据不足", "pending": "未复核"}


def answer_record_digest(answer_artifact):
    """摘要绑定原始回答、请求、证据和来源；相同文字不代表同一次回答。"""
    if not isinstance(answer_artifact, dict):
        raise ValueError("原始回答记录必须是字典")
    validate_cited_answer_record(
        answer_artifact, answer_artifact.get("question"), answer_artifact.get("final_results"),
        history=answer_artifact.get("history", []),
    )
    payload = json.dumps(answer_artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def apply_claim_review(answer_artifact, review):
    """返回带人工判读的独立展示数据，不修改模型原始响应或其审核状态。"""
    digest = answer_record_digest(answer_artifact)
    if (not isinstance(review, dict)
            or set(review) != {"schema_version", "purpose", "source_answer_sha256", "reviewer", "entries"}
            or type(review["schema_version"]) is not int or review["schema_version"] != 1
            or review["purpose"] != "manual_claim_review"):
        raise ValueError("人工复核记录的结构或用途不正确")
    if review["source_answer_sha256"] != digest:
        raise ValueError("人工复核不对应当前原始回答与证据快照")
    if not isinstance(review["reviewer"], str) or not review["reviewer"].strip():
        raise ValueError("人工复核必须记录复核者")
    if not isinstance(review["entries"], list):
        raise ValueError("人工复核 entries 必须是列表")

    claims = answer_artifact["result"]["claims"]
    entries_by_number = {}
    for entry in review["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"claim_number", "original_text", "chunk_ids", "verdict", "reason"}:
            raise ValueError("每条复核必须提供结论编号、原结论、引用标识、判断和理由")
        number = entry["claim_number"]
        if type(number) is not int or not 1 <= number <= len(claims):
            raise ValueError("复核结论编号超出当前回答")
        if number in entries_by_number:
            raise ValueError("同一结论不能重复复核")
        claim = claims[number - 1]
        if entry["original_text"] != claim["text"] or entry["chunk_ids"] != claim["chunk_ids"]:
            raise ValueError("复核的原结论或引用标识与当前结论不一致")
        if not isinstance(entry["verdict"], str) or entry["verdict"] not in REVIEW_VERDICTS:
            raise ValueError("不支持的人工复核判断")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise ValueError("人工复核必须说明原文依据、冲突或缺失信息")
        entries_by_number[number] = entry

    reviewed_claims = []
    for number, resolved in enumerate(answer_artifact["claims_with_sources"], start=1):
        entry = entries_by_number.get(number)
        reviewed_claims.append({
            "claim_number": number, "text": resolved["text"],
            "chunk_ids": deepcopy(claims[number - 1]["chunk_ids"]), "sources": deepcopy(resolved["sources"]),
            "verdict": entry["verdict"] if entry is not None else "pending",
            "reason": entry["reason"] if entry is not None else "",
        })
    return {
        "schema_version": 1, "method": "manual", "source_answer_sha256": digest,
        "review_record": deepcopy(review), "claim_count": len(claims),
        "reviewed_count": len(entries_by_number), "claims": reviewed_claims,
    }
