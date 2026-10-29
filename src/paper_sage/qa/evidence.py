"""问答结论与本次提供片段的关联；不执行检索或判断结论是否得到原文支持。"""

import json


SOURCE_FIELDS = ("chunk_id", "project_id", "document_id", "source", "pdf_page", "text")


def build_evidence_catalog(final_results):
    """沿用已完成练习：仅为本次提供的最终片段建立独立来源记录。"""
    if not isinstance(final_results, list):
        raise ValueError("本次最终片段必须是列表")
    evidence_by_id = {}
    for result in final_results:
        if not isinstance(result, dict) or not isinstance(result.get("chunk"), dict):
            raise ValueError("候选记录必须包含 chunk 字典")
        chunk = result["chunk"]
        for field in SOURCE_FIELDS:
            value = chunk.get(field)
            if field == "pdf_page":
                if type(value) is not int or value < 1:
                    raise ValueError("PDF 页码必须是从 1 开始的整数")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"片段缺少有效的 {field}")
        chunk_id = chunk["chunk_id"]
        if chunk_id in evidence_by_id:
            raise ValueError("本次证据表中出现重复的片段标识")
        evidence_record = {field: chunk[field] for field in SOURCE_FIELDS}
        evidence_by_id[chunk_id] = evidence_record
    return evidence_by_id


def parse_cited_answer(raw_response, evidence_by_id):
    """校验新协议的结构、引用范围及资料不足约定；返回模型生成的数据。"""
    result = json.loads(raw_response)
    if not isinstance(result, dict) or set(result) != {"claims", "insufficient_reason"}:
        raise ValueError("回答必须且只能包含 claims 和 insufficient_reason")
    claims, reason = result["claims"], result["insufficient_reason"]
    if not isinstance(claims, list) or not isinstance(reason, str):
        raise ValueError("claims 必须是列表，insufficient_reason 必须是字符串")
    if reason.strip():
        if claims:
            raise ValueError("资料不足时，claims 必须为空列表")
        return result
    if not claims:
        raise ValueError("没有结论时，必须说明资料不足的原因")
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"text", "chunk_ids"}:
            raise ValueError("每条结论必须且只能包含 text 和 chunk_ids")
        if not isinstance(claim["text"], str) or not claim["text"].strip():
            raise ValueError("结论 text 必须是非空字符串")
        chunk_ids = claim["chunk_ids"]
        if not isinstance(chunk_ids, list) or not chunk_ids:
            raise ValueError("每条结论必须提供非空的 chunk_ids 列表")
        seen = set()
        for chunk_id in chunk_ids:
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("引用的片段标识必须是非空字符串")
            if chunk_id in seen:
                raise ValueError("同一结论不能重复引用同一个片段标识")
            seen.add(chunk_id)
            # 保留学习者完成的范围判断。
            chunk_is_allowed = True if chunk_id in evidence_by_id else False
            if not chunk_is_allowed:
                raise ValueError(f"引用片段未提供给本次回答：{chunk_id}")
    return result


def resolve_claim_sources(result, evidence_by_id):
    """接收已校验结果，按引用顺序返回来源副本，原始模型数据保持不变。"""
    return [
        {"text": claim["text"],
         "sources": [dict(evidence_by_id[chunk_id]) for chunk_id in claim["chunk_ids"]]}
        for claim in result["claims"]
    ]
