"""核对证据与当前索引、PDF 页文字的一致性；不判断结论支持关系。"""

from paper_sage.qa.evidence import SOURCE_FIELDS, build_evidence_catalog


def validate_source_record(source, indexed_chunk, pdf_page_record):
    """调用方须从已确认的当前文档读取索引和 PDF，不以报告副本作为校验依据。"""
    # 沿用来源字段约定，特别是页码必须为正整数，不能用布尔值代替。
    build_evidence_catalog([{"chunk": source}])
    if indexed_chunk is None:
        raise ValueError("引用片段不在当前索引中")
    build_evidence_catalog([{"chunk": indexed_chunk}])
    different = [field for field in SOURCE_FIELDS if source[field] != indexed_chunk[field]]
    if different:
        raise ValueError("报告来源与当前索引不一致：" + ", ".join(different))

    if not isinstance(pdf_page_record, dict):
        raise ValueError("当前 PDF 中没有对应的页面记录")
    if (type(pdf_page_record.get("pdf_page")) is not int
            or any(pdf_page_record.get(field) != source[field]
                   for field in ("document_id", "source", "pdf_page"))):
        raise ValueError("PDF 页面不属于该文档、来源或页码")
    page_text = pdf_page_record.get("text")
    if not isinstance(page_text, str) or not page_text.strip():
        raise ValueError("对应 PDF 页未提取到可用文字，需要进一步核对")
    # 折叠空白差异，不删除词间空格、不变更数字、不做语义近似匹配。
    normalized_source = " ".join(source["text"].split())
    normalized_page = " ".join(page_text.split())
    if normalized_source not in normalized_page:
        raise ValueError("片段文字未在对应 PDF 页的提取文字中找到，需要进一步核对")
    return {
        "chunk_id": source["chunk_id"], "pdf_page": source["pdf_page"],
        "index_fields_match": True, "page_text_found": True,
    }
