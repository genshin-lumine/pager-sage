"""校验比较结果的结构与分组引用范围；不自动判断原文是否支持文字结论。"""

from copy import deepcopy
import json

from paper_sage.retrieval.tools import EMPTY_REFERENCE_TEXT
from paper_sage.retrieval.context import build_reference_text


MAX_FINDINGS_PER_PAPER = 3


def _text(value, label, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{label} 必须是{'可为空的' if allow_empty else '非空'}字符串")


def _fields(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{label} 必须且只能包含 {', '.join(fields)}")


def index_comparison_material(material):
    """以（论文标识，维度）建立原文索引，检查两篇的来源和各维度覆盖。"""
    if not isinstance(material, dict):
        raise ValueError("比较输入必须是字典")
    _text(material.get("project_id"), "project_id")
    dimensions = material.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("比较输入需要非空维度列表")
    for dimension in dimensions:
        _text(dimension, "比较维度")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("比较输入的维度不能重复")
    groups = material.get("groups")
    if not isinstance(groups, list) or len(groups) != len(dimensions):
        raise ValueError("比较输入缺少完整的维度分组")
    cells, canonical_chunks, sources_by_document = {}, {}, {}
    document_ids = None
    for dimension, group in zip(dimensions, groups, strict=True):
        if not isinstance(group, dict) or group.get("dimension") != dimension:
            raise ValueError("比较输入的分组与维度顺序不一致")
        papers = group.get("papers")
        if not isinstance(papers, list) or len(papers) != 2:
            raise ValueError("每个维度需要两篇论文的资料，空候选也必须保留")
        row_ids = []
        for paper in papers:
            if not isinstance(paper, dict):
                raise ValueError("论文资料必须是字典")
            for field in ("document_id", "name", "source", "question"):
                _text(paper.get(field), f"论文资料的 {field}")
            document_id = paper["document_id"]
            row_ids.append(document_id)
            if document_id in sources_by_document and sources_by_document[document_id] != paper["source"]:
                raise ValueError("同一论文在不同维度中的来源不一致")
            sources_by_document[document_id] = paper["source"]
            results = paper.get("retrieval_results")
            if not isinstance(results, list):
                raise ValueError("每篇论文必须保留候选列表")
            chunks_by_id = {}
            for result in results:
                if not isinstance(result, dict) or type(result.get("rank")) is not int or result["rank"] < 1:
                    raise ValueError("候选记录缺少有效排名")
                chunk = result.get("chunk")
                if (not isinstance(chunk, dict) or chunk.get("project_id") != material["project_id"]
                        or chunk.get("document_id") != document_id or chunk.get("source") != paper["source"]):
                    raise ValueError("比较输入混入其他项目、论文或来源的片段")
                _text(chunk.get("chunk_id"), "chunk_id")
                _text(chunk.get("text"), "片段正文")
                if type(chunk.get("pdf_page")) is not int or chunk["pdf_page"] < 1:
                    raise ValueError("片段的 PDF 页码无效")
                chunk_id = chunk["chunk_id"]
                if chunk_id in chunks_by_id:
                    raise ValueError("同一论文和维度中的片段标识重复")
                if chunk_id in canonical_chunks and canonical_chunks[chunk_id] != chunk:
                    raise ValueError("同一片段标识对应不同的原文记录")
                chunks_by_id[chunk_id] = chunk
                canonical_chunks[chunk_id] = chunk
            if paper.get("reference_text") != (build_reference_text(results) or EMPTY_REFERENCE_TEXT):
                raise ValueError("比较输入的参考正文与原始片段不一致")
            cells[(document_id, dimension)] = chunks_by_id
        if len(set(row_ids)) != 2:
            raise ValueError("比较双方的文档标识不能重复")
        if document_ids is None:
            document_ids = row_ids
        elif document_ids != row_ids:
            raise ValueError("各维度的两篇论文及排列顺序必须一致")
    return document_ids, cells


def _validate_chunk_ids(chunk_ids, allowed_chunk_ids):
    if not isinstance(chunk_ids, list) or not chunk_ids:
        raise ValueError("每条发现或比较结论必须提供非空的 chunk_ids 列表")
    seen = set()
    for chunk_id in chunk_ids:
        _text(chunk_id, "引用片段标识")
        # 判断 chunk_id 是否属于本次传入的 allowed_chunk_ids。
        chunk_is_allowed = True if chunk_id in allowed_chunk_ids else False
        if not chunk_is_allowed:
            raise ValueError(f"引用片段不属于当前论文与维度的允许范围：{chunk_id}")
        if chunk_id in seen:
            raise ValueError("同一条发现或比较结论的引用片段不能重复")
        seen.add(chunk_id)
    return seen


def parse_comparison(raw_response, material):
    """读取各篇发现、缺失信息与带双方来源的比较，返回校验后的字典。"""
    if not isinstance(raw_response, str):
        raise ValueError("比较响应必须是 JSON 字符串")
    document_ids, cells = index_comparison_material(material)
    result = json.loads(raw_response)
    _fields(result, ("comparisons",), "比较响应")
    rows = result["comparisons"]
    if not isinstance(rows, list) or len(rows) != len(material["dimensions"]):
        raise ValueError("比较响应必须覆盖全部指定维度")
    rows_by_dimension = {}
    for row in rows:
        _fields(row, ("dimension", "papers", "comparison", "comparison_limit"), "每个维度的比较")
        dimension = row["dimension"]
        _text(dimension, "输出维度")
        if dimension not in material["dimensions"] or dimension in rows_by_dimension:
            raise ValueError("比较响应出现未知或重复维度")
        papers = row["papers"]
        if not isinstance(papers, list) or len(papers) != 2:
            raise ValueError("每个维度必须分别返回两篇论文的结果")
        papers_by_id, cited_by_document = {}, {}
        for paper in papers:
            _fields(paper, ("document_id", "findings", "missing_information"), "每篇论文的结果")
            document_id = paper["document_id"]
            _text(document_id, "输出文档标识")
            if document_id not in document_ids or document_id in papers_by_id:
                raise ValueError("比较响应出现未知或重复论文")
            findings = paper["findings"]
            if not isinstance(findings, list) or len(findings) > MAX_FINDINGS_PER_PAPER:
                raise ValueError(f"findings 必须是最多 {MAX_FINDINGS_PER_PAPER} 条的列表")
            _text(paper["missing_information"], "missing_information", allow_empty=True)
            if not findings and not paper["missing_information"].strip():
                raise ValueError("没有可支持的发现时，必须说明缺失信息")
            allowed_chunk_ids = set(cells[(document_id, dimension)])
            cited = set()
            for finding in findings:
                _fields(finding, ("text", "chunk_ids"), "每条发现")
                _text(finding["text"], "发现正文")
                cited.update(_validate_chunk_ids(finding["chunk_ids"], allowed_chunk_ids))
            cited_by_document[document_id] = cited
            papers_by_id[document_id] = paper

        _text(row["comparison_limit"], "comparison_limit", allow_empty=True)
        comparison = row["comparison"]
        if comparison is None:
            if not row["comparison_limit"].strip():
                raise ValueError("无法形成比较时，必须说明比较限制")
        else:
            _fields(comparison, ("text", "chunk_ids"), "比较结论")
            _text(comparison["text"], "比较结论正文")
            if any(not cited for cited in cited_by_document.values()):
                raise ValueError("一方没有有依据的发现时，comparison 必须为 null")
            cited = _validate_chunk_ids(comparison["chunk_ids"], set().union(*cited_by_document.values()))
            if any(not cited.intersection(ids) for ids in cited_by_document.values()):
                raise ValueError("比较结论必须引用双方已列出的发现来源")
        if any(paper["missing_information"].strip() for paper in papers) and not row["comparison_limit"].strip():
            raise ValueError("有缺失信息时，必须说明比较限制")
        rows_by_dimension[dimension] = {
            **deepcopy(row), "papers": [deepcopy(papers_by_id[doc_id]) for doc_id in document_ids],
        }
    return {"comparisons": [rows_by_dimension[dimension] for dimension in material["dimensions"]]}
