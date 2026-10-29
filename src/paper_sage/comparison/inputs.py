"""准备逐篇、逐维度的对比任务；本模块只组织输入，不执行检索。"""

import json

from paper_sage.comparison.plan import parse_comparison_plan


def validate_comparison_documents(project_id, documents):
    """在请求计划或准备任务前，检查程序提供的项目和两篇文档。"""
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("项目标识必须是非空字符串")
    if not isinstance(documents, list) or len(documents) != 2:
        raise ValueError("当前对比流程需要恰好两篇所选论文")
    seen_ids = set()
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("文档信息必须是字典")
        for field in ("document_id", "source"):
            if not isinstance(document.get(field), str) or not document[field].strip():
                raise ValueError(f"文档信息缺少有效的 {field}")
        if document["document_id"] in seen_ids:
            raise ValueError("不能用同一文档重复充当对比双方")
        seen_ids.add(document["document_id"])


def build_comparison_tasks(project_id, documents, dimensions, *, queries_by_dimension=None):
    """所选范围由程序提供；可分别指定展示维度与实际检索问题。"""
    validate_comparison_documents(project_id, documents)
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("至少需要一个明确的比较维度")
    if any(not isinstance(value, str) or not value.strip() for value in dimensions):
        raise ValueError("比较维度必须是非空字符串")
    clean_dimensions = [value.strip() for value in dimensions]
    if len(set(clean_dimensions)) != len(clean_dimensions):
        raise ValueError("比较维度不能重复")
    if queries_by_dimension is not None:
        if not isinstance(queries_by_dimension, dict) or set(queries_by_dimension) != set(clean_dimensions):
            raise ValueError("检索问题映射必须恰好覆盖当前比较维度")
        if any(not isinstance(query, str) or not query.strip() for query in queries_by_dimension.values()):
            raise ValueError("每个比较维度的检索问题必须是非空字符串")
        queries_by_dimension = {dimension: query.strip() for dimension, query in queries_by_dimension.items()}

    tasks = []
    for document in documents:
        # 每次循环使用当前文档的标识。
        document_id = document["document_id"]

        for dimension in clean_dimensions:
            question = f"这篇论文在“{dimension}”方面报告了哪些具体信息？请查找原文依据。"
            if queries_by_dimension is not None:
                # 以当前 dimension 为键，取出 queries_by_dimension 中对应的检索问题。
                question = queries_by_dimension[dimension]
            tasks.append({
                "project_id": project_id,
                "document_id": document_id,
                "source": document["source"],
                "dimension": dimension,
                "question": question,
            })
    return tasks


def prepare_comparison_tasks(project_id, documents, plan):
    """将经过校验的计划展开为任务；需要澄清时直接返回追问。"""
    validate_comparison_documents(project_id, documents)
    plan = parse_comparison_plan(json.dumps(plan, ensure_ascii=False))
    if plan["clarification_question"]:
        return {"response_kind": "clarify", "reply": plan["clarification_question"], "tasks": []}

    dimensions = []
    queries_by_dimension = {}
    for dimension in plan["dimensions"]:
        name = dimension["name"]
        dimensions.append(name)
        # 保留学习者完成的维度名称到检索问题的映射。
        queries_by_dimension[name] = dimension["query"]

    tasks = build_comparison_tasks(
        project_id, documents, dimensions, queries_by_dimension=queries_by_dimension,
    )
    return {"response_kind": "tasks", "reply": "", "tasks": tasks}
