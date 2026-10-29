"""将检索记录组织为逐维度的双篇证据输入，不判断片段是否支持比较结论。"""

from copy import deepcopy
from pathlib import Path

from langchain_core.messages import ToolMessage

from paper_sage.comparison.inputs import build_comparison_tasks
from paper_sage.comparison.retrieval import validate_comparison_result


def build_comparison_material(project_id, documents, dimensions, records):
    """保留每个“论文、维度”的独立参考资料；缺少整项记录时拒绝继续。"""
    expected_tasks = build_comparison_tasks(project_id, documents, dimensions)
    clean_dimensions = list(dict.fromkeys(task["dimension"] for task in expected_tasks))
    if not isinstance(records, list) or len(records) != len(expected_tasks):
        raise ValueError("需要每篇论文在每个维度上的完整检索记录，空候选也须保留记录")
    documents_by_id = {document["document_id"]: document for document in documents}
    records_by_scope = {}
    for number, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("task"), dict):
            raise ValueError("对比检索记录缺少任务")
        task = record["task"]
        document_id, dimension = task.get("document_id"), task.get("dimension")
        if not isinstance(document_id, str) or document_id not in documents_by_id:
            raise ValueError("检索任务不属于所选论文")
        if not isinstance(dimension, str) or dimension not in clean_dimensions:
            raise ValueError("检索任务不属于当前比较维度")
        if task.get("project_id") != project_id or task.get("source") != documents_by_id[document_id]["source"]:
            raise ValueError("检索任务的项目或来源不一致")
        if not isinstance(task.get("question"), str) or not task["question"].strip():
            raise ValueError("检索任务缺少有效问题")

        # 用 document_id 和 dimension 按此顺序组成一个元组，作为字典键。
        key = (document_id, dimension)
        if key is None:
            raise NotImplementedError("请先填写同时包含论文与维度的键")
        if key in records_by_scope:
            raise ValueError("同一论文和维度的检索记录重复")
        if not isinstance(record.get("reference_text"), str) or not isinstance(record.get("retrieval_artifact"), dict):
            raise ValueError("检索记录缺少参考正文或结构化数据")
        # 重放本地数据以校验正文与来源，不执行工具或模型。
        tool_call = {"name": "search_papers", "id": f"comparison_material_{number}"}
        message = ToolMessage(
            content=record["reference_text"], artifact=record["retrieval_artifact"],
            tool_call_id=tool_call["id"], name=tool_call["name"],
        )
        validate_comparison_result(task, message, tool_call)
        records_by_scope[key] = record

    groups = []
    for dimension in clean_dimensions:
        papers = []
        for document in documents:
            record = records_by_scope.get((document["document_id"], dimension))
            if record is None:
                raise ValueError("缺少对应论文与维度的检索记录，请检查键的字段和顺序")
            papers.append({
                "document_id": document["document_id"], "name": Path(document["source"]).name,
                "source": document["source"], "question": record["task"]["question"],
                "reference_text": record["reference_text"],
                "retrieval_results": deepcopy(record["retrieval_artifact"]["retrieval_results"]),
            })
        groups.append({"dimension": dimension, "papers": papers})
    return {"project_id": project_id, "dimensions": clean_dimensions, "groups": groups}
