"""逐项执行对比检索，保留任务、原文及检索记录；不生成比较结论。"""

from copy import deepcopy
from uuid import uuid4

from langchain_core.messages import ToolMessage

from paper_sage.retrieval.tools import EMPTY_REFERENCE_TEXT
from paper_sage.retrieval.context import build_reference_text


def validate_comparison_result(task, message, tool_call):
    """检查工具消息对应关系、正文一致性以及片段的项目和单篇文档归属。"""
    if not isinstance(message, ToolMessage) or message.status != "success":
        raise ValueError("对比检索未返回成功的 ToolMessage")
    if message.tool_call_id != tool_call["id"] or message.name != tool_call["name"]:
        raise ValueError("检索结果与本次工具调用不匹配")
    artifact = message.artifact
    if not isinstance(artifact, dict) or artifact.get("question") != task["question"]:
        raise ValueError("检索记录与当前任务的问题不一致")
    results = artifact.get("retrieval_results")
    if not isinstance(results, list):
        raise ValueError("对比检索缺少候选列表")

    # 所有被保留的候选都检查归属，包括最后没有入选的中间记录。
    result_lists = [results]
    for field in ("retrieval_candidates", "rrf_results", "rerank_results"):
        if field in artifact:
            result_lists.append(artifact[field])
    if "retrieval_channels" in artifact:
        channels = artifact["retrieval_channels"]
        if not isinstance(channels, dict):
            raise ValueError("双路检索记录格式不正确")
        result_lists.extend(channels.values())
    if "retrieval_trace" in artifact:
        trace = artifact["retrieval_trace"]
        if not isinstance(trace, dict) or not isinstance(trace.get("stages"), list):
            raise ValueError("检索阶段记录格式不正确")
        for stage in trace["stages"]:
            if not isinstance(stage, dict):
                raise ValueError("检索阶段必须是字典")
            result_lists.append(stage.get("results"))
        # 路径表的来源也必须属于本篇；完整项目与文档信息在上述片段中检查。
        paths = trace.get("candidate_paths", [])
        if not isinstance(paths, list):
            raise ValueError("片段排名路径格式不正确")
        for path in paths:
            if not isinstance(path, dict) or path.get("source") != task["source"]:
                raise ValueError("片段排名路径来源与当前任务不一致")

    for candidates in result_lists:
        if not isinstance(candidates, list):
            raise ValueError("候选记录必须是列表")
        for candidate in candidates:
            chunk = candidate.get("chunk") if isinstance(candidate, dict) else None
            if not isinstance(chunk, dict):
                raise ValueError("候选记录缺少原始片段")
            if any(chunk.get(key) != task[key] for key in ("project_id", "document_id", "source")):
                raise ValueError("检索片段不属于当前任务的项目、文档或来源")
            for key in ("chunk_id", "text"):
                if not isinstance(chunk.get(key), str) or not chunk[key].strip():
                    raise ValueError(f"检索片段缺少有效的 {key}")
            if type(chunk.get("pdf_page")) is not int or chunk["pdf_page"] < 1:
                raise ValueError("检索片段缺少有效的 PDF 页码")
    if any(type(result.get("rank")) is not int or result["rank"] < 1 for result in results):
        raise ValueError("最终候选缺少有效排名")
    expected_text = build_reference_text(results) or EMPTY_REFERENCE_TEXT
    if message.content != expected_text:
        raise ValueError("工具正文与原始片段不一致")


def collect_comparison_evidence(tasks, tools_by_document):
    """调用方提供每篇论文各自的检索工具；失败时不返回部分成功的结果。

    正式调用方须用单篇索引创建各工具；此处再校验返回片段，不能只信工具名称。
    """
    if not isinstance(tasks, list) or not tasks or not isinstance(tools_by_document, dict):
        raise ValueError("需要非空任务列表与按文档标识组织的检索工具")
    tasks = deepcopy(tasks)
    identities, seen_tasks = {}, set()
    projects = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("比较任务必须是字典")
        for field in ("project_id", "document_id", "source", "dimension", "question"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                raise ValueError(f"比较任务缺少有效的 {field}")
        document_id = task["document_id"]
        projects.add(task["project_id"])
        if document_id in identities and identities[document_id] != task["source"]:
            raise ValueError("同一文档的任务来源不一致")
        identities[document_id] = task["source"]
        key = (document_id, task["dimension"])
        if key in seen_tasks:
            raise ValueError("同一论文的比较维度重复")
        seen_tasks.add(key)
        tool = tools_by_document.get(document_id)
        if (tool is None or not callable(getattr(tool, "invoke", None))
                or getattr(tool, "response_format", None) != "content_and_artifact"):
            raise ValueError("每篇论文都需要返回原文及 artifact 的检索工具")
    if len(projects) != 1:
        raise ValueError("比较任务必须属于同一个项目")

    records = []
    for task in tasks:
        document_id = task["document_id"]
        # 以 document_id 为键，从 tools_by_document 取出本篇的工具。
        search_tool = tools_by_document[document_id]
        tool_call = {
            "name": search_tool.name, "args": {"question": task["question"]},
            "id": f"compare_{uuid4().hex}", "type": "tool_call",
        }
        # 完整 ToolCall 会让 LangChain 返回同时带正文和 artifact 的 ToolMessage。
        message = search_tool.invoke(tool_call)
        validate_comparison_result(task, message, tool_call)
        records.append({
            "task": deepcopy(task), "reference_text": message.content,
            "retrieval_artifact": deepcopy(message.artifact),
        })
    return records
