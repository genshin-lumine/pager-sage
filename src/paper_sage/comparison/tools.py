"""为论文对比读取独立的单篇索引，供后续创建各篇的检索工具。"""

from copy import deepcopy

from paper_sage.retrieval.tools import create_search_tool
from paper_sage.retrieval.scope import get_selected_documents, load_scope_index


def load_comparison_indexes(project, documents, *, workspace_root=None):
    """只读现有索引；每篇论文获得各自的 index 与配套 metadata。"""
    if (not isinstance(project, dict) or not isinstance(project.get("project_id"), str)
            or not project["project_id"].strip() or not isinstance(project.get("documents"), list)):
        raise ValueError("需要有效的当前项目记录")
    if not isinstance(documents, list) or len(documents) != 2:
        raise ValueError("当前对比流程需要恰好两篇所选论文")
    project = deepcopy(project)
    documents = deepcopy(documents)
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("所选文档记录必须是字典")
        for field in ("document_id", "source", "index_version"):
            if not isinstance(document.get(field), str) or not document[field].strip():
                raise ValueError(f"所选文档记录缺少有效的 {field}")
    document_ids = [document["document_id"] for document in documents]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("对比双方不能是重复的文档记录")
    selected = get_selected_documents(project, document_ids)
    for requested, stored in zip(documents, selected, strict=True):
        if any(requested[field] != stored[field] for field in ("document_id", "source", "index_version")):
            raise ValueError("所选文档的来源或索引版本与当前项目不一致")

    resources = {}
    for document in selected:
        document_id = document["document_id"]
        # load_scope_index 的第二个参数是列表。
        # 本次循环只加载 document_id 这一篇。
        selected_document_ids = [document_id]
        if workspace_root is None:
            index, metadata = load_scope_index(project, selected_document_ids)
        else:
            index, metadata = load_scope_index(project, selected_document_ids, workspace_root=workspace_root)

        # 检查实际加载内容，不能只凭 resources 的字典键判断隔离成功。
        if not metadata["chunks"] or any(
            chunk["project_id"] != project["project_id"]
            or chunk["document_id"] != document_id
            or chunk["source"] != document["source"]
            for chunk in metadata["chunks"]
        ):
            raise ValueError("对比工具的索引必须只包含当前这一篇论文")
        resources[document_id] = {
            "project_id": project["project_id"], "document": document,
            "index": index, "metadata": metadata,
        }
    return resources


def create_comparison_tools(resources, embedding_model, rerank_model=None, *, rerank_model_loader=None):
    """共用模型，分别绑定单篇索引；向量模型须与资源中的编码配置对应。

    资源来自 load_comparison_indexes；当前共享向量模型的版本要求两篇编码配置一致。
    此函数不执行检索；各工具仍只接收 question，文档范围由程序绑定。
    """
    if not isinstance(resources, dict) or len(resources) != 2:
        raise ValueError("需要两篇论文各自的索引资源")
    project_ids, common_config = set(), None
    for document_id, resource in resources.items():
        if not isinstance(resource, dict) or not isinstance(resource.get("document"), dict):
            raise ValueError("单篇索引资源缺少文档记录")
        document = resource["document"]
        project_id = resource.get("project_id")
        if not isinstance(project_id, str) or not project_id.strip() or document.get("document_id") != document_id:
            raise ValueError("索引资源与对应项目或文档标识不一致")
        if not isinstance(document.get("source"), str) or not document["source"].strip():
            raise ValueError("索引资源缺少文档来源")
        project_ids.add(project_id)
        metadata, index = resource.get("metadata"), resource.get("index")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("chunks"), list) or not metadata["chunks"]:
            raise ValueError("单篇索引资源缺少有效片段")
        if getattr(index, "ntotal", None) != len(metadata["chunks"]) or getattr(index, "d", None) != metadata.get("dimension"):
            raise ValueError("单篇索引与元数据不配套")
        for chunk in metadata["chunks"]:
            if (not isinstance(chunk, dict) or chunk.get("project_id") != project_id
                    or chunk.get("document_id") != document_id or chunk.get("source") != document["source"]):
                raise ValueError("单篇工具资源混入其他项目、文档或来源")
        config = metadata.get("embedding_config")
        if (not isinstance(config, dict) or not isinstance(config.get("model_name"), str)
                or not config["model_name"].strip() or type(config.get("normalize_embeddings")) is not bool
                or not isinstance(config.get("query_prefix"), str) or not isinstance(config.get("passage_prefix"), str)):
            raise ValueError("单篇索引的编码配置不完整")
        if common_config is None:
            common_config = config
        elif config != common_config:
            raise ValueError("共享向量模型要求两篇论文的编码配置一致")
    if len(project_ids) != 1:
        raise ValueError("两篇论文的索引资源必须属于同一项目")

    tools_by_document = {}
    for document_id, resource in resources.items():
        tools_by_document[document_id] = create_search_tool(
            embedding_model=embedding_model,
            index=resource["index"],
            metadata=deepcopy(resource["metadata"]),
            rerank_model=rerank_model,
            rerank_model_loader=rerank_model_loader,
            recall_top_k=6, rrf_top_k=6, final_top_k=3, rank_constant=60,
        )
    return tools_by_document
