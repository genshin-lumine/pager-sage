import faiss
from pathlib import Path

from paper_sage.retrieval.index_store import load_index
from paper_sage.projects.store import get_index_dir


def get_selected_documents(project, selected_document_ids):
    documents_by_id = {}

    for document in project["documents"]:
        documents_by_id[document["document_id"]] = document

    selected_documents = []

    for document_id in selected_document_ids:
        if document_id not in documents_by_id:
            raise ValueError("所选文档不属于当前项目")

        # 根据 document_id 取出完整的文档记录
        # 把这条记录加入 selected_documents
        selected_documents.append(documents_by_id[document_id])

    return selected_documents


def load_scope_index(project, selected_document_ids, *, workspace_root=None):
    """在内存中合并所选文档的索引，不写入或修改原索引文件。"""
    selected_documents = get_selected_documents(project, selected_document_ids)
    if not selected_documents:
        raise ValueError("请至少选择一篇文档")

    combined_index = None
    combined_chunks = []
    common_config = None
    seen_chunk_ids = set()

    for document in selected_documents:
        output_dir = get_index_dir(
            project["project_id"],
            document["document_id"],
            document["index_version"],
        )
        if workspace_root is not None:
            root = Path(workspace_root).resolve()
            output_dir = (root / output_dir).resolve()
            if not output_dir.is_relative_to(root):
                raise ValueError("索引路径超出当前工作区")
        try:
            index, metadata = load_index(output_dir)
        except (KeyError, TypeError) as error:
            raise ValueError("索引元数据格式不完整，请检查对应文档的索引") from error

        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("当前范围检索只支持 IndexFlatIP 索引")

        config = metadata.get("embedding_config")
        if (
            not isinstance(config, dict)
            or not isinstance(config.get("model_name"), str)
            or not config["model_name"].strip()
            or not isinstance(config.get("normalize_embeddings"), bool)
            or not isinstance(config.get("passage_prefix"), str)
            or not isinstance(config.get("query_prefix"), str)
        ):
            raise ValueError("索引中的向量编码配置不完整")

        if combined_index is None:
            combined_index = faiss.IndexFlatIP(index.d)
            common_config = config.copy()
        else:
            if index.d != combined_index.d:
                raise ValueError("所选文档的向量维度不一致，不能合并检索")
            if config != common_config:
                raise ValueError("所选文档的向量编码配置不一致，不能合并检索")

        chunks = metadata["chunks"]
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("所选文档的索引没有可用片段")

        for chunk in chunks:
            if (
                not isinstance(chunk, dict)
                or chunk.get("project_id") != project["project_id"]
                or chunk.get("document_id") != document["document_id"]
                or chunk.get("source") != document["source"]
            ):
                raise ValueError("索引片段的项目、文档或来源与所选记录不一致")

            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("索引片段缺少有效的片段标识")
            if chunk_id in seen_chunk_ids:
                raise ValueError("所选索引中存在重复的片段标识")
            seen_chunk_ids.add(chunk_id)

        # 按原索引顺序读取所有已保存向量，不调用向量模型。
        vectors = index.reconstruct_n(0, index.ntotal)

        # 把 vectors 加入 combined_index
        combined_index.add(vectors)
        # 把 chunks 中的每条记录加入 combined_chunks，保持一维列表
        combined_chunks.extend(chunks)

    if combined_index.ntotal != len(combined_chunks):
        raise ValueError("合并后的向量数量与片段数量不一致")

    combined_metadata = {
        "schema_version": 1,
        "dimension": combined_index.d,
        "embedding_config": common_config,
        "chunks": combined_chunks,
    }
    return combined_index, combined_metadata
