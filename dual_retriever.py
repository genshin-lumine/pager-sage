from retriever import retrieve_dense
from bm25_retriever import retrieve_bm25
from retrieval_trace import run_stage


def _scope_chunks_by_id(index, metadata):
    """调用方提供 load_scope_index 返回的配套索引与元数据。"""
    chunks = metadata.get("chunks")
    if not isinstance(chunks, list) or index.ntotal != len(chunks):
        raise ValueError("索引数量与当前范围的片段数量不一致")

    chunks_by_id = {}
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
            raise ValueError("当前范围存在无效的片段记录")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("当前范围存在缺少标识的片段")
        if chunk_id in chunks_by_id:
            raise ValueError("当前范围存在重复片段标识")
        chunks_by_id[chunk_id] = chunk
    return chunks_by_id


def retrieve_dual(question, model, index, metadata, top_k=3, *, stage_records=None):
    chunks_by_id = _scope_chunks_by_id(index, metadata)
    # 只有两路结果都通过范围校验，才向调用方交付阶段快照。
    pending_records = [] if stage_records is not None else None
    dense_results = run_stage(pending_records, "dense", len(chunks_by_id), lambda: retrieve_dense(
        question=question,
        model=model,
        index=index,
        metadata=metadata,
        top_k=top_k,
    ))

    bm25_results = run_stage(pending_records, "bm25", len(chunks_by_id), lambda: retrieve_bm25(
        question=question,
        # 取出同一份 metadata 中的片段列表
        chunks=metadata["chunks"],
        top_k=top_k,
    ))

    dual_results = {
        "dense": dense_results,
        # 保存 BM25 返回的候选列表
        "bm25": bm25_results,
    }

    # 两路结果都必须对应本次传入的原始记录，包含相同的来源和正文。
    for method, results in dual_results.items():
        for result in results:
            chunk = result["chunk"]
            chunk_id = chunk.get("chunk_id")
            if chunk_id not in chunks_by_id or chunk != chunks_by_id[chunk_id]:
                raise ValueError(f"{method} 返回的片段与当前检索范围不一致")
    if stage_records is not None:
        stage_records.extend(pending_records)
    return dual_results


def merge_candidates(dual_results):
    candidates_by_id = {}

    for method in ("dense", "bm25"):
        for result in dual_results[method]:
            chunk = result["chunk"]

            # 取出当前片段的唯一标识
            chunk_id = result["chunk"]["chunk_id"]

            if chunk_id not in candidates_by_id:
                candidates_by_id[chunk_id] = {
                    "chunk": chunk,
                    "retrievals": {},
                }

            candidate = candidates_by_id[chunk_id]

            if candidate["chunk"] != chunk:
                raise ValueError("同一片段标识对应不同的内容或来源")

            if method in candidate["retrievals"]:
                raise ValueError("同一路检索结果出现重复片段标识")

            # 以当前检索方式作为键，分别保存两路信息
            candidate["retrievals"][method] = {
                "rank": result["rank"],
                "score": result["score"],
            }

    return list(candidates_by_id.values())
