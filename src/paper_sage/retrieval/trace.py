
from copy import deepcopy
from time import perf_counter


def run_stage(records, stage, input_count, operation):
    """只计 operation 的耗时；复制快照和页面展示不在计时内。

    records 为 None 时沿用原调用行为。失败向上抛出，不生成成功记录。
    """
    if records is None:
        return operation()
    started = perf_counter()
    results = operation()
    elapsed_seconds = perf_counter() - started
    records.append({
        "stage": stage,
        "input_count": input_count,
        "output_count": len(results),
        "elapsed_seconds": elapsed_seconds,
        "results": deepcopy(results),
    })
    return results


def build_candidate_paths(stage_records):
    """接收本轮成功完成的六个阶段；按去重时的原顺序汇总。"""
    stages = {record["stage"]: record["results"] for record in stage_records}
    rrf_by_id = {result["chunk"]["chunk_id"]: result for result in stages["rrf"]}
    rerank_by_id = {result["chunk"]["chunk_id"]: result for result in stages["rerank"]}
    selected_ids = {result["chunk"]["chunk_id"] for result in stages["select"]}

    rows = []
    for candidate in stages["merge"]:
        chunk = candidate["chunk"]
        chunk_id = chunk["chunk_id"]
        rrf_result = rrf_by_id.get(chunk_id, {})
        rerank_result = rerank_by_id.get(chunk_id, {})
        selected = chunk_id in selected_ids
        rows.append({
            "chunk_id": chunk_id,
            "source": chunk["source"],
            "pdf_page": chunk["pdf_page"],
            "dense_rank": candidate["retrievals"].get("dense", {}).get("rank"),
            "dense_score": candidate["retrievals"].get("dense", {}).get("score"),
            "bm25_rank": candidate["retrievals"].get("bm25", {}).get("rank"),
            "bm25_score": candidate["retrievals"].get("bm25", {}).get("score"),
            "rrf_rank": rrf_result.get("rank"),
            "rrf_score": rrf_result.get("score"),
            "rerank_rank": rerank_result.get("rank"),
            "rerank_score": rerank_result.get("rerank_score"),
            "final_selected": selected,
            "selection_status": (
                "selected" if selected else
                "excluded_by_rrf" if not rrf_result else "excluded_by_final_top_k"
            ),
        })
    return rows
