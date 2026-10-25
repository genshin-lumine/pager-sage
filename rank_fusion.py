from copy import deepcopy


def calculate_rrf_score(retrievals, rank_constant=60):
    if type(rank_constant) is not int or rank_constant < 1:
        raise ValueError("rank_constant 必须是正整数")
    if not isinstance(retrievals, dict):
        raise ValueError("各路检索信息必须是字典")

    total = 0.0

    for retrieval in retrievals.values():
        if not isinstance(retrieval, dict) or "rank" not in retrieval:
            raise ValueError("检索信息缺少排名")
        rank = retrieval["rank"]

        if type(rank) is not int or rank < 1:
            raise ValueError("排名必须是正整数")

        # 计算当前这一路的贡献，累加到 total
        total += 1.0 / (rank_constant + rank)

    return total


def fuse_rrf(candidates, top_k=3, rank_constant=60):
    """接收去重候选，返回含 RRF 分数、最终名次及完整来源的前 top_k 条。"""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须是正整数")
    if type(rank_constant) is not int or rank_constant < 1:
        raise ValueError("rank_constant 必须是正整数")

    scored = []
    seen_ids = set()
    for candidate in candidates:
        chunk = candidate["chunk"]
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("融合候选缺少有效的片段标识")
        if chunk_id in seen_ids:
            raise ValueError("融合前必须先按片段标识去重")
        seen_ids.add(chunk_id)
        retrievals = candidate["retrievals"]
        if not retrievals:
            raise ValueError("融合候选至少需要一路检索排名")

        score = calculate_rrf_score(retrievals, rank_constant=rank_constant)
        scored.append({
            "score": score,
            "chunk": deepcopy(chunk),
            "retrievals": deepcopy(retrievals),
        })

    # 同分时按片段标识固定顺序，避免候选输入顺序影响结果。
    scored.sort(key=lambda item: (-item["score"], item["chunk"]["chunk_id"]))
    results = scored[:top_k]
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results
