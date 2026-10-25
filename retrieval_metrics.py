

def calculate_recall_at_k(retrieved_ids, relevant_ids, k=3):
    """输入按排名排列的片段标识、全部相关片段标识和检查数量。"""
    if type(k) is not int or k < 1:
        raise ValueError("k 必须是正整数")

    # set 是集合：同一个片段标识只保留一次。
    relevant_ids = set(relevant_ids)
    if not relevant_ids:
        # 分母为零时不计算 Recall；None 表示本指标不适用。
        return None

    # 先按原排名取前 k 条。
    top_ids = retrieved_ids[:k]

    # 收集前 k 条中命中的相关片段，重复标识不重复计数。
    hit_ids = set()
    for chunk_id in top_ids:
        if chunk_id in relevant_ids:
            hit_ids.add(chunk_id)

    # 统计 hit_ids 中命中的不同片段数量。
    hit_count = len(hit_ids)

    # 用命中数量除以全部相关片段数量。
    recall = hit_count / len(relevant_ids)

    return recall


def calculate_rr_at_k(retrieved_ids, relevant_ids, k=3):
    """计算单个问题的 RR；不存在相关片段时返回 None 表示不适用。"""
    if type(k) is not int or k < 1:
        raise ValueError("k 必须是正整数")

    relevant_ids = set(relevant_ids)
    if not relevant_ids:
        return None

    # enumerate 同时给出名次和片段标识；start=1 让名次从 1 开始。
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            # 找到第一个相关片段，返回当前名次 rank 的倒数。
            return 1.0 / rank

    # 有相关片段，但检查完前 k 条仍未命中，返回本题的 RR。
    return 0.0


def calculate_mrr(rr_values):
    """对同一检索方法、同一 K 下各问题的 RR 求平均。"""
    # None 表示指标不适用，排除；未命中的 0 必须保留并参与平均。
    applicable_values = [value for value in rr_values if value is not None]
    if not applicable_values:
        return None

    # 把 applicable_values 的总和除以它的元素数量。
    mrr = sum(applicable_values) / len(applicable_values)
    return mrr


def evaluate_retrieval_results(results, relevant_ids, k=3):
    """将按排名排列的完整检索记录转换为标识，再调用指标函数。"""
    retrieved_ids = []
    for result in results:
        # 从 result 的 chunk 字典中读取完整片段标识。
        chunk_id = result.get("chunk", {}).get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("检索记录缺少有效的片段标识")
        retrieved_ids.append(chunk_id)

    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError("评估结果中不能重复返回同一个片段标识")
    relevant_ids = set(relevant_ids)
    recall = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
    rr = calculate_rr_at_k(retrieved_ids, relevant_ids, k)
    return {
        "retrieved_chunk_ids": retrieved_ids[:k],
        "hit_chunk_ids": [cid for cid in retrieved_ids[:k] if cid in relevant_ids],
        "relevant_count": len(relevant_ids),
        "recall_at_k": recall,
        "rr_at_k": rr,
    }
