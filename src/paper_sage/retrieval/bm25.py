import re

from rank_bm25 import BM25Okapi


def tokenize_bm25(text):
    # 把 text 转成小写，赋给 normalized_text
    normalized_text = text.lower()

    tokens = re.findall(r"[a-z0-9]+", normalized_text)

    return tokens


def retrieve_bm25(question, chunks, top_k=3):
    """在传入片段中按 BM25 排名，返回有查询词匹配的完整片段记录。"""
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须是正整数")

    query_tokens = tokenize_bm25(question)
    if not query_tokens or not chunks:
        return []

    tokenized_corpus = []
    for chunk in chunks:
        tokenized_corpus.append(tokenize_bm25(chunk["text"]))

    # 至少出现一个查询词才参与排名；不以分数是否大于 0 判断匹配。
    matching_positions = []
    for position, tokens in enumerate(tokenized_corpus):
        if any(token in tokens for token in query_tokens):
            matching_positions.append(position)

    if not matching_positions:
        return []

    # 用全部传入片段统计词频与长度，再对匹配的片段进行排名。
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)
    ranked_positions = sorted(
        matching_positions,
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for rank, position in enumerate(ranked_positions[:top_k], start=1):
        result = {
            "rank": rank,          # 当前排名
            "score": float(scores[position]),  # scores 中当前位置的分数
            "chunk": chunks[position],         # chunks 中当前位置的完整片段
        }
        results.append(result)

    return results
