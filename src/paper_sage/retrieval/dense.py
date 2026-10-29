def retrieve_dense(question, model, index, metadata, top_k=3):
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须是正整数")

    if index.ntotal == 0:
        return []

    actual_k = min(top_k, index.ntotal)

    embedding_config = metadata["embedding_config"]
    chunks = metadata["chunks"]

    query_text = embedding_config["query_prefix"] + question

    query_vectors = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=embedding_config["normalize_embeddings"],
    )
    query_vectors = query_vectors.astype("float32")

    scores, positions = index.search(query_vectors, k=actual_k)

    results = []

    for rank in range(actual_k):
        position = int(positions[0][rank])
        score = float(scores[0][rank])
        chunk = chunks[position]
        # 字典包含 rank、score、chunk 三个字段
        result_records = {
            "rank": rank + 1,
            "score": score,
            "chunk": chunk,
        }
        results.append(result_records)

    # TODO 2：全部结果收集完成后，返回 results
    return results