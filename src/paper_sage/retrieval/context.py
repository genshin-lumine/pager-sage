def build_reference_text(results):
    blocks = []

    for result in results:
        chunk = result["chunk"]

        block = (
            f"[片段 {result['rank']}]\n"
            f"片段标识：{chunk['chunk_id']}\n"
            f"来源：{chunk['source']}\n"
            f"PDF 页码：{chunk['pdf_page']}\n"
            f"原文：\n{chunk['text']}"
        )

        # TODO：把当前 block 加入 blocks 列表
        blocks.append(block)
    return "\n\n".join(blocks)