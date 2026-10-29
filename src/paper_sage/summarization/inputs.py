
def build_summary_batches(pages, project_id, max_chars=6000):
    """覆盖可提取的文字；字符上限不是 token 上限，也不代表图片已被读取。"""
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("项目标识不能为空")
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("批次字符上限必须是正整数")
    if not isinstance(pages, list) or not pages:
        raise ValueError("需要单篇论文的完整页面记录")
    if not isinstance(pages[0], dict):
        raise ValueError("页面记录必须是字典")
    document_id = pages[0].get("document_id")
    source = pages[0].get("source")
    if not all(isinstance(value, str) and value.strip() for value in (document_id, source)):
        raise ValueError("页面记录缺少文档标识或来源")

    batches = []
    current_parts = []
    current_chars = 0
    text_pages = []
    empty_pages = []

    def save_batch(parts, char_count):
        batches.append({
            "batch_id": f"{project_id}:{document_id}:summary:b{len(batches) + 1}",
            "parts": parts,
            "char_count": char_count,
        })

    for expected_page, page in enumerate(pages, start=1):
        if (not isinstance(page, dict)
                or page.get("document_id") != document_id or page.get("source") != source
                or type(page.get("pdf_page")) is not int or page["pdf_page"] != expected_page):
            raise ValueError("请提供同一文档从第 1 页开始、连续且有序的页面记录")
        text = page.get("text")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError("页面文字必须是字符串或 None")
        if not text.strip():
            empty_pages.append(expected_page)
            continue
        text_pages.append(expected_page)
        # 单页超过上限时，先拆成不重叠的文字段；start/end 对应提取文本的切片位置。
        for start in range(0, len(text), max_chars):
            end = min(start + max_chars, len(text))
            part = {
                "project_id": project_id, "document_id": document_id, "source": source,
                "pdf_page": expected_page,
                "part_id": f"{project_id}:{document_id}:p{expected_page}:chars{start}-{end}",
                "start_char": start, "end_char": end, "text": text[start:end],
            }

            # 判断加入本段后，当前批次的字符数是否会超过 max_chars。
            # 已有字符数是 current_chars，本段文字是 part["text"]；结果应为 bool。
            would_overflow = True if current_chars + len(part["text"]) > max_chars else False
            if type(would_overflow) is not bool:
                raise ValueError("would_overflow 必须是布尔值")
            if current_parts and would_overflow:
                save_batch(current_parts, current_chars)
                current_parts = []
                current_chars = 0

            current_parts.append(part)
            current_chars += len(part["text"])

    if current_parts:
        save_batch(current_parts, current_chars)
    if not batches:
        raise ValueError("未提取到可用于总结的文字；这不代表 PDF 没有信息")
    return {
        "project_id": project_id, "document_id": document_id, "source": source,
        "total_pages": len(pages), "text_pages": text_pages, "empty_pages": empty_pages,
        "total_char_count": sum(batch["char_count"] for batch in batches),
        "max_chars": max_chars, "batches": batches,
    }
