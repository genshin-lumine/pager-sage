"""检查已保存回答并生成聊天正文；不依赖页面组件。"""

from pathlib import Path

from paper_sage.qa.model import validate_cited_answer_record


def validate_saved_answer(artifact):
    return validate_cited_answer_record(
        artifact, artifact["question"], artifact["final_results"], history=artifact.get("history", []),
    )


def format_cited_answer(artifact):
    """聊天正文由通过校验的结论生成；来源标签来自程序保存的记录。"""
    result = validate_saved_answer(artifact)
    if result["insufficient_reason"].strip():
        return ("**结论**\n\n资料不足，无法依据本次检索资料回答。\n\n"
                f"**资料不足说明**\n\n{result['insufficient_reason']}")
    lines = ["**结论**"]
    for number, claim in enumerate(artifact["claims_with_sources"], start=1):
        labels = []
        seen = set()
        for source in claim["sources"]:
            key = (source["document_id"], source["source"], source["pdf_page"])
            if key not in seen:
                seen.add(key)
                labels.append(f"{Path(source['source']).name} · PDF 第 {source['pdf_page']} 页")
        lines.append(f"{number}. {claim['text']}（{'；'.join(labels)}）")
    return "\n\n".join(lines)
