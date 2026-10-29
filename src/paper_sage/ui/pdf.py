"""按需准备并展示当前引用的 PDF 原页，复用文档范围检查。"""

from pathlib import Path

import streamlit as st

from paper_sage.qa.pdf import load_pdf_page_preview


def render_pdf_source_control(source, *, project_id, selected_documents, key_prefix):
    """只在打开本处引用的开关时读取当前文档；不把 PDF 数据写入聊天记录。"""
    if not st.toggle("查看 PDF 原页", key=f"{key_prefix}_show"):
        return
    try:
        preview = load_pdf_page_preview(source, project_id, selected_documents)
    except (OSError, ValueError) as error:
        st.warning(f"无法查看 PDF 原页：{error}")
        return
    render_pdf_page_preview(preview, key_prefix=key_prefix)


def render_pdf_page_preview(preview, *, key_prefix):
    pdf_bytes = preview["pdf_bytes"]
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("PDF 查看器需要 PDF 的 bytes 数据")
    filename = Path(preview["source"]).name
    original_page = preview["pdf_page"]
    st.caption(f"{filename} · 原 PDF 第 {original_page} 页 / 共 {preview['total_pages']} 页")
    st.caption(f"这里展示原论文第 {original_page} 页的单页副本；查看器中的 1/1 是副本页码。")
    st.download_button(
        "下载当前证据页", data=pdf_bytes, file_name=preview["download_name"],
        mime="application/pdf", on_click="ignore", key=f"{key_prefix}_download",
    )
    st.pdf(
        pdf_bytes, height=900, key=f"{key_prefix}_viewer",
        alt=f"{filename} 的原 PDF 第 {original_page} 页",
    )
