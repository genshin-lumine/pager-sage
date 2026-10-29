"""展示已保存的结构化回答；可由调用方提供 PDF 查看入口。"""

from pathlib import Path

import streamlit as st

from paper_sage.qa.report import validate_saved_answer


def render_cited_answer_details(artifact, *, pdf_source_renderer=None):
    result = validate_saved_answer(artifact)
    with st.expander("查看结论原文依据"):
        if result["insufficient_reason"].strip():
            st.info("本次资料不足，没有带引用的结论；检索候选仍可在检索面板查看。")
            return
        st.caption("按结论展示所引片段的原文与来源；引用关联通过，不代表结论已被自动核实。")
        for number, claim in enumerate(artifact["claims_with_sources"], start=1):
            st.markdown(f"**结论 {number}**")
            st.write(claim["text"])
            for source_number, source in enumerate(claim["sources"], start=1):
                st.caption(f"文档：{Path(source['source']).name} · PDF 第 {source['pdf_page']} 页")
                st.caption(f"来源：{source['source']}")
                st.caption(f"文档标识：{source['document_id']}")
                st.caption(f"片段标识：{source['chunk_id']}")
                st.code(source["text"], language=None, wrap_lines=True)
                if pdf_source_renderer is not None:
                    pdf_source_renderer(source, f"qa_{number}_{source_number}")
