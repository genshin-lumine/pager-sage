"""展示已保存的比较记录，可提供 PDF 查看入口，不重新执行比较。"""

from pathlib import Path

import streamlit as st

from paper_sage.comparison.report import validate_comparison_artifact


def render_comparison_details(artifact, *, pdf_source_renderer=None):
    validate_comparison_artifact(artifact)
    names = {doc["document_id"]: Path(doc["source"]).name for doc in artifact["documents"]}
    plan = artifact["plan_record"]["plan"]
    with st.expander("查看比较计划与执行过程"):
        st.write("参与比较的论文：" + "、".join(names.values()))
        for dimension in plan["dimensions"]:
            st.markdown(f"**{dimension['name']}**")
            st.write(f"检索问题：{dimension['query']}")
        if plan["clarification_question"]:
            st.write(plan["clarification_question"])
        stage_names = {"plan": "识别比较维度", "tasks": "生成逐篇任务", "load_tools": "准备检索工具",
                       "retrieve": "逐篇检索", "material": "按维度组织资料", "generate": "生成比较"}
        st.dataframe([
            {"阶段": stage_names[stage["stage"]], "耗时（秒）": f"{stage['elapsed_seconds']:.3f}"}
            for stage in artifact["stages"]
        ], hide_index=True)
        st.caption("以上是比较流程的内部阶段；入口分类路径见“本轮工作流程”。")
        if artifact.get("report_path"):
            st.caption(f"比较报告：{artifact['report_path']}")

    if artifact["response_kind"] == "clarify":
        st.caption("本轮需要补充比较维度，尚未执行比较检索和生成。")
        return

    review = artifact.get("content_review")
    resolved = (review["resolved_comparison"] if review is not None
                else artifact["generation_record"]["resolved_comparison"])
    if review is None:
        st.caption("本轮比较由模型生成，可展开原文核对发现、缺失信息与比较条件。")
    else:
        st.caption(f"本轮展示已记录的 {len(review['record']['corrections'])} 处内容修订，原始模型响应保留。")
        with st.expander("查看比较内容修订"):
            for correction in review["record"]["corrections"]:
                st.write(f"原文：{correction['original_text']}")
                st.write(f"修订：{correction['replacement_text']}")
                st.caption(f"理由：{correction['reason']}")
            st.caption(f"修订记录：{review['path']}")

    with st.expander("查看比较原文依据"):
        for row_number, row in enumerate(resolved["comparisons"], start=1):
            st.markdown(f"**比较维度：{row['dimension']}**")
            for paper_number, paper in enumerate(row["papers"], start=1):
                st.markdown(f"**{paper['name']}**")
                sources = {}
                for number, finding in enumerate(paper["findings"], 1):
                    st.write(f"发现 {number}：{finding['text']}")
                    st.caption("引用片段：" + "、".join(finding["chunk_ids"]))
                    for source in finding["sources"]:
                        sources.setdefault(source["chunk_id"], source)
                if paper["missing_information"]:
                    st.write(f"缺失信息：{paper['missing_information']}")
                for source_number, source in enumerate(sources.values(), start=1):
                    st.caption(f"来源：{source['source']} · PDF 第 {source['pdf_page']} 页")
                    st.caption(f"片段标识：{source['chunk_id']}")
                    st.code(source["text"], language=None, wrap_lines=True)
                    if pdf_source_renderer is not None:
                        pdf_source_renderer(source, f"compare_{row_number}_{paper_number}_{source_number}")
            if row["comparison"] is not None:
                st.caption("本维度比较引用：" + "、".join(row["comparison"]["chunk_ids"]))

    with st.expander("查看比较检索记录"):
        st.caption("按论文和维度分别展示本轮送入比较模型的候选，保留未被结论引用的候选。")
        for record in artifact["records"]:
            task = record["task"]
            retrieval = record["retrieval_artifact"]
            results = retrieval["retrieval_results"]
            st.markdown(f"**{names[task['document_id']]} · {task['dimension']}**")
            st.write(f"检索问题：{task['question']}")
            st.caption(f"候选数量：{len(results)}")
            if not results:
                st.write("本项没有检索到候选片段。")
            for result in results:
                source = result["chunk"]
                st.caption(f"排名 {result['rank']} · PDF 第 {source['pdf_page']} 页 · 片段：{source['chunk_id']}")
                st.caption(f"来源：{source['source']}")
                st.code(source["text"], language=None, wrap_lines=True)
            if retrieval.get("retrieval_trace") is not None:
                st.caption("各阶段的数量、耗时、排名与原文记录")
                st.json(retrieval["retrieval_trace"], expanded=False)
