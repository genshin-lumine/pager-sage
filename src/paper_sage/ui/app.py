import streamlit as st
from paper_sage.ui.theme import (
    apply_theme, render_sidebar_brand, render_workspace_header,
    render_scope_overview, render_empty_conversation,
)
from paper_sage.llm.client import create_chat_model, IncompleteAnswerError
import sqlite3
from paper_sage.workflows.runner import start_workflow_run, resume_workflow_run, WorkflowRunError
from paper_sage.ui.workflow import clear_loaded_workflow, remember_workflow_result, render_workflow_controls
from paper_sage.workflows.observation import observe_model, observe_operation
from paper_sage.summarization.service import create_summary_runner
from paper_sage.comparison.service import create_comparison_runner, ComparisonServiceError
from paper_sage.ui.comparison import render_comparison_details
from paper_sage.ui.citations import render_cited_answer_details
from paper_sage.ui.pdf import render_pdf_source_control
from paper_sage.sessions.recovery import capture_scope_inputs
from paper_sage.ui.sessions import render_session_controls
from paper_sage.qa.retry import QARetryError
from paper_sage.ui.qa_retry import render_qa_retry_details, render_qa_retry_error
from paper_sage.summarization.parser import SUMMARY_FIELDS, SUMMARY_LABELS
from paper_sage.retrieval.tools import create_search_tool, EmptyRetrievalError
from langgraph.errors import GraphRecursionError
from json import JSONDecodeError
from sentence_transformers import SentenceTransformer
from paper_sage.retrieval.reranker import get_rerank_cache_folder, load_reranker
from pathlib import Path
from pypdf.errors import PyPdfError
from paper_sage.documents.pdf import load_pdf_pages
from paper_sage.retrieval.scope import get_selected_documents, load_scope_index
from paper_sage.sessions.chat import (
    create_chat_session,
    ensure_chat_session,
    build_history_messages,
)
from paper_sage.documents.deleter import (
    delete_document,
    retry_document_cleanup,
    begin_document_import,
    finish_document_import,
)
from paper_sage.documents.importer import (
    get_file_hash,
    find_existing_document,
    save_uploaded_pdf,
    build_document_index,
)
from paper_sage.projects.store import (
    load_project,
    get_index_dir,
    list_projects,
    create_project,
)
from openai import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
    LengthFinishReasonError,
    ContentFilterFinishReasonError,
)

# Day 28：开发对照与固定新题复验后，采用质量优先的 B 配置。
RECALL_TOP_K = 6
RRF_TOP_K = 6
FINAL_TOP_K = 3
RRF_RANK_CONSTANT = 60
# 问答的尝试次数与结论引用协议统一由 workflow_runner 定义并绑定到运行快照。

@st.cache_resource(show_spinner="正在加载向量模型……")
def load_embedding_model(model_name):
    return SentenceTransformer(model_name, device="cpu")


@st.cache_resource(show_spinner="正在加载所选论文的索引……")
def load_retrieval_resources(project, selected_document_ids, input_files=None):
    # 文件摘要参与缓存键；同名同版本文件改变后，不再使用旧索引资源。
    index, metadata = load_scope_index(project, selected_document_ids)
    model = load_embedding_model(metadata["embedding_config"]["model_name"])

    return model, index, metadata


@st.cache_resource(show_spinner="正在加载精排模型……")
def load_rerank_resources(cache_folder):
    # 页面仅复用已准备好的缓存，缺少模型时提示准备，避免交互时大文件下载。
    return load_reranker(cache_folder=cache_folder, local_files_only=True)


def render_candidate(chunk, label, retrievals, fusion_score=None, rerank_score=None):
    st.markdown(f"**{label} · PDF 第 {chunk['pdf_page']} 页**")
    st.caption(f"来源：{chunk['source']}")
    st.caption(f"片段标识：{chunk['chunk_id']}")
    if fusion_score is not None:
        st.caption(f"RRF 分数：{fusion_score:.6f}")
    if rerank_score is not None:
        st.caption(f"精排分数：{rerank_score:.4f}")
    for method, retrieval in retrievals.items():
        st.caption(
            f"{method.upper()} · 排名：{retrieval['rank']}"
            f" · 分数：{retrieval['score']:.4f}"
        )
    st.code(chunk["text"], language=None, wrap_lines=True)


def render_message(message, *, message_key, project_id, selected_documents):
    def render_source_pdf(source, source_key):
        render_pdf_source_control(
            source, project_id=project_id, selected_documents=selected_documents,
            key_prefix=f"{message_key}_{source_key}",
        )

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 旧消息没有该字段时继续展示原正文；页面重运行只读取已保存的证据。
        answer_artifact = message.get("answer_artifact")
        if answer_artifact is not None:
            render_cited_answer_details(answer_artifact, pdf_source_renderer=render_source_pdf)

        workflow = message.get("workflow_trace")
        if workflow is not None:
            route = workflow["route"]
            names = {"qa": "论文问答", "clarify": "需要澄清", "summary": "论文总结", "compare": "论文对比"}
            with st.expander("查看本轮工作流程"):
                st.write(f"任务分类：{names[route['intent']]}")
                st.write(f"分类理由：{route['reason']}")
                st.code(" → ".join(workflow["path"]) + " → END", language=None)
                if message.get("retrieval_query"):
                    st.write(f"本轮检索问题：{message['retrieval_query']}")

        summary = message.get("summary_artifact")
        if summary is not None:
            st.caption("已复用保存的总结。" if summary["reused"] else "本次已生成并保存总结。")
            with st.expander("查看总结原文依据"):
                coverage = summary["coverage"]
                st.caption(f"PDF 共 {coverage['total_pages']} 页 · 已处理文字 {len(coverage['text_pages'])} 页 · 分为 {coverage['batch_count']} 部分")
                if coverage["empty_pages"]:
                    st.write("未提取到可用文字的页码：", coverage["empty_pages"])
                st.caption("以下保留所引文字段，可对照原文核对总结。")
                notes_by_id = {note["note_id"]: note for note in summary["notes"]}
                for field in SUMMARY_FIELDS:
                    for number, point in enumerate(summary["resolved_summary"][field], start=1):
                        st.markdown(f"**{SUMMARY_LABELS[field]} {number}**")
                        st.write(point["text"])
                        for note_id in point["note_ids"]:
                            st.caption(f"支持要点：{notes_by_id[note_id]['text']}")
                        for source_number, source in enumerate(point["sources"], start=1):
                            st.caption(f"来源：{source['source']} · PDF 第 {source['pdf_page']} 页")
                            st.caption(f"文字段标识：{source['part_id']}")
                            st.code(source["text"], language=None, wrap_lines=True)
                            render_source_pdf(source, f"summary_{field}_{number}_{source_number}")
                if summary["review_corrections"]:
                    st.caption(f"汇总输入包含 {len(summary['review_corrections'])} 条人工校正；原始模型报告保留。")
                st.caption(f"总结报告：{summary['report_path']}")

        comparison = message.get("comparison_artifact")
        if comparison is not None:
            render_comparison_details(comparison, pdf_source_renderer=render_source_pdf)

        retry_artifact = message.get("qa_retry_artifact")
        if retry_artifact is not None:
            render_qa_retry_details(retry_artifact)
            if retry_artifact["attempts_used"] > 1:
                st.caption("下方检索片段与流程对应最后一次尝试；此前资料见“查看本轮检索尝试”。")

        results = message.get("retrieval_results", [])
        channels = message.get("retrieval_channels")

        if channels is not None:
            candidates = message["retrieval_candidates"]
            with st.expander("查看本轮检索片段"):
                st.caption(
                    f"Dense：{len(channels['dense'])} 条"
                    f" · BM25：{len(channels['bm25'])} 条"
                    f" · 去重后：{len(candidates)} 个片段"
                )
                if not channels["bm25"]:
                    st.info("BM25 未找到可用的词项匹配。当前分词仅处理英文字母和数字。")
                if message.get("reference_method") == "rerank":
                    st.caption(f"本轮回答使用 Rerank 精排后的 {len(results)} 个片段作为参考资料。")
                    st.caption(
                        f"参与精排：{len(message['rrf_results'])} 条"
                        f" · 最终参考：{len(results)} 条"
                        f" · 精排耗时：{message['rerank_elapsed_seconds']:.3f} 秒（不含模型加载）"
                    )
                    for result in results:
                        render_candidate(
                            result["chunk"],
                            f"精排排名 {result['rank']}（原 RRF 排名 {result['rrf_rank']}）",
                            result["retrievals"], fusion_score=result["rrf_score"],
                            rerank_score=result["rerank_score"],
                        )
                elif message.get("reference_method", "dense") == "rrf":
                    st.caption(f"本轮回答使用 RRF 融合后的 {len(results)} 个片段作为参考资料。")
                    st.caption(
                        f"排名常数：{message['fusion_config']['rank_constant']}"
                        " · 下方按融合排名展示选中的片段，各路原始排名与分数仍保留。"
                    )
                    for result in results:
                        render_candidate(
                            result["chunk"], f"融合排名 {result['rank']}",
                            result["retrievals"], fusion_score=result["score"],
                        )
                else:
                    # 旧消息继续展示当时实际使用的参考资料规则。
                    st.caption("本轮回答使用 Dense 候选作为参考资料，具体依据见上方原文引用。")
                    st.caption("以下按片段首次出现顺序展示，各路排名和分数分别保留。")
                    for number, candidate in enumerate(candidates, start=1):
                        render_candidate(
                            candidate["chunk"], f"候选 {number}", candidate["retrievals"]
                        )

        elif results:
            # 兼容页面更新前已经保存在当前会话中的 Dense 历史。
            with st.expander("查看本轮检索片段"):
                st.caption("以下为本轮 Dense 候选片段，具体回答依据见上方原文引用。")
                for result in results:
                    render_candidate(
                        result["chunk"], f"片段 {result['rank']}", {"dense": result}
                    )

        trace = message.get("retrieval_trace")
        if trace is not None:
            render_retrieval_trace(trace)


def render_retrieval_trace(trace):
    stage_names = {
        "dense": "Dense 召回", "bm25": "BM25 召回", "merge": "候选去重",
        "rrf": "RRF 融合", "rerank": "Rerank 精排", "select": "最终截取",
    }
    selection_names = {
        "selected": "已入选参考资料",
        "excluded_by_rrf": "未进入 RRF 保留范围",
        "excluded_by_final_top_k": "未进入精排前 K 条",
    }
    with st.expander("查看本轮检索流程"):
        st.markdown("**阶段数量与耗时**")
        st.dataframe([
            {
                "阶段": stage_names[record["stage"]],
                "输入数量": record["input_count"],
                "输出数量": record["output_count"],
                "耗时（秒）": f"{record['elapsed_seconds']:.6f}",
            }
            for record in trace["stages"]
        ], hide_index=True)
        st.caption(
            "Dense 和 BM25 的输入是所选范围的全部片段；去重输入是两路返回的记录总数。"
            "阶段耗时包含该函数的处理，不含模型资源加载、快照复制、页面展示和回答生成。"
        )
        timings = trace.get("other_elapsed_seconds", {})
        if "rerank_resource" in timings and "answer" in timings:
            st.caption(
                f"本轮精排资源获取：{timings['rerank_resource']:.6f} 秒（含缓存查找，未命中时含加载）"
                f" · 回答调用与校验：{timings['answer']:.6f} 秒"
            )
        st.markdown("**片段排名路径**")
        st.dataframe([
            {
                "片段标识": row["chunk_id"],
                "PDF 页码": row["pdf_page"],
                "Dense 排名": str(row["dense_rank"]) if row["dense_rank"] is not None else "—",
                "BM25 排名": str(row["bm25_rank"]) if row["bm25_rank"] is not None else "—",
                "RRF 排名": str(row["rrf_rank"]) if row["rrf_rank"] is not None else "—",
                "精排排名": str(row["rerank_rank"]) if row["rerank_rank"] is not None else "—",
                "最终状态": selection_names[row["selection_status"]],
                "来源": row["source"],
            }
            for row in trace["candidate_paths"]
        ], hide_index=True)
        st.caption("按去重后的原顺序展示；— 表示该阶段的输出中没有这个片段。")
        st.info("入选参考资料表示片段已发送给回答模型；是否支持回答，还需核对回答中的原文依据。")
        st.markdown("**原始阶段记录（含分数与片段原文）**")
        st.json(trace, expanded=False)



def clear_current_chat(scope_key):
    """按钮回调先创建新会话，再重绘页面，避免留下本轮旧聊天或输入。"""
    st.session_state["chat_session"] = create_chat_session(scope_key)
    clear_loaded_workflow()
    st.session_state.pop("submitted_question", None)
    st.session_state.pop("question_submission_notice", None)


def consume_question_input(input_key):
    """表单回调在控件重建前保存提交值并清空输入。"""
    st.session_state["submitted_question"] = st.session_state.get(input_key, "")
    st.session_state[input_key] = ""


def main():
    st.set_page_config(page_title="Paper Sage · 论文研究工作台", page_icon="📚", layout="wide")
    apply_theme()
    render_sidebar_brand()
    render_workspace_header()

    with st.sidebar.popover("新建项目", icon=":material/create_new_folder:", width="stretch"):
        with st.form("create_project_form", clear_on_submit=True, border=False):
            new_project_name = st.text_input("新项目名称", placeholder="例如：注意力机制研究")
            create_submitted = st.form_submit_button("创建项目", type="primary", width="stretch")

    if create_submitted:
        try:
            new_project = create_project(
                # 传入上面输入框中的名称
                new_project_name
            )
        except (ValueError, OSError) as error:
            st.sidebar.error(f"创建项目失败：{error}")
        else:
            # 替换新项目字典中的项目标识
            st.session_state["selected_project_id"] = new_project["project_id"]

            st.sidebar.success(f"已创建：{new_project['name']}")

    try:
        projects = list_projects()

        if not projects:
            st.session_state["chat_session"] = None
            st.info("目前还没有项目。")
            st.stop()

        project_names = {}
        for item in projects:
            project_names[item["project_id"]] = item["name"]

        def format_project_name(project_id):
            return project_names[project_id]

        selected_project_id = st.sidebar.selectbox(
            "选择项目",
            options=list(project_names.keys()),
            format_func=format_project_name,
            key="selected_project_id",
        )

        # 先解除旧项目的会话归属，读取失败或切入空项目时也不保留旧会话。
        chat_session = st.session_state.get("chat_session")
        if (
            chat_session is not None
            and chat_session["scope_key"][0] != selected_project_id
        ):
            st.session_state["chat_session"] = None

        project = load_project(
            # 传入下拉框返回的项目标识
            selected_project_id
        )

        # 管理入口在范围控件之前，删除后才能先修正多选值，再创建控件。
        with st.sidebar.expander("文档管理"):
            notice = st.session_state.pop("document_action_notice", None)
            if notice is not None and notice["project_id"] == selected_project_id:
                if notice["error"]:
                    st.warning(notice["message"])
                else:
                    st.success(notice["message"])
            management_names = {
                doc["document_id"]: Path(doc["source"]).name
                for doc in project["documents"]
            }
            if management_names:
                delete_key = f"delete_document_{selected_project_id}"
                if st.session_state.get(delete_key) not in management_names:
                    st.session_state[delete_key] = next(iter(management_names))
                delete_document_id = st.selectbox(
                    "选择要删除的论文",
                    options=list(management_names),
                    format_func=lambda doc_id: management_names[doc_id],
                    key=delete_key,
                )
                st.caption("删除会移除项目记录、上传副本与全部索引版本，并清空当前会话。公共原始 PDF 会保留。")
                if st.button("删除所选文档", key=f"delete_submit_{selected_project_id}"):
                    try:
                        updated_project, cleanup_error = delete_document(
                            selected_project_id, delete_document_id
                        )
                    except (OSError, ValueError, RuntimeError) as error:
                        st.error(f"删除失败，文档记录保留：{error}")
                    else:
                        project = updated_project
                        # 只清除范围索引缓存，模型对象继续复用。
                        load_retrieval_resources.clear()
                        st.session_state["chat_session"] = None
                        scope_key = f"scope_documents_{selected_project_id}"
                        if scope_key in st.session_state:
                            st.session_state[scope_key] = [
                                doc_id for doc_id in st.session_state[scope_key]
                                if doc_id != delete_document_id
                            ]
                        st.session_state["document_action_notice"] = {
                            "project_id": selected_project_id,
                            "error": bool(cleanup_error),
                            "message": (
                                f"文档已退出项目，清理流程尚未完成：{cleanup_error}"
                                if cleanup_error else "文档已删除，当前会话已清空。"
                            ),
                        }
                        st.rerun()

            # 空项目也必须能重试，不能把入口放到空项目的 st.stop() 后面。
            for pending_document in list(project.get("pending_cleanup", [])):
                pending_id = pending_document["document_id"]
                st.warning(f"待清理文件：{Path(pending_document['source']).name}")
                if st.button(
                    "重试清理", key=f"cleanup_{selected_project_id}_{pending_id}"
                ):
                    try:
                        project, cleanup_error = retry_document_cleanup(
                            selected_project_id, pending_id
                        )
                    except (OSError, ValueError, RuntimeError) as error:
                        st.error(f"清理失败：{error}")
                    else:
                        if cleanup_error:
                            st.error(f"清理未完成，可稍后重试：{cleanup_error}")
                        else:
                            load_retrieval_resources.clear()
                            st.session_state["document_action_notice"] = {
                                "project_id": selected_project_id,
                                "error": False,
                                "message": "残留文件已清理。",
                            }
                            st.rerun()

        upload_panel = st.sidebar.expander("导入论文", icon=":material/upload_file:", expanded=not project["documents"])
        with upload_panel:
            uploaded_file = upload_panel.file_uploader(
                "上传 PDF 论文",
                type=["pdf"],
                accept_multiple_files=False,
                key=f"pdf_upload_{project['project_id']}",
            )

            if uploaded_file is not None:
                # 调用 uploaded_file 的 getvalue() 方法
                file_bytes = uploaded_file.getvalue()
                file_hash = get_file_hash(file_bytes)
                try:
                    existing_document = find_existing_document(project, file_hash)
                except (OSError, ValueError) as error:
                    st.error(f"导入失败（检查重复文件）：{error}")
                    st.stop()

                if existing_document is not None:
                    st.info("这份论文已存在于当前项目中。")
                elif any(
                    doc["document_id"] == f"doc_{file_hash}"
                    for doc in project.get("pending_cleanup", [])
                ):
                    st.warning("这份论文仍有待清理文件，请在文档管理中完成清理后重新导入。")
                else:
                    if st.button(
                        "导入论文",
                        key=f"parse_pdf_{project['project_id']}",
                    ):
                        import_stage = "登记导入待办"
                        import_started = False
                        try:
                            document_id = f"doc_{file_hash}"
                            index_version = "v1"
                            planned_path = (
                                Path("data") / "papers" / project["project_id"]
                                / document_id / Path(uploaded_file.name).name
                            )
                            project = begin_document_import(project["project_id"], {
                                "document_id": document_id,
                                "source": planned_path.as_posix(),
                                "index_version": index_version,
                            })
                            import_started = True
                            import_stage = "保存 PDF"
                            pdf_path = save_uploaded_pdf(
                                project["project_id"],
                                document_id,
                                uploaded_file.name,
                                file_bytes,
                            )

                            import_stage = "解析 PDF"
                            pages = load_pdf_pages(pdf_path.as_posix(), document_id)

                            import_stage = "构建索引"
                            new_index_dir = get_index_dir(
                                project["project_id"], document_id, index_version
                            )
                            with st.spinner("正在分块并建立索引……"):
                                index_info = build_document_index(
                                    pages, project["project_id"], new_index_dir
                                )

                            import_stage = "登记文档"
                            new_document = {
                                "document_id": document_id,
                                "source": pdf_path.as_posix(),
                                "index_version": index_version,
                                "file_hash": file_hash,
                                "page_count": len(pages),
                                "chunk_count": index_info["chunk_count"],
                            }
                            # 成功登记文档时，一起移除这次导入的清理待办。
                            project = finish_document_import(project["project_id"], new_document)
                        except (OSError, ValueError, RuntimeError, PyPdfError) as error:
                            st.error(f"导入失败（{import_stage}）：{error}")
                            if import_started:
                                try:
                                    project, cleanup_error = retry_document_cleanup(
                                        project["project_id"], document_id
                                    )
                                except (OSError, ValueError, RuntimeError) as cleanup_failure:
                                    cleanup_error = str(cleanup_failure)
                                if cleanup_error:
                                    st.warning(f"残留文件待清理：{cleanup_error}。重新运行页面后可在文档管理中重试。")
                        else:
                            load_retrieval_resources.clear()
                            st.success(
                                f"已导入 {pdf_path.name}："
                                f"{len(pages)} 页，{index_info['chunk_count']} 个片段。"
                            )

        st.sidebar.caption(
            f"本项目文档数量：{len(project['documents'])}"
        )

        if not project["documents"]:
            st.session_state["chat_session"] = ensure_chat_session(
                st.session_state.get("chat_session"),
                (project["project_id"], ()),
            )
            st.info("当前项目还没有论文。")
            st.stop()

        with st.sidebar.expander("项目论文", icon=":material/library_books:"):
            for document_record in project["documents"]:
                st.caption(Path(document_record["source"]).name)

        document_names = {}
        for document_record in project["documents"]:
            document_names[document_record["document_id"]] = Path(
                document_record["source"]
            ).name

        def format_document_name(document_id):
            return document_names[document_id]

        st.sidebar.markdown("**阅读范围**")
        scope_mode = st.sidebar.radio(
            "检索范围",
            options=["全部文档", "指定文档"],
            key=f"scope_mode_{project['project_id']}",
        )

        if scope_mode == "全部文档":
            selected_document_ids = list(document_names.keys())
        else:
            selected_document_ids = st.sidebar.multiselect(
                "选择参与问答的论文",
                options=list(document_names.keys()),
                format_func=format_document_name,
                key=f"scope_documents_{project['project_id']}",
            )

        selected_documents = get_selected_documents(
            project,
            selected_document_ids,  # 传入上面得到的文档标识列表
        )

        st.sidebar.caption(f"已选文档数量：{len(selected_documents)}")

        # 对实际参与检索的文档排序，选择顺序不同仍视为同一范围。
        scope_documents = tuple(sorted(
            (document["document_id"], document["index_version"], document["source"])
            for document in selected_documents
        ))
        current_scope = (project["project_id"], scope_documents)

        # 范围变化时新建会话；空选择也先更新会话，再停止执行。
        chat_session = ensure_chat_session(
            st.session_state.get("chat_session"), current_scope
        )
        st.session_state["chat_session"] = chat_session

        if not selected_documents:
            st.info("请至少选择一篇论文参与问答。")
            st.stop()

        try:
            scope_inputs = capture_scope_inputs(project["project_id"], selected_documents)
            # 缓存参数包含项目、所选文档和当前文件摘要。
            embedding_model, index, metadata = load_retrieval_resources(
                project,
                tuple(sorted(selected_document_ids)),
                scope_inputs,
            )
        except (OSError, ValueError, RuntimeError) as error:
            if not chat_session["messages"]:
                raise
            # 来源损坏时保留当前范围的已有回答；不给旧资料开启新提交或续跑。
            st.warning(f"当前论文资源不可用，暂时只能查看已有记录：{error}")
            for message_number, message in enumerate(chat_session["messages"]):
                render_message(
                    message, message_key=f"pdf_{chat_session['session_id']}_{message_number}",
                    project_id=project["project_id"], selected_documents=selected_documents,
                )
            with st.container(key="paper_composer"):
                st.info("恢复原文件后刷新页面，或切换到可用的论文范围后继续。")
                with st.container(horizontal=True, key="paper_actions"):
                    st.button("提交问题", disabled=True, type="primary", icon=":material/arrow_upward:")
                    st.button("清空会话", on_click=clear_current_chat, args=(current_scope,),
                              icon=":material/restart_alt:", key="clear_chat")
            st.stop()
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"加载检索资源失败：{error}")
        st.stop()

    render_scope_overview(project, selected_documents, index.ntotal)
    submission_notice = st.session_state.pop("question_submission_notice", None)
    if submission_notice and submission_notice["scope"] == current_scope:
        st.success(submission_notice["text"])

    chat_session = render_session_controls(
        chat_session, project["project_id"], selected_documents, current_scope,
        container=st.sidebar.expander("会话存档", icon=":material/forum:"),
    )
    st.session_state["chat_session"] = chat_session


    def execute_workflow(session, *, question=None, run_id=None, pause_before_answer=False, retry_failed=False):
        # 恢复和新提交共用当前范围资源；查看已完成运行不会调用本函数。
        if capture_scope_inputs(project["project_id"], selected_documents) != scope_inputs:
            raise ValueError("当前文件已变化，请刷新页面后再处理")
        documents = sorted(selected_documents, key=lambda doc: doc["document_id"])
        search_tool = create_search_tool(
            embedding_model, index, metadata,
            rerank_model_loader=lambda: observe_operation("resource_reranker",
                lambda: load_rerank_resources(get_rerank_cache_folder())),
            recall_top_k=RECALL_TOP_K, rrf_top_k=RRF_TOP_K,
            final_top_k=FINAL_TOP_K, rank_constant=RRF_RANK_CONSTANT, require_results=True,
        )
        with st.status("正在处理论文请求，请稍候……", expanded=True) as running_status:
            st.write("已开始运行：正在判断任务并处理论文资料。")
            progress = st.empty()
            try:
                model = observe_model(create_chat_model(st.secrets["DEEPSEEK_API_KEY"]))
                summary_runner = create_summary_runner(project["project_id"], documents, model, on_progress=progress.info)
                comparison_runner = create_comparison_runner(
                    project, documents, model,
                    embedding_model_loader=lambda name: observe_operation("resource_embedding", lambda: load_embedding_model(name)),
                    rerank_model_loader=lambda: observe_operation("resource_reranker",
                        lambda: load_rerank_resources(get_rerank_cache_folder())), on_progress=progress.info,
                )
                if run_id is None:
                    outcome = start_workflow_run(question, session, project["project_id"], documents, model, search_tool,
                        summary_runner=summary_runner, comparison_runner=comparison_runner,
                        pause_before_answer=pause_before_answer)
                else:
                    outcome = resume_workflow_run(project["project_id"], run_id, session, documents, model, search_tool,
                        summary_runner=summary_runner, comparison_runner=comparison_runner, retry_failed=retry_failed)
                running_status.update(label="处理完成" if outcome["status"] == "completed" else "检索已保存，等待继续生成回答",
                                      state="complete", expanded=False)
                return outcome
            except WorkflowRunError as error:
                running_status.update(label="处理失败，请查看错误提示", state="error", expanded=True)
                original_question = question or st.session_state.get("workflow_loaded", {}).get(
                    "question", "原问题保存在运行记录中",
                )
                remember_workflow_result({"run_id": error.run_id, "session": session, "status": "failed",
                                          "log_write_errors": error.log_write_errors},
                                         current_scope, original_question, select=True)
                st.caption("运行进度已保存，可在侧栏刷新运行记录后查看。")
                raise error.cause
            except Exception:
                running_status.update(label="处理失败，请查看错误提示", state="error", expanded=True)
                raise
            finally:
                progress.empty()


    chat_session = render_workflow_controls(
        chat_session, project["project_id"], selected_documents, current_scope,
        container=st.sidebar.expander("运行记录", icon=":material/history:"),
        on_resume=lambda run_id, session, retry_failed: execute_workflow(
            session, run_id=run_id, retry_failed=retry_failed),
    )
    st.session_state["chat_session"] = chat_session

    # 展示聊天、准备模型历史前，检查当前会话的范围归属。
    try:
        build_history_messages(chat_session, current_scope)
    except ValueError as error:
        st.error(f"会话检查失败：{error}")
        st.stop()

    if not chat_session["messages"]:
        render_empty_conversation()

    # 每次页面重新运行时，重新显示已有记录
    for message_number, message in enumerate(chat_session["messages"]):
        render_message(
            message, message_key=f"pdf_{chat_session['session_id']}_{message_number}",
            project_id=project["project_id"], selected_documents=selected_documents,
        )


    with st.container(key="paper_composer"):
        st.html('<div class="sage-composer-heading">向论文提问<span>回答仅依据当前所选论文</span></div>')
        input_key = f"question_{chat_session['session_id']}"
        with st.expander("回答选项", icon=":material/tune:"):
            st.caption("论文问答资料不足时，会在当前所选范围内最多补充检索一次。")
            pause_before_answer = st.checkbox(
                "先检索，稍后生成回答（问答）", value=False, key="pause_before_answer",
                help="仅对论文问答生效。保存检索进度后暂停，可重启应用，再从侧栏继续；总结和比较按原流程完成。",
            )
        with st.form(f"question_form_{chat_session['session_id']}", clear_on_submit=False,
                     enter_to_submit=True, border=False):
            question = st.text_input(
                "你想了解论文的什么内容？", label_visibility="collapsed",
                placeholder="输入问题，按 Enter 提交……",
                key=input_key,
            )
            with st.container(horizontal=True, key="paper_actions"):
                submitted = st.form_submit_button("提交问题", type="primary", icon=":material/arrow_upward:",
                    key="submit_question", on_click=consume_question_input, args=(input_key,))
                st.form_submit_button("清空会话", icon=":material/restart_alt:", key="clear_chat",
                          on_click=clear_current_chat, args=(current_scope,),
                          help="开启新的聊天；保留当前论文范围和已保存的存档。")

    st.html('<div class="sage-footnote">结论值得追问，引用值得核对。展开回答中的原文依据，可查看对应 PDF 页面。</div>')

    if submitted:
        question = st.session_state.pop("submitted_question", question)
        if not question.strip():
            st.warning("请先输入一个问题")
        else:
            try:
                outcome = execute_workflow(chat_session, question=question, pause_before_answer=pause_before_answer)
                # 后端返回完整会话并按运行编号去重；页面不再重复追加消息。
                chat_session = outcome["session"]
                st.session_state["chat_session"] = chat_session
                remember_workflow_result(outcome, current_scope, question.strip(), select=True)
                st.session_state["workflow_notice"] = {"scope_key": current_scope,
                    "text": "运行已完成。" if outcome["status"] == "completed" else "检索进度已保存，可稍后继续生成回答。"}
                st.session_state["question_submission_notice"] = {"scope": current_scope,
                    "text": "处理完成，回答已显示在下方。" if outcome["status"] == "completed" else "检索进度已保存。可在侧栏运行记录中继续生成回答。"}
                st.rerun()
            except QARetryError as error:
                render_qa_retry_error(error)
            except EmptyRetrievalError as error:
                st.warning(str(error))
            except IncompleteAnswerError as error:
                st.error(str(error))
            except ComparisonServiceError as error:
                st.error(f"{error}。本轮未加入会话。")
            except (LengthFinishReasonError, ContentFilterFinishReasonError):
                st.error("模型未返回完整内容，本轮未保存；请缩小问题范围或调整问题后重试。")
            except AuthenticationError:
                st.error("身份验证失败，请检查 DeepSeek API Key。")
            except RateLimitError:
                st.error("请求受到限流，请稍后重试。")
            except APITimeoutError:
                st.error("请求超时，请稍后重试。")
            except APIConnectionError:
                st.error("无法连接 DeepSeek，请检查网络连接。")
            except APIError as error:
                st.error(f"模型调用失败：{error}")
            except GraphRecursionError:
                st.error("工作流未在规定步数内结束，请检查节点连线。本轮未保存。")
            except JSONDecodeError:
                st.error("模型返回的内容不是有效的 JSON，本轮未保存。")
            except ValueError as error:
                st.error(f"流程数据不符合要求：{error}")
            except (OSError, RuntimeError, PyPdfError, sqlite3.Error) as error:
                st.error(f"工作流执行失败：{error}")
