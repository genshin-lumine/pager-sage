import streamlit as st
from llm import ask_deepseek, IncompleteAnswerError
from json import JSONDecodeError
from sentence_transformers import SentenceTransformer
from dual_retriever import retrieve_dual, merge_candidates
from rank_fusion import fuse_rrf
from retrieval_trace import run_stage, build_candidate_paths
from reranker import (
    get_rerank_cache_folder, load_reranker, rerank_candidates,
    RERANK_MODEL_NAME, RERANK_MODEL_REVISION,
)
from rag_context import build_reference_text
from pathlib import Path
from time import perf_counter
from pypdf.errors import PyPdfError
from pdf_loader import load_pdf_pages
from retrieval_scope import get_selected_documents, load_scope_index
from chat_session import (
    create_chat_session,
    ensure_chat_session,
    build_history_messages,
)
from document_deleter import (
    delete_document,
    retry_document_cleanup,
    begin_document_import,
    finish_document_import,
)
from document_importer import (
    get_file_hash,
    find_existing_document,
    save_uploaded_pdf,
    build_document_index,
)
from project_store import (
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
)

st.set_page_config(page_title="Paper Sage", page_icon="📚")

# Day 28：开发对照与固定新题复验后，采用质量优先的 B 配置。
RECALL_TOP_K = 6
RRF_TOP_K = 6
FINAL_TOP_K = 3
RRF_RANK_CONSTANT = 60

with st.sidebar.form("create_project_form", clear_on_submit=True):
    new_project_name = st.text_input("新项目名称")
    create_submitted = st.form_submit_button("创建项目")

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

@st.cache_resource(show_spinner="正在加载向量模型……")
def load_embedding_model(model_name):
    return SentenceTransformer(model_name, device="cpu")


@st.cache_resource(show_spinner="正在加载所选论文的索引……")
def load_retrieval_resources(project, selected_document_ids):
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


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
        timings = trace["other_elapsed_seconds"]
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

st.title("Paper Sage")

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

    uploaded_file = st.sidebar.file_uploader(
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
            st.sidebar.error(f"导入失败（检查重复文件）：{error}")
            st.stop()

        if existing_document is not None:
            st.sidebar.info("这份论文已存在于当前项目中。")
        elif any(
            doc["document_id"] == f"doc_{file_hash}"
            for doc in project.get("pending_cleanup", [])
        ):
            st.sidebar.warning("这份论文仍有待清理文件，请在文档管理中完成清理后重新导入。")
        else:
            if st.sidebar.button(
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
                    st.sidebar.error(f"导入失败（{import_stage}）：{error}")
                    if import_started:
                        try:
                            project, cleanup_error = retry_document_cleanup(
                                project["project_id"], document_id
                            )
                        except (OSError, ValueError, RuntimeError) as cleanup_failure:
                            cleanup_error = str(cleanup_failure)
                        if cleanup_error:
                            st.sidebar.warning(f"残留文件待清理：{cleanup_error}。重新运行页面后可在文档管理中重试。")
                else:
                    load_retrieval_resources.clear()
                    st.sidebar.success(
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

    for document_record in project["documents"]:
        # 取得当前文档的 source，用 Path(...).name 得到文件名
        # 用 st.sidebar.write() 显示这个文件名
        file_name = Path(document_record["source"]).name
        st.sidebar.write(file_name)

    document_names = {}
    for document_record in project["documents"]:
        document_names[document_record["document_id"]] = Path(
            document_record["source"]
        ).name

    def format_document_name(document_id):
        return document_names[document_id]

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

    # 缓存参数包含项目记录（含索引版本）和排序后的所选文档标识。
    embedding_model, index, metadata = load_retrieval_resources(
        project,
        tuple(sorted(selected_document_ids)),
    )
except (OSError, ValueError, RuntimeError) as error:
    st.error(f"加载检索资源失败：{error}")
    st.stop()

st.caption(
    f"当前项目：{project['name']}"
    f" · 检索文档：{len(selected_documents)} / {len(project['documents'])}"
    f" · 可检索片段：{index.ntotal}"
)
st.caption(
    "参与问答的论文："
    + "、".join(Path(document["source"]).name for document in selected_documents)
)

# 清空会话会生成新编号，检索范围与已加载资源继续保留。
if st.button("清空会话"):
    chat_session = create_chat_session(current_scope)
    st.session_state["chat_session"] = chat_session

# 展示聊天、准备模型历史前，检查当前会话的范围归属。
try:
    history_messages = build_history_messages(chat_session, current_scope)
except ValueError as error:
    st.error(f"会话检查失败：{error}")
    st.stop()

# 每次页面重新运行时，重新显示已有记录
for message in chat_session["messages"]:
    render_message(message)


question = st.text_input(
    "你想了解论文的什么内容？",
    placeholder="例如：这篇论文的主要贡献是什么？",
    key=f"question_{chat_session['session_id']}",
)


if st.button("提交问题"):
    if not question.strip():
        st.warning("请先输入一个问题")
    else:
        stage_records = []
        try:
            with st.spinner("正在检索论文片段……"):
                dual_results = retrieve_dual(
                    question=question,
                    model=embedding_model,
                    index=index,
                    metadata=metadata,
                    top_k=RECALL_TOP_K,
                    stage_records=stage_records,
                )
                retrieval_candidates = run_stage(
                    stage_records, "merge", sum(len(route) for route in dual_results.values()),
                    lambda: merge_candidates(dual_results),
                )
                rrf_results = run_stage(
                    stage_records, "rrf", len(retrieval_candidates),
                    lambda: fuse_rrf(
                        retrieval_candidates, top_k=RRF_TOP_K, rank_constant=RRF_RANK_CONSTANT,
                    ),
                )

        except (ValueError, RuntimeError) as error:
            st.error(f"论文检索失败：{error}")
            st.stop()

        if not rrf_results:
            st.warning("当前索引没有可用的论文片段。")
            st.stop()

        try:
            resource_started = perf_counter()
            rerank_model = load_rerank_resources(get_rerank_cache_folder())
            rerank_resource_seconds = perf_counter() - resource_started
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"精排资源加载失败，请检查本机缓存配置并确认模型已下载：{error}")
            st.stop()

        try:
            with st.spinner("正在精排论文片段……"):
                # 只打分一次，保存所有精排结果后，再截取真正进入参考资料的候选。
                rerank_results = run_stage(
                    stage_records, "rerank", len(rrf_results),
                    lambda: rerank_candidates(question, rrf_results, rerank_model, top_k=len(rrf_results)),
                )
                rerank_elapsed_seconds = stage_records[-1]["elapsed_seconds"]
                retrieval_results = run_stage(
                    stage_records, "select", len(rerank_results),
                    lambda: rerank_results[:FINAL_TOP_K],
                )
            reference_text = build_reference_text(retrieval_results)
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"论文精排失败：{error}")
            st.stop()

        user_message = {
            "role": "user",
            "content": (
              f"<reference>\n"
              f"{reference_text.strip()}\n"
              f"</reference>\n\n"
              f"问题：{question.strip()}"  
            ),
        }
        display_user_message = {
            "role": "user",
            "content": question.strip(),
        }

        # 页面负责组装本次请求的完整消息
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位论文阅读助手，请用中文回答。"
                    "论文事实只能依据本次用户消息中 <reference> 内的资料。"
                    "聊天历史仅用于理解追问，不作为事实依据。"
                    "只输出一个 JSON 对象，不要输出代码块或额外文字。"
                    "必须包含 answer、evidence、insufficient_reason 三个字段。"
                    "answer 是字符串，存放回答正文。"
                    "evidence 是字符串数组，每条依据必须逐字摘录参考资料原文。"
                    "insufficient_reason 是字符串，存放资料不足的原因。"
                    "资料足够时，evidence 提供支持回答的原文，"
                    "insufficient_reason 为空字符串。"
                    "资料为空或不足时，不要猜测；answer 明确说明无法依据资料回答，"
                    "evidence 使用空数组，insufficient_reason 说明原因。"
                    'JSON 格式示例：{"answer": "资料不足，无法回答。", '
                    '"evidence": [], '
                    '"insufficient_reason": "没有提供参考资料。"}'
                ),
            },
        ] + history_messages + [user_message]

        try:

            with st.spinner("正在等待 DeepSeek 回答……"):
                answer_started = perf_counter()
                result = ask_deepseek(
                    messages,
                    st.secrets["DEEPSEEK_API_KEY"],
                )
                answer_elapsed_seconds = perf_counter() - answer_started

            answer = f"**结论**\n\n{result['answer']}"
            answer += "\n\n**原文依据**"

            if result["evidence"]:
                for evidence in result["evidence"]:
                    answer += f"\n\n> {evidence}"
            else:
                answer += "\n\n无。"

            if result["insufficient_reason"]:
                answer += (
                    "\n\n**资料不足说明**\n\n"
                    f"{result['insufficient_reason']}"
                )

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "retrieval_results": retrieval_results,
                "retrieval_channels": dual_results,
                "retrieval_candidates": retrieval_candidates,
                "reference_method": "rerank",
                "rrf_results": rrf_results,
                "rerank_elapsed_seconds": rerank_elapsed_seconds,
                "retrieval_trace": {
                    "schema_version": 1,
                    "question": question.strip(),
                    "scope_key": chat_session["scope_key"],
                    "stages": stage_records,
                    "candidate_paths": build_candidate_paths(stage_records),
                    "other_elapsed_seconds": {
                        "rerank_resource": rerank_resource_seconds,
                        "answer": answer_elapsed_seconds,
                    },
                },
                "rerank_config": {
                    "model_name": RERANK_MODEL_NAME,
                    "model_revision": RERANK_MODEL_REVISION,
                    "device": "cpu",
                    "max_length": 512,
                    "batch_size": 3,
                    "score_type": "logit",
                },
                "fusion_config": {
                    "rank_constant": RRF_RANK_CONSTANT,
                    "recall_top_k": RECALL_TOP_K,
                    "rrf_top_k": RRF_TOP_K,
                    "final_top_k": FINAL_TOP_K,
                },
            }

            # 页面负责保存成功完成的一轮对话
            chat_session["messages"].extend(
                [display_user_message, assistant_message]
            )

            # 页面负责展示新消息
            for message in [display_user_message, assistant_message]:
                render_message(message)

        except IncompleteAnswerError as error:
            st.error(str(error))
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

        except JSONDecodeError:
            st.error("模型返回的内容不是有效的 JSON，请重试。")

        except ValueError as error:
            st.error(f"回答结构不符合要求：{error}")
