import streamlit as st
from llm import ask_deepseek, IncompleteAnswerError
from json import JSONDecodeError
from sentence_transformers import SentenceTransformer
from retriever import retrieve_dense
from rag_context import build_reference_text
from pathlib import Path
from pypdf.errors import PyPdfError
from pdf_loader import load_pdf_pages
from retrieval_scope import get_selected_documents, load_scope_index
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
    save_project,
)
from openai import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

st.set_page_config(page_title="Paper Sage", page_icon="📚")

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

def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        results = message.get("retrieval_results", [])

        if results:
            with st.expander("查看本轮检索片段"):
                st.caption(
                    "以下是本轮检索到的候选片段，"
                    "具体回答依据请看上方的原文引用。"
                )

                for result in results:
                    chunk = result["chunk"]

                    st.markdown(
                        f"**片段 {result['rank']} "
                        f"· PDF 第 {chunk['pdf_page']} 页**"
                    )
                    st.caption(f"来源：{chunk['source']}")
                    st.caption(f"片段标识：{chunk['chunk_id']}")
                    st.caption(f"相似度：{result['score']:.4f}")

                    st.code(
                        chunk["text"],
                        language=None,
                        wrap_lines=True,
                    )

st.title("Paper Sage")

try:
    projects = list_projects()

    if not projects:
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

    project = load_project(
        # 传入下拉框返回的项目标识
        selected_project_id
    )

    # 切换项目时，清空当前聊天。
    if st.session_state.get("active_project_id") != project["project_id"]:
        st.session_state.messages = []
        st.session_state.active_project_id = project["project_id"]

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
        else:
            if st.sidebar.button(
                "导入论文",
                key=f"parse_pdf_{project['project_id']}",
            ):
                import_stage = "保存 PDF"
                try:
                    document_id = f"doc_{file_hash}"
                    pdf_path = save_uploaded_pdf(
                        project["project_id"],
                        document_id,
                        uploaded_file.name,
                        file_bytes,
                    )

                    import_stage = "解析 PDF"
                    pages = load_pdf_pages(pdf_path.as_posix(), document_id)

                    import_stage = "构建索引"
                    index_version = "v1"
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
                    # 保存成功后再更新页面使用的项目，失败时保留原文档列表。
                    updated_project = project.copy()
                    updated_project["documents"] = project["documents"] + [new_document]
                    save_project(updated_project)
                    project = updated_project
                except (OSError, ValueError, RuntimeError, PyPdfError) as error:
                    st.sidebar.error(f"导入失败（{import_stage}）：{error}")
                else:
                    st.sidebar.success(
                        f"已导入 {pdf_path.name}："
                        f"{len(pages)} 页，{index_info['chunk_count']} 个片段。"
                    )

    st.sidebar.caption(
        f"本项目文档数量：{len(project['documents'])}"
    )

    if not project["documents"]:
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

    # 在空选择停止执行之前清空历史，避免恢复选择时带回旧上下文。
    if st.session_state.get("active_scope") != current_scope:
        st.session_state.messages = []
        st.session_state.active_scope = current_scope

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

# 第一次打开会话时，创建聊天记录列表
if "messages" not in st.session_state:
    st.session_state.messages = []

# 清除回话记录
if st.button("清空会话"):
    st.session_state.messages = []

# 每次页面重新运行时，重新显示已有记录
for message in st.session_state.messages:
    render_message(message)


question = st.text_input(
    "你想了解论文的什么内容？",
    placeholder="例如：这篇论文的主要贡献是什么？",
)


if st.button("提交问题"):
    if not question.strip():
        st.warning("请先输入一个问题")
    else:
        try:
            with st.spinner("正在检索论文片段……"):
                retrieval_results = retrieve_dense(
                    question=question,
                    model=embedding_model,
                    index=index,
                    metadata=metadata,
                    top_k=3,
                )

                reference_text = build_reference_text(retrieval_results)

        except (ValueError, RuntimeError) as error:
            st.error(f"论文检索失败：{error}")
            st.stop()

        if not retrieval_results:
            st.warning("当前索引没有可用的论文片段。")
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

        history_messages = []

        for message in st.session_state.messages:
            history_messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )
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
                result = ask_deepseek(
                    messages,
                    st.secrets["DEEPSEEK_API_KEY"],
                )

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
            }

            # 页面负责保存成功完成的一轮对话
            st.session_state.messages.extend(
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
