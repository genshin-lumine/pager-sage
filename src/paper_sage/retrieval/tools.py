"""将已选范围内的混合检索包装成 LangChain 工具。"""

from copy import deepcopy
from time import perf_counter

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from paper_sage.retrieval.dual import retrieve_dual, merge_candidates
from paper_sage.retrieval.context import build_reference_text
from paper_sage.retrieval.fusion import fuse_rrf
from paper_sage.retrieval.reranker import rerank_candidates, RERANK_MODEL_NAME, RERANK_MODEL_REVISION
from paper_sage.retrieval.trace import run_stage, build_candidate_paths


EMPTY_REFERENCE_TEXT = "当前所选论文中没有检索到可用片段。"


class EmptyRetrievalError(ValueError):
    pass


def create_search_tool(
    embedding_model, index, metadata, rerank_model=None,
    *, recall_top_k=6, rrf_top_k=6, final_top_k=3, rank_constant=60,
    rerank_model_loader=None, require_results=False,
):
    """接收 load_scope_index() 的配套索引与元数据，创建使用该范围的工具。

    调用方负责按用户选择加载范围及配套模型；切换范围时重新创建工具。
    这些资源及检索参数由程序配置，不进入模型看到的输入参数定义。
    传入完整 ToolCall 时返回含 artifact 的 ToolMessage；只传参数时沿用文本返回。
    页面可传入 rerank_model_loader，实际找到候选后再取得精排模型。
    require_results=True 时，空候选抛出异常供页面提示，不继续生成回答。
    """
    for name, value in (
        ("recall_top_k", recall_top_k), ("rrf_top_k", rrf_top_k),
        ("final_top_k", final_top_k), ("rank_constant", rank_constant),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} 必须是正整数")
    if rerank_model is not None and rerank_model_loader is not None:
        raise ValueError("精排模型与延迟加载函数只能提供一个")
    if rerank_model is None and not callable(rerank_model_loader):
        raise ValueError("需要精排模型或可调用的加载函数")

    @tool(response_format="content_and_artifact")
    def search_papers(question: str) -> tuple[str, dict]:
        """在当前项目已选中的论文中检索与问题相关的原文片段。

        用于查找论文事实依据。返回片段标识、来源、PDF 页码和原文；
        候选可能不足以回答问题，需根据原文判断，不能仅凭被检索到就作答。
        """
        if not question.strip():
            raise ValueError("问题不能为空")
        question = question.strip()
        # 每次调用使用独立的记录列表；失败会抛出异常，不返回半成品。
        stage_records = []

        dual_results = retrieve_dual(
            # 将本工具接收到的问题传给双路检索。
            question=question,
            model=embedding_model,
            index=index,
            metadata=metadata,
            top_k=recall_top_k,
            stage_records=stage_records,
        )
        candidates = run_stage(
            stage_records, "merge", sum(len(route) for route in dual_results.values()),
            lambda: merge_candidates(dual_results),
        )
        fused = run_stage(
            stage_records, "rrf", len(candidates),
            lambda: fuse_rrf(candidates, top_k=rrf_top_k, rank_constant=rank_constant),
        )
        if require_results and not fused:
            raise EmptyRetrievalError("当前索引没有可用的论文片段。")
        # 进入检索分支且确有候选时才取得精排资源；资源获取不计入打分耗时。
        active_reranker = rerank_model
        resource_seconds = 0.0
        if fused and rerank_model_loader is not None:
            started = perf_counter()
            try:
                active_reranker = rerank_model_loader()
            except (OSError, ValueError, RuntimeError) as error:
                raise RuntimeError(f"精排资源加载失败，请检查本机模型缓存：{error}") from error
            resource_seconds = perf_counter() - started
        # 精排只执行一次，保留全部分数，再截取最终参考片段。
        try:
            ranked = run_stage(
                stage_records, "rerank", len(fused),
                lambda: rerank_candidates(question, fused, active_reranker, top_k=max(1, len(fused))),
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(f"论文精排失败：{error}") from error
        rerank_elapsed_seconds = stage_records[-1]["elapsed_seconds"]
        results = run_stage(stage_records, "select", len(ranked), lambda: ranked[:final_top_k])
        reference_text = build_reference_text(results)

        if not reference_text:
            reference_text = EMPTY_REFERENCE_TEXT
        artifact = {
            "question": question,
            "retrieval_results": results,
            "retrieval_channels": dual_results,
            "retrieval_candidates": candidates,
            "rrf_results": fused,
            "rerank_results": ranked,
            "reference_method": "rerank",
            "rerank_elapsed_seconds": rerank_elapsed_seconds,
            "retrieval_trace": {
                "schema_version": 1,
                "question": question,
                "stages": stage_records,
                "candidate_paths": build_candidate_paths(stage_records),
                "other_elapsed_seconds": {"rerank_resource": resource_seconds},
            },
            "fusion_config": {
                "recall_top_k": recall_top_k,
                "rrf_top_k": rrf_top_k,
                "final_top_k": final_top_k,
                "rank_constant": rank_constant,
            },
            "rerank_config": {
                "model_name": RERANK_MODEL_NAME,
                "model_revision": RERANK_MODEL_REVISION,
                "device": "cpu",
                "max_length": 512,
                "batch_size": 3,
                "score_type": "logit",
            },
        }
        # 返回独立记录，避免调用方修改结果时影响缓存的索引元数据。
        return reference_text, deepcopy(artifact)

    return search_papers


def build_tool_message(reference_text, tool_call):
    """把实际检索结果包装成消息，对应当前模型发出的那次调用。"""
    return ToolMessage(
        # 放入本次检索得到的原文字符串。
        content=reference_text,
        # 从当前 tool_call 读取调用标识，关联请求与结果。
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )
