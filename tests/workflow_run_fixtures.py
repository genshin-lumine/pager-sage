"""持久化后端与教学脚本共用的固定响应；不加载模型或访问 API。"""

from copy import deepcopy
import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from paper_sage.qa.model import CITED_ANSWER_MAX_TOKENS
from paper_sage.retrieval.context import build_reference_text


def make_fixed_resources(question, chunk, *, route="qa", answer_error=None):
    calls = {"classify": 0, "request_tool": 0, "search": 0, "answer": 0}

    def response(value):
        return AIMessage(content=json.dumps(value, ensure_ascii=False), response_metadata={"finish_reason": "stop"})

    @tool(response_format="content_and_artifact")
    def search_papers(question: str) -> tuple[str, dict]:
        """返回一个指定片段及固定流程记录，仅验证保存、恢复和来源传递。"""
        calls["search"] += 1
        result = {"rank": 1, "score": 1.0, "chunk": deepcopy(chunk)}
        results = [result]
        stages = [{"stage": stage, "input_count": 1, "output_count": 1,
                   "elapsed_seconds": 0.0, "results": deepcopy(results)}
                  for stage in ("dense", "bm25", "merge", "rrf", "rerank", "select")]
        artifact = {
            "question": question, "retrieval_results": deepcopy(results),
            "retrieval_channels": {"dense": deepcopy(results), "bm25": deepcopy(results)},
            "retrieval_candidates": deepcopy(results), "rrf_results": deepcopy(results),
            "rerank_results": deepcopy(results), "reference_method": "rerank", "rerank_elapsed_seconds": 0.0,
            "fusion_config": {"rank_constant": 60}, "rerank_config": {"mode": "simulation"},
            "retrieval_trace": {"question": question, "mode": "simulation", "stages": stages,
                                "candidate_paths": [{key: chunk[key] for key in ("chunk_id", "source", "pdf_page")}]},
        }
        return build_reference_text(results), artifact

    class FixedModel:
        def bind_tools(self, tools, **kwargs):
            def request(messages):
                calls["request_tool"] += 1
                return AIMessage(content="", response_metadata={"finish_reason": "tool_calls"},
                    tool_calls=[{"name": "search_papers", "args": {"question": question}, "id": "fixed_call"}])
            return RunnableLambda(request)

        def bind(self, **kwargs):
            def invoke(messages):
                if kwargs.get("max_tokens") == CITED_ANSWER_MAX_TOKENS:
                    calls["answer"] += 1
                    if answer_error is not None:
                        raise answer_error
                    payload = json.loads(messages[-1].content)
                    return response({"claims": [{"text": "教学固定结论，仅用于验证保存与恢复。",
                                                  "chunk_ids": [payload["reference"][0]["chunk_id"]]}],
                                     "insufficient_reason": ""})
                calls["classify"] += 1
                return response({"intent": route, "reason": "固定教学分类",
                                 "clarification_question": "你希望了解什么内容？" if route == "clarify" else ""})
            return RunnableLambda(invoke)

    return FixedModel(), search_papers, calls
