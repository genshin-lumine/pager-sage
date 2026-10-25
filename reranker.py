
from copy import deepcopy
from math import isfinite
from numbers import Real
from pathlib import Path
import tomllib

import numpy as np


RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
RERANK_MODEL_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"


def get_rerank_cache_folder(config_path=None):
    """读取本机缓存目录；未配置时沿用 Hugging Face 的缓存设置。"""
    path = Path(config_path) if config_path is not None else Path(__file__).with_name("config.local.toml")
    if not path.exists():
        return None
    with path.open("rb") as file:
        config = tomllib.load(file)
    settings = config.get("rerank", {})
    if not isinstance(settings, dict) or not isinstance(settings.get("cache_folder", ""), str):
        raise ValueError("本机配置中的 rerank.cache_folder 必须是字符串")
    folder = settings.get("cache_folder", "").strip()
    if not folder:
        return None
    cache_path = Path(folder).expanduser()
    if not cache_path.is_absolute():
        cache_path = path.resolve().parent / cache_path
    return str(cache_path)


def load_reranker(cache_folder=None, local_files_only=False):
    # 延迟导入，读取候选或运行纯逻辑检查时不加载模型依赖。
    from sentence_transformers import CrossEncoder
    import torch

    return CrossEncoder(
        RERANK_MODEL_NAME,
        revision=RERANK_MODEL_REVISION,
        device="cpu",
        max_length=512,
        activation_fn=torch.nn.Identity(),
        model_kwargs={"use_safetensors": True},
        cache_folder=cache_folder,
        local_files_only=local_files_only,
    )


def rerank_candidates(question, candidates, model, top_k=3):
    """接收 fuse_rrf() 的有序结果；返回精排后的前 top_k 条新记录。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("精排问题不能为空")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须是正整数")
    if not candidates:
        return []

    pairs = []
    seen_ids = set()
    for candidate in candidates:
        chunk = candidate["chunk"]
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("精排候选缺少有效的片段标识")
        if chunk_id in seen_ids:
            raise ValueError("精排候选包含重复片段标识")
        seen_ids.add(chunk_id)
        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            raise ValueError("精排候选原文不能为空")
        if type(candidate.get("rank")) is not int or candidate["rank"] < 1:
            raise ValueError("精排候选缺少有效的 RRF 排名")
        rrf_score = candidate.get("score")
        if isinstance(rrf_score, bool) or not isinstance(rrf_score, Real) or not isfinite(rrf_score):
            raise ValueError("精排候选缺少有效的 RRF 分数")
        pairs.append([question, chunk["text"]])

    scores = np.asarray(model.predict(
        pairs,
        batch_size=3,
        show_progress_bar=False,
        convert_to_numpy=True,
    ))
    # 一条候选必须恰好对应一个有限数值；异常输出不能进入参考资料。
    if scores.shape != (len(candidates),):
        raise ValueError("精排分数必须是一维数组，且与候选数量一致")
    if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
        raise ValueError("精排分数必须全部是有限数值")

    scored_candidates = []
    for candidate, score in zip(candidates, scores, strict=True):
        result = deepcopy(candidate)
        result["rrf_rank"] = candidate["rank"]
        result["rrf_score"] = float(candidate["score"])
        result["rerank_score"] = float(score)
        scored_candidates.append(result)

    # 保留学习者的降序写法；Python 稳定排序使同分项沿用输入的 RRF 顺序。
    scored_candidates.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )
    selected_results = scored_candidates[:top_k]
    for rank, result in enumerate(selected_results, start=1):
        result["rank"] = rank
        # 通用 score 表示当前阶段的分数，原 RRF 分数已另存。
        result["score"] = result["rerank_score"]
    return selected_results
