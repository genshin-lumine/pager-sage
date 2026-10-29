
import argparse
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

from paper_sage.retrieval.bm25 import tokenize_bm25
from paper_sage.retrieval.dual import retrieve_dual, merge_candidates
from paper_sage.retrieval.index_store import load_index
from paper_sage.projects.store import get_index_dir
from paper_sage.retrieval.fusion import fuse_rrf
from paper_sage.retrieval.reranker import (
    get_rerank_cache_folder, load_reranker, rerank_candidates,
    RERANK_MODEL_NAME, RERANK_MODEL_REVISION,
)
from paper_sage.evaluation.metrics import evaluate_retrieval_results, calculate_mrr
from paper_sage.retrieval.trace import run_stage


WORKSPACE = Path.cwd().resolve()
METHODS = ("dense", "bm25", "rrf", "rerank")
RECALL_TOP_K = 3
RRF_TOP_K = 6
EVALUATION_K = 3
RRF_RANK_CONSTANT = 60


def file_sha256(path):
    return sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_evaluation_inputs(root=WORKSPACE):
    """固定本次标注的项目、文档、题集与分块，发现变化就停止。"""
    root = Path(root).resolve()
    labels_path = root / "data/evaluations/day27_relevance_labels.json"
    labels = read_json(labels_path)
    scope = labels["scope"]
    if labels["status"] != "complete":
        raise ValueError("相关性标注尚未完成")
    paths = {name: root / scope[name] for name in ("metadata_path", "question_path", "source")}
    for name, hash_name in (
        ("metadata_path", "metadata_sha256"), ("question_path", "questions_sha256"),
        ("source", "source_sha256"),
    ):
        if file_sha256(paths[name]) != scope[hash_name]:
            raise ValueError(f"评估输入 {name} 已变化，请先复核标注版本")

    project_path = root / "data/projects" / scope["project_id"] / "project.json"
    project = read_json(project_path)
    documents = [doc for doc in project["documents"] if doc["document_id"] == scope["document_id"]]
    if project["project_id"] != scope["project_id"] or len(documents) != 1:
        raise ValueError("标注文档不属于当前项目，或存在重复登记")
    document = documents[0]
    if document["source"] != scope["source"] or document["index_version"] != scope["index_version"]:
        raise ValueError("当前文档来源或索引版本与标注不一致")
    index_dir = root / get_index_dir(scope["project_id"], scope["document_id"], scope["index_version"])
    if (index_dir / "metadata.json").resolve() != paths["metadata_path"].resolve():
        raise ValueError("标注中的元数据路径与项目索引目录不一致")
    index, metadata = load_index(index_dir)
    chunks = metadata["chunks"]
    manifest = labels["scope"]["chunk_manifest"]
    if len(chunks) != scope["chunk_count"] or len(manifest) != len(chunks):
        raise ValueError("索引片段数量与标注不一致")
    chunk_ids = set()
    for chunk, saved in zip(chunks, manifest, strict=True):
        if (
            chunk["project_id"] != scope["project_id"]
            or chunk["document_id"] != scope["document_id"]
            or chunk["source"] != scope["source"]
            or chunk["chunk_id"] != saved["chunk_id"]
            or sha256(chunk["text"].encode()).hexdigest() != saved["text_sha256"]
            or chunk["pdf_page"] != saved["pdf_page"]
            or chunk["chunk_id"] in chunk_ids
        ):
            raise ValueError("片段顺序、标识、来源或原文与标注不一致")
        chunk_ids.add(chunk["chunk_id"])

    questions = read_json(paths["question_path"])["cases"]
    cases = labels["cases"]
    if len(cases) != len(questions) or len(cases) != scope["question_count"]:
        raise ValueError("评估题目数量不一致")
    seen_ids = set()
    for case, question in zip(cases, questions, strict=True):
        positives, negatives = set(case["relevant_chunk_ids"]), set(case["non_relevant_chunk_ids"])
        if (
            case["id"] != question["id"] or case["id"] in seen_ids
            or case["question"] != question["question"]
            or case["answerable"] != question["answerable"]
            or bool(positives) != case["answerable"]
            or case["include_in_recall_mrr"] != case["answerable"]
            or positives & negatives or positives | negatives != chunk_ids
            or len(positives) != len(case["relevant_chunk_ids"])
        ):
            raise ValueError(f"题目 {case['id']} 的范围或标签不一致")
        seen_ids.add(case["id"])
    if sum(case["answerable"] for case in cases) != scope["answerable_count"]:
        raise ValueError("有答案题目的汇总数量不一致")

    inputs = {labels_path, project_path, index_dir / "index.faiss", *paths.values()}
    fingerprints = {path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(inputs)}
    return index, metadata, labels, fingerprints


def verify_metric_wiring():
    """检查学习者填写的连接代码，确保在模型加载前发现问题。"""
    result = evaluate_retrieval_results(
        [{"rank": 1, "score": 0.9, "chunk": {"chunk_id": "check:hit"}}],
        {"check:hit"}, k=EVALUATION_K,
    )
    if result["recall_at_k"] != 1.0 or result["rr_at_k"] != 1.0:
        raise ValueError("指标连接检查未通过，请检查片段标识读取")


def evaluate_case(case, embedding_model, index, metadata, rerank_model, *, recall_top_k=RECALL_TOP_K):
    """按传入的每路召回数量运行；融合上限和最终评价 K 保持不变。"""
    if type(recall_top_k) is not int or recall_top_k < 1:
        raise ValueError("recall_top_k 必须是正整数")
    stages = []
    question = case["question"]
    routes = retrieve_dual(
        question, embedding_model, index, metadata,
        # 将本次调用的召回数量传给双路检索，原默认配置保持不变。
        top_k=recall_top_k,
        stage_records=stages,
    )
    merged = run_stage(stages, "merge", sum(map(len, routes.values())), lambda: merge_candidates(routes))
    fused = run_stage(stages, "rrf", len(merged), lambda: fuse_rrf(merged, RRF_TOP_K, RRF_RANK_CONSTANT))
    ranked = run_stage(
        stages, "rerank", len(fused),
        lambda: rerank_candidates(question, fused, rerank_model, top_k=max(1, len(fused))),
    )
    methods = {}
    for name, results in {**routes, "rrf": fused, "rerank": ranked}.items():
        selected = results[:EVALUATION_K]
        score = evaluate_retrieval_results(selected, case["relevant_chunk_ids"], EVALUATION_K)
        methods[name] = {"results": selected, **score}
    return {
        "id": case["id"], "question": question, "answerable": case["answerable"],
        "include_in_recall_mrr": case["include_in_recall_mrr"],
        "relevant_chunk_ids": case["relevant_chunk_ids"],
        "bm25_query_tokens": tokenize_bm25(question),
        "methods": methods, "stages": stages,
        "rerank_prediction_executed": bool(fused),
    }


def summarize_results(records):
    """只对有答案题求均值，未命中题的 0 必须保留在分母中。"""
    eligible = [record for record in records if record["include_in_recall_mrr"]]
    summary = {}
    for method in METHODS:
        recalls = [record["methods"][method]["recall_at_k"] for record in eligible]
        rrs = [record["methods"][method]["rr_at_k"] for record in eligible]
        if any(value is None for value in recalls + rrs):
            raise ValueError("有答案题缺少指标，不能跳过后计算平均值")
        summary[method] = {
            "question_count": len(eligible),
            "mean_recall_at_k": sum(recalls) / len(recalls) if recalls else None,
            "mrr_at_k": calculate_mrr(rrs),
            "empty_result_count": sum(not record["methods"][method]["results"] for record in records),
            "unanswerable_questions_with_candidates": sum(
                not record["answerable"] and bool(record["methods"][method]["results"])
                for record in records
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只检查指标连接与输入，不加载模型、不保存结果")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "data/evaluations/day27_retrieval_comparison.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.check and output.exists():
        parser.error("输出文件已存在，请用 --output 指定新文件名，旧报告不会被覆盖。")
    try:
        verify_metric_wiring()
        index, metadata, labels, fingerprints = load_evaluation_inputs()
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    scope = labels["scope"]
    print(f"输入检查通过：{scope['question_count']} 题 · 有答案 {scope['answerable_count']} 题"
          f" · 资料不足 {scope['unanswerable_count']} 题 · 片段 {scope['chunk_count']} 个", flush=True)
    if args.check:
        print("仅检查完成：未加载模型，未执行检索，未生成报告。")
        return

    from sentence_transformers import SentenceTransformer

    print("正在从本地缓存加载向量模型与精排模型……", flush=True)
    started = perf_counter()
    embedding_model = SentenceTransformer(metadata["embedding_config"]["model_name"], device="cpu", local_files_only=True)
    embedding_load_seconds = perf_counter() - started
    started = perf_counter()
    rerank_model = load_reranker(cache_folder=get_rerank_cache_folder(), local_files_only=True)
    rerank_load_seconds = perf_counter() - started

    records = []
    started = perf_counter()
    for case in labels["cases"]:
        records.append(evaluate_case(case, embedding_model, index, metadata, rerank_model))
        print(f"完成 {len(records)}/{len(labels['cases'])}：{case['id']}", flush=True)
    elapsed_seconds = perf_counter() - started
    summary = summarize_results(records)
    for relative_path, expected in fingerprints.items():
        current_path = WORKSPACE / relative_path
        if file_sha256(current_path) != expected:
            raise ValueError(f"运行期间输入发生变化，结果未保存：{current_path.name}")
    report = {
        "schema_version": 1, "purpose": "day27_four_method_retrieval_comparison", "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": scope, "input_sha256": fingerprints,
        "label_policy": labels["policy"],
        "config": {"recall_top_k": RECALL_TOP_K, "rrf_top_k": RRF_TOP_K,
                   "evaluation_k": EVALUATION_K, "rank_constant": RRF_RANK_CONSTANT},
        "models": {
            "embedding": metadata["embedding_config"],
            "rerank": {"model_name": RERANK_MODEL_NAME, "revision": RERANK_MODEL_REVISION,
                       "device": "cpu", "max_length": 512, "score_type": "logit"},
        },
        "package_versions": {name: version(name) for name in ["sentence-transformers", "rank-bm25", "faiss-cpu", "numpy"]},
        "execution": {
            "dense_queries": len(records), "bm25_queries": len(records),
            "rerank_predictions": sum(r["rerank_prediction_executed"] for r in records),
            "deepseek_calls": 0, "local_files_only": True,
            "embedding_load_seconds": embedding_load_seconds, "rerank_load_seconds": rerank_load_seconds,
            "evaluation_seconds_excluding_model_loading": elapsed_seconds,
        },
        "summary": summary, "results": records,
        "limitations": [
            "指标基于助手按直接依据标准给出的片段标签，尚无独立人工复核，不是盲测或回答正确率。",
            "按片段计数，重叠原文可能贡献多个相关片段；复合题命中一个单元不表示全部事实都有依据。",
            "资料不足题不进入 Recall / MRR 平均值；其返回候选不等于系统作出了错误回答。",
            "当前 BM25 只处理英文字母和数字，本次使用原题文本，不做翻译或查询改写。",
            "四种方法共享同一范围与双路结果，最终均评价前三条；Rerank 处理最多六条融合候选。",
            "耗时是单次顺序运行的观测；各阶段耗时与完整方法耗时不可混为一谈。",
            "索引未保存向量模型 revision，本次使用本地缓存的同名模型。",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print("\n方法       有答案题数  Mean Recall@3  MRR@3")
    for method, metrics in summary.items():
        print(f"{method:<10} {metrics['question_count']:>5} {metrics['mean_recall_at_k']:>14.4f} {metrics['mrr_at_k']:>7.4f}")
    print(f"已保存：{output}")


if __name__ == "__main__":
    main()
