
import argparse
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

from paper_sage.evaluation.retrieval import (
    WORKSPACE, RECALL_TOP_K, RRF_TOP_K, EVALUATION_K, RRF_RANK_CONSTANT,
    evaluate_case, file_sha256, load_evaluation_inputs, read_json, summarize_results,
)
from paper_sage.retrieval.reranker import get_rerank_cache_folder, load_reranker, RERANK_MODEL_NAME, RERANK_MODEL_REVISION


PLAN_PATH = WORKSPACE / "data/evaluations/day28_tuning_plan.json"
OUTPUT_PATH = WORKSPACE / "data/evaluations/day28_development_comparison.json"
CODE_FILES = (
    'src/paper_sage/evaluation/recall_limits.py', 'src/paper_sage/evaluation/retrieval.py', 'src/paper_sage/evaluation/metrics.py',
    'src/paper_sage/retrieval/dense.py', 'src/paper_sage/retrieval/bm25.py', 'src/paper_sage/retrieval/dual.py', 'src/paper_sage/retrieval/fusion.py',
    'src/paper_sage/retrieval/reranker.py', 'src/paper_sage/retrieval/trace.py',
)


def verify_fingerprints(fingerprints):
    for relative_path, expected in fingerprints.items():
        current_path = WORKSPACE / relative_path
        if file_sha256(current_path) != expected:
            raise ValueError(f"实验输入已变化，请复核固定计划：{current_path.name}")


def load_experiment():
    plan = read_json(PLAN_PATH)
    verify_fingerprints(plan["input_sha256"])
    index, metadata, labels, fingerprints = load_evaluation_inputs()
    if (RECALL_TOP_K, RRF_TOP_K, EVALUATION_K, RRF_RANK_CONSTANT) != (3, 6, 3, 60):
        raise ValueError("请保留原默认配置，本实验仅通过函数参数改变每路召回数量")
    if plan["fixed_parameters"] != {"rrf_top_k": 6, "evaluation_k": 3, "rank_constant": 60}:
        raise ValueError("固定参数与实验约定不一致")
    if plan["configurations"] != [{"name": "A", "recall_top_k": 3}, {"name": "B", "recall_top_k": 6}]:
        raise ValueError("本轮仅比较已约定的 A=3 与 B=6")
    if plan["label_policy_version"] != labels["policy"]["version"]:
        raise ValueError("标注规则版本不一致")
    by_id = {case["id"]: case for case in labels["cases"]}
    if plan["development_case_ids"] != ["q5", "q12", "q13"]:
        raise ValueError("开发题与预先约定不一致")
    cases = [by_id[case_id] for case_id in plan["development_case_ids"]]

    # 复验题在真实调参前固定；此脚本只核对标签与引用，不对它们调用模型。
    chunks = {chunk["chunk_id"]: chunk for chunk in metadata["chunks"]}
    seen_ids = set(by_id)
    seen_questions = {case["question"] for case in labels["cases"]}
    validation = plan["validation_cases"]
    if len(validation) != 5 or sum(c["answerable"] for c in validation) != 4:
        raise ValueError("复验题应为 4 道有答案题和 1 道资料不足题")
    for case in validation:
        positives = set(case["relevant_chunk_ids"])
        negatives = set(case["non_relevant_chunk_ids"])
        if (
            case["id"] in seen_ids or case["question"] in seen_questions
            or positives & negatives or positives | negatives != set(chunks)
            or len(positives) != len(case["relevant_chunk_ids"])
            or len(negatives) != len(case["non_relevant_chunk_ids"])
            or bool(positives) != case["answerable"]
            or case["include_in_recall_mrr"] != case["answerable"]
            or {e["chunk_id"] for e in case["evidence"]} != positives
        ):
            raise ValueError(f"复验题 {case['id']} 的范围、标签或引用不一致")
        for evidence in case["evidence"]:
            chunk = chunks[evidence["chunk_id"]]
            if (
                evidence["source"] != chunk["source"] or evidence["pdf_page"] != chunk["pdf_page"]
                or evidence["text_sha256"] != sha256(chunk["text"].encode()).hexdigest()
                or not evidence["quote"]
                or chunk["text"][evidence["start"]:evidence["end"]] != evidence["quote"]
            ):
                raise ValueError("复验题引用位置或原文不一致")
        seen_ids.add(case["id"])
        seen_questions.add(case["question"])
    fingerprints.update(plan["input_sha256"])
    fingerprints[PLAN_PATH.relative_to(WORKSPACE).as_posix()] = file_sha256(PLAN_PATH)
    return index, metadata, plan, cases, fingerprints


def describe_candidates(results, relevant_ids):
    ids = [result["chunk"]["chunk_id"] for result in results]
    gold = set(relevant_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("同一阶段存在重复片段")
    hits = [cid for cid in ids if cid in gold]
    return {"candidate_count": len(ids), "hit_chunk_ids": hits,
            "candidate_recall": len(hits) / len(gold) if gold else None}


def run_comparison(cases, configurations, embedding_model, index, metadata, rerank_model):
    records = []
    for position, case in enumerate(cases):
        # 交替 A/B 顺序，减少始终让某一组先运行的偏差；仍只是一次观测。
        ordered_configs = configurations if position % 2 == 0 else configurations[::-1]
        for config in ordered_configs:
            started = perf_counter()
            record = evaluate_case(
                case, embedding_model, index, metadata, rerank_model,
                recall_top_k=config["recall_top_k"],
            )
            elapsed = perf_counter() - started
            diagnostics = {
                stage["stage"]: describe_candidates(stage["results"], case["relevant_chunk_ids"])
                for stage in record["stages"]
            }
            diagnostics["selected"] = describe_candidates(
                record["methods"]["rerank"]["results"], case["relevant_chunk_ids"],
            )
            record.update(group=config["name"], recall_top_k=config["recall_top_k"],
                          elapsed_seconds=elapsed, stage_diagnostics=diagnostics)
            records.append(record)
            final = record["methods"]["rerank"]
            recall_text = "不适用" if final["recall_at_k"] is None else f"{final['recall_at_k']:.4f}"
            rr_text = "不适用" if final["rr_at_k"] is None else f"{final['rr_at_k']:.4f}"
            print(f"{case['id']}  {config['name']}  每路 {config['recall_top_k']} 条"
                  f"  召回池命中 {len(diagnostics['merge']['hit_chunk_ids'])}"
                  f"  精排输入命中 {len(diagnostics['rrf']['hit_chunk_ids'])}"
                  f"  Recall@3 {recall_text}  RR@3 {rr_text}"
                  f"  耗时 {elapsed:.3f} 秒", flush=True)
    summary = {}
    for config in configurations:
        subset = [record for record in records if record["group"] == config["name"]]
        summary[config["name"]] = {
            "recall_top_k": config["recall_top_k"], "methods": summarize_results(subset),
            "mean_pipeline_seconds": sum(record["elapsed_seconds"] for record in subset) / len(subset),
        }
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只检查实验计划、输入和标注")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.check and output.exists():
        parser.error("报告已存在；确需新实验时请用 --output 指定新文件名。")
    try:
        index, metadata, plan, cases, fingerprints = load_experiment()
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print("计划检查通过：开发题 3 道 × 两组；新复验题 5 道已固定，本次不运行。", flush=True)
    if args.check:
        print("仅检查完成：未加载模型，未执行检索，未生成实验结果。")
        return

    code_fingerprints = {name: file_sha256(WORKSPACE / name) for name in CODE_FILES}
    from sentence_transformers import SentenceTransformer

    print("正在从本地缓存加载模型……", flush=True)
    started = perf_counter()
    embedding_model = SentenceTransformer(metadata["embedding_config"]["model_name"], device="cpu", local_files_only=True)
    embedding_load_seconds = perf_counter() - started
    started = perf_counter()
    rerank_model = load_reranker(cache_folder=get_rerank_cache_folder(), local_files_only=True)
    rerank_load_seconds = perf_counter() - started
    records, summary = run_comparison(cases, plan["configurations"], embedding_model, index, metadata, rerank_model)
    verify_fingerprints(fingerprints)
    verify_fingerprints(code_fingerprints)
    report = {
        "schema_version": 1, "purpose": "day28_development_recall_limit_comparison", "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(),
        "input_sha256": fingerprints, "code_sha256": code_fingerprints,
        "plan_path": PLAN_PATH.relative_to(WORKSPACE).as_posix(),
        "configurations": plan["configurations"], "fixed_parameters": plan["fixed_parameters"],
        "development_case_ids": plan["development_case_ids"],
        "validation_case_ids": [case["id"] for case in plan["validation_cases"]],
        "models": {"embedding": metadata["embedding_config"],
                   "rerank": {"model_name": RERANK_MODEL_NAME, "revision": RERANK_MODEL_REVISION,
                              "device": "cpu", "max_length": 512, "score_type": "logit"}},
        "package_versions": {name: version(name) for name in ["sentence-transformers", "rank-bm25", "faiss-cpu", "numpy"]},
        "execution": {"development_pipeline_runs": len(records), "validation_pipeline_runs": 0,
                      "deepseek_calls": 0, "local_files_only": True,
                      "embedding_load_seconds": embedding_load_seconds, "rerank_load_seconds": rerank_load_seconds,
                      "run_order": [{"id": r["id"], "group": r["group"]} for r in records]},
        "summary": summary, "results": records,
        "limitations": plan["limitations"] + [
            "耗时含单次检索、快照和指标计算，不含模型加载与 Python 导入；未专门预热，不能作为稳定性能基准。",
            "这里的召回池与精排输入指标按各阶段完整集合计算，不能当作最终 Recall@3。",
        ],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write(serialized)
    print("\n组别  每路召回  平均 Recall@3  MRR@3  平均检索耗时（秒）")
    for name, values in summary.items():
        metrics = values["methods"]["rerank"]
        print(f"{name}     {values['recall_top_k']}       {metrics['mean_recall_at_k']:.4f}       "
              f"{metrics['mrr_at_k']:.4f}    {values['mean_pipeline_seconds']:.3f}")
    print(f"已保存：{output}")


if __name__ == "__main__":
    main()
