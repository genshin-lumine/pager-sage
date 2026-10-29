"""Day 28：记录暂选配置后，在预先固定的五道新问题上复验 A / B。"""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

from paper_sage.evaluation.recall_limits import CODE_FILES, PLAN_PATH, load_experiment, run_comparison, verify_fingerprints
from paper_sage.evaluation.retrieval import WORKSPACE, file_sha256, read_json
from paper_sage.retrieval.reranker import get_rerank_cache_folder, load_reranker, RERANK_MODEL_NAME, RERANK_MODEL_REVISION


SELECTION_PATH = WORKSPACE / "data/evaluations/day28_selection.json"
DEVELOPMENT_PATH = WORKSPACE / "data/evaluations/day28_development_comparison.json"
OUTPUT_PATH = WORKSPACE / "data/evaluations/day28_validation_comparison.json"


def load_validation():
    index, metadata, plan, _, fingerprints = load_experiment()
    selection = read_json(SELECTION_PATH)
    verify_fingerprints(selection["input_sha256"])
    development = read_json(DEVELOPMENT_PATH)
    selected = next((c for c in plan["configurations"] if c["name"] == selection["selected_group"]), None)
    if (
        selection["status"] != "selected_before_validation" or selection["selected_by"] != "user"
        or selection["validation_runs_at_selection"] != 0 or selected is None
        or selection["selected_config"] != {"recall_top_k": selected["recall_top_k"], **plan["fixed_parameters"]}
        or selection["validation_case_ids"] != [case["id"] for case in plan["validation_cases"]]
        or development["status"] != "complete" or development["execution"]["validation_pipeline_runs"] != 0
        or selection["development_summary"] != development["summary"]
    ):
        raise ValueError("暂选配置、开发报告或固定复验题不一致")
    if not (datetime.fromisoformat(plan["created_at"]) < datetime.fromisoformat(development["created_at"])
            <= datetime.fromisoformat(selection["created_at"])):
        raise ValueError("应先固定实验计划，完成开发对照，再记录暂选配置")
    verify_fingerprints(development["input_sha256"])
    # 共用驱动只增加 None 指标的打印处理；实际检索与指标模块必须保持原版本。
    verify_fingerprints({path: digest for path, digest in development["code_sha256"].items()
                         if path != 'src/paper_sage/evaluation/recall_limits.py'})
    fingerprints.update(selection["input_sha256"])
    fingerprints[SELECTION_PATH.relative_to(WORKSPACE).as_posix()] = file_sha256(SELECTION_PATH)
    return index, metadata, plan, selection, fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只核对选择记录与固定题目，不加载模型")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.check and output.exists():
        parser.error("复验报告已存在；请先阅读原报告，避免在同一组题目上反复挑参数。")
    try:
        index, metadata, plan, selection, fingerprints = load_validation()
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"复验检查通过：暂选 {selection['selected_group']}；固定新问题 5 道（4 道有答案、1 道资料不足）。", flush=True)
    if args.check:
        print("仅检查完成：未加载模型，未执行检索，未生成复验结果。")
        return

    code_fingerprints = {name: file_sha256(WORKSPACE / name) for name in (*CODE_FILES, 'src/paper_sage/evaluation/validate_recall.py')}
    from sentence_transformers import SentenceTransformer

    print("正在从本地缓存加载模型……", flush=True)
    started = perf_counter()
    embedding_model = SentenceTransformer(metadata["embedding_config"]["model_name"], device="cpu", local_files_only=True)
    embedding_load_seconds = perf_counter() - started
    started = perf_counter()
    rerank_model = load_reranker(cache_folder=get_rerank_cache_folder(), local_files_only=True)
    rerank_load_seconds = perf_counter() - started
    records, summary = run_comparison(
        plan["validation_cases"], plan["configurations"], embedding_model, index, metadata, rerank_model,
    )
    verify_fingerprints(fingerprints)
    verify_fingerprints(code_fingerprints)
    report = {
        "schema_version": 1, "purpose": "day28_validation_recall_limit_comparison", "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(),
        "input_sha256": fingerprints, "code_sha256": code_fingerprints,
        "plan_path": PLAN_PATH.relative_to(WORKSPACE).as_posix(),
        "selection_path": SELECTION_PATH.relative_to(WORKSPACE).as_posix(),
        "selected_group_before_validation": selection["selected_group"],
        "configurations": plan["configurations"], "fixed_parameters": plan["fixed_parameters"],
        "validation_case_ids": selection["validation_case_ids"],
        "models": {"embedding": metadata["embedding_config"],
                   "rerank": {"model_name": RERANK_MODEL_NAME, "revision": RERANK_MODEL_REVISION,
                              "device": "cpu", "max_length": 512, "score_type": "logit"}},
        "package_versions": {name: version(name) for name in ["sentence-transformers", "rank-bm25", "faiss-cpu", "numpy"]},
        "execution": {"development_pipeline_runs": 0, "validation_pipeline_runs": len(records),
                      "deepseek_calls": 0, "local_files_only": True,
                      "embedding_load_seconds": embedding_load_seconds, "rerank_load_seconds": rerank_load_seconds,
                      "run_order": [{"id": r["id"], "group": r["group"]} for r in records]},
        "aggregation": {"metric_question_count_per_group": 4, "timing_question_count_per_group": 5,
                        "unanswerable_question_ids": [c["id"] for c in plan["validation_cases"] if not c["answerable"]]},
        "summary": summary, "results": records,
        "limitations": plan["limitations"] + [
            "只有四道有答案题参与质量指标，五道题都参与耗时平均；资料不足题指标为 null，不补成 0。",
            "资料不足题返回候选不表示生成了错误回答，本次没有调用回答模型或测量拒答正确率。",
            "模型加载一次，按题交替 A/B，未专门预热；单次耗时不代表稳定性能。",
            "与开发运行相比，共用驱动仅增加资料不足指标的显示处理；检索与指标模块通过版本摘要校验。",
        ],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write(serialized)
    print("\n复验汇总：质量指标平均 4 题，耗时平均 5 题。")
    print("组别  每路召回  平均 Recall@3  MRR@3  平均检索耗时（秒）")
    for name, values in summary.items():
        metrics = values["methods"]["rerank"]
        print(f"{name}     {values['recall_top_k']}       {metrics['mean_recall_at_k']:.4f}       "
              f"{metrics['mrr_at_k']:.4f}    {values['mean_pipeline_seconds']:.3f}")
    print(f"已保存：{output}")


if __name__ == "__main__":
    main()
