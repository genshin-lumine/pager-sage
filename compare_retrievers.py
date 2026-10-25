
import argparse
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

from bm25_retriever import retrieve_bm25, tokenize_bm25
from project_store import get_index_dir, load_project
from retrieval_scope import load_scope_index
from retriever import retrieve_dense


CASES = [
    {"case_id": "day22_keyword", "question": "NVIDIA P100"},
    {"case_id": "day22_chinese", "question": "编码器由多少个相同的层堆叠而成？"},
    {
        "case_id": "day22_energy",
        "question": "How much electrical energy in kWh was consumed during training?",
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluations/day22_retrieval_comparison.json"),
        help="结果文件路径；已有文件不会被覆盖。",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("结果文件已存在，请用 --output 指定新的文件名。")

    project = load_project("demo_project")
    documents = project["documents"]
    selected_ids = [document["document_id"] for document in documents]
    index, metadata = load_scope_index(project, selected_ids)
    chunks = metadata["chunks"]
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    config = metadata["embedding_config"]
    inputs = []
    for document in documents:
        index_dir = get_index_dir(
            project["project_id"], document["document_id"], document["index_version"]
        )
        inputs.append({
            "document": document,
            "sha256": {
                name: sha256((index_dir / name).read_bytes()).hexdigest()
                for name in ["index.faiss", "metadata.json"]
            },
        })

    print(f"Scope: {project['project_id']} | chunks: {len(chunks)}", flush=True)
    print("Loading cached embedding model on CPU...", flush=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config["model_name"], device="cpu", local_files_only=True
    )

    records = []
    for case in CASES:
        question = case["question"]
        dense = retrieve_dense(question, model, index, metadata, top_k=3)
        bm25 = retrieve_bm25(question, chunks, top_k=3)
        for result in dense + bm25:
            chunk = result["chunk"]
            assert chunk["project_id"] == project["project_id"]
            assert chunk["document_id"] in selected_ids
            assert chunk == chunks_by_id[chunk["chunk_id"]]
        records.append({
            **case,
            "bm25_query_tokens": tokenize_bm25(question),
            "dense": dense,
            "bm25": bm25,
        })
        print(f"{case['case_id']}: Dense={len(dense)}, BM25={len(bm25)}", flush=True)
        for method, results in [("Dense", dense), ("BM25", bm25)]:
            for result in results:
                print(
                    f"  {method} {result['rank']}: "
                    f"{result['score']:.4f} | {result['chunk']['chunk_id']}",
                    flush=True,
                )

    report = {
        "schema_version": 1,
        "purpose": "day22_retrieval_learning_comparison",
        "created_at": datetime.now().astimezone().isoformat(),
        "project_id": project["project_id"],
        "selected_document_ids": selected_ids,
        "chunk_count": len(chunks),
        "top_k": 3,
        "input_indexes": inputs,
        "embedding_config": config,
        "bm25_config": {
            "algorithm": "BM25Okapi", "k1": 1.5, "b": 0.75, "epsilon": 0.25,
            "tokenizer": "lowercase_then_[a-z0-9]+",
            "candidate_filter": "at_least_one_query_term_occurs",
        },
        "package_versions": {
            name: version(name)
            for name in ["rank-bm25", "sentence-transformers", "numpy", "faiss-cpu"]
        },
        "execution": {
            "dense_queries": len(records),
            "bm25_queries": len(records),
            "deepseek_calls": 0,
            "embedding_device": "cpu",
            "embedding_local_files_only": True,
        },
        "limitations": [
            "三个固定查询用于学习观察，不代表整体检索性能，也不替代 Day 14 评估。",
            "只保存候选；没有生成回答、计算回答正确率或进行自动证据支持判定。",
            "BM25 当前只提取英文字母和数字，不做翻译、词形还原或同义词扩展。",
            "Dense 与 BM25 的原始分数不能直接比较或相加。",
            "索引未记录模型 revision，查询使用本地已缓存的同名模型。",
        ],
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved: {args.output.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
