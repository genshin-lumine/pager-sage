import json
from pathlib import Path

import faiss


def save_index(index, chunks, output_dir, embedding_config):
    if index.ntotal != len(chunks):
        raise ValueError("索引中的向量数量与片段数量不一致")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": 1,
        "dimension": index.d,
        "embedding_config": embedding_config,
        "chunks": chunks,
    }

    faiss.write_index(index, str(output_dir / "index.faiss"))

    with (output_dir / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_index(output_dir):
    output_dir = Path(output_dir)

    with (output_dir / "metadata.json").open(
        "r", encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if metadata["schema_version"] != 1:
        raise ValueError("不支持的索引保存格式版本")

    index = faiss.read_index(str(output_dir / "index.faiss"))

    if index.d != metadata["dimension"]:
        raise ValueError("索引维度与记录的维度不一致")

    if index.ntotal != len(metadata["chunks"]):
        raise ValueError("索引中的向量数量与片段数量不一致")

    return index, metadata
