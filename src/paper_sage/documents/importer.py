import faiss
from sentence_transformers import SentenceTransformer

from paper_sage.documents.chunking import split_pages
from paper_sage.retrieval.index_store import save_index

from hashlib import sha256
from pathlib import Path

def get_file_hash(file_bytes):
    # 计算 file_bytes 的 SHA-256，返回十六进制摘要字符串
    return sha256(file_bytes).hexdigest()


def find_existing_document(project, file_hash):
    for document in project["documents"]:
        existing_hash = document.get("file_hash")

        if existing_hash is None:
            existing_bytes = Path(document["source"]).read_bytes()
            existing_hash = get_file_hash(existing_bytes)

        # 如果 existing_hash 与 file_hash 相等，
        # 返回当前 document 字典
        if existing_hash == file_hash:
            return document

    return None

def save_uploaded_pdf(project_id, document_id, file_name, file_bytes):
    output_dir = Path("data") / "papers" / project_id / document_id
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / Path(file_name).name

    # 使用 pdf_path.write_bytes()，把 file_bytes 写入文件
    pdf_path.write_bytes(file_bytes)

    return pdf_path

def build_document_index(pages, project_id, output_dir):
    chunks = split_pages(
        pages,
        project_id=project_id,
        chunk_size=800,
        chunk_overlap=100,
    )

    if not chunks:
        raise ValueError("PDF 未提取到可用文字，无法建立索引。")

    embedding_config = {
        "model_name": "intfloat/multilingual-e5-small",
        "normalize_embeddings": True,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
    }

    passage_texts = []
    for chunk in chunks:
        passage_texts.append(
            embedding_config["passage_prefix"] + chunk["text"]
        )

    model = SentenceTransformer(
        embedding_config["model_name"],
        device="cpu",
    )

    embeddings = model.encode(
        passage_texts,
        convert_to_numpy=True,
        normalize_embeddings=embedding_config["normalize_embeddings"],
        show_progress_bar=False,
    )
    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])

    # 调用 index 的 add()，将 embeddings 加入索引
    index.add(embeddings)

    save_index(
        index=index,
        chunks=chunks,
        output_dir=output_dir,
        embedding_config=embedding_config,
    )

    return {
        "chunk_count": len(chunks),
        "dimension": index.d,
    }
