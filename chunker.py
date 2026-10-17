from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pages(pages, project_id, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []

    for page_record in pages:
        page_chunks = splitter.split_text(page_record["text"])

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            chunk_id = f"{project_id}:{page_record['document_id']}:p{page_record['pdf_page']}:c{chunk_index}"

            chunk_record = {
                "project_id": project_id,
                "document_id": page_record['document_id'],
                "source": page_record['source'],
                "pdf_page": page_record['pdf_page'],
                "chunk_id": chunk_id,
                "text": chunk_text,
            }
            chunks.append(chunk_record)

    return chunks