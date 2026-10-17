from pypdf import PdfReader


def load_pdf_pages(pdf_path, document_id):
    # 逐页读取 PDF 文件内容
    reader = PdfReader(pdf_path)
    
    pages = []
    
    for pdf_page, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        page_record = {
            "document_id": document_id,
            "source": pdf_path,
            "pdf_page": pdf_page,
            "text": text,
        }

        pages.append(page_record)

    return pages