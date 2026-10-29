"""根据当前所选文档定位证据，并准备内存中的单页 PDF。"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path, PureWindowsPath

from pypdf import PdfReader, PdfWriter
from pypdf.errors import PyPdfError


def build_pdf_page_target(source, project_id, selected_documents):
    """核对项目、文档与路径后，返回包含物理页码和零起始下标的定位信息。"""
    if not isinstance(source, dict):
        raise ValueError("证据来源必须是字典")
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("当前项目标识无效")
    if source.get("project_id") != project_id:
        raise ValueError("证据不属于当前项目")
    document_id = source.get("document_id")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("证据缺少有效的文档标识")
    pdf_page = source.get("pdf_page")
    if type(pdf_page) is not int or pdf_page < 1:
        raise ValueError("PDF 页码必须是从 1 开始的整数")
    if not isinstance(selected_documents, list):
        raise ValueError("所选文档必须是列表")

    documents_by_id = {}
    for document in selected_documents:
        if not isinstance(document, dict):
            raise ValueError("所选文档记录必须是字典")
        selected_id, path = document.get("document_id"), document.get("source")
        if not isinstance(selected_id, str) or not selected_id.strip() or not isinstance(path, str) or not path.strip():
            raise ValueError("所选文档缺少有效的标识或来源路径")
        if selected_id in documents_by_id:
            raise ValueError("所选文档标识重复")
        documents_by_id[selected_id] = document
    document = documents_by_id.get(document_id)
    if document is None:
        raise ValueError("证据文档不在当前所选范围内")
    if source.get("source") != document["source"]:
        raise ValueError("证据来源路径与当前文档记录不一致")

    return {
        "project_id": project_id,
        "document_id": document_id,
        "source": document["source"],
        "pdf_page": pdf_page,
        "page_index": pdf_page - 1,
    }


def _checked_pdf_path(source_path, workspace_root):
    """仅从项目的 papers 目录读取当前文档，保留早期 Demo 的公共 PDF 路径。"""
    root = Path(workspace_root).resolve()
    path = (root / source_path).absolute()
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        raise ValueError("PDF 路径超出当前工作目录") from None
    current = root
    for part in parts:
        if part == ".." or ":" in part or part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved():
            raise ValueError("PDF 路径包含不允许的路径片段")
        current /= part
        if current.is_symlink() or current.is_junction():
            raise ValueError("PDF 路径不能包含符号链接或目录联接")
    resolved = path.resolve()
    if not resolved.is_relative_to(root / "data" / "papers"):
        raise ValueError("PDF 必须位于当前项目工作目录的 data/papers 中")
    if resolved.suffix.lower() != ".pdf":
        raise ValueError("文档来源必须是 PDF 文件")
    if not resolved.is_file():
        raise FileNotFoundError("当前文档的 PDF 文件不存在或不是普通文件")
    return resolved


def load_pdf_page_preview(source, project_id, selected_documents, *, workspace_root=None):
    """读取当前 PDF 的一个页面，返回 bytes、原页码和文件摘要，不写入磁盘。"""
    target = build_pdf_page_target(source, project_id, selected_documents)
    path = _checked_pdf_path(target["source"], Path.cwd() if workspace_root is None else workspace_root)
    original_bytes = path.read_bytes()
    try:
        reader = PdfReader(BytesIO(original_bytes))
        if reader.is_encrypted:
            raise ValueError("暂不支持查看加密 PDF")
        total_pages = len(reader.pages)
        if target["pdf_page"] > total_pages:
            raise ValueError(f"PDF 页码超出文件总页数：请求第 {target['pdf_page']} 页，文件共 {total_pages} 页")
        page = reader.pages[target["page_index"]]
        page_text = page.extract_text() or ""
        source_text_found = None
        if "text" in source:
            expected_text = source["text"]
            if not isinstance(expected_text, str) or not expected_text.strip():
                raise ValueError("证据原文必须是非空字符串")
            source_text_found = " ".join(expected_text.split()) in " ".join(page_text.split())
            if not source_text_found:
                raise ValueError("证据原文未在当前 PDF 页的提取文字中找到，请核对文档版本或页码")
        writer = PdfWriter()
        writer.add_page(page)
        with BytesIO() as buffer:
            writer.write(buffer)
            pdf_bytes = buffer.getvalue()
    except PyPdfError as error:
        raise ValueError("无法读取 PDF 页面，请检查文件是否有效") from error
    return {
        **target,
        "total_pages": total_pages,
        "pdf_bytes": pdf_bytes,
        "page_text": page_text,
        "source_text_found": source_text_found,
        "source_sha256": sha256(original_bytes).hexdigest(),
        "download_name": f"{path.stem}_p{target['pdf_page']}.pdf",
    }
