import os
import re
import shutil
from pathlib import Path, PureWindowsPath
from paper_sage.projects.store import get_index_dir, load_project, save_project

def remove_document_record(project, document_id):
    remaining_documents = []
    removed_document = None

    for document in project["documents"]:
        if document["document_id"] == document_id:  # 判断这是否是要移除的文档
            if removed_document is not None:
                raise ValueError("项目中存在重复的文档标识")

            removed_document = document
        else:
            remaining_documents.append(document)  # 把其他文档加入 remaining_documents

    if removed_document is None:
        raise ValueError("当前项目中不存在这份文档")

    updated_project = project.copy()
    updated_project["documents"] = remaining_documents

    return updated_project, removed_document


def get_document_paths(project_id, document):
    pdf_path = Path(
        document["source"]  # 传入文档记录中的来源路径
    ).resolve()

    index_dir = get_index_dir(
        project_id,
        document["document_id"],  # 传入文档记录中的文档标识
        document["index_version"],
    ).parent.resolve()

    return pdf_path, index_dir


def _check_identifier(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None
        or PureWindowsPath(value).is_reserved()
    ):
        raise ValueError("项目、文档或版本标识不合法")


def _checked_path(path):
    """检查原始路径的每一级，不能只检查 resolve() 后的目标。"""
    workspace = Path.cwd().resolve()
    path = Path(path).absolute()
    try:
        parts = path.relative_to(workspace).parts
    except ValueError:
        raise ValueError("文件路径超出当前工作目录") from None
    current = workspace
    for part in parts:
        if (
            part == ".." or ":" in part or part.endswith((" ", "."))
            or PureWindowsPath(part).is_reserved()
        ):
            raise ValueError("文件路径包含不允许的路径片段")
        current = current / part
        if current.is_symlink() or current.is_junction():
            raise ValueError("清理路径不能包含符号链接或目录联接")
    resolved = path.resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError("文件路径超出当前工作目录")
    return resolved


def _check_index_tree(index_dir):
    if not index_dir.exists():
        return
    if not index_dir.is_dir():
        raise ValueError("文档索引路径不是目录")

    def raise_walk_error(error):
        raise error

    # rglob/os.walk 默认不跟随链接，但仍需明确拒绝树内的联接和链接。
    for root, directories, files in os.walk(
        index_dir, followlinks=False, onerror=raise_walk_error
    ):
        for name in directories + files:
            _checked_path(Path(root) / name)


def get_cleanup_paths(project_id, document):
    """返回可清理的 PDF 与文档索引目录；公共 PDF 返回 None。"""
    _check_identifier(project_id)
    _check_identifier(document.get("document_id"))
    _check_identifier(document.get("index_version"))
    source = document.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("文档缺少 source 路径")

    document_id = document["document_id"]
    papers_root = _checked_path(Path("data") / "papers")
    owned_pdf_dir = _checked_path(papers_root / project_id / document_id)
    # 校验前保留原始路径，防止 resolve() 隐藏中间的目录联接。
    _checked_path(source)
    _checked_path(Path("data") / "indexes" / project_id / document_id)
    pdf_path, index_dir = get_document_paths(project_id, document)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("文档来源必须是 PDF 文件")
    if pdf_path.exists() and not pdf_path.is_file():
        raise ValueError("PDF 来源不是普通文件")
    if pdf_path.parent == papers_root:
        pdf_path = None  # 兼容早期 Demo 的公共原始 PDF，保留原件。
    elif pdf_path.parent != owned_pdf_dir:
        raise ValueError("PDF 不属于当前项目和文档目录")
    _check_index_tree(index_dir)
    return pdf_path, index_dir


def _load_managed_project(project_id):
    _check_identifier(project_id)
    _checked_path(Path("data") / "projects" / project_id / "project.json")
    project = load_project(project_id)
    if project.get("project_id") != project_id:
        raise ValueError("项目记录与当前项目标识不匹配")
    for field in ("documents", "pending_cleanup"):
        records = project.get(field, [])
        if not isinstance(records, list):
            raise ValueError(f"{field} 必须是列表")
        ids = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{field} 中的文档记录必须是字典")
            _check_identifier(record.get("document_id"))
            ids.append(record["document_id"])
        if len(set(ids)) != len(ids):
            raise ValueError(f"{field} 中存在重复文档标识")
    return project


def _with_pending(project, records):
    updated = project.copy()
    if records:
        updated["pending_cleanup"] = records
    else:
        updated.pop("pending_cleanup", None)
    return updated


def _cleanup_files(project_id, document):
    # 每次尝试都重新验证，不能信任上一次检查或 JSON 中保存的路径。
    pdf_path, index_dir = get_cleanup_paths(project_id, document)
    if pdf_path is not None:
        pdf_path.unlink(missing_ok=True)
        if pdf_path.parent.exists() and not any(pdf_path.parent.iterdir()):
            pdf_path.parent.rmdir()
    if index_dir.exists():
        # 递归删除前核对最终绝对目标并拒绝树内链接，包含所有索引版本。
        _checked_path(index_dir)
        _check_index_tree(index_dir)
        shutil.rmtree(index_dir)


def retry_document_cleanup(project_id, document_id):
    """清理成功才移除待办；失败返回提示，已删除的文件允许再次清理。"""
    project = _load_managed_project(project_id)
    if any(doc["document_id"] == document_id for doc in project["documents"]):
        raise ValueError("文档仍在项目中，不能按残留文件清理")
    pending = project.get("pending_cleanup", [])
    document = next((doc for doc in pending if doc["document_id"] == document_id), None)
    if document is None:
        raise ValueError("没有这份文档的待清理记录")
    try:
        _cleanup_files(project_id, document)
        updated = _with_pending(
            project, [doc for doc in pending if doc["document_id"] != document_id]
        )
        save_project(updated)
    except (OSError, ValueError, RuntimeError) as error:
        # 包括最后的待办保存失败：磁盘仍有待办，重启后可以安全重试。
        return project, str(error)
    return updated, None


def delete_document(project_id, document_id):
    project = _load_managed_project(project_id)
    updated, document = remove_document_record(project, document_id)
    get_cleanup_paths(project_id, document)  # 保存前先验证归属。
    pending = project.get("pending_cleanup", [])
    if any(doc["document_id"] == document_id for doc in pending):
        raise ValueError("这份文档已有待清理记录")
    updated = _with_pending(updated, pending + [document])

    # 本次课的关键顺序：登记退出检索与待清理事项，成功后才动文件。
    save_project(updated)
    try:
        return retry_document_cleanup(project_id, document_id)
    except (OSError, ValueError, RuntimeError) as error:
        return updated, str(error)


def begin_document_import(project_id, document):
    """在写入 PDF 前登记清理待办，导入失败或中断也能找回残留。"""
    project = _load_managed_project(project_id)
    pdf_path, _ = get_cleanup_paths(project_id, document)
    if pdf_path is None:
        raise ValueError("新导入必须使用项目专属 PDF 路径")
    records = project["documents"] + project.get("pending_cleanup", [])
    if any(doc["document_id"] == document["document_id"] for doc in records):
        raise ValueError("文档已存在，或仍有待清理文件；请先完成清理")
    updated = _with_pending(project, project.get("pending_cleanup", []) + [document])
    save_project(updated)
    return updated


def finish_document_import(project_id, document):
    project = _load_managed_project(project_id)
    pending = project.get("pending_cleanup", [])
    staged = next(
        (doc for doc in pending if doc["document_id"] == document["document_id"]), None
    )
    if staged is None or any(
        staged[key] != document[key] for key in ("source", "index_version")
    ):
        raise ValueError("导入记录与待办不匹配")
    if any(doc["document_id"] == document["document_id"] for doc in project["documents"]):
        raise ValueError("项目中已存在这份文档")
    updated = _with_pending(
        project, [doc for doc in pending if doc["document_id"] != document["document_id"]]
    )
    updated["documents"] = project["documents"] + [document]
    save_project(updated)
    return updated
