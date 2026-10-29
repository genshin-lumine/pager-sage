"""绑定真实项目、索引与模型加载函数，执行比较并单独保存本次报告。"""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from openai import APIError

from paper_sage.comparison.inputs import validate_comparison_documents
from paper_sage.comparison.model import resolve_comparison_sources
from paper_sage.comparison.report import validate_comparison_artifact
from paper_sage.comparison.review import apply_comparison_review
from paper_sage.comparison.runner import ComparisonRunError, run_comparison
from paper_sage.comparison.tools import create_comparison_tools, load_comparison_indexes
from paper_sage.projects.store import get_index_dir


class ComparisonServiceError(RuntimeError):
    def __init__(self, message, report_path):
        super().__init__(message)
        self.report_path = report_path


def _now():
    return datetime.now(timezone.utc).isoformat()


def _component(value):
    if (not isinstance(value, str) or not value.strip() or value in {".", ".."}
            or any(character in value for character in '/\\:')):
        raise ValueError("项目、文档和索引版本必须是有效的目录标识")
    return value


def _within_root(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("比较数据路径超出当前工作区")
    return path


def _select_documents(project, documents):
    if not isinstance(project, dict) or not isinstance(project.get("documents"), list):
        raise ValueError("需要有效的项目记录")
    _component(project.get("project_id"))
    if not isinstance(documents, list) or not documents:
        raise ValueError("需要所选文档记录")
    stored = {}
    for doc in project["documents"]:
        if not isinstance(doc, dict):
            raise ValueError("项目文档记录必须是字典")
        doc_id = _component(doc.get("document_id"))
        if doc_id in stored:
            raise ValueError("项目包含重复文档标识")
        stored[doc_id] = doc
    selected, seen = [], set()
    for doc in documents:
        if not isinstance(doc, dict):
            raise ValueError("所选文档记录必须是字典")
        doc_id = _component(doc.get("document_id"))
        _component(doc.get("index_version"))
        if not isinstance(doc.get("source"), str) or not doc["source"].strip():
            raise ValueError("所选文档缺少来源")
        if doc_id in seen or doc_id not in stored:
            raise ValueError("所选文档重复或不属于当前项目")
        if any(doc.get(field) != stored[doc_id].get(field) for field in ("source", "index_version")):
            raise ValueError("所选文档的来源或索引版本与项目记录不一致")
        selected.append(deepcopy(stored[doc_id]))
        seen.add(doc_id)
    return selected


def _check_current_project(root, project_id, documents):
    path = _within_root(root, Path("data/projects") / project_id / "project.json")
    project = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(project, dict) or project.get("project_id") != project_id:
        raise ValueError("当前项目记录与绑定范围不一致")
    if _select_documents(project, documents) != documents:
        raise ValueError("所选论文记录已经变化，请重新选择后提交")


def _fingerprints(root, project_id, documents):
    records = []
    for document in documents:
        index_dir = get_index_dir(project_id, document["document_id"], document["index_version"])
        paths = [Path(document["source"]), index_dir / "metadata.json", index_dir / "index.faiss"]
        files = []
        for value in paths:
            path = _within_root(root, value)
            digest = sha256(path.read_bytes()).hexdigest()
            files.append({"path": path.relative_to(root).as_posix(), "sha256": digest})
        if document.get("file_hash") and files[0]["sha256"] != document["file_hash"]:
            raise ValueError("PDF 内容与项目中登记的文件摘要不一致")
        records.append({"document_id": document["document_id"], "index_version": document["index_version"], "files": files})
    return records


def _write_report(path, report, *, new=False):
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if new:
        with path.open("x", encoding="utf-8") as file:
            file.write(payload)
        return
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.stem + "_",
                                suffix=".tmp", delete=False) as file:
            temporary = Path(file.name).resolve()
            file.write(payload)
        if temporary.parent != path.parent:
            raise ValueError("报告临时文件不在目标目录")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.parent == path.parent:
            temporary.unlink(missing_ok=True)


def load_saved_comparison(report_path, project, documents, *, workspace_root=None, review_path=None):
    """显式读取已完成的比较，并可在副本应用修订；不运行模型、检索或写入文件。

    只接受与当前项目、PDF 和索引一致的报告；这是当前工作区的复用入口，
    不是查看已删除或更换来源的历史档案入口，也不证明比较文字事实正确。
    """
    root = Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    project = deepcopy(project)
    documents = _select_documents(project, deepcopy(documents))
    project_id = project["project_id"]
    validate_comparison_documents(project_id, documents)
    _check_current_project(root, project_id, documents)
    path = _within_root(root, report_path)
    raw = path.read_bytes()
    report = json.loads(raw)
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("purpose") != "day34_comparison_run" or report.get("mode") != "real"
            or report.get("status") != "completed_comparison" or report.get("accepted") is not True
            or report.get("error_type") or report.get("failed_stage") or report.get("error")
            or report.get("project_id") != project_id or report.get("documents") != documents):
        raise ValueError("报告未完成比较或不属于当前项目与所选论文")
    artifact = deepcopy(report.get("result"))
    if (not isinstance(artifact, dict) or artifact.get("project_id") != project_id
            or artifact.get("documents") != documents or artifact.get("response_kind") != "compare"
            or artifact.get("report_path") != path.relative_to(root).as_posix()
            or "content_review" in artifact):
        raise ValueError("需要与报告路径及当前论文一致的原始比较产物")
    validate_comparison_artifact(artifact)
    fingerprints = _fingerprints(root, project_id, documents)
    if report.get("input_files") != fingerprints or artifact.get("input_files") != fingerprints:
        raise ValueError("当前 PDF 或索引与原始比较报告不一致")
    resources = load_comparison_indexes(project, documents, workspace_root=root)
    for record in artifact["records"]:
        chunks = resources[record["task"]["document_id"]]["metadata"]["chunks"]
        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        if len(by_id) != len(chunks):
            raise ValueError("当前索引包含重复片段标识")
        retrieval = record["retrieval_artifact"]
        lists = [retrieval.get(key, []) for key in
                 ("retrieval_results", "retrieval_candidates", "rrf_results", "rerank_results")]
        lists.extend(retrieval.get("retrieval_channels", {}).values())
        lists.extend(stage["results"] for stage in retrieval.get("retrieval_trace", {}).get("stages", []))
        for results in lists:
            for result in results:
                chunk = result["chunk"]
                if by_id.get(chunk["chunk_id"]) != chunk:
                    raise ValueError("报告片段与当前索引原文或元数据不一致")
    artifact["source_report_sha256"] = sha256(raw).hexdigest()
    if review_path is not None:
        review_file = _within_root(root, review_path)
        review_bytes = review_file.read_bytes()
        review = json.loads(review_bytes)
        if (not isinstance(review, dict) or review.get("source_report") != artifact["report_path"]
                or review.get("source_report_sha256") != artifact["source_report_sha256"]):
            raise ValueError("修订记录不对应这份原始比较报告")
        reviewed = apply_comparison_review(artifact["generation_record"]["comparison"], artifact["material"], review)
        artifact["content_review"] = {
            "path": review_file.relative_to(root).as_posix(), "sha256": sha256(review_bytes).hexdigest(),
            "record": review, "comparison": reviewed,
            "resolved_comparison": resolve_comparison_sources(reviewed, artifact["material"]),
        }
    validate_comparison_artifact(artifact)
    _check_current_project(root, project_id, documents)
    if _fingerprints(root, project_id, documents) != fingerprints:
        raise ValueError("读取期间 PDF 或索引发生变化")
    return artifact


def create_comparison_runner(project, documents, model, *, embedding_model_loader, rerank_model_loader,
                             workspace_root=None, on_progress=None):
    """仅绑定配置；返回的函数才执行计划、检索与生成。模型加载函数由调用方缓存。

    embedding_model_loader 接收索引中的模型名称；rerank_model_loader 不接收参数。
    每次执行新建报告，不自动复用旧检索或比较；可显式传入经核对的 plan_record。
    """
    root = Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    project = deepcopy(project)
    documents = _select_documents(project, deepcopy(documents))
    project_id = project["project_id"]
    for document in documents:
        _within_root(root, document["source"])
        _within_root(root, get_index_dir(project_id, document["document_id"], document["index_version"]))
    if not callable(embedding_model_loader) or not callable(rerank_model_loader):
        raise ValueError("需要传入向量模型和精排模型的加载函数")
    if on_progress is not None and not callable(on_progress):
        raise ValueError("进度处理函数必须可调用")

    def progress(message):
        if on_progress is not None:
            on_progress(message)

    def run(question, *, history=None, plan_record=None):
        validate_comparison_documents(project_id, documents)
        _check_current_project(root, project_id, documents)
        directory = _within_root(root, "data/evaluations")
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = directory / f"day34_comparison_run_{stamp}_{uuid4().hex[:8]}.json"
        relative_path = path.relative_to(root).as_posix()
        report = {"schema_version": 1, "purpose": "day34_comparison_run", "mode": "real",
                  "status": "in_progress", "recorded_at": _now(), "project_id": project_id,
                  "documents": deepcopy(documents), "input_files": [], "result": None,
                  "accepted": False, "semantic_review": "pending", "error_type": None}
        _write_report(path, report, new=True)
        artifact = None
        fingerprints = []

        def load_tools():
            nonlocal fingerprints
            progress("比较维度已明确，正在读取两篇论文的独立索引……")
            _check_current_project(root, project_id, documents)
            fingerprints = _fingerprints(root, project_id, documents)
            report["input_files"] = fingerprints
            resources = load_comparison_indexes(project, documents, workspace_root=root)
            configs = [resource["metadata"]["embedding_config"] for resource in resources.values()]
            if any(config != configs[0] for config in configs):
                raise ValueError("两篇索引的编码配置不一致，无法共用向量模型")
            progress("正在加载本地检索模型……")
            embedding = embedding_model_loader(configs[0]["model_name"])
            reranker = rerank_model_loader()
            tools = create_comparison_tools(resources, embedding, reranker)
            progress("正在逐篇检索并生成比较……")
            return tools

        service_stage = "run"
        try:
            progress("正在核对已有比较计划……" if plan_record is not None else "正在识别比较维度……")
            artifact = run_comparison(question, project_id, documents, model, tools_loader=load_tools,
                                      history=history, plan_record=plan_record)
            service_stage = "validate_result"
            validate_comparison_artifact(artifact)
            service_stage = "verify_inputs"
            _check_current_project(root, project_id, documents)
            if fingerprints and _fingerprints(root, project_id, documents) != fingerprints:
                raise ValueError("运行期间 PDF 或索引发生变化，请重新提交比较")
            artifact["report_path"] = relative_path
            artifact["input_files"] = deepcopy(fingerprints)
            report.update(result=artifact, status="completed_comparison" if artifact["response_kind"] == "compare" else "completed_clarification",
                          accepted=True, updated_at=_now())
            service_stage = "save_report"
            _write_report(path, report)
        except Exception as error:
            if isinstance(error, ComparisonRunError):
                artifact = error.artifact
            elif isinstance(artifact, dict):
                artifact.update(status="failed", failed_stage=service_stage, error_type=type(error).__name__)
            cause = error.__cause__ if isinstance(error, ComparisonRunError) and error.__cause__ is not None else error
            error_text = (f"{type(cause).__name__}，HTTP 状态：{getattr(cause, 'status_code', None)}"
                          if isinstance(cause, APIError) else f"{type(cause).__name__}: {cause}")
            failed_stage = artifact.get("failed_stage", service_stage) if isinstance(artifact, dict) else service_stage
            report.update(status="failed", result=artifact, accepted=False, error_type=type(error).__name__,
                          error=error_text, failed_stage=failed_stage, updated_at=_now())
            _write_report(path, report)
            raise ComparisonServiceError(f"论文比较在 {failed_stage} 阶段失败：{error_text}。记录已保存：{relative_path}", relative_path) from error
        progress("比较记录已保存。" if artifact["response_kind"] == "compare" else "需要补充比较维度，追问记录已保存。")
        return artifact

    return run
