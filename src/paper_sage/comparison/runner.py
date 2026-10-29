"""串联比较计划、单篇检索与结构化生成；由调用方提供延后加载工具的函数。"""

from copy import deepcopy
from pathlib import Path
from time import perf_counter

from paper_sage.sessions.chat import validate_history_messages
from paper_sage.comparison.context import build_comparison_material
from paper_sage.comparison.inputs import prepare_comparison_tasks, validate_comparison_documents
from paper_sage.comparison.model import ComparisonGenerationError, compare_papers
from paper_sage.comparison.plan import ComparisonPlanError, request_comparison_plan, validate_comparison_plan_record
from paper_sage.comparison.retrieval import collect_comparison_evidence


class ComparisonRunError(RuntimeError):
    """保留失败前完整阶段和收到的响应，不把部分结果当作完成。"""

    def __init__(self, message, artifact):
        super().__init__(message)
        self.artifact = deepcopy(artifact)


def run_comparison(question, project_id, documents, model, *, tools_loader, history=None, plan_record=None):
    """tools_loader 是零参数函数，只有明确比较维度后才会调用。

    可以复用与本次输入完全关联的计划记录，但检索始终使用当前计划中的查询。
    本函数不读写报告、不修改会话；成功时返回完整 artifact，失败时抛出异常。
    """
    validate_comparison_documents(project_id, documents)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("用户问题必须是非空字符串")
    if not callable(tools_loader):
        raise ValueError("tools_loader 必须是加载工具的函数，不能是已经加载的工具字典")
    question = question.strip()
    documents = deepcopy(documents)
    history = validate_history_messages(history)
    document_info = [{"document_id": doc["document_id"], "name": doc.get("title") or Path(doc["source"]).name}
                     for doc in documents]
    artifact = {
        "project_id": project_id, "documents": documents, "question": question,
        "history": history, "response_kind": None, "reply": "", "status": "running",
        "plan_reused": plan_record is not None, "plan_record": None,
        "tasks": [], "records": [], "material": None, "generation_record": None,
        "stages": [],
    }
    started = perf_counter()
    stage = "plan"

    def completed(name, stage_start):
        artifact["stages"].append({"stage": name, "elapsed_seconds": perf_counter() - stage_start})

    try:
        stage_started = perf_counter()
        if plan_record is None:
            plan_record = request_comparison_plan(question, document_info, model, history=history)
        artifact["plan_record"] = deepcopy(plan_record)
        plan = validate_comparison_plan_record(plan_record, question, document_info, history=history)
        completed(stage, stage_started)

        stage, stage_started = "tasks", perf_counter()
        prepared = prepare_comparison_tasks(project_id, documents, plan)
        artifact["tasks"] = prepared["tasks"]
        completed(stage, stage_started)
        if prepared["response_kind"] == "clarify":
            artifact.update(response_kind="clarify", reply=prepared["reply"], status="completed",
                            elapsed_seconds=perf_counter() - started)
            return artifact

        # 加载器由外部绑定项目与所选论文；澄清时不会执行这里。
        stage, stage_started = "load_tools", perf_counter()
        tools_by_document = tools_loader()
        if not isinstance(tools_by_document, dict) or set(tools_by_document) != {doc["document_id"] for doc in documents}:
            raise ValueError("工具字典必须恰好覆盖当前两篇论文")
        completed(stage, stage_started)

        stage, stage_started = "retrieve", perf_counter()
        records = collect_comparison_evidence(artifact["tasks"], tools_by_document)
        artifact["records"] = records
        completed(stage, stage_started)

        stage, stage_started = "material", perf_counter()
        dimensions = [item["name"] for item in plan["dimensions"]]
        material = build_comparison_material(project_id, documents, dimensions, records)
        artifact["material"] = material
        completed(stage, stage_started)

        stage, stage_started = "generate", perf_counter()
        artifact["generation_record"] = compare_papers(material, model)
        completed(stage, stage_started)
        artifact.update(response_kind="compare", status="completed", elapsed_seconds=perf_counter() - started)
        return artifact
    except Exception as error:
        if isinstance(error, ComparisonPlanError):
            artifact["plan_record"] = deepcopy(error.record)
        if isinstance(error, ComparisonGenerationError):
            artifact["generation_record"] = deepcopy(error.record)
        artifact.update(status="failed", failed_stage=stage, error_type=type(error).__name__,
                        elapsed_seconds=perf_counter() - started)
        raise ComparisonRunError(f"比较流程在 {stage} 阶段失败（{type(error).__name__}）。", artifact) from error
