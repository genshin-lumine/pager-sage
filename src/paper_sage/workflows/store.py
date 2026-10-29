"""工作流运行目录与不可覆盖的输入记录；检查点由 SqliteSaver 管理。"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath
import re
from tempfile import NamedTemporaryFile

from filelock import FileLock, Timeout

from paper_sage.sessions.store import _read_snapshot, normalize_session_scope, restore_session_snapshot
from paper_sage.workflows.checkpoint import prepare_bound_run


def run_paths(project_id, run_id, *, workspace_root=None):
    if (not isinstance(project_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", project_id)
            or PureWindowsPath(project_id).is_reserved()):
        raise ValueError("项目标识不能作为运行目录名")
    if not isinstance(run_id, str) or not re.fullmatch(r"run_[0-9a-f]{32}", run_id):
        raise ValueError("运行编号格式无效")
    root = Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    directory = root / "data" / "checkpoints" / "runs" / project_id / run_id
    paths = {"directory": directory, "context": directory / "context.json",
             "database": directory / "checkpoints.sqlite", "lock": directory / "run.lock",
             "events": directory / "events.jsonl", "events_lock": directory / "events.jsonl.lock"}
    for target in paths.values():
        current = root
        for part in target.relative_to(root).parts:
            current /= part
            if current.is_symlink() or current.is_junction():
                raise ValueError("运行目录不能包含链接或目录联接")
        if not target.resolve().is_relative_to(root):
            raise ValueError("运行存储路径超出工作区")
    return paths


def create_run_storage(checked_context, *, workspace_root=None):
    """输入记录先完整落盘，再允许启动图；同一运行目录绝不覆盖。"""
    prepare_bound_run(checked_context)
    project_id = checked_context["session_snapshot"]["session"]["scope_key"][0]
    paths = run_paths(project_id, checked_context["run_id"], workspace_root=workspace_root)
    payload = json.dumps({"schema_version": 1, "purpose": "paper_sage_workflow_run",
                          "created_at": datetime.now(timezone.utc).isoformat(), "context": checked_context},
                         ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    paths["directory"].parent.mkdir(parents=True, exist_ok=True)
    paths["directory"].mkdir()  # 已存在即失败，保留原运行的输入和数据库。
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=paths["directory"],
                                prefix=".context-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, paths["context"])
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return paths


def load_run_record(project_id, run_id, *, workspace_root=None):
    paths = run_paths(project_id, run_id, workspace_root=workspace_root)
    record = _read_snapshot(paths["context"])
    if (not isinstance(record, dict) or set(record) != {"schema_version", "purpose", "created_at", "context"}
            or type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["purpose"] != "paper_sage_workflow_run"
            or not isinstance(record["created_at"], str)):
        raise ValueError("运行存档格式不支持")
    context = record["context"]
    prepare_bound_run(context)
    if (context["run_id"] != run_id
            or context["session_snapshot"]["session"]["scope_key"][0] != project_id):
        raise ValueError("运行存档与请求的项目或运行编号不一致")
    return record


def list_workflow_runs(scope_key, *, workspace_root=None):
    """只列同范围的运行输入；不加载模型，不把存档存在等同于运行完成。"""
    scope = normalize_session_scope(scope_key)
    parent = run_paths(scope[0], "run_" + "0" * 32, workspace_root=workspace_root)["directory"].parent
    records = []
    for path in parent.glob("run_*/context.json"):
        try:
            record = load_run_record(scope[0], path.parent.name, workspace_root=workspace_root)
            session = restore_session_snapshot(record["context"]["session_snapshot"], scope)
            records.append({"run_id": record["context"]["run_id"], "question": record["context"]["question"],
                            "session_id": session["session_id"], "created_at": record["created_at"]})
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return sorted(records, key=lambda item: (item["created_at"], item["run_id"]), reverse=True)


@contextmanager
def lock_workflow_run(project_id, run_id, *, workspace_root=None):
    paths = run_paths(project_id, run_id, workspace_root=workspace_root)
    if not paths["context"].is_file():
        raise FileNotFoundError("没有完整的运行输入存档")
    # OS 文件锁在进程退出时释放；不依赖手工删除锁文件来恢复。
    lock = FileLock(str(paths["lock"]), timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError("这一轮正在另一个进程中处理，请稍后再试") from error
    try:
        yield paths
    finally:
        lock.release()
