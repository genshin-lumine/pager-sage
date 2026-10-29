"""逐次调用的 JSONL 诊断事件；记录失败不能替换业务返回值或异常。"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

from filelock import FileLock


FIELDS = {"schema_version", "run_id", "call_id", "step", "phase", "status",
          "timestamp", "elapsed_seconds", "error_type"}


def validate_event(event, run_id):
    if not isinstance(event, dict) or set(event) != FIELDS:
        raise ValueError("事件字段不符合约定")
    if type(event["schema_version"]) is not int or event["schema_version"] != 1:
        raise ValueError("事件版本不支持")
    if event["run_id"] != run_id or not re.fullmatch(r"run_[0-9a-f]{32}", run_id):
        raise ValueError("事件不属于当前运行")
    if not isinstance(event["call_id"], str) or not re.fullmatch(r"call_[0-9a-f]{32}", event["call_id"]):
        raise ValueError("步骤调用编号无效")
    if not isinstance(event["step"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", event["step"]):
        raise ValueError("步骤名称无效")
    if not isinstance(event["timestamp"], str):
        raise ValueError("事件时间无效")
    stamp = datetime.fromisoformat(event["timestamp"])
    if stamp.utcoffset() is None:
        raise ValueError("事件时间必须带时区")
    if event["phase"] == "start":
        if event["status"] != "running" or event["elapsed_seconds"] is not None or event["error_type"] is not None:
            raise ValueError("开始事件不应包含结束结果")
    elif event["phase"] == "end":
        elapsed = event["elapsed_seconds"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("结束事件耗时无效")
        if event["status"] == "ok":
            if event["error_type"] is not None:
                raise ValueError("成功事件不应包含异常类型")
        elif event["status"] in ("failed", "cancelled"):
            if not isinstance(event["error_type"], str) or not event["error_type"].isidentifier():
                raise ValueError("失败或中断事件缺少异常类型")
        else:
            raise ValueError("结束事件状态无效")
    else:
        raise ValueError("事件阶段无效")


def append_event(path, event):
    """一行一个 JSON 对象；只追加，不重写已有记录。"""
    validate_event(event, event["run_id"])
    encoded = (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=1):
        # 若上次进程留下半行，先以换行隔开，保留坏行供读取方明确报告。
        with path.open("a+b") as file:
            file.seek(0, 2)
            if file.tell():
                file.seek(-1, 2)
                if file.read(1) != b"\n":
                    file.write(b"\n")
            file.write(encoded)
            file.flush()


def read_events(path, run_id):
    """返回合法事件和逐行问题；损坏的日志不修复、不作为图恢复依据。"""
    events, issues = [], []
    path = Path(path)
    if not path.exists():
        return events, issues
    # 与追加方使用相同锁，避免把另一个正常写入者的半行当作损坏。
    with FileLock(str(path) + ".lock", timeout=1):
        lines = path.read_bytes().splitlines()
    for number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line.decode("utf-8"))
            validate_event(event, run_id)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            # 只报行号与异常类型，不回显可能混入日志的正文。
            issues.append({"line": number, "error_type": type(error).__name__})
        else:
            events.append(event)
    return events, issues


def summarize_calls(events):
    """按独立调用配对；缺失和重复事件单独展示，不推测节点是否仍在运行。"""
    groups = {}
    for event in events:
        groups.setdefault(event["call_id"], []).append(event)
    rows = []
    for call_id, items in groups.items():
        first = items[0]
        phases = [item["phase"] for item in items]
        same_identity = all((item["run_id"], item["step"]) == (first["run_id"], first["step"]) for item in items)
        row = {"call_id": call_id, "step": first["step"], "timestamp": first["timestamp"],
               "status": "invalid_pair", "elapsed_seconds": None, "error_type": None}
        if same_identity and phases == ["start", "end"]:
            end = items[-1]
            row.update(status=end["status"], elapsed_seconds=end["elapsed_seconds"], error_type=end["error_type"])
        elif phases == ["start"]:
            row["status"] = "unfinished"
        elif phases == ["end"]:
            row["status"] = "missing_start"
            row["error_type"] = first["error_type"]
        rows.append(row)
    return rows


class RunEventLog:
    def __init__(self, run_id, path, *, clock=perf_counter):
        if not isinstance(run_id, str) or not re.fullmatch(r"run_[0-9a-f]{32}", run_id):
            raise ValueError("日志需要合法的单轮运行编号")
        self.run_id = run_id
        self.path = Path(path)
        self.clock = clock
        self.write_errors = []

    def _emit(self, event):
        try:
            append_event(self.path, event)
        except OSError as error:
            # 日志 IO 错误另行报告，不能掩盖 operation 的结果或原异常。
            self.write_errors.append({"call_id": event["call_id"], "step": event["step"],
                                      "phase": event["phase"], "error_type": type(error).__name__})

    def begin(self, step):
        event = {"schema_version": 1, "run_id": self.run_id, "call_id": "call_" + uuid4().hex,
                 "step": step, "phase": "start", "status": "running",
                 "timestamp": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": None, "error_type": None}
        validate_event(event, self.run_id)
        self._emit(event)
        return event

    def call(self, step, operation):
        event = self.begin(step)
        started = self.clock()
        try:
            result = operation()
        except BaseException as error:
            elapsed = self.clock() - started
            self._emit({**event, "phase": "end",
                        "status": "failed" if isinstance(error, Exception) else "cancelled",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": elapsed, "error_type": type(error).__name__})
            raise
        else:
            elapsed = self.clock() - started
            self._emit({**event, "phase": "end", "status": "ok",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": elapsed, "error_type": None})
            return result
