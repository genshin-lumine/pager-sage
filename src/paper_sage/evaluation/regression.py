"""固定场景回归入口：显式列出或执行，复用临时项目与受控模型测试。"""

import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import platform
import sys
from time import perf_counter
import unittest
from uuid import uuid4


ROOT = Path.cwd().resolve()
PLAN_PATH = ROOT / "tests/regression_cases.json"
GROUPS = ("core", "documents", "workflow")


def load_plan():
    raw = PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    if plan.get("schema_version") != 1 or plan.get("purpose") != "day41_regression_plan":
        raise ValueError("回归清单版本不支持")
    ids, test_ids = set(), set()
    for case in plan["cases"]:
        if case["case_id"] in ids or case["group"] not in GROUPS:
            raise ValueError("场景编号重复或分组无效")
        ids.add(case["case_id"])
        if not all(isinstance(case[key], str) and case[key].strip() for key in ("name", "trigger", "expected")):
            raise ValueError("场景缺少名称、触发条件或预期")
        test_id = case["test_id"]
        if test_id is None:
            continue
        if test_id in test_ids:
            raise ValueError("同一测试不能重复登记为多个独立场景")
        test_ids.add(test_id)
        module, class_name, method_name = test_id.split(".")
        if not module.startswith("test_") or not module.isidentifier():
            raise ValueError("清单只支持 tests 下的固定测试模块")
        tree = ast.parse((ROOT / "tests" / f"{module}.py").read_text(encoding="utf-8"))
        if not any(isinstance(node, ast.ClassDef) and node.name == class_name
                   and any(isinstance(item, ast.FunctionDef) and item.name == method_name for item in node.body)
                   for node in tree.body):
            raise ValueError(f"登记的测试不存在：{test_id}")
    return plan, sha256(raw).hexdigest()


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.current = None

    def startTest(self, test):
        super().startTest(test)
        self.started = perf_counter()
        self.current = {"test_id": test.id(), "status": "running", "issues": []}

    def addSuccess(self, test):
        super().addSuccess(test)
        if self.current["status"] == "running":
            self.current["status"] = "passed"

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.current.update(status="failed")
        self.current["issues"].append(self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        if self.current is not None:
            self.current.update(status="error")
            self.current["issues"].append(self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.current is not None:
            self.current.update(status="skipped")
            self.current["issues"].append(reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.current.update(status="expected_failure")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.current.update(status="unexpected_success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.current.update(status="failed" if issubclass(err[0], test.failureException) else "error")
            self.current["issues"].append(self._exc_info_to_string(err, subtest))

    def stopTest(self, test):
        self.current["elapsed_seconds"] = perf_counter() - self.started
        self.records.append(self.current)
        self.current = None
        super().stopTest(test)


def execute_cases(cases, group, plan_digest, output_dir):
    if not cases:
        raise ValueError("没有可执行的场景")
    if any(case["test_id"] is None for case in cases):
        raise ValueError("所选组仍有未覆盖场景，请先补齐用例；不会执行一部分后把整组标为通过")
    sys.path.insert(0, str(ROOT / "tests"))
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames([case["test_id"] for case in cases])
    if loader.errors:
        raise ValueError("测试加载失败：\n" + "\n".join(loader.errors))
    if suite.countTestCases() != len(cases):
        raise ValueError("加载的测试数量与所选场景不一致")
    print("模式：固定回归场景、临时项目与受控模型，不请求 DeepSeek。", flush=True)
    logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2, resultclass=RecordingResult).run(suite)
    finally:
        logger.setLevel(old_level)
    by_id = {record["test_id"]: record for record in result.records}
    records = [{"case_id": case["case_id"], "name": case["name"], "expected": case["expected"],
                **by_id.get(case["test_id"], {"test_id": case["test_id"], "status": "not_run", "issues": []})}
               for case in cases]
    passed = sum(record["status"] == "passed" for record in records)
    accepted = result.wasSuccessful() and passed == len(cases)
    created = datetime.now(timezone.utc)
    report = {"schema_version": 1, "purpose": "day41_regression_report",
              "created_at": created.isoformat(), "python_version": platform.python_version(),
              "mode": "offline_controlled_tests", "group": group, "plan_sha256": plan_digest,
              "selected_count": len(cases), "passed_count": passed, "accepted": accepted,
              "runner_errors": [{"test_id": test.id(), "traceback": detail} for test, detail in result.errors],
              "runner_skips": [{"test_id": test.id(), "reason": reason} for test, reason in result.skipped],
              "results": records}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"day41_regression_{created.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}.json"
    with path.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(f"所选场景：{len(cases)} · 通过：{passed} · 未通过或未执行：{len(cases) - passed}")
    print("本组通过：", accepted)
    print("报告已保存：", path.as_posix())
    print("固定响应检查验证程序行为，不评价真实模型回答质量；本组结果不代表未选择的场景也通过。")
    return 0 if accepted else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="只列出清单与覆盖准备状态")
    action.add_argument("--run", choices=(*GROUPS, "all"), help="执行所选固定场景")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluations")
    args = parser.parse_args()
    plan, digest = load_plan()
    if args.list:
        ready = sum(case["test_id"] is not None for case in plan["cases"])
        for case in plan["cases"]:
            state = "已有用例，尚未在本次执行" if case["test_id"] else "待补用例"
            print(f"[{case['group']}] {case['case_id']} · {case['name']} · {state}")
            print("  预期：", case["expected"])
        print(f"场景总数：{len(plan['cases'])} · 已有用例：{ready} · 待补用例：{len(plan['cases']) - ready}")
        print("仅列出清单，未导入测试模块、执行测试或生成报告；已有用例不等于本次通过。")
        return 0
    cases = [case for case in plan["cases"] if args.run == "all" or case["group"] == args.run]
    return execute_cases(cases, args.run, digest, args.output_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"回归入口未完成：{error}", file=sys.stderr)
        raise SystemExit(2)
