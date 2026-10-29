"""防止回归报告把子测试失败、跳过或准备失败误报成通过。"""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paper_sage.evaluation.regression import RecordingResult, execute_cases


class RegressionRunnerTests(unittest.TestCase):
    def test_subtest_failure_remains_failed(self):
        class Sample(unittest.TestCase):
            def runTest(self):
                for number in (1, 2):
                    with self.subTest(number=number):
                        self.assertEqual(number, 1)

        result = unittest.TextTestRunner(stream=StringIO(), resultclass=RecordingResult).run(
            unittest.TestSuite([Sample()])
        )
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.records[0]["status"], "failed")
        self.assertEqual(len(result.records[0]["issues"]), 1)

    def test_class_setup_skip_or_error_is_reported_and_not_accepted(self):
        for failure in (unittest.SkipTest("controlled skip"), RuntimeError("controlled setup failure")):
            with self.subTest(error_type=type(failure).__name__):
                class Sample(unittest.TestCase):
                    @classmethod
                    def setUpClass(cls):
                        raise failure

                    def runTest(self):
                        raise AssertionError("must not run")

                test = Sample()
                loader = unittest.TestLoader()
                cases = [{"case_id": "sample", "name": "sample", "expected": "sample", "test_id": test.id()}]
                with TemporaryDirectory(prefix="paper-sage-regression-unit-") as directory:
                    with patch('paper_sage.evaluation.regression.unittest.TestLoader', return_value=loader), \
                         patch.object(loader, "loadTestsFromNames", return_value=unittest.TestSuite([test])), \
                         redirect_stdout(StringIO()):
                        exit_code = execute_cases(cases, "core", "sample", directory)
                    report = json.loads(next(Path(directory).glob("*.json")).read_text(encoding="utf-8"))
                self.assertEqual(exit_code, 1)
                self.assertFalse(report["accepted"])
                self.assertEqual(report["passed_count"], 0)
                self.assertEqual(report["results"][0]["status"], "not_run")
                field = "runner_skips" if isinstance(failure, unittest.SkipTest) else "runner_errors"
                self.assertEqual(len(report[field]), 1)

    def test_incomplete_or_empty_group_does_not_start_tests_or_write_report(self):
        for cases in ([], [{"test_id": None}]):
            with self.subTest(cases=cases), TemporaryDirectory(prefix="paper-sage-regression-unit-") as directory:
                with patch('paper_sage.evaluation.regression.unittest.TestLoader') as loader:
                    with self.assertRaises(ValueError):
                        execute_cases(cases, "documents", "sample", directory)
                    loader.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
