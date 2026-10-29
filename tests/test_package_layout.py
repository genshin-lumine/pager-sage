"""Protect package installation, workspace paths, and nested source fingerprinting."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from paper_sage.evaluation.demo import source_fingerprints
from paper_sage.retrieval.reranker import get_rerank_cache_folder
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutTests(unittest.TestCase):
    def test_package_is_importable_outside_repository(self):
        with tempfile.TemporaryDirectory(prefix="paper-sage-package-import-") as directory:
            result = subprocess.run([sys.executable, "-c",
                "from paper_sage.documents.chunking import split_pages; "
                "from paper_sage.sessions.chat import create_chat_session; "
                "assert callable(split_pages) and callable(create_chat_session)"],
                cwd=directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_package_does_not_import_old_root_or_practice_modules(self):
        import json
        mapping = json.loads((ROOT / "docs/layout_migration.json").read_text(encoding="utf-8"))["formal_modules"]
        for path in (ROOT / "src/paper_sage").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                           else [item.name for item in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    self.assertNotIn(module.split(".")[0], mapping, str(path))
                    self.assertFalse(module.startswith("practice_"), str(path))

    def test_default_cache_configuration_follows_workspace(self):
        with tempfile.TemporaryDirectory(prefix="paper-sage-package-config-") as directory:
            root = Path(directory)
            (root / "config.local.toml").write_text('[rerank]\ncache_folder = "cache"\n', encoding="utf-8")
            with patch("paper_sage.retrieval.reranker.Path.cwd", return_value=root):
                self.assertEqual(Path(get_rerank_cache_folder()), root / "cache")

    def test_demo_fingerprint_covers_nested_module_changes(self):
        with tempfile.TemporaryDirectory(prefix="paper-sage-package-hash-") as directory:
            root = Path(directory)
            for name in ("app.py", "pyproject.toml", "requirements.txt", "scripts/demo.py", "src/paper_sage/qa/model.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original", encoding="utf-8")
            before = source_fingerprints(root)
            (root / "src/paper_sage/qa/model.py").write_text("changed", encoding="utf-8")
            self.assertNotEqual(before, source_fingerprints(root))
            self.assertIn("scripts/demo.py", before)


if __name__ == "__main__":
    unittest.main()
