"""Destructive operations run only inside a new temporary workspace."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import paper_sage.documents.deleter as deletion
from paper_sage.projects.store import load_project, save_project


class DocumentDeletionTests(unittest.TestCase):
    def setUp(self):
        self.original_cwd = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory(prefix="paper-sage-delete-")
        self.root = Path(self.temporary.name).resolve()
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        assert self.root.name.startswith("paper-sage-delete-")
        os.chdir(self.root)
        self.a = self.make_document("project_a", "paper_a")
        self.b = self.make_document("project_a", "paper_b")
        other = self.make_document("project_b", "paper_b")
        save_project({"project_id": "project_a", "name": "A", "documents": [self.a, self.b]})
        save_project({"project_id": "project_b", "name": "B", "documents": [other]})
        Path("data/fixtures").mkdir()
        Path("data/fixtures/original.pdf").write_bytes(b"fixture master")

    def tearDown(self):
        os.chdir(self.original_cwd)
        assert self.root.resolve() == self.root
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        assert self.root.name.startswith("paper-sage-delete-")
        self.temporary.cleanup()

    def make_document(self, project_id, document_id):
        source = Path("data/papers") / project_id / document_id / "paper.pdf"
        source.parent.mkdir(parents=True)
        source.write_bytes(f"{project_id}:{document_id}".encode())
        for version in ("v1", "v2"):
            index = Path("data/indexes") / project_id / document_id / version
            index.mkdir(parents=True)
            (index / "index.faiss").write_bytes(b"test index")
        return {"document_id": document_id, "source": source.as_posix(), "index_version": "v1"}

    def snapshot(self):
        return {p.as_posix(): p.read_bytes() for p in Path("data").rglob("*") if p.is_file()}

    def test_delete_all_versions_and_preserve_other_data(self):
        before = self.snapshot()
        updated, error = deletion.delete_document("project_a", "paper_b")
        self.assertIsNone(error)
        self.assertEqual(updated["documents"], [self.a])
        self.assertNotIn("pending_cleanup", updated)
        self.assertEqual(load_project("project_a"), updated)
        self.assertFalse(Path(self.b["source"]).exists())
        self.assertFalse(Path("data/indexes/project_a/paper_b").exists())
        for path, content in before.items():
            if path.startswith(("data/papers/project_a/paper_b/", "data/indexes/project_a/paper_b/")):
                continue
            if path == "data/projects/project_a/project.json":
                continue
            self.assertEqual(Path(path).read_bytes(), content, path)

    def test_first_save_failure_changes_nothing(self):
        before = self.snapshot()
        with patch.object(deletion, "save_project", side_effect=OSError("save blocked")):
            with self.assertRaises(OSError):
                deletion.delete_document("project_a", "paper_b")
        self.assertEqual(self.snapshot(), before)

    def test_partial_cleanup_survives_reload_and_retries(self):
        with patch.object(deletion.shutil, "rmtree", side_effect=PermissionError("index busy")):
            updated, error = deletion.delete_document("project_a", "paper_b")
        self.assertIn("index busy", error)
        self.assertEqual(updated["documents"], [self.a])
        self.assertFalse(Path(self.b["source"]).exists())
        self.assertTrue(Path("data/indexes/project_a/paper_b/v2").exists())
        self.assertEqual(load_project("project_a")["pending_cleanup"], [self.b])
        with self.assertRaises(ValueError):
            deletion.begin_document_import("project_a", self.b)
        updated, error = deletion.retry_document_cleanup("project_a", "paper_b")
        self.assertIsNone(error)
        self.assertNotIn("pending_cleanup", updated)
        self.assertFalse(Path("data/indexes/project_a/paper_b").exists())

    def test_final_save_failure_leaves_retryable_job(self):
        calls = 0

        def fail_second_save(project):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("final save blocked")
            save_project(project)

        with patch.object(deletion, "save_project", side_effect=fail_second_save):
            _, error = deletion.delete_document("project_a", "paper_b")
        self.assertIn("final save blocked", error)
        self.assertFalse(Path("data/indexes/project_a/paper_b").exists())
        self.assertEqual(load_project("project_a")["pending_cleanup"], [self.b])
        _, error = deletion.retry_document_cleanup("project_a", "paper_b")
        self.assertIsNone(error)

    def test_shared_pdf_preserved_when_last_document_deleted(self):
        shared = Path("data/papers/shared.pdf")
        shared.write_bytes(b"public original")
        project = load_project("project_a")
        document = {**self.a, "source": shared.as_posix()}
        project["documents"] = [document]
        save_project(project)
        updated, error = deletion.delete_document("project_a", "paper_a")
        self.assertIsNone(error)
        self.assertEqual(updated["documents"], [])
        self.assertEqual(shared.read_bytes(), b"public original")

    def test_unsafe_source_and_identifiers_rejected_before_save(self):
        for source in (
            "data/papers/project_b/paper_b/paper.pdf",
            "data/fixtures/original.pdf",
            "../outside.pdf",
            "data/papers/project_a/paper_b/../paper_a/paper.pdf",
        ):
            with self.subTest(source=source):
                project = load_project("project_a")
                project["documents"] = [self.a, {**self.b, "source": source}]
                save_project(project)
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    deletion.delete_document("project_a", "paper_b")
                self.assertEqual(self.snapshot(), before)
        for identifier in ("../project_a", "CON", "project_a/other", "C:outside"):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                deletion.delete_document(identifier, "paper_b")

    def test_unknown_or_duplicate_record_rejected(self):
        with self.assertRaises(ValueError):
            deletion.delete_document("project_a", "missing")
        project = load_project("project_a")
        project["documents"].append(self.b)
        save_project(project)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            deletion.delete_document("project_a", "paper_b")
        self.assertEqual(self.snapshot(), before)

    def test_import_journal_success_and_failed_import_cleanup(self):
        document = {"document_id": "paper_c", "source": "data/papers/project_a/paper_c/new.pdf", "index_version": "v1"}
        deletion.begin_document_import("project_a", document)
        self.assertEqual(load_project("project_a")["pending_cleanup"], [document])
        path = Path(document["source"])
        path.parent.mkdir(parents=True)
        path.write_bytes(b"partial import")
        updated, error = deletion.retry_document_cleanup("project_a", "paper_c")
        self.assertIsNone(error)
        self.assertFalse(path.exists())
        self.assertNotIn("pending_cleanup", updated)
        deletion.begin_document_import("project_a", document)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"complete import")
        updated = deletion.finish_document_import("project_a", document)
        self.assertEqual(updated["documents"][-1], document)
        self.assertNotIn("pending_cleanup", updated)
        with self.assertRaises(ValueError):
            deletion.retry_document_cleanup("project_a", "paper_c")

    def test_missing_files_are_already_clean(self):
        deletion._cleanup_files("project_a", self.b)
        updated, error = deletion.delete_document("project_a", "paper_b")
        self.assertIsNone(error)
        self.assertEqual(updated["documents"], [self.a])

    def test_index_junction_cannot_delete_other_document(self):
        link = Path("data/indexes/project_a/paper_b/foreign").absolute()
        target = Path("data/indexes/project_b/paper_b").resolve()
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                check=True, capture_output=True,
            )
        else:
            link.symlink_to(target, target_is_directory=True)
        try:
            with self.assertRaises(ValueError):
                deletion.delete_document("project_a", "paper_b")
            self.assertEqual(len(load_project("project_a")["documents"]), 2)
            self.assertTrue((target / "v1/index.faiss").exists())
            self.assertTrue(Path(self.b["source"]).exists())
        finally:
            # Remove the junction itself, never recurse through its target.
            if os.name == "nt":
                link.rmdir()
            else:
                link.unlink()


if __name__ == "__main__":
    unittest.main()
