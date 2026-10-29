"""项目范围验收：真实页面与 FAISS，控制向量与模拟回答，无模型调用。"""
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

WORKSPACE = Path(__file__).resolve().parents[1]

import faiss
import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest

from paper_sage.documents.chunking import split_pages
from paper_sage.retrieval.index_store import save_index
from paper_sage.documents.pdf import load_pdf_pages
from paper_sage.projects.store import get_index_dir, load_project, save_project
from paper_sage.retrieval.scope import load_scope_index
import paper_sage.retrieval.reranker as reranker


def assert_results_in_scope(results, project_id, selected_document_ids):
    for result in results:
        chunk = result["chunk"]

        # 片段的 project_id 必须与传入的 project_id 相同
        assert chunk["project_id"] == project_id, "片段来自其他项目"

        # 片段的 document_id 必须属于 selected_document_ids
        assert chunk["document_id"] in selected_document_ids, "片段来自未选中的文档"


def data_snapshot(root):
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


class ProjectScopeTests(unittest.TestCase):
    observations = []

    @classmethod
    def setUpClass(cls):
        cls.data_root = WORKSPACE / "data"
        cls.original_data = data_snapshot(cls.data_root)
        cls.observations = []

    @classmethod
    def tearDownClass(cls):
        if data_snapshot(cls.data_root) != cls.original_data:
            raise AssertionError("实际工作区的数据发生了变化")

    def setUp(self):
        self.previous_cwd = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory(prefix="paper-sage-scope-")
        self.root = Path(self.temporary.name).resolve()
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        assert self.root.name.startswith("paper-sage-scope-")
        self.patches = ExitStack()
        self.addCleanup(self.cleanup_workspace)
        os.chdir(self.root)
        self.requests = []
        self.classifications = []
        self.tool_requests = []
        self.answer_protocols = []
        self.route_override = None
        self.classify_failure = None
        self.request_failure = None
        self.search_query_fn = lambda question: question
        self.answer_finish_reason = "stop"
        self.rewrite_requests = []
        self.rewrite_override = {"query": "verification code test document", "reason": "固定改写，仅检查流程"}
        self.rewrite_failure = None
        self.summary_requests = []
        self.summary_failure = None
        self.queries = []
        self.encode_calls = 0
        self.model_instances = 0
        self.rerank_model_instances = 0
        self.rerank_calls = []
        self.rerank_score_fn = None
        self.rerank_failure = None
        self.project_a, self.project_b = "scope_project_a", "scope_project_b"
        manifest = json.loads(
            (WORKSPACE / "data/fixtures/day21/manifest.json").read_text(encoding="utf-8")
        )
        self.question = manifest["question"]
        self.fixture_a, self.fixture_b = manifest["documents"]
        self.doc_a = self.seed_document(self.project_a, self.fixture_a, "scope_note.pdf", 0)
        self.doc_b_in_a = self.seed_document(self.project_a, self.fixture_b, "scope_note_b.pdf", 1)
        self.doc_b = self.seed_document(self.project_b, self.fixture_b, "scope_note.pdf", 1)
        self.aid, self.bid = self.doc_a["document_id"], self.doc_b["document_id"]
        save_project({"project_id": self.project_a, "name": "Scope A", "documents": [self.doc_a, self.doc_b_in_a]})
        save_project({"project_id": self.project_b, "name": "Scope B", "documents": [self.doc_b]})

        # Reimport the page with this test's controlled dependencies.
        sys.modules.pop("paper_sage.ui.app", None)
        owner = self

        class ControlModel:
            def __init__(self, model_name, device):
                owner.model_instances += 1

            def encode(self, texts, **kwargs):
                owner.assertTrue(all(text.startswith("query: ") for text in texts))
                owner.encode_calls += 1
                # 故意让 B 文档的相似度最高，验证仅选 A 时仍不会返回 B。
                vectors = np.zeros((len(texts), 384), dtype="float32")
                vectors[:, 1] = 1.0
                return vectors

        def simulated_answer(messages, api_key):
            self.assertEqual(api_key, "test-only-key")
            self.requests.append(deepcopy(messages))
            match = re.search(
                r"The verification code of this test document is (PROJECT-[AB]-\d+)\.",
                messages[-1]["content"],
            )
            self.assertIsNotNone(match)
            return {"answer": match.group(1), "evidence": [match.group(0)], "insufficient_reason": ""}

        def controlled_rerank(pairs, **kwargs):
            self.rerank_calls.append(deepcopy(pairs))
            if self.rerank_failure is not None:
                raise self.rerank_failure
            if self.rerank_score_fn is not None:
                return self.rerank_score_fn(pairs)
            # 范围测试默认保持 RRF 顺序；专门的精排场景覆盖改序。
            return -np.arange(1, len(pairs) + 1, dtype="float32")

        def controlled_rerank_loader(cache_folder=None, local_files_only=False):
            self.assertTrue(local_files_only)
            self.rerank_model_instances += 1
            return types.SimpleNamespace(predict=controlled_rerank)

        fake_embedding = types.ModuleType("sentence_transformers")
        fake_embedding.SentenceTransformer = ControlModel
        fake_llm = types.ModuleType('paper_sage.llm.client')
        # 仅供测试替换最终回答；页面使用 create_chat_model 和真实图。
        fake_llm.answer_response = simulated_answer
        fake_llm.IncompleteAnswerError = type("IncompleteAnswerError", (Exception,), {})

        def simulated_cited_answer(messages, api_key):
            # 将原有固定回答场景映射成新协议；这是测试替身，不是应用的兼容转换。
            roles = {"system": "system", "human": "user", "ai": "assistant"}
            records = [{"role": roles[m.type], "content": m.content} for m in messages]
            result = fake_llm.answer_response(records, api_key)
            if set(result) != {"answer", "evidence", "insufficient_reason"}:
                return result  # 故意无效的测试响应继续保持无效。
            if result["insufficient_reason"]:
                return {"claims": [], "insufficient_reason": result["insufficient_reason"]}
            reference = json.loads(messages[-1].content)["reference"]
            ids = [item["chunk_id"] for item in reference if any(quote in item["text"] for quote in result["evidence"])]
            return {"claims": [{"text": result["answer"], "chunk_ids": ids}], "insufficient_reason": ""}

        fake_llm.cited_answer_response = simulated_cited_answer

        class ControlledChatModel:
            def __init__(self, api_key):
                owner.assertEqual(api_key, "test-only-key")
                self.api_key = api_key

            def bind_tools(self, tools, **kwargs):
                owner.assertEqual(set(tools[0].args), {"question"})

                def request(messages):
                    owner.tool_requests.append(deepcopy(messages))
                    if owner.request_failure:
                        raise owner.request_failure
                    return AIMessage(content="", tool_calls=[{
                        "name": tools[0].name,
                        "args": {"question": owner.search_query_fn(messages[-1].content)},
                        "id": "controlled_call", "type": "tool_call",
                    }], response_metadata={"finish_reason": "tool_calls"})
                return RunnableLambda(request)

            def bind(self, **kwargs):
                owner.assertIn(kwargs, [
                    {"response_format": {"type": "json_object"}},
                    {"response_format": {"type": "json_object"}, "max_tokens": 2048},
                    {"response_format": {"type": "json_object"}, "max_tokens": 4096},
                    {"response_format": {"type": "json_object"}, "max_tokens": 512},
                ])

                def respond(messages):
                    if kwargs.get("max_tokens") == 512:
                        owner.rewrite_requests.append(json.loads(messages[-1].content))
                        if owner.rewrite_failure:
                            raise owner.rewrite_failure
                        return AIMessage(content=json.dumps(owner.rewrite_override, ensure_ascii=False),
                                         response_metadata={"finish_reason": "stop"})
                    if isinstance(messages[-1], ToolMessage):
                        owner.answer_protocols.append(deepcopy(messages))
                        # 原有检索验收读取正文；另外保留完整协议供 Agent 验收核对。
                        roles = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}
                        records = [{"role": roles[m.type], "content": m.content} for m in messages]
                        result = fake_llm.answer_response(records, self.api_key)
                        return AIMessage(content=json.dumps(result, ensure_ascii=False),
                                         response_metadata={"finish_reason": owner.answer_finish_reason})
                    payload = json.loads(messages[-1].content)
                    if "reference" in payload and "question" in payload:
                        owner.answer_protocols.append(deepcopy(messages))
                        result = fake_llm.cited_answer_response(messages, self.api_key)
                        return AIMessage(content=json.dumps(result, ensure_ascii=False),
                                         response_metadata={"finish_reason": owner.answer_finish_reason})
                    if "parts" in payload or "notes" in payload:
                        owner.summary_requests.append(deepcopy(payload))
                        if owner.summary_failure:
                            raise owner.summary_failure
                        result = {field: [] for field in ("research_question", "method", "results", "limitations")}
                        if "parts" in payload:
                            text = "\n".join(part["text"] for part in payload["parts"])
                            match = re.search(r"The verification code of this test document is (PROJECT-[AB]-\d+)\.", text)
                            owner.assertIsNotNone(match)
                            result["method"] = [{"text": f"测试文档提供的校验码是 {match.group(1)}。",
                                                 "part_ids": [part["part_id"] for part in payload["parts"]]}]
                        else:
                            result["method"] = [{"text": payload["notes"][0]["text"],
                                                 "note_ids": [payload["notes"][0]["note_id"]]}]
                        return AIMessage(content=json.dumps(result, ensure_ascii=False), response_metadata={"finish_reason": "stop"})
                    owner.classifications.append(payload)
                    if owner.classify_failure:
                        raise owner.classify_failure
                    route = owner.route_override or {
                        "intent": "qa", "reason": "固定分类，仅验证工作流连接", "clarification_question": "",
                    }
                    return AIMessage(content=json.dumps(route, ensure_ascii=False),
                                     response_metadata={"finish_reason": "stop"})
                return RunnableLambda(respond)

        fake_llm.create_chat_model = ControlledChatModel
        self.patches.enter_context(patch.dict(sys.modules, {
            "sentence_transformers": fake_embedding, 'paper_sage.llm.client': fake_llm,
        }))
        self.patches.enter_context(patch.object(st, "secrets", {"DEEPSEEK_API_KEY": "test-only-key"}))
        self.patches.enter_context(patch.object(reranker, "load_reranker", side_effect=controlled_rerank_loader))
        st.cache_resource.clear()

    def cleanup_workspace(self):
        st.cache_resource.clear()
        self.patches.close()
        os.chdir(self.previous_cwd)
        # 删除临时目录前核对绝对目标；不清理任何实际项目目录。
        assert self.root.resolve() == self.root
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        assert self.root.name.startswith("paper-sage-scope-")
        self.temporary.cleanup()

    def seed_document(self, project_id, fixture, filename, coordinate):
        raw = (WORKSPACE / fixture["path"]).read_bytes()
        self.assertEqual(sha256(raw).hexdigest(), fixture["sha256"])
        document_id = "doc_" + fixture["sha256"]
        source = Path("data/papers") / project_id / document_id / filename
        source.parent.mkdir(parents=True)
        source.write_bytes(raw)
        chunks = split_pages(load_pdf_pages(source.as_posix(), document_id), project_id)
        self.assertEqual(len(chunks), 1)
        vector = np.zeros((1, 384), dtype="float32")
        vector[0, coordinate] = 1.0
        index = faiss.IndexFlatIP(384)
        index.add(vector)
        save_index(index, chunks, get_index_dir(project_id, document_id, "v1"), {
            "model_name": "intfloat/multilingual-e5-small", "normalize_embeddings": True,
            "passage_prefix": "passage: ", "query_prefix": "query: ",
        })
        return {"document_id": document_id, "source": source.as_posix(), "index_version": "v1", "file_hash": fixture["sha256"]}

    def checked(self, at):
        self.assertFalse(at.exception, [item.message for item in at.exception])
        self.assertFalse(at.error, [item.value for item in at.error])
        return at

    def app(self, project_id, selected_ids):
        at = AppTest.from_file(str(WORKSPACE / "app.py"), default_timeout=30)
        at.session_state["selected_project_id"] = project_id
        at.session_state[f"scope_mode_{project_id}"] = "指定文档"
        at.session_state[f"scope_documents_{project_id}"] = selected_ids
        return self.checked(at.run())

    @staticmethod
    def widget(widgets, key):
        return next(widget for widget in widgets if widget.key == key)

    def select(self, at, document_ids):
        project_id = at.session_state["selected_project_id"]
        radio = self.widget(at.radio, f"scope_mode_{project_id}")
        if radio.value != "指定文档":
            self.checked(radio.set_value("指定文档").run())
        self.checked(self.widget(at.multiselect, f"scope_documents_{project_id}").set_value(document_ids).run())

    def switch(self, at, project_id):
        self.checked(self.widget(at.selectbox, "selected_project_id").set_value(project_id).run())

    def assert_empty_chat(self, at):
        self.assertEqual(at.session_state["chat_session"]["messages"], [])
        self.assertFalse(at.chat_message)
        for widget in at.text_input:
            if str(widget.key).startswith("question_"):
                self.assertEqual(widget.value, "")

    def ask(self, at, expected_code):
        project_id, documents = at.session_state["chat_session"]["scope_key"]
        selected_ids = [record[0] for record in documents]
        question = next(widget for widget in at.text_input if str(widget.key).startswith("question_"))
        question.set_value(self.question)
        submit = next(widget for widget in at.button if widget.label == "提交问题")
        self.checked(submit.click().run())
        self.assertEqual(len(self.requests[-1]), 2, "切换后结构化回答只收到系统消息和本次问题与片段")
        self.assertNotIn("history", self.classifications[-1], "切换后分类也不能携带旧范围历史")
        answer = at.session_state["chat_session"]["messages"][-1]
        results = answer["retrieval_results"]
        self.assertTrue(results, "有答案样例必须实际返回候选")
        assert_results_in_scope(results, project_id, selected_ids)
        registered = {doc["document_id"]: doc for doc in load_project(project_id)["documents"]}
        for result in results:
            chunk = result["chunk"]
            self.assertEqual(chunk["source"], registered[chunk["document_id"]]["source"])
        self.assertIn(expected_code, answer["content"])
        self.queries.append({
            "project_id": project_id, "selected_document_ids": selected_ids,
            "candidate_chunk_ids": [r["chunk"]["chunk_id"] for r in results],
            "candidate_sources": [r["chunk"]["source"] for r in results],
            "simulated_answer_code": expected_code,
        })
        return results

    def record_case(self, name):
        self.observations.append({
            "case": name, "status": "passed", "queries": deepcopy(self.queries),
            "simulated_answer_count": len(self.requests), "controlled_query_count": self.encode_calls,
            "controlled_rerank_call_count": len(self.rerank_calls),
            "rerank_model_instances": self.rerank_model_instances,
        })

    def test_project_switch(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        original_session = at.session_state["chat_session"]["session_id"]
        self.switch(at, self.project_b)
        self.assert_empty_chat(at)
        self.assertNotEqual(at.session_state["chat_session"]["session_id"], original_session)
        results = self.ask(at, self.fixture_b["expected_code"])
        # 相同文档标识在另一个项目中也存在，仍不能混用项目归属。
        with self.assertRaises(AssertionError):
            assert_results_in_scope(results, self.project_a, [self.bid])
        with self.assertRaises(AssertionError):
            assert_results_in_scope(results, self.project_b, [self.aid])
        self.switch(at, self.project_a)
        self.assert_empty_chat(at)
        self.assertNotEqual(at.session_state["chat_session"]["session_id"], original_session)
        self.record_case("project_switch")

    def test_document_selection(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        self.select(at, [self.bid])
        self.assert_empty_chat(at)
        self.ask(at, self.fixture_b["expected_code"])
        self.select(at, [self.aid, self.bid])
        self.assert_empty_chat(at)
        results = self.ask(at, self.fixture_b["expected_code"])
        self.assertEqual({r["chunk"]["document_id"] for r in results}, {self.aid, self.bid})
        session = deepcopy(at.session_state["chat_session"])
        self.select(at, [self.bid, self.aid])
        self.assertEqual(at.session_state["chat_session"], session)
        self.record_case("document_selection")

    def test_empty_selection(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = (len(self.requests), self.encode_calls, len(self.rerank_calls))
        self.select(at, [])
        self.assert_empty_chat(at)
        self.assertFalse(any(widget.label == "提交问题" for widget in at.button))
        self.assertFalse(any(str(widget.key).startswith("question_") for widget in at.text_input))
        self.assertEqual((len(self.requests), self.encode_calls, len(self.rerank_calls)), before)
        with self.assertRaises(ValueError):
            load_scope_index(load_project(self.project_a), [])
        self.select(at, [self.aid])
        self.assert_empty_chat(at)
        self.ask(at, self.fixture_a["expected_code"])
        self.record_case("empty_selection")

    def test_deletion_isolation(self):
        at = self.app(self.project_b, [self.bid])
        self.ask(at, self.fixture_b["expected_code"])
        protected_before = {
            name: data_snapshot(Path("data") / name / self.project_b)
            for name in ("projects", "papers", "indexes")
        }
        self.switch(at, self.project_a)
        self.select(at, [self.bid])
        self.ask(at, self.fixture_b["expected_code"])
        self.widget(at.selectbox, f"delete_document_{self.project_a}").set_value(self.bid).run()
        self.checked(self.widget(at.button, f"delete_submit_{self.project_a}").click().run())
        self.assert_empty_chat(at)
        self.assertEqual(self.widget(at.multiselect, f"scope_documents_{self.project_a}").value, [])
        self.assertFalse(any(widget.label == "提交问题" for widget in at.button))
        self.assertEqual(load_project(self.project_a)["documents"], [self.doc_a])
        self.assertNotIn("pending_cleanup", load_project(self.project_a))
        self.assertFalse(Path(self.doc_b_in_a["source"]).exists())
        self.assertFalse(get_index_dir(self.project_a, self.bid, "v1").parent.exists())
        for name, snapshot in protected_before.items():
            self.assertEqual(data_snapshot(Path("data") / name / self.project_b), snapshot)
        self.select(at, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        self.switch(at, self.project_b)
        self.assert_empty_chat(at)
        self.ask(at, self.fixture_b["expected_code"])
        self.assertEqual(self.model_instances, 1, "删除只清除索引缓存，模型缓存继续复用")
        self.assertEqual(self.rerank_model_instances, 1, "删除后继续复用精排模型")
        self.record_case("deletion_isolation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
