"""真实问答图的结构化证据隔离与保存校验；模型和检索均为固定替身。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from paper_sage.agents.graph import build_paper_graph, make_initial_state
from paper_sage.qa.model import CITED_ANSWER_MAX_TOKENS
from paper_sage.qa.retry import QARetryError
from paper_sage.qa.retry_records import build_qa_retry_artifact
from paper_sage.retrieval.context import build_reference_text


QUESTION = "训练模型使用了什么硬件？"
NEW_QUERY = "training hardware used for the model"
DOCUMENTS = [{"document_id": "paper_a", "source": "practice/paper_a.pdf", "index_version": "v1"}]
SCOPE = ("practice", (("paper_a", "v1", "practice/paper_a.pdf"),))


def response(value):
    return AIMessage(content=json.dumps(value, ensure_ascii=False), response_metadata={"finish_reason": "stop"})


def make_chunk(label, page):
    return {"project_id": "practice", "document_id": "paper_a", "source": "practice/paper_a.pdf",
            "pdf_page": page, "chunk_id": f"practice:paper_a:{label}", "text": f"ONLY_THIS_ATTEMPT_{label}"}


class CitedResources:
    def __init__(self, outcomes, *, route="qa", empty=False, rewrite_query=NEW_QUERY):
        self.outcomes = outcomes
        self.searches = []
        self.answer_messages = []
        self.states = []
        self.model, self.answer, self.classify, self.request, self.rewrite = [Mock() for _ in range(5)]
        self.classify.invoke.return_value = response({"intent": route, "reason": "固定分类",
                                                      "clarification_question": "想了解什么？" if route == "clarify" else ""})
        self.request.invoke.return_value = AIMessage(content="", response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": "search_papers", "args": {"question": QUESTION}, "id": "initial"}])
        self.rewrite.invoke.return_value = response({"query": rewrite_query, "reason": "固定改写"})
        self.model.bind_tools.return_value = self.request

        def bind(**kwargs):
            if kwargs.get("max_tokens") == CITED_ANSWER_MAX_TOKENS:
                return self.answer
            return self.rewrite if kwargs.get("max_tokens") == 512 else self.classify

        self.model.bind.side_effect = bind

        def answer(messages):
            self.answer_messages.append(deepcopy(messages))
            outcome = self.outcomes[len(self.answer_messages) - 1]
            if isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, AIMessage):
                return outcome
            data = json.loads(messages[-1].content)
            if outcome == "insufficient":
                return response({"claims": [], "insufficient_reason": "当前片段没有提供硬件信息。"})
            chunk_id = ("practice:paper_a:A" if outcome == "old_citation" else
                        "practice:paper_a:E" if outcome == "pool_citation" else data["reference"][0]["chunk_id"])
            return response({"claims": [{"text": "模拟结论，仅验证来源传递。", "chunk_ids": [chunk_id]}],
                             "insufficient_reason": ""})

        self.answer.invoke.side_effect = answer

        @tool(response_format="content_and_artifact")
        def search_papers(question: str) -> tuple[str, dict]:
            """返回固定片段和范围记录，不执行真实检索。"""
            self.searches.append(question)
            labels = ("A", "B") if len(self.searches) == 1 else ("C", "D")
            selected = [] if empty else [{"rank": i, "chunk": make_chunk(label, i)} for i, label in enumerate(labels, 1)]
            pool = selected + [{"rank": 3, "chunk": make_chunk("E", 9)}]
            artifact = {
                "question": question, "retrieval_results": deepcopy(selected),
                "retrieval_channels": {"dense": deepcopy(pool), "bm25": []},
                "retrieval_candidates": deepcopy(pool), "rrf_results": deepcopy(pool), "rerank_results": deepcopy(pool),
                "retrieval_trace": {"question": question,
                    "stages": [{"stage": "simulation_select", "results": deepcopy(selected)}],
                    "candidate_paths": [{key: item["chunk"][key] for key in ("chunk_id", "pdf_page", "source")} for item in pool]},
            }
            from paper_sage.retrieval.tools import EMPTY_REFERENCE_TEXT
            return build_reference_text(selected) or EMPTY_REFERENCE_TEXT, artifact

        self.tool = search_papers

    def run(self, *, history=None, max_attempts=2):
        graph = build_paper_graph(self.model, self.tool, documents=[{"document_id": "paper_a", "name": "paper_a.pdf"}],
                                  max_attempts=max_attempts, cited_answers=True)
        for state in graph.stream(make_initial_state(QUESTION, history=history),
                                  config={"recursion_limit": 24}, stream_mode="values"):
            self.states.append(state)
        return self.states[-1]


class CitedAnswerGraphTests(unittest.TestCase):
    def test_first_answer_records_sources_and_snapshot_is_independent(self):
        resources = CitedResources(["answered"])
        state = resources.run()
        artifact = state["answer_artifact"]
        self.assertEqual(state["attempt_records"][0]["answer_artifact"], artifact)
        self.assertEqual(state["result"], artifact["result"])
        self.assertEqual(list(artifact["evidence_by_id"]), ["practice:paper_a:A", "practice:paper_a:B"])
        saved = build_qa_retry_artifact(state, SCOPE, DOCUMENTS, 2)
        self.assertEqual(saved["schema_version"], 2)
        self.assertEqual((saved["attempts_used"], saved["stop_reason"]), (1, "answered"))
        artifact["claims_with_sources"][0]["sources"][0]["pdf_page"] = 99
        self.assertEqual(saved["attempt_records"][0]["answer_artifact"]["claims_with_sources"][0]["sources"][0]["pdf_page"], 1)
        resources.answer.invoke.assert_called_once()
        resources.rewrite.invoke.assert_not_called()

    def test_retry_rebuilds_catalog_and_keeps_only_same_scope_history(self):
        history = [{"role": "user", "content": "之前问过模型。"},
                   {"role": "assistant", "content": "之前的回答仅作指代上下文。", "extra_artifact": "DO_NOT_SEND"}]
        before = deepcopy(history)
        resources = CitedResources(["insufficient", "answered"])
        state = resources.run(history=history)
        records = state["attempt_records"]
        self.assertEqual([list(r["answer_artifact"]["evidence_by_id"]) for r in records],
                         [["practice:paper_a:A", "practice:paper_a:B"], ["practice:paper_a:C", "practice:paper_a:D"]])
        self.assertEqual([r["status"] for r in records], ["insufficient", "answered"])
        for s in resources.states:
            if s["last_node"] in ("begin_attempt", "execute_tool"):
                self.assertIsNone(s["answer_artifact"])
        second_messages = resources.answer_messages[1]
        self.assertEqual([m.type for m in second_messages], ["system", "human", "ai", "human"])
        self.assertEqual(second_messages[1].content, history[0]["content"])
        second_input = json.loads(second_messages[-1].content)
        self.assertEqual(second_input["question"], QUESTION)
        self.assertEqual([r["chunk_id"] for r in second_input["reference"]], ["practice:paper_a:C", "practice:paper_a:D"])
        self.assertNotIn("ONLY_THIS_ATTEMPT_A", str(second_messages))
        self.assertNotIn("DO_NOT_SEND", str(second_messages))
        self.assertEqual(history, before)
        resources.request.invoke.assert_called_once()
        resources.rewrite.invoke.assert_called_once()
        self.assertEqual(build_qa_retry_artifact(state, SCOPE, DOCUMENTS, 2)["attempts_used"], 2)

    def test_old_same_document_citation_is_answer_error_with_failed_record(self):
        resources = CitedResources(["insufficient", "old_citation"])
        with self.assertRaises(QARetryError) as caught:
            resources.run()
        state = caught.exception.state
        self.assertEqual(state["retry_status"], "answer_error")
        self.assertEqual([r["status"] for r in state["attempt_records"]], ["insufficient", "answer_error"])
        self.assertIsNone(state["result"])
        self.assertEqual(state["response_kind"], "error")
        failed = state["attempt_records"][-1]["answer_artifact"]
        self.assertEqual(list(failed["evidence_by_id"]), ["practice:paper_a:C", "practice:paper_a:D"])
        self.assertIn("practice:paper_a:A", failed["raw_response"])
        self.assertFalse(failed["structural_checks_passed"])
        self.assertEqual(state["attempt_records"][0]["answer_artifact"]["status"], "accepted")
        self.assertEqual(len(resources.searches), 2)
        with self.assertRaises(ValueError):
            build_qa_retry_artifact(state, SCOPE, DOCUMENTS, 2)

    def test_recalled_but_not_selected_chunk_is_not_allowed(self):
        resources = CitedResources(["pool_citation"])
        with self.assertRaises(QARetryError) as caught:
            resources.run()
        state = caught.exception.state
        self.assertIn("practice:paper_a:E", str(state["retrieval_artifact"]["retrieval_candidates"]))
        self.assertNotIn("practice:paper_a:E", state["answer_artifact"]["evidence_by_id"])
        resources.rewrite.invoke.assert_not_called()

    def test_truncation_and_request_failure_stop_without_retrying_answer(self):
        truncated = response({"claims": [], "insufficient_reason": "未完成"})
        truncated.response_metadata["finish_reason"] = "length"
        for outcome in (truncated, TimeoutError("private-sdk-message")):
            resources = CitedResources([outcome])
            with self.subTest(outcome=type(outcome).__name__), self.assertRaises(QARetryError) as caught:
                resources.run()
            state = caught.exception.state
            self.assertEqual(state["retry_status"], "answer_error")
            self.assertEqual(state["answer_artifact"]["status"], "failed")
            self.assertNotIn("private-sdk-message", str(state))
            resources.answer.invoke.assert_called_once()
            resources.rewrite.invoke.assert_not_called()

    def test_save_rejects_source_result_history_or_protocol_tampering(self):
        state = CitedResources(["insufficient", "answered"]).run()
        for field in ("source", "result", "history", "protocol", "final"):
            altered = deepcopy(state)
            first = altered["attempt_records"][0]["answer_artifact"]
            if field == "source":
                first["evidence_by_id"]["practice:paper_a:A"]["pdf_page"] = 88
            elif field == "result":
                first["result"]["insufficient_reason"] = "Changed after generation"
            elif field == "history":
                first["history"] = [{"role": "user", "content": "Injected"}]
            elif field == "protocol":
                altered["attempt_records"][0]["answer_artifact"] = None
            else:
                altered["answer_artifact"]["claims_with_sources"][0]["sources"][0]["pdf_page"] = 88
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_qa_retry_artifact(altered, SCOPE, DOCUMENTS, 2)

    def test_empty_retrieval_and_duplicate_query_finish_with_insufficient_result(self):
        for options, expected_used, reason in (({"empty": True}, 2, "attempt_limit"),
                                               ({"rewrite_query": QUESTION}, 1, "duplicate_query")):
            resources = CitedResources(["insufficient", "insufficient"], **options)
            state = resources.run()
            self.assertEqual((state["attempts_used"], state["stop_reason"]), (expected_used, reason))
            self.assertEqual(state["answer_artifact"]["claims_with_sources"], [])
            self.assertEqual(build_qa_retry_artifact(state, SCOPE, DOCUMENTS, 2)["status"], "insufficient")

    def test_clarification_does_not_request_answer_or_create_evidence(self):
        resources = CitedResources([], route="clarify")
        state = resources.run()
        self.assertEqual(state["response_kind"], "clarify")
        self.assertIsNone(state["answer_artifact"])
        resources.request.invoke.assert_not_called()
        resources.answer.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
