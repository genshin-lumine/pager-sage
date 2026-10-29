"""结构化问答工作台：真实页面与图，固定模型及临时文档，无外部请求。"""

from copy import deepcopy
import json
import sys
from unittest.mock import patch

from paper_sage.qa.report import format_cited_answer
import test_project_scope as scope_tests


class CitedAnswerPageTests(scope_tests.ProjectScopeTests):
    def submit(self, at, question=None):
        next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(question or self.question)
        return next(w for w in at.button if w.label == "提交问题").click().run()

    def panel(self, at):
        return next(item for item in at.expander if item.label == "查看结论原文依据")

    def counts(self):
        return (len(self.classifications), len(self.tool_requests), len(self.requests), len(self.rewrite_requests),
                self.encode_calls, len(self.rerank_calls))

    def test_saved_answer_shows_program_sources_and_rerun_has_no_calls(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        self.assertEqual(next(w for w in at.text_input if str(w.key).startswith("question_")).value, "")
        self.assertTrue(any("处理完成" in item.value for item in at.success))
        message = at.session_state["chat_session"]["messages"][-1]
        self.assertEqual(at.session_state["chat_session"]["messages"][-2]["content"], self.question)
        record = message["answer_artifact"]
        self.assertEqual(record["result"], json.loads(record["raw_response"]))
        self.assertEqual(record["final_results"], message["retrieval_results"])
        self.assertEqual(message["qa_retry_artifact"]["schema_version"], 2)
        self.assertEqual(message["qa_retry_artifact"]["attempt_records"][-1]["answer_artifact"], record)
        source = record["claims_with_sources"][0]["sources"][0]
        self.assertEqual((source["document_id"], source["source"], source["pdf_page"]),
                         (self.aid, self.doc_a["source"], 1))
        self.assertIn("PDF 第 1 页", message["content"])
        self.assertIn("scope_note.pdf", message["content"])
        self.assertEqual([item.value for item in self.panel(at).get("code")], [source["text"]])
        before = deepcopy(at.session_state["chat_session"])
        counts = self.counts()
        self.checked(at.run())
        self.assertEqual(at.session_state["chat_session"], before)
        self.assertEqual(self.counts(), counts)
        altered = deepcopy(record)
        altered["claims_with_sources"][0]["sources"][0]["pdf_page"] = 99
        with self.assertRaises(ValueError):
            format_cited_answer(altered)

    def test_evidence_panel_contains_only_cited_snippets_not_all_candidates(self):
        at = self.app(self.project_a, [self.aid, self.bid])
        self.checked(self.submit(at))
        message = at.session_state["chat_session"]["messages"][-1]
        record = message["answer_artifact"]
        self.assertEqual(len(record["evidence_by_id"]), 2)
        sources = record["claims_with_sources"][0]["sources"]
        self.assertEqual(len(sources), 1)
        self.assertEqual([item.value for item in self.panel(at).get("code")], [sources[0]["text"]])
        retrieval = next(item for item in at.expander if item.label == "查看本轮检索片段")
        self.assertEqual(len(retrieval.get("code")), 2)

    def test_one_claim_can_preserve_both_documents_and_pages(self):
        original = sys.modules['paper_sage.llm.client'].cited_answer_response

        def respond(messages, key):
            original(messages, key)
            refs = json.loads(messages[-1].content)["reference"]
            return {"claims": [{"text": "两份测试文档分别提供校验码。", "chunk_ids": [r["chunk_id"] for r in refs]}],
                    "insufficient_reason": ""}

        at = self.app(self.project_a, [self.aid, self.bid])
        with patch.object(sys.modules['paper_sage.llm.client'], "cited_answer_response", side_effect=respond):
            self.checked(self.submit(at))
        message = at.session_state["chat_session"]["messages"][-1]
        sources = message["answer_artifact"]["claims_with_sources"][0]["sources"]
        self.assertEqual({s["document_id"] for s in sources}, {self.aid, self.bid})
        self.assertEqual([item.value for item in self.panel(at).get("code")], [s["text"] for s in sources])
        for document in (self.doc_a, self.doc_b_in_a):
            self.assertTrue(any(document["source"] in item.value for item in self.panel(at).get("caption")))

    def test_insufficient_answer_keeps_candidates_without_presenting_them_as_citations(self):
        original = sys.modules['paper_sage.llm.client'].cited_answer_response

        def respond(messages, key):
            original(messages, key)
            return {"claims": [], "insufficient_reason": "当前片段未提供所需信息。"}

        at = self.app(self.project_a, [self.aid])
        with patch.object(sys.modules['paper_sage.llm.client'], "cited_answer_response", side_effect=respond):
            self.checked(self.submit(at))
        message = at.session_state["chat_session"]["messages"][-1]
        record = message["answer_artifact"]
        self.assertEqual(record["claims_with_sources"], [])
        self.assertTrue(record["evidence_by_id"])
        self.assertIn("资料不足", message["content"])
        self.assertEqual(len(self.panel(at).get("code")), 0)
        self.assertFalse(at.toggle)
        self.assertTrue(self.panel(at).get("info"))
        self.assertEqual((message["qa_retry_artifact"]["attempts_used"], message["qa_retry_artifact"]["stop_reason"]),
                         (2, "attempt_limit"))
        self.assertEqual(len(at.session_state["chat_session"]["messages"]), 2)

    def test_bad_second_citation_preserves_prior_chat_and_displays_failure(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = deepcopy(at.session_state["chat_session"])
        original = sys.modules['paper_sage.llm.client'].cited_answer_response
        calls = []

        def respond(messages, key):
            result = original(messages, key)
            calls.append(1)
            if len(calls) == 1:
                return {"claims": [], "insufficient_reason": "模拟首次资料不足。"}
            result["claims"][0]["chunk_ids"] = [f"{self.project_a}:{self.bid}:s800:o100:p1:c1"]
            return result

        with patch.object(sys.modules['paper_sage.llm.client'], "cited_answer_response", side_effect=respond):
            self.submit(at)
        self.assertFalse(at.exception)
        self.assertTrue(any("引用检查" in item.value for item in at.error))
        self.assertEqual(len(calls), 2)
        self.assertEqual(at.session_state["chat_session"], before)
        self.assertEqual(len([item for item in at.expander if item.label == "查看结论原文依据"]), 1)

    def test_followup_retains_old_snapshot_and_uses_plain_history(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        before = deepcopy(at.session_state["chat_session"]["messages"])
        self.search_query_fn = lambda question: "verification code"
        self.checked(self.submit(at, "校验码呢？"))
        messages = at.session_state["chat_session"]["messages"]
        self.assertEqual(messages[:2], before)
        expected_history = [{"role": m["role"], "content": m["content"]} for m in before]
        record = messages[-1]["answer_artifact"]
        self.assertEqual(record["history"], expected_history)
        self.assertEqual(record["question"], "校验码呢？")
        self.assertEqual(len(self.answer_protocols[-1]), 4)
        payload = json.loads(self.answer_protocols[-1][-1].content)
        self.assertEqual(payload["question"], "校验码呢？")
        self.assertEqual(set(payload), {"question", "reference"})
        self.assertEqual(len([item for item in at.expander if item.label == "查看结论原文依据"]), 2)

    def test_scope_clear_and_legacy_history_do_not_keep_new_evidence_panels(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        self.switch(at, self.project_b)
        self.assert_empty_chat(at)
        self.assertFalse(any(item.label == "查看结论原文依据" for item in at.expander))
        self.ask(at, self.fixture_b["expected_code"])
        record = at.session_state["chat_session"]["messages"][-1]["answer_artifact"]
        self.assertTrue(all(s["project_id"] == self.project_b for s in record["evidence_by_id"].values()))
        self.checked(next(w for w in at.button if w.label == "清空会话").click().run())
        self.assert_empty_chat(at)
        self.assertFalse(any(item.label == "查看结论原文依据" for item in at.expander))
        session = deepcopy(at.session_state["chat_session"])
        session["messages"] = [{"role": "user", "content": "旧问题"},
                               {"role": "assistant", "content": "旧版回答及原文引用。"}]
        at.session_state["chat_session"] = session
        counts = self.counts()
        self.checked(at.run())
        self.assertEqual(self.counts(), counts)
        self.assertTrue(any(item.value == "旧版回答及原文引用。" for item in at.markdown))
        self.assertFalse(at.toggle)
        self.assertFalse(any(item.label == "查看结论原文依据" for item in at.expander))
