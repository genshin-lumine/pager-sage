"""工作台手动存档恢复：真实页面及本地索引，所有模型响应受控。"""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import test_project_scope as scope_tests


class SessionRecoveryPageTests(scope_tests.ProjectScopeTests):
    def counters(self):
        return (len(self.classifications), len(self.tool_requests), len(self.requests),
                self.encode_calls, len(self.rerank_calls), len(self.summary_requests))

    def button(self, at, key):
        return next(widget for widget in at.button if widget.key == key)

    def choose(self, at, session_id):
        select = next(widget for widget in at.selectbox if str(widget.key).startswith("saved_session_"))
        return self.checked(select.set_value(session_id).run())

    def submit_followup(self, at):
        next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(self.question)
        self.checked(next(w for w in at.button if w.label == "提交问题").click().run())
        self.assertIn(self.fixture_a["expected_code"], at.session_state["chat_session"]["messages"][-1]["content"])

    def saved_path(self, session):
        return Path("data/sessions") / session["scope_key"][0] / f"{session['session_id']}.json"

    def seed_saved(self):
        at = self.app(self.project_a, [self.aid])
        self.ask(at, self.fixture_a["expected_code"])
        session = deepcopy(at.session_state["chat_session"])
        before = self.counters()
        self.checked(self.button(at, "save_chat_session").click().run())
        self.assertEqual(self.counters(), before)
        self.assertTrue(self.saved_path(session).is_file())
        return at, session

    def test_fresh_page_restores_exact_messages_without_model_calls_and_can_continue(self):
        at, saved = self.seed_saved()
        original_file = self.saved_path(saved).read_bytes()
        counters = self.counters()
        fresh = self.app(self.project_a, [self.aid])
        self.assert_empty_chat(fresh)
        self.choose(fresh, saved["session_id"])
        self.assert_empty_chat(fresh)
        self.checked(self.button(fresh, "restore_chat_session").click().run())
        self.assertEqual(fresh.session_state["chat_session"], saved)
        self.assertEqual(self.counters(), counters)
        self.assertEqual(self.saved_path(saved).read_bytes(), original_file)
        self.assertTrue(any(exp.label == "查看结论原文依据" for exp in fresh.expander))
        self.assertTrue(any(exp.label == "查看本轮检索尝试" for exp in fresh.expander))
        question = next(w for w in fresh.text_input if str(w.key).startswith("question_"))
        self.assertEqual(question.value, "")
        self.submit_followup(fresh)
        self.assertEqual(len(fresh.session_state["chat_session"]["messages"]), 4)
        self.assertEqual(self.classifications[-1]["history"], [
            {"role": m["role"], "content": m["content"]} for m in saved["messages"]
        ])
        self.checked(self.button(fresh, "save_chat_session").click().run())
        record = json.loads(self.saved_path(saved).read_bytes())
        self.assertEqual(len(record["session"]["messages"]), 4)

    def test_clear_and_scope_switch_do_not_automatically_restore(self):
        at, saved = self.seed_saved()
        original = self.saved_path(saved).read_bytes()
        self.checked(next(w for w in at.button if w.label == "清空会话").click().run())
        self.assert_empty_chat(at)
        self.checked(at.run())
        self.assert_empty_chat(at)
        self.assertEqual(self.saved_path(saved).read_bytes(), original)
        self.select(at, [self.bid])
        self.assert_empty_chat(at)
        self.assertFalse(any(str(w.key).startswith("saved_session_") for w in at.selectbox))
        self.switch(at, self.project_b)
        self.assert_empty_chat(at)
        self.assertFalse(any(str(w.key).startswith("saved_session_") for w in at.selectbox))
        self.switch(at, self.project_a)
        self.select(at, [self.aid])
        self.assert_empty_chat(at)
        self.choose(at, saved["session_id"])
        self.checked(self.button(at, "restore_chat_session").click().run())
        self.assertEqual(at.session_state["chat_session"], saved)

    def test_empty_and_failed_saves_keep_memory_and_existing_file(self):
        at, saved = self.seed_saved()
        original = self.saved_path(saved).read_bytes()
        self.submit_followup(at)
        current = deepcopy(at.session_state["chat_session"])
        counters = self.counters()
        with patch('paper_sage.sessions.store.os.replace', side_effect=OSError("controlled disk failure")):
            self.button(at, "save_chat_session").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertEqual(at.session_state["chat_session"], current)
        self.assertEqual(self.saved_path(saved).read_bytes(), original)
        self.assertEqual(self.counters(), counters)
        self.checked(next(w for w in at.button if w.label == "清空会话").click().run())
        self.button(at, "save_chat_session").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("为空" in error.value for error in at.error))
        self.assert_empty_chat(at)
        self.assertEqual(self.saved_path(saved).read_bytes(), original)

    def test_changed_index_and_tampered_answer_do_not_replace_current_session(self):
        at, saved = self.seed_saved()
        path = self.saved_path(saved)
        original = path.read_bytes()
        fresh = self.app(self.project_a, [self.aid])
        self.choose(fresh, saved["session_id"])
        current = deepcopy(fresh.session_state["chat_session"])
        counters = self.counters()
        changed = json.loads(original)
        changed["session"]["messages"][-1]["content"] = "Changed answer text"
        path.write_text(json.dumps(changed), encoding="utf-8")
        self.button(fresh, "restore_chat_session").click().run()
        self.assertFalse(fresh.exception)
        self.assertTrue(any("正文" in error.value for error in fresh.error))
        self.assertEqual(fresh.session_state["chat_session"], current)
        self.assertEqual(self.counters(), counters)
        path.write_bytes(original)
        metadata = Path("data/indexes") / self.project_a / self.aid / "v1/metadata.json"
        metadata.write_bytes(metadata.read_bytes() + b"\n")
        self.button(fresh, "restore_chat_session").click().run()
        self.assertFalse(fresh.exception)
        self.assertTrue(any("版本不一致" in error.value for error in fresh.error))
        self.assertEqual(fresh.session_state["chat_session"], current)
        self.assertEqual(self.counters(), counters)

    def test_legacy_missing_input_snapshot_and_corrupt_json_are_not_restored(self):
        at, saved = self.seed_saved()
        path = self.saved_path(saved)
        original = path.read_bytes()
        fresh = self.app(self.project_a, [self.aid])
        self.choose(fresh, saved["session_id"])
        changed = json.loads(original)
        changed["session"]["messages"][-1].pop("input_files")
        path.write_text(json.dumps(changed), encoding="utf-8")
        self.button(fresh, "restore_chat_session").click().run()
        self.assertFalse(fresh.exception)
        self.assertTrue(any("缺少文件快照" in error.value for error in fresh.error))
        self.assert_empty_chat(fresh)
        path.write_text("{", encoding="utf-8")
        self.checked(fresh.run())
        self.assert_empty_chat(fresh)
        self.assertFalse(any(str(w.key).startswith("saved_session_") for w in fresh.selectbox))

    def test_clarification_can_be_saved_and_restored_without_retrieval(self):
        at = self.app(self.project_a, [self.aid])
        self.route_override = {"intent": "clarify", "reason": "固定分类", "clarification_question": "请说明想了解什么？"}
        question = next(w for w in at.text_input if str(w.key).startswith("question_"))
        question.set_value("帮我分析一下")
        self.checked(next(w for w in at.button if w.label == "提交问题").click().run())
        saved = deepcopy(at.session_state["chat_session"])
        self.checked(self.button(at, "save_chat_session").click().run())
        counters = self.counters()
        fresh = self.app(self.project_a, [self.aid])
        self.choose(fresh, saved["session_id"])
        self.checked(self.button(fresh, "restore_chat_session").click().run())
        self.assertEqual(fresh.session_state["chat_session"], saved)
        self.assertEqual(self.counters(), counters)
        self.assertEqual((self.encode_calls, len(self.requests)), (0, 0))
