"""真实页面、图、比较服务和临时索引；全部模型响应受控，不调用外部 API。"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from paper_sage.comparison.report import format_paper_comparison
from paper_sage.comparison.review import comparison_digest
from paper_sage.comparison.service import load_saved_comparison
import test_project_scope as scope_tests
from paper_sage.workflows.events import read_events, summarize_calls
from paper_sage.workflows.store import run_paths


class ComparisonPageTests(scope_tests.ProjectScopeTests):
    def test_saved_comparison_restores_both_paper_sources_without_new_calls(self):
        at = self.app(self.project_a, [self.aid, self.bid])
        self.checked(self.submit(at))
        saved = deepcopy(at.session_state["chat_session"])
        self.checked(next(w for w in at.button if w.key == "save_chat_session").click().run())
        counts = self.counters()
        # 选择顺序不同，仍是同一项目与文档范围。
        fresh = self.app(self.project_a, [self.bid, self.aid])
        self.assert_empty_chat(fresh)
        widget = next(w for w in fresh.selectbox if str(w.key).startswith("saved_session_"))
        self.checked(widget.set_value(saved["session_id"]).run())
        self.checked(next(w for w in fresh.button if w.key == "restore_chat_session").click().run())
        self.assertEqual(fresh.session_state["chat_session"], saved)
        self.assertEqual(self.counters(), counts)
        self.assertTrue(any(e.label == "查看比较原文依据" for e in fresh.expander))
        self.assertTrue(fresh.toggle)

        # 图检查点恢复与聊天存档恢复保留同一份双方来源，选择顺序不改变范围。
        run_page = self.app(self.project_a, [self.bid, self.aid])
        widget = next(w for w in run_page.selectbox if str(w.key).startswith("workflow_run_"))
        self.checked(widget.set_value(saved["messages"][-1]["workflow_run_id"]).run())
        self.checked(next(w for w in run_page.button if w.key == "load_workflow_run").click().run())
        self.assertEqual(run_page.session_state["chat_session"], saved)
        self.assertEqual(self.counters(), counts)
        self.assertTrue(any(e.label == "查看比较原文依据" for e in run_page.expander))

    def setUp(self):
        super().setUp()
        self.comparison_plans = []
        self.comparison_generations = []
        self.plan_override = None
        self.plan_failure = None
        self.comparison_failure = False
        owner = self
        original_factory = sys.modules['paper_sage.llm.client'].create_chat_model

        class ComparisonModel:
            def __init__(self, api_key):
                self.base = original_factory(api_key)

            def bind_tools(self, *args, **kwargs):
                return self.base.bind_tools(*args, **kwargs)

            def bind(self, **kwargs):
                if kwargs.get("max_tokens") == 1024:
                    return RunnableLambda(owner.respond_plan)
                bound = self.base.bind(**kwargs)
                if kwargs.get("max_tokens") != 4096:
                    return bound

                def respond(messages):
                    payload = json.loads(messages[-1].content)
                    if "groups" in payload:
                        return owner.respond_comparison(payload)
                    return bound.invoke(messages)
                return RunnableLambda(respond)

        self.patches.enter_context(patch("paper_sage.ui.app.create_chat_model", ComparisonModel))

    def respond_plan(self, messages):
        self.comparison_plans.append(json.loads(messages[-1].content))
        if self.plan_failure:
            raise self.plan_failure
        plan = self.plan_override or {"dimensions": [
            {"name": "训练数据", "query": "Which training data is used?"},
            {"name": "实验结果", "query": "Which experimental results are reported?"},
        ], "clarification_question": ""}
        return AIMessage(content=json.dumps(plan), response_metadata={"finish_reason": "stop"})

    def respond_comparison(self, payload):
        self.comparison_generations.append(deepcopy(payload))
        if self.comparison_failure:
            return AIMessage(content="{", response_metadata={"finish_reason": "stop"})
        rows = []
        # 测试仅用校验码验证来源与页面流转，不评价这两个维度上的论文内容质量。
        for group in payload["groups"]:
            papers, cited = [], []
            for paper in group["papers"]:
                chunk = paper["chunks"][0]
                match = re.search(r"PROJECT-[AB]-\d+", chunk["text"])
                self.assertIsNotNone(match)
                papers.append({"document_id": paper["document_id"],
                               "findings": [{"text": f"测试校验码：{match.group(0)}", "chunk_ids": [chunk["chunk_id"]]}],
                               "missing_information": "测试片段没有完整的论文实验信息。"})
                cited.append(chunk["chunk_id"])
            rows.append({"dimension": group["dimension"], "papers": papers,
                         "comparison": {"text": "两份测试文档报告不同校验码。", "chunk_ids": cited},
                         "comparison_limit": "固定响应，仅验证比较流程与来源展示。"})
        return AIMessage(content=json.dumps({"comparisons": rows}), response_metadata={"finish_reason": "stop"})

    def submit(self, at, question="比较这两篇论文的训练数据和实验结果。", *, route="compare"):
        # 只对本次比较提交设置分类；继承的普通问答范围测试继续使用基类默认值。
        self.route_override = {"intent": route, "reason": "固定页面测试路由", "clarification_question": ""}
        next(w for w in at.text_input if str(w.key).startswith("question_")).set_value(question)
        return next(w for w in at.button if w.label == "提交问题").click().run()

    def counters(self):
        return (len(self.classifications), len(self.comparison_plans), len(self.comparison_generations),
                self.encode_calls, len(self.rerank_calls))

    def test_complete_comparison_sources_rerun_and_followup_history(self):
        at = self.app(self.project_a, [self.aid, self.bid])
        self.checked(self.submit(at))
        message = at.session_state["chat_session"]["messages"][-1]
        self.assertEqual(message["response_kind"], "compare")
        self.assertEqual(message["workflow_trace"]["path"], ["start", "classify_request", "compare_handler"])
        artifact = message["comparison_artifact"]
        self.assertEqual(len(artifact["records"]), 4)
        self.assertEqual(self.counters(), (1, 1, 1, 4, 4))
        self.assertEqual((self.model_instances, self.rerank_model_instances), (1, 1))
        run_id = message["workflow_run_id"]
        events, issues = read_events(run_paths(self.project_a, run_id)["events"], run_id)
        self.assertEqual(issues, [])
        self.assertEqual(sum(r["step"] == "model_compare_handler" for r in summarize_calls(events)), 2)
        for record in artifact["records"]:
            for result in record["retrieval_artifact"]["retrieval_results"]:
                self.assertEqual(result["chunk"]["document_id"], record["task"]["document_id"])
                self.assertEqual(result["chunk"]["project_id"], self.project_a)
        for label in ("查看本轮工作流程", "查看比较原文依据", "查看比较计划与执行过程", "查看比较检索记录"):
            self.assertTrue(any(exp.label == label for exp in at.expander), label)
        for fixture in (self.fixture_a, self.fixture_b):
            self.assertIn(fixture["expected_code"], message["content"])
            self.assertTrue(any(fixture["expected_code"] in code.value for code in at.code))
        path = Path(artifact["report_path"])
        before_bytes, before_calls = path.read_bytes(), self.counters()
        saved = deepcopy(at.session_state["chat_session"]["messages"])
        self.assertEqual(len(at.toggle), 4)
        self.checked(at.toggle[0].set_value(True).run())
        self.checked(at.toggle[1].set_value(True).run())
        self.assertEqual(len(at.get("bidi_component")), 2)
        self.assertEqual(len(at.get("download_button")), 2)
        self.assertEqual(self.counters(), before_calls)
        self.assertEqual(at.session_state["chat_session"]["messages"], saved)
        self.checked(at.toggle[0].set_value(False).run())
        self.checked(at.toggle[1].set_value(False).run())
        self.checked(at.run())
        self.assertEqual(self.counters(), before_calls)
        self.assertEqual(path.read_bytes(), before_bytes)
        self.assertEqual(at.session_state["chat_session"]["messages"], saved)
        self.checked(self.submit(at, self.question, route="qa"))
        self.assertEqual(self.classifications[-1]["history"], [{"role": m["role"], "content": m["content"]} for m in saved])
        self.assertEqual(at.session_state["chat_session"]["messages"][-1]["response_kind"], "answer")
        self.assertEqual(len(self.comparison_generations), 1)

    def test_missing_dimensions_clarify_then_use_plain_history(self):
        self.plan_override = {"dimensions": [], "clarification_question": "希望比较哪些方面？"}
        at = self.app(self.project_a, [self.aid, self.bid])
        self.checked(self.submit(at, "比较一下这两篇论文。"))
        messages = deepcopy(at.session_state["chat_session"]["messages"])
        artifact = messages[-1]["comparison_artifact"]
        self.assertEqual(messages[-1]["response_kind"], "clarify")
        self.assertEqual(artifact["records"], [])
        self.assertEqual(self.counters(), (1, 1, 0, 0, 0))
        self.assertEqual(self.rerank_model_instances, 0)
        self.assertFalse(any(exp.label == "查看比较原文依据" for exp in at.expander))
        self.plan_override = None
        self.checked(self.submit(at, "训练数据和实验结果"))
        self.assertEqual(self.comparison_plans[-1]["history"], [{"role": m["role"], "content": m["content"]} for m in messages])
        self.assertEqual(at.session_state["chat_session"]["messages"][-1]["response_kind"], "compare")
        self.assertEqual(self.counters(), (2, 2, 1, 4, 4))

    def test_single_document_stops_before_comparison_service(self):
        at = self.app(self.project_a, [self.aid])
        self.checked(self.submit(at))
        message = at.session_state["chat_session"]["messages"][-1]
        self.assertEqual(message["response_kind"], "clarify")
        self.assertIn("恰好勾选", message["content"])
        self.assertNotIn("comparison_artifact", message)
        self.assertEqual(self.counters(), (1, 0, 0, 0, 0))
        self.assertEqual(list(Path("data/evaluations").glob("day34_comparison_run_*.json")), [])

    def test_failure_preserves_history_and_saves_failed_report(self):
        at = self.app(self.project_a, [self.aid, self.bid])
        self.checked(self.submit(at))
        before = deepcopy(at.session_state["chat_session"])
        for stage in ("plan", "generate"):
            with self.subTest(stage=stage):
                self.plan_failure = ValueError("controlled plan failure") if stage == "plan" else None
                self.comparison_failure = stage == "generate"
                existing = set(Path("data/evaluations").glob("day34_comparison_run_*.json"))
                self.submit(at)
                self.assertFalse(at.exception)
                self.assertTrue(any("本轮未加入会话" in error.value for error in at.error))
                self.assertEqual(at.session_state["chat_session"], before)
                created = set(Path("data/evaluations").glob("day34_comparison_run_*.json")) - existing
                self.assertEqual(len(created), 1)
                report = json.loads(created.pop().read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["accepted"])
                self.assertEqual(report["failed_stage"], stage)
                if stage == "generate":
                    self.assertEqual(report["result"]["generation_record"]["raw_response"], "{")
                    self.assertEqual(len(report["result"]["records"]), 4)

    def test_scope_change_and_clear_remove_saved_comparison_panels(self):
        at = self.app(self.project_a, [self.aid, self.bid])
        self.checked(self.submit(at))
        self.checked(next(w for w in at.button if w.label == "清空会话").click().run())
        self.assert_empty_chat(at)
        self.assertFalse(any(exp.label.startswith("查看比较") for exp in at.expander))
        self.checked(self.submit(at))
        self.select(at, [self.aid])
        self.assert_empty_chat(at)
        self.assertFalse(any(exp.label.startswith("查看比较") for exp in at.expander))
        self.switch(at, self.project_b)
        self.assert_empty_chat(at)
        self.assertFalse(any(exp.label.startswith("查看比较") for exp in at.expander))

    def test_explicit_review_in_history_renders_copy_without_new_requests(self):
        at = self.app(self.project_a, [self.aid, self.bid])
        self.checked(self.submit(at))
        session = deepcopy(at.session_state["chat_session"])
        message = session["messages"][-1]
        artifact = message["comparison_artifact"]
        path = Path(artifact["report_path"])
        original_bytes = path.read_bytes()
        result = artifact["generation_record"]["comparison"]
        finding = result["comparisons"][0]["papers"][0]["findings"][0]
        review = {"schema_version": 1, "purpose": "manual_comparison_review", "reviewer": "test",
                  "source_report": artifact["report_path"], "source_report_sha256": sha256(original_bytes).hexdigest(),
                  "source_comparison_sha256": comparison_digest(result),
                  "corrections": [{"path": ["comparisons", 0, "papers", 0, "findings", 0, "text"],
                                   "original_text": finding["text"], "replacement_text": "测试修订副本：" + finding["text"],
                                   "reason": "测试展示已记录修订", "evidence_chunk_ids": finding["chunk_ids"]}]}
        review_path = path.with_name("test_review.json")
        review_path.write_text(json.dumps(review), encoding="utf-8")
        project = scope_tests.load_project(self.project_a)
        reviewed = load_saved_comparison(path, project, artifact["documents"], review_path=review_path)
        message["comparison_artifact"] = reviewed
        message["content"] = format_paper_comparison(reviewed)
        at.session_state["chat_session"] = session
        calls = self.counters()
        self.checked(at.run())
        self.assertEqual(self.counters(), calls)
        self.assertTrue(any(exp.label == "查看比较内容修订" for exp in at.expander))
        self.assertTrue(any("测试修订副本" in block.value for block in at.markdown))
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(json.loads(original_bytes)["semantic_review"], "pending")
