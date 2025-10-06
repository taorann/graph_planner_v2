import json
from typing import Dict, Any
from aci.runner import execute
from memory.memorizer import Memorizer, MemorizerConfig
from memory.types import Feedback
from actor.cgm_adapter import CGMAdapter
# 可选：启用 LLM 选择器
from memory.selector import LLMSelector
from memory.llm_provider import call_model

class Orchestrator:
    def __init__(self, repo_root: str, use_llm_selector: bool=False):
        selector = LLMSelector(call_model) if use_llm_selector else None
        self.repo = repo_root
        self.mem  = Memorizer(repo_root, MemorizerConfig(), selector=selector)
        self.actor = CGMAdapter()

    def run_once(self, issue: str, max_rounds: int = 6) -> Dict[str, Any]:
        S = self.mem.bootstrap(issue)
        failing_hint = ""
        events = []
        for step in range(1, max_rounds+1):
            plan = self.mem.plan(failing_hint)
            patch = self.actor.generate(S, plan)
            if not patch:
                break

            # ACI 执行：lint → edit → pytest
            if plan.budget.get("lint_required", True):
                lint = execute({"tool":"lint_check","args":{"path": patch["path"]}})
                if not lint["ok"]:
                    events.append({"step":step, "lint_error": lint})
            edit = execute({"tool":"edit_lines","args": patch})
            test = execute({"tool":"run_tests","args":{"targets":["-q"], "timeout_s":180}})
            fb = self._to_feedback(edit, test)

            events.append({"step":step, "plan": plan, "patch": patch, "feedback": fb.__dict__})

            # 更新记忆
            self.mem.update(fb)

            if fb.tests_failed == 0:
                return {"ok": True, "events": events}

            failing_hint = (test.get("observations",{}).get("test_report",{}) or {}).get("top_assert","")

        return {"ok": False, "events": events}

    @staticmethod
    def _to_feedback(edit: Dict[str,Any], test: Dict[str,Any]) -> Feedback:
        diff = (edit.get("effects",{}) or {}).get("diff_summary","")
        changed = (edit.get("effects",{}) or {}).get("lines_changed",[])
        trep = (test.get("observations",{}) or {}).get("test_report",{}) or {}
        return Feedback(
            diff_summary = diff,
            lines_changed = changed,
            top_assert = trep.get("top_assert"),
            tests_failed = trep.get("failed", 1),
            first_failure_frame = trep.get("first_failure_frame"),
        )
