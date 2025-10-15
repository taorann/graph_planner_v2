# graph_planner/env/planner_env.py
from typing import Dict, Any, Tuple
from runtime.sandbox import SandboxConfig, SandboxRuntime
from core.actions import (
    ActionUnion, ExploreAction, MemoryAction, RepairAction, SubmitAction
)

class PlannerEnv:
    """
    第一步：把操作落到容器；奖励=Submit 是否通过。
    后续在 _do_repair 接入 Collater→CGM→Guard。
    """
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlannerEnv":
        cfg = SandboxConfig(**d["sandbox"])
        return cls(issue=d["issue"], sandbox_cfg=cfg)

    def __init__(self, issue: Dict[str, Any], sandbox_cfg: SandboxConfig):
        self.issue = issue
        self.box = SandboxRuntime(sandbox_cfg)
        self.steps = 0
        self.last_info: Dict[str, Any] = {}

    def reset(self) -> Dict[str, Any]:
        self.steps = 0
        self.last_info = {"reset": True}
        return self._obs()

    def step(self, action: ActionUnion) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if isinstance(action, ExploreAction):
            info = {"kind": "explore", "op": action.op}
        elif isinstance(action, MemoryAction):
            info = {"kind": "memory"}
        elif isinstance(action, RepairAction):
            info = self._do_repair(action)
        elif isinstance(action, SubmitAction):
            info = self._do_submit()
        else:
            info = {"kind": "noop"}

        reward = 1.0 if info.get("submit") and info.get("tests", {}).get("passed", False) else 0.0
        done = bool(info.get("submit"))
        self.steps += 1
        self.last_info = info
        return self._obs(), reward, done, info

    # —— Repair：后续把 Collater→CGM→Guard 接在这里 ——
    def _do_repair(self, act: RepairAction) -> Dict[str, Any]:
        info = {"kind": "repair", "apply": act.apply}
        if not act.apply:
            return info
        # Step1：最小 NOOP；后续用 CGM 产出的 unified diff
        applied = True  # 或 self.box.apply_patch(minimal_diff)
        info["applied"] = applied
        info["lint_ok"] = self.box.lint()
        info["tests"] = self.box.test()
        return info

    def _do_submit(self) -> Dict[str, Any]:
        tests = self.box.test()
        return {"submit": True, "tests": tests, "patch": self.box.get_patch()}

    def _obs(self) -> Dict[str, Any]:
        return {"issue": self.issue, "steps": self.steps}

    def close(self):
        try:
            self.box.close()
        except Exception:
            pass