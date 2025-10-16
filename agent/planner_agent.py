# graph_planner/agent/planner_agent.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.actions import ExploreAction, MemoryAction, RepairAction, SubmitAction
from infra.config import Config, load as load_config
from memory import anchor_planner, subgraph_store
from aci.schema import Plan, PlanTarget
from agents.rule_based import cgm_adapter


@dataclass
class AgentState:
    issue: Dict[str, Any] = field(default_factory=dict)
    phase: str = "expand"
    last_candidates: List[Dict[str, Any]] = field(default_factory=list)
    last_snippets: List[Dict[str, Any]] = field(default_factory=list)
    last_memory: Dict[str, Any] = field(default_factory=dict)
    last_repair: Dict[str, Any] = field(default_factory=dict)
    plan_targets: List[Dict[str, Any]] = field(default_factory=list)
    plan_text: str = ""


class PlannerAgent:
    """基于图规划的最小决策器。

    流程：
      1. expand：根据 Anchor Planner 决定 1-hop 扩展；
      2. memory：调用环境维护子图；
      3. read：阅读新增节点所在片段；
      4. plan → repair：生成自然语言计划并调用 CGM 产出补丁；
      5. submit：运行测试获取 reward。
    """

    def __init__(self) -> None:
        self.cfg: Config = load_config()
        self.state = AgentState()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.state.issue or obs.get("reset") or obs.get("steps") == 0:
            self._on_reset(obs)
        self._update_state(obs)

        if self.state.phase == "expand":
            return self._act_expand(obs)
        if self.state.phase == "memory":
            return self._wrap_action(MemoryAction())
        if self.state.phase == "read":
            return self._act_read()
        if self.state.phase == "plan":
            return self._act_repair(obs)
        return self._wrap_action(SubmitAction())

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------
    def _on_reset(self, obs: Dict[str, Any]) -> None:
        self.state = AgentState(issue=obs.get("issue") or {})
        self.state.phase = "expand"

    def _update_state(self, obs: Dict[str, Any]) -> None:
        info = obs.get("last_info") or {}
        kind = info.get("kind")
        if kind == "explore" and info.get("op") == "expand":
            self.state.last_candidates = info.get("candidates", [])
            self.state.phase = "memory"
        elif kind == "memory":
            self.state.last_memory = info
            self.state.phase = "read"
        elif kind == "explore" and info.get("op") == "read":
            self.state.last_snippets = info.get("snippets", [])
            self.state.phase = "plan"
        elif kind == "repair":
            self.state.last_repair = info
            if info.get("applied"):
                self.state.phase = "submit"
            else:
                # Guard 拒绝或补丁失败时回到扩展阶段重新定位
                self.state.phase = "expand"

    # ------------------------------------------------------------------
    # Phase actions
    # ------------------------------------------------------------------
    def _act_expand(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        observation_pack = obs.get("observation_pack") or {}
        proposal = anchor_planner.propose(observation_pack)
        anchors = proposal.get("anchors", [])
        if not anchors:
            failure = observation_pack.get("failure_frame", {})
            fallback = failure.get("path") or (self.state.issue.get("path") if isinstance(self.state.issue, dict) else None)
            if fallback:
                anchors = [{"kind": "file", "text": fallback}]
        hop = max(1, int(proposal.get("hop", 1)))
        limit = int(self.cfg.max_nodes_per_anchor)
        self.state.phase = "memory"
        action = ExploreAction(op="expand", anchors=anchors, hop=hop, limit=limit)
        return self._wrap_action(action)

    def _act_read(self) -> Dict[str, Any]:
        candidates = self.state.last_candidates or []
        sorted_cands = sorted(
            [c for c in candidates if c.get("id")],
            key=lambda c: float(c.get("score", 0.0)),
            reverse=True,
        )
        if not sorted_cands:
            self.state.phase = "plan"
            return self._wrap_action(ExploreAction(op="read", nodes=[], limit=0))
        top_ids = [c["id"] for c in sorted_cands[:3]]
        self.state.phase = "plan"
        return self._wrap_action(ExploreAction(op="read", nodes=top_ids, limit=len(top_ids)))

    def _act_repair(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        snippets = self.state.last_snippets or []
        if not snippets:
            snippets = self._fallback_snippets()

        plan_targets = []
        plan_lines = []
        for snip in snippets:
            path = snip.get("path")
            start = int(snip.get("start", 1))
            end = int(snip.get("end", start))
            node_id = snip.get("node_id") or f"{path}::{start}-{end}"
            plan_targets.append(
                {
                    "path": path,
                    "start": start,
                    "end": end,
                    "id": node_id,
                    "confidence": 1.0,
                    "why": "graph_planner-agent",
                }
            )
            preview = " | ".join(line.split(":", 1)[-1].strip() for line in (snip.get("snippet") or [])[:3])
            plan_lines.append(f"- {path} L{start}-{end}: {preview[:160]}")

        if not plan_targets:
            self.state.phase = "submit"
            return self._wrap_action(SubmitAction())

        plan_text = "Plan to address the following locations:\n" + "\n".join(plan_lines)
        plan_obj = Plan(
            targets=[
                PlanTarget(
                    path=pt["path"],
                    start=int(pt["start"]),
                    end=int(pt["end"]),
                    id=str(pt["id"]),
                    why=pt.get("why", "graph_planner-agent"),
                )
                for pt in plan_targets
            ],
            budget={"mode": self.cfg.mode},
            priority_tests=[],
        )

        subgraph = subgraph_store.wrap(obs.get("subgraph") or {})
        linearized = subgraph_store.linearize(subgraph, mode=self.cfg.collate.mode)
        patch = cgm_adapter.generate(
            subgraph_linearized=linearized,
            plan=plan_obj,
            constraints={"max_edits": max(1, len(plan_targets))},
            snippets=snippets,
            plan_text=plan_text,
            issue=self.state.issue,
        )
        patch.setdefault("summary", plan_text)

        self.state.plan_targets = plan_targets
        self.state.plan_text = plan_text
        self.state.phase = "submit"

        action = RepairAction(
            apply=True,
            issue=self.state.issue,
            plan=plan_text,
            plan_targets=plan_targets,
            patch=patch,
        )
        return self._wrap_action(action)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _wrap_action(self, action) -> Dict[str, Any]:
        return {
            "prompt": f"issue={self.state.issue.get('id', 'NA')} phase={self.state.phase}",
            "response": action.type,
            "action_obj": action,
        }

    def _fallback_snippets(self) -> List[Dict[str, Any]]:
        snippets: List[Dict[str, Any]] = []
        for cand in self.state.last_candidates[:1]:
            path = cand.get("path")
            span = cand.get("span") or {}
            start = int(span.get("start", 1))
            end = int(span.get("end", start))
            snippet_line = f"{start:04d}: {cand.get('name', '')}"
            snippets.append(
                {
                    "path": path,
                    "start": start,
                    "end": end,
                    "node_id": cand.get("id"),
                    "snippet": [snippet_line],
                }
            )
        return snippets

