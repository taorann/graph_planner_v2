# graph_planner/env/planner_env.py
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from difflib import unified_diff

from ..agents.common import text_protocol
from ..agents.common.contracts import ProtocolError, validate_planner_action
from ..core.actions import (
    ActionUnion,
    ExploreAction,
    MemoryAction,
    NoopAction,
    RepairAction,
    SubmitAction,
)
from ..infra.config import Config, load as load_config
from ..memory import graph_adapter, mem_candidates, subgraph_store, text_memory
from ..memory.subgraph_store import WorkingSubgraph
from ..runtime.sandbox import SandboxConfig, SandboxRuntime
from aci.schema import Plan, PlanTarget
from aci.guard import GuardError, enforce_patch_guard
from ..agents.rule_based.test_prioritizer import prioritize_tests


DEFAULT_MEMORY_CAPS = {
    "nodes": 200,
    "edges": 1000,
    "frontier": 50,
    "planner_tokens": 2000,
    "cgm_tokens": 16000,
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


class PlannerEnv:
    """Graph Planner 环境封装。

    负责：
      * 调用 SandboxRuntime 与容器交互；
      * 维护工作子图与记忆日志；
      * 将 Explore/Memory/Repair/Submit 动作映射为具体容器操作；
      * 将状态打包为 Observation 供上层 Agent 使用。
    """

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannerEnv":
        cfg = SandboxConfig(**payload["sandbox"])
        return cls(issue=payload.get("issue", {}), sandbox_cfg=cfg)

    def __init__(self, issue: Dict[str, Any], sandbox_cfg: SandboxConfig):
        self.issue: Dict[str, Any] = issue or {}
        self.issue_id: str = str(self.issue.get("id") or "__default__")
        self.sandbox_cfg = sandbox_cfg
        self.box = SandboxRuntime(sandbox_cfg)
        self.config: Config = load_config()
        self.config_dict: Dict[str, Any] = self.config.to_dict()
        io_cfg = self.config_dict.get("io") if isinstance(self.config_dict, Mapping) else {}
        if not isinstance(io_cfg, Mapping):
            io_cfg = {}
        strict_env = os.environ.get("GRAPH_PLANNER_STRICT_IO") or os.environ.get("STRICT_PLANNER_IO")
        if strict_env is not None:
            self._strict_io = str(strict_env).strip().lower() in {"1", "true", "yes"}
        else:
            self._strict_io = bool(io_cfg.get("strict_planner_io", False))

        self.steps: int = 0
        self.last_info: Dict[str, Any] = {}
        self.repo_root_in_container: str = sandbox_cfg.workdir or "."

        self.subgraph: WorkingSubgraph = subgraph_store.new()
        self.last_candidates: List[Dict[str, Any]] = []
        self.last_reads: List[Dict[str, Any]] = []
        raw_caps = dict(getattr(self.config, "memory_caps", {}) or {})
        self.memory_caps: Dict[str, int] = {**DEFAULT_MEMORY_CAPS, **raw_caps}
        self.memory_graph_store: Optional[text_memory.WorkingGraphStore] = None
        self.memory_text_store: Optional[text_memory.NoteTextStore] = None
        self.memory_state: Optional[text_memory.TurnState] = None

    # ------------------------------------------------------------------
    # 环境基础
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.steps = 0
        self.last_info = {"reset": True}

        # 准备图句柄与子图
        graph_adapter.connect()
        try:
            self.subgraph = subgraph_store.load(self.issue_id)
        except Exception:
            self.subgraph = subgraph_store.new()

        self.memory_graph_store = text_memory.WorkingGraphStore(self.subgraph)
        self.memory_text_store = text_memory.NoteTextStore()
        self.memory_state = text_memory.TurnState(
            graph_store=self.memory_graph_store,
            text_store=self.memory_text_store,
        )
        self.memory_state.size = text_memory.Size(
            nodes=len(self.subgraph.nodes),
            edges=len(self.subgraph.edges),
            frontier=0,
            planner_tokens_est=0,
            cgm_tokens_est=0,
        )

        # 记录容器内工作目录（pwd 优先，其次配置）
        try:
            out, _ = self.box.run("pwd", timeout=10)
            pwd = (out or "").strip().splitlines()[-1]
            if pwd:
                self.repo_root_in_container = pwd
        except Exception:
            pass

        self.last_candidates = []
        self.last_reads = []
        return self._obs()

    def close(self) -> None:
        try:
            subgraph_store.save(self.issue_id, self.subgraph)
        except Exception:
            pass
        try:
            self.box.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 主 step
    # ------------------------------------------------------------------
    def step(
        self, action: ActionUnion | Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        validation_meta: Dict[str, Any] = {}
        action_obj: ActionUnion

        if isinstance(action, Mapping):
            try:
                action_obj = validate_planner_action(action)
                validation_meta = getattr(action_obj, "_meta", {}) or {}
            except ProtocolError as exc:
                if self._strict_io:
                    info = {"error": exc.code, "detail": exc.detail}
                    return self._obs(), -0.05, False, info
                raise
        else:
            action_obj = action
            if self._strict_io:
                try:
                    payload = self._serialise_action_for_validation(action_obj)
                except Exception as exc:
                    info = {"error": "invalid-action", "detail": str(exc)}
                    return self._obs(), -0.05, False, info
                try:
                    action_obj = validate_planner_action(payload)
                    validation_meta = getattr(action_obj, "_meta", {}) or {}
                except ProtocolError as exc:
                    info = {"error": exc.code, "detail": exc.detail}
                    return self._obs(), -0.05, False, info

        if isinstance(action_obj, ExploreAction):
            info = self._handle_explore(action_obj)
        elif isinstance(action_obj, MemoryAction):
            info = self._handle_memory(action_obj)
        elif isinstance(action_obj, RepairAction):
            info = self._handle_repair(action_obj)
        elif isinstance(action_obj, SubmitAction):
            info = self._handle_submit()
        elif isinstance(action_obj, NoopAction):
            info = {"kind": "noop"}
        else:
            info = {"kind": "noop"}

        if validation_meta.get("capped"):
            info["capped"] = True
            capped_fields = validation_meta.get("capped_fields") or {}
            if capped_fields:
                info["capped_fields"] = capped_fields
        warnings = validation_meta.get("warnings")
        if warnings:
            existing = info.get("warnings")
            if isinstance(existing, list):
                info["warnings"] = existing + [w for w in warnings if w not in existing]
            else:
                info["warnings"] = list(warnings)

        if self.memory_state:
            kind = info.get("kind")
            if kind == "explore":
                self.memory_state.latest_explore = info
            elif kind and kind != "memory":
                self.memory_state.latest_observation = info

        reward = 1.0 if info.get("submit") and info.get("tests", {}).get("passed") else 0.0
        done = bool(info.get("submit"))
        self.steps += 1
        self.last_info = info
        return self._obs(), reward, done, info

    @staticmethod
    def _serialise_action_for_validation(action: ActionUnion) -> Dict[str, Any]:
        payload = action.dict(exclude={"schema_version"}, exclude_none=True)
        name = payload.pop("type", getattr(action, "type", None))
        if not name:
            raise ValueError("action missing type field")
        return {"name": name, "params": payload}

    # ------------------------------------------------------------------
    # Explore / Memory handlers
    # ------------------------------------------------------------------
    def _handle_explore(self, act: ExploreAction) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": "explore", "op": act.op}

        if act.op == "find":
            nodes: List[Dict[str, Any]] = []
            for anchor in act.anchors:
                try:
                    nodes.extend(graph_adapter.find_nodes_by_anchor(anchor))
                except Exception:
                    continue
            info["nodes"] = nodes
            return info

        if act.op == "expand":
            max_per_anchor = max(1, int(act.limit or self.config.max_nodes_per_anchor))
            total_limit = int(self.config.candidate_total_limit)
            dir_k = int(self.config.dir_diversity_k)
            candidates = mem_candidates.build_mem_candidates(
                subgraph=self.subgraph,
                anchors=act.anchors,
                max_nodes_per_anchor=max_per_anchor,
                total_limit=total_limit,
                dir_diversity_k=dir_k,
            )
            self.last_candidates = candidates
            info["candidates"] = candidates
            info["subgraph_stats"] = subgraph_store.stats(self.subgraph)
            return info

        if act.op == "read":
            resolved: List[Dict[str, Any]] = []
            for node_id in act.nodes:
                node = self._resolve_node(node_id)
                if node:
                    resolved.append(node)
            snippets: List[Dict[str, Any]] = []
            for node in resolved[: max(1, int(act.limit or 3))]:
                snippet = self._read_node_snippet(node)
                if snippet:
                    snippets.append(snippet)
            self.last_reads = snippets
            info["snippets"] = snippets
            return info

        info["warning"] = f"unknown explore op: {act.op}"
        return info

    def _handle_memory(self, act: MemoryAction) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": "memory"}
        if not self.memory_state:
            info.update({"ok": False, "error": "memory-uninitialized"})
            return info

        caps = self.memory_caps or {}
        try:
            if act.intent == "delete":
                result = text_memory.memory_delete(
                    self.memory_state, act.target, act.scope, act.selector
                )
            else:
                result = text_memory.memory_commit(
                    self.memory_state, act.target, act.scope, act.selector, caps
                )
        except Exception as exc:
            info.update({"ok": False, "error": f"memory-exception:{exc}"})
            return info

        info.update(result)
        info["subgraph_stats"] = subgraph_store.stats(self.subgraph)
        if info.get("ok"):
            try:
                subgraph_store.save(self.issue_id, self.subgraph)
            except Exception:
                pass
        return info

    # ------------------------------------------------------------------
    # Repair / Submit
    # ------------------------------------------------------------------
    def _handle_repair(self, act: RepairAction) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": "repair", "apply": act.apply, "plan": act.plan}
        if not act.apply:
            return info

        if act.patch and isinstance(act.patch, dict) and act.patch.get("edits"):
            patch: Dict[str, Any] = dict(act.patch)
            if "summary" not in patch:
                patch["summary"] = act.plan or ""
            plan = self._build_plan(act.plan_targets)
            try:
                enforce_patch_guard(patch, plan, self.config)
            except GuardError as ge:
                info["guard_error"] = str(ge)
                info["applied"] = False
                return info

            apply_result = self._apply_patch_edits(patch.get("edits") or [])
            info.update(apply_result)
            info["plan_targets"] = act.plan_targets

            lint_report = self.box.lint()
            tests_report = self.box.test()
            info["lint"] = lint_report
            info["tests"] = tests_report
            info["applied"] = bool(apply_result.get("success"))
            info["priority_tests"] = prioritize_tests(
                self._observation_pack(),
                subgraph=self.subgraph,
            ).get("priority_tests", [])
            return info

        subplan_text = (act.plan or "").strip()
        focus_ids = [str(t.get("id")) for t in act.plan_targets or [] if isinstance(t, dict) and t.get("id")]
        try:
            runtime_state = self._build_repair_state(subplan_text, focus_ids)
        except Exception as exc:
            info["applied"] = False
            info["error"] = f"repair-state-failed:{exc}"
            return info

        action_payload = {"name": "repair", "params": {"subplan": subplan_text, "focus_ids": focus_ids, "apply": act.apply}}
        try:
            result = text_protocol.handle_planner_repair(action_payload, runtime_state)
        except ProtocolError as exc:
            info["applied"] = False
            info["error"] = f"text-repair-error:{exc.code}"
            info["fallback_reason"] = exc.code
            info["msg"] = exc.detail
            return info
        except Exception as exc:
            info["applied"] = False
            info["error"] = f"text-repair-error:{exc}"
            return info

        info.update(result)
        info.setdefault("plan_targets", act.plan_targets)
        info.setdefault("applied", bool(result.get("applied")))
        return info

    def _handle_submit(self) -> Dict[str, Any]:
        tests = self.box.test()
        patch_text = self.box.get_patch()
        return {"submit": True, "tests": tests, "patch": patch_text}

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------
    def _obs(self) -> Dict[str, Any]:
        stats = subgraph_store.stats(self.subgraph)
        return {
            "issue": self.issue,
            "steps": self.steps,
            "subgraph": self.subgraph.to_json_obj(),
            "subgraph_stats": stats,
            "last_info": self.last_info,
            "observation_pack": self._observation_pack(stats),
            "reset": bool(self.last_info.get("reset")),
        }

    def _observation_pack(self, stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        stats = stats or subgraph_store.stats(self.subgraph)
        failure = self.issue.get("failure_frame") or {}
        issue_text = " ".join(
            str(x)
            for x in (
                self.issue.get("title"),
                self.issue.get("body"),
                self.issue.get("description"),
            )
            if x
        ).strip()
        pack = {
            "issue": issue_text,
            "top_assert": self.issue.get("top_assert"),
            "error_kind": self.issue.get("error_kind"),
            "failure_frame": failure,
            "subgraph_stats": stats,
            "cost": {"tokens": int(self.last_info.get("collate_meta", {}).get("est_tokens", 0))},
            "cfg": self.config_dict,
        }
        return pack

    def _resolve_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if not node_id:
            return None
        node = self.subgraph.get_node(node_id)
        if node:
            return dict(node)
        for cand in self.last_candidates:
            if cand.get("id") == node_id:
                return dict(cand)
        return None

    def _read_node_snippet(self, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        path = node.get("path")
        if not path:
            return None
        span = node.get("span") or {}
        start = _safe_int(span.get("start"), 1)
        end = max(start, _safe_int(span.get("end"), start))
        pad = 3
        start0 = max(1, start - pad)
        end0 = end + pad

        payload = json.dumps(
            {
                "path": path,
                "base": self.repo_root_in_container,
                "start": start0,
                "end": end0,
            }
        )
        script = (
            "python - <<'PY'\n"
            "import json\nfrom pathlib import Path\n"
            "req=json.loads('''" + payload.replace("'", "\\'") + "''')\n"
            "path=Path(req['path'])\n"
            "if not path.is_absolute():\n"
            "    path = Path(req['base']).joinpath(path)\n"
            "try:\n"
            "    text = path.read_text(encoding='utf-8', errors='ignore').splitlines()\n"
            "except Exception:\n"
            "    text = []\n"
            "start=max(1,int(req['start']))\n"
            "end=max(start,int(req['end']))\n"
            "snippet=[]\n"
            "for idx in range(start-1, min(len(text), end)):\n"
            "    snippet.append(f'{idx+1:04d}: {text[idx]}')\n"
            "print(json.dumps({'path': str(path), 'start': start, 'end': min(end, len(text)), 'snippet': snippet}))\n"
            "PY"
        )
        out, rc = self.box.run(script, timeout=30)
        if rc != 0:
            return None
        try:
            data = json.loads(out.strip().splitlines()[-1])
        except Exception:
            return None
        abs_path = data.get("path")
        data.update({"node_id": node.get("id"), "path": path, "span": span})
        if abs_path:
            data["abs_path"] = abs_path
        return data

    def _build_repair_state(self, subplan: str, focus_ids: List[str]) -> text_protocol.RepairRuntimeState:
        if not subplan:
            raise ValueError("repair subplan is required for CGM execution")
        text_memory = self._build_text_memory_snapshot()
        related_files = self._collect_related_files(focus_ids)
        if not related_files:
            related_files = self._collect_related_files(self._default_focus_ids())
        default_focus = focus_ids or self._default_focus_ids()
        token_budget = int(getattr(self.config.cgm, "max_input_tokens", 8192))
        state = text_protocol.RepairRuntimeState(
            issue=dict(self.issue),
            subgraph=subgraph_store.wrap(self.subgraph),
            sandbox=self.box,
            repo_root=self.repo_root_in_container,
            text_memory=text_memory,
            related_files=related_files,
            default_focus_ids=default_focus,
            snippets=self.last_reads or [],
            token_budget=token_budget,
            max_graph_nodes=min(128, int(self.config.candidate_total_limit or 200)),
            max_files=4,
            max_file_bytes=40000,
            cgm_top_k=1,
            cgm_generate=self._generate_cgm_candidates,
        )
        return state

    def _build_text_memory_snapshot(self) -> Dict[str, str]:
        stats = subgraph_store.stats(self.subgraph)
        summary_parts = [
            f"steps={self.steps}",
            f"nodes={stats.get('nodes', 0)}",
            f"edges={stats.get('edges', 0)}",
        ]
        session_summary = " | ".join(summary_parts)
        turn_notes = json.dumps(self.last_info, ensure_ascii=False) if self.last_info else ""
        memory = {"session_summary": session_summary, "turn_notes": turn_notes}
        return memory

    def _collect_related_files(self, focus_ids: Iterable[str]) -> Dict[str, str]:
        paths: List[str] = []
        for node_id in focus_ids:
            node = self._resolve_node(node_id)
            if node and node.get("path"):
                paths.append(str(node.get("path")))
        if not paths:
            paths.extend([snip.get("path") for snip in self.last_reads or [] if snip.get("path")])
        unique_paths = []
        seen = set()
        for path in paths:
            if path and path not in seen:
                seen.add(path)
                unique_paths.append(path)
        files: Dict[str, str] = {}
        for path in unique_paths:
            text = self._read_full_file_text(path)
            if text:
                files[path] = text
        return files

    def _read_full_file_text(self, path: str) -> str:
        payload = json.dumps({"path": path, "base": self.repo_root_in_container})
        script = (
            "python - <<'PY'\n"
            "import json\nfrom pathlib import Path\n"
            "req=json.loads('''" + payload.replace("'", "\\'") + "''')\n"
            "path=Path(req['path'])\n"
            "if not path.is_absolute():\n"
            "    path = Path(req['base']).joinpath(path)\n"
            "try:\n"
            "    text = path.read_text(encoding='utf-8', errors='ignore')\n"
            "except Exception:\n"
            "    text = ''\n"
            "print(json.dumps({'text': text}))\n"
            "PY"
        )
        out, rc = self.box.run(script, timeout=30)
        if rc != 0:
            return ""
        lines = [line for line in out.strip().splitlines() if line.strip()]
        if not lines:
            return ""
        try:
            data = json.loads(lines[-1])
        except Exception:
            return ""
        return str(data.get("text") or "")

    def _default_focus_ids(self) -> List[str]:
        ids = [snip.get("node_id") for snip in self.last_reads or [] if snip.get("node_id")]
        if ids:
            return [str(i) for i in ids if i]
        return [str(nid) for nid in getattr(self.subgraph, "node_ids", [])]

    def _plan_targets_from_focus(self, focus_ids: Iterable[str]) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []
        for node_id in focus_ids:
            node = self._resolve_node(node_id)
            if not node:
                continue
            path = node.get("path")
            span = node.get("span") or {}
            if not path:
                continue
            start = _safe_int(span.get("start"), 1)
            end = max(start, _safe_int(span.get("end"), start))
            targets.append(
                {
                    "path": str(path),
                    "start": start,
                    "end": end,
                    "id": node.get("id") or node_id,
                    "why": "graph_planner-focus",
                }
            )
        return targets

    def _generate_cgm_candidates(self, payload: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        focus_ids = payload.get("graph", {}).get("focus_ids") or []
        plan_targets = self._plan_targets_from_focus(focus_ids)
        if not plan_targets:
            plan_targets = self._plan_targets_from_snippets(self.last_reads or [])
        if not plan_targets:
            return []
        plan = self._build_plan(plan_targets)
        plan_text = payload.get("plan_text") or "\n".join(payload.get("plan") or [])
        linearized = subgraph_store.linearize(self.subgraph, mode=getattr(self.config.collate, "mode", "wsd"))
        patch = cgm_adapter.generate(
            subgraph_linearized=linearized,
            plan=plan,
            constraints={"max_edits": max(1, len(plan_targets))},
            snippets=self.last_reads or [],
            plan_text=plan_text,
            issue=self.issue,
        )
        edits = list(patch.get("edits") or [])
        if not edits:
            return []
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for edit in edits:
            path = str(edit.get("path") or "")
            if not path:
                continue
            start = _safe_int(edit.get("start"), 1)
            end = max(start, _safe_int(edit.get("end"), start))
            new_text = str(edit.get("new_text") or "")
            if new_text and not new_text.endswith("\n"):
                new_text = new_text + "\n"
            grouped.setdefault(path, []).append(
                {
                    "path": path,
                    "start": start,
                    "end": end,
                    "new_text": new_text,
                }
            )

        summary = patch.get("summary", "")
        confidence = float(patch.get("confidence", 0.5))
        tests = patch.get("tests", [])
        candidates: List[Dict[str, Any]] = []
        for path, group in grouped.items():
            candidates.append(
                {
                    "patch": {"edits": group},
                    "summary": summary,
                    "confidence": confidence,
                    "tests": tests,
                    "path": path,
                }
            )
        return candidates[:k]

    def _build_plan(self, targets: List[Dict[str, Any]]) -> Plan:
        plan_targets: List[PlanTarget] = []
        for idx, target in enumerate(targets or []):
            try:
                path = str(target.get("path"))
                start = _safe_int(target.get("start"), 1)
                end = max(start, _safe_int(target.get("end"), start))
                plan_targets.append(
                    PlanTarget(
                        path=path,
                        start=start,
                        end=end,
                        id=str(target.get("id") or f"{path}::{start}-{end}::#{idx}"),
                        confidence=float(target.get("confidence", 1.0)),
                        why=str(target.get("why", "agent-plan")),
                    )
                )
            except Exception:
                continue
        return Plan(targets=plan_targets, budget={"mode": self.config.mode}, priority_tests=[])

    def _apply_patch_edits(self, edits: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not edits:
            return {"success": False, "message": "no_edits", "changed_files": []}

        changed_files: List[str] = []
        logs: List[str] = []
        success = True

        for edit in edits:
            path = edit.get("path")
            start = _safe_int(edit.get("start"), 1)
            end = max(start, _safe_int(edit.get("end"), start))
            new_text = edit.get("new_text", "")
            if not path:
                success = False
                logs.append("missing_path")
                continue
            encoded = base64.b64encode(str(new_text).encode("utf-8")).decode("ascii")
            payload = json.dumps(
                {
                    "path": path,
                    "base": self.repo_root_in_container,
                    "start": start,
                    "end": end,
                    "content": encoded,
                }
            )
            script = (
                "python - <<'PY'\n"
                "import base64, json\nfrom pathlib import Path\n"
                "req=json.loads('''" + payload.replace("'", "\\'") + "''')\n"
                "path=Path(req['path'])\n"
                "if not path.is_absolute():\n"
                "    path = Path(req['base']).joinpath(path)\n"
                "lines = []\n"
                "try:\n"
                "    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines(True)\n"
                "except Exception:\n"
                "    pass\n"
                "start=max(1,int(req['start']))\n"
                "end=max(start,int(req['end']))\n"
                "content = base64.b64decode(req['content']).decode('utf-8')\n"
                "replacement = content.splitlines(True)\n"
                "if lines and lines[end-1:end] and lines[end-1].endswith('\n') and (not replacement or not replacement[-1].endswith('\n')):\n"
                "    if replacement:\n"
                "        replacement[-1] = replacement[-1] + '\n'\n"
                "    else:\n"
                "        replacement = ['\n']\n"
                "if end > len(lines):\n"
                "    raise SystemExit(1)\n"
                "lines[start-1:end] = replacement\n"
                "path.write_text(''.join(lines), encoding='utf-8')\n"
                "print(json.dumps({'path': str(path), 'changed_lines': len(replacement)}))\n"
                "PY"
            )
            out, rc = self.box.run(script, timeout=60)
            if rc != 0:
                success = False
                logs.append(f"edit_failed:{path}:{start}-{end}")
                continue
            changed_files.append(path)
            logs.append(out.strip())

        diff_text = self.box.get_patch()
        return {
            "success": success,
            "changed_files": sorted({p for p in changed_files}),
            "git_diff": diff_text,
            "logs": logs,
        }

