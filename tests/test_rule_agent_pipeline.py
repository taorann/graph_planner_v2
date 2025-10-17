import base64
import json
import os
from pathlib import Path

from graph_planner.core.actions import ExploreAction, MemoryAction, RepairAction, SubmitAction
from graph_planner.env.planner_env import PlannerEnv
from graph_planner.infra import telemetry
from graph_planner.infra.config import load as load_config
from graph_planner.runtime.sandbox import SandboxConfig


class FakeSandbox:
    """Minimal sandbox runtime that mimics container behaviour for tests."""

    def __init__(self, cfg: SandboxConfig) -> None:  # pragma: no cover - signature parity
        self.cfg = cfg
        self.repo_root = "/repo"
        # 初始文件带有 Bug：return a - b
        self.files = {
            os.path.join(self.repo_root, "app", "calc.py"): """def add(a, b):\n    return a - b\n"""
        }
        self.tests_invocations = 0
        self._trace = {"reads": [], "edits": [], "commands": []}

    def _relpath(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.repo_root)
        except Exception:
            return path

    # ---- container command helpers -------------------------------------------------
    def run(self, command: str, timeout: int | None = None):  # pragma: no cover - exercised via test
        command = command.strip()
        if command == "pwd":
            self._trace["commands"].append({"kind": "pwd"})
            return f"{self.repo_root}\n", 0

        if "json.loads('''" in command:
            payload = command.split("json.loads('''", 1)[1].split("''')", 1)[0]
            data = json.loads(payload)
            rel_path = data.get("path", "")
            abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.repo_root, rel_path)

            if "content" in data:  # patch application
                content = base64.b64decode(data["content"]).decode("utf-8")
                lines = self.files.get(abs_path, "").splitlines(True)
                start = int(data.get("start", 1))
                end = max(start, int(data.get("end", start)))
                if end > len(lines):
                    return "", 1
                before = "".join(lines[start - 1 : end])
                replacement = content.splitlines(True)
                if replacement and not replacement[-1].endswith("\n"):
                    replacement[-1] += "\n"
                if not replacement:
                    replacement = ["\n"]
                lines[start - 1 : end] = replacement
                self.files[abs_path] = "".join(lines)
                self._trace["edits"].append(
                    {
                        "path": self._relpath(abs_path),
                        "start": start,
                        "end": end,
                        "before": before,
                        "after": "".join(replacement),
                    }
                )
                self._trace["commands"].append(
                    {
                        "kind": "apply_patch",
                        "path": self._relpath(abs_path),
                        "start": start,
                        "end": end,
                        "changed_lines": len(replacement),
                    }
                )
                response = {"path": abs_path, "changed_lines": len(replacement)}
                return json.dumps(response) + "\n", 0

            # snippet request
            start = int(data.get("start", 1))
            end = max(start, int(data.get("end", start)))
            lines = self.files.get(abs_path, "").splitlines()
            snippet = [
                f"{idx + 1:04d}: {lines[idx]}"
                for idx in range(start - 1, min(len(lines), end))
            ]
            response = {
                "path": str(Path(abs_path)),
                "start": start,
                "end": min(end, len(lines)),
                "snippet": snippet,
            }
            self._trace["reads"].append(
                {
                    "path": self._relpath(abs_path),
                    "start": start,
                    "end": min(end, len(lines)),
                }
            )
            self._trace["commands"].append(
                {
                    "kind": "read_snippet",
                    "path": self._relpath(abs_path),
                    "start": start,
                    "end": min(end, len(lines)),
                }
            )
            return json.dumps(response) + "\n", 0

        # 默认返回空输出（例如工具命令）
        self._trace["commands"].append({"kind": "noop", "command": command})
        return "", 0

    def lint(self):  # pragma: no cover - trivial
        return {"ok": True}

    def test(self):  # pragma: no cover - trivial
        self.tests_invocations += 1
        passed = "return a + b" in self.files[os.path.join(self.repo_root, "app", "calc.py")]
        payload = {
            "kind": "test_run",
            "backend": "fake",
            "command": "python -m pytest -q",
            "selector": [],
            "duration_sec": 0.0,
            "result": {"mode": "pytest", "rc": 0 if passed else 1, "passed": passed},
            "stdout": "FakeSandbox pytest output",
            "repair_trace": {
                "reads": list(self._trace["reads"]),
                "edits": list(self._trace["edits"]),
                "commands": list(self._trace["commands"]),
                "final_files": {
                    self._relpath(path): content
                    for path, content in self.files.items()
                },
            },
        }
        try:
            telemetry.log_test_result(payload)
        except Exception:
            pass
        return {"passed": passed, "runs": self.tests_invocations}

    def get_patch(self):  # pragma: no cover - trivial
        return "diff --git a/app/calc.py b/app/calc.py"

    def close(self):  # pragma: no cover - trivial
        return None

    def reset_soft(self):  # pragma: no cover - signature parity
        return None


def test_rule_agent_pipeline_happy_path(monkeypatch):
    """Ensure the rule-based planner pipeline can drive a fake container end-to-end."""

    cfg_obj = load_config()
    test_log = Path(cfg_obj.telemetry.test_runs_path)
    if test_log.exists():
        test_log.unlink()

    # Patch sandbox runtime and graph/memory helpers to deterministic stubs
    monkeypatch.setattr("graph_planner.env.planner_env.SandboxRuntime", FakeSandbox)
    monkeypatch.setattr("graph_planner.env.planner_env.graph_adapter.connect", lambda: None)
    monkeypatch.setattr("graph_planner.env.planner_env.subgraph_store.save", lambda *args, **kwargs: None)

    def fake_load(issue_id):
        raise FileNotFoundError(issue_id)

    monkeypatch.setattr("graph_planner.env.planner_env.subgraph_store.load", fake_load)

    candidate = {
        "id": "node-1",
        "path": "app/calc.py",
        "span": {"start": 2, "end": 2},
        "score": 1.0,
        "kind": "function",
    }
    monkeypatch.setattr(
        "graph_planner.env.planner_env.mem_candidates.build_mem_candidates",
        lambda *args, **kwargs: [candidate],
    )

    def fake_apply_ops(*, ops, subgraph, policy):
        for op in ops:
            if op.get("op") == "ADD" and op.get("id"):
                node = {
                    "id": op["id"],
                    "path": op.get("path"),
                    "kind": op.get("kind"),
                    "span": op.get("span"),
                }
                subgraph.add_nodes([node])
        return {"applied": True, "ops": ops}

    monkeypatch.setattr("graph_planner.env.planner_env.apply_memory_ops", fake_apply_ops)

    cfg = SandboxConfig(
        docker_image="local/test", workdir="/repo", mounts={}, env={}, backend="docker"
    )
    env = PlannerEnv(
        issue={"id": "demo", "title": "Fix calc", "failure_frame": {"path": "app/calc.py"}},
        sandbox_cfg=cfg,
    )

    observation = env.reset()
    assert observation["reset"] is True

    obs, reward, done, info = env.step(
        ExploreAction(op="expand", anchors=[{"kind": "file", "text": "app/calc.py"}], limit=1)
    )
    assert info["kind"] == "explore"
    assert info["candidates"][0]["id"] == candidate["id"]
    assert reward == 0.0 and done is False

    ops = [
        {
            "op": "ADD",
            "id": candidate["id"],
            "path": candidate["path"],
            "kind": candidate["kind"],
            "span": candidate["span"],
        }
    ]
    obs, reward, done, info = env.step(MemoryAction(ops=ops))
    assert info["kind"] == "memory"
    assert info["summary"]["applied"] is True
    assert reward == 0.0 and done is False

    obs, reward, done, info = env.step(
        ExploreAction(op="read", nodes=[candidate["id"]], limit=1)
    )
    assert info["kind"] == "explore"
    assert info["snippets"] and "0002:" in info["snippets"][0]["snippet"][1]

    edit = {"path": candidate["path"], "start": 2, "end": 2, "new_text": "    return a + b\n"}
    plan_targets = [{"path": candidate["path"], "start": 2, "end": 2, "id": candidate["id"]}]
    obs, reward, done, info = env.step(
        RepairAction(
            apply=True,
            issue=env.issue,
            plan="Fix addition",
            plan_targets=plan_targets,
            patch={"summary": "fix add", "edits": [edit]},
        )
    )
    assert info["kind"] == "repair"
    assert info["tests"]["passed"] is True
    assert reward == 0.0 and done is False

    obs, reward, done, info = env.step(SubmitAction())
    assert info["submit"] is True
    assert info["tests"]["passed"] is True
    assert done is True and reward == 1.0

    env.close()

    assert test_log.exists()
    with test_log.open("r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    assert entries, "test run log should contain at least one entry"
    last = entries[-1]
    trace = last.get("repair_trace", {})
    assert trace.get("edits"), "repair trace should record applied edits"
    final_calc = trace.get("final_files", {}).get("app/calc.py", "")
    assert "return a + b" in final_calc
