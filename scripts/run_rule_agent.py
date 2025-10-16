"""Run the rule-based planner agent end-to-end inside a container."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

from agents import PlannerAgent
from core.actions import ActionUnion
from env.planner_env import PlannerEnv
from infra.config import load as load_config

ACTION_TYPES = tuple(ActionUnion.__args__)  # type: ignore[attr-defined]


def _parse_mounts(values: Tuple[str, ...]) -> Dict[str, str]:
    mounts: Dict[str, str] = {}
    for item in values or ():
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"invalid mount: {item!r}; expected host:container")
        host, container = item.split(":", 1)
        host = os.path.abspath(os.path.expanduser(host))
        mounts[host] = container
    return mounts


def build_env(args: argparse.Namespace) -> PlannerEnv:
    if args.backend == "repoenv":
        ds_json = args.ds_json or os.environ.get("R2E_DS_JSON")
        if not ds_json:
            raise SystemExit("--ds-json or R2E_DS_JSON is required when backend=repoenv")
        ds_json = os.path.abspath(os.path.expanduser(ds_json))
        if not os.path.exists(ds_json):
            raise SystemExit(f"RepoEnv dataset json not found: {ds_json}")
    else:
        ds_json = None

    mounts = _parse_mounts(tuple(args.mount or ()))
    sandbox_payload = {
        "docker_image": args.docker_image,
        "workdir": args.workdir,
        "mounts": mounts,
        "env": {},
        "backend": args.backend,
        "r2e_ds_json": ds_json,
    }

    issue = {"id": args.issue_id, "title": args.issue_title, "body": args.issue_body}

    payload = {"issue": issue, "sandbox": sandbox_payload}
    return PlannerEnv.from_dict(payload)


def run_episode(env: PlannerEnv, max_steps: int) -> Dict[str, object]:
    agent = PlannerAgent()
    obs = env.reset()
    trajectory = []
    reward = 0.0
    done = False
    steps = 0

    while not done and steps < max_steps:
        decision = agent.step(obs)
        action_obj = decision["action_obj"]
        if not isinstance(action_obj, ACTION_TYPES):
            raise RuntimeError(f"Agent returned invalid action object: {type(action_obj)}")
        obs, reward, done, info = env.step(action_obj)
        trajectory.append({
            "step": steps,
            "phase": agent.state.phase,
            "action": action_obj.type,
            "info": info,
        })
        steps += 1
        if done:
            break

    return {
        "reward": reward,
        "done": done,
        "steps": steps,
        "trajectory": trajectory,
        "last_info": obs.get("last_info"),
        "tests": trajectory[-1]["info"].get("tests") if trajectory else {},
        "patch": trajectory[-1]["info"].get("patch") if trajectory else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rule-driven agent against a sandbox")
    parser.add_argument("--backend", default="repoenv", choices=["repoenv", "docker"], help="Sandbox backend to use")
    parser.add_argument("--ds-json", help="Path to the RepoEnv dataset JSON")
    parser.add_argument("--docker-image", default="python:3.10", help="Docker image when backend=docker")
    parser.add_argument("--workdir", default="/testbed", help="Container workdir")
    parser.add_argument("--mount", action="append", help="Mount specification host:container; repeatable")
    parser.add_argument("--issue-id", default="ISSUE-1")
    parser.add_argument("--issue-title", default="rule-agent demo")
    parser.add_argument("--issue-body", default="")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--report", help="Optional path to dump the trajectory JSON")

    args = parser.parse_args()
    env = build_env(args)
    try:
        result = run_episode(env, args.max_steps)
    finally:
        env.close()

    cfg = load_config()

    print("=== Rule Agent Episode Summary ===")
    print("steps:", result["steps"], "reward:", result["reward"], "done:", result["done"])
    tests = result.get("tests") or {}
    print("tests passed:", tests.get("passed"), "rc:", tests.get("rc"))
    if tests.get("stdout"):
        print("--- tests stdout tail ---")
        print(tests["stdout"][-800:])
    if result.get("patch"):
        print("--- git diff ---")
        print(result["patch"])
    print(f"test logs appended to {cfg.telemetry.test_runs_path}")

    if args.report:
        report_path = os.path.abspath(os.path.expanduser(args.report))
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Trajectory report written to {report_path}")


if __name__ == "__main__":
    main()
