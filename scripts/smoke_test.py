# /map-vepfs/taoran/graph_planner/scripts/smoke_test.py
import os
from env.planner_env import PlannerEnv
from agent.planner_agent import PlannerAgent

def main():
    extra = {
        "issue": {"id": "DEMO-A", "title": "docker-backend-smoke"},
        "sandbox": {
            "docker_image": os.environ.get("R2E_IMAGE", "python:3.10"),
            "workdir": os.environ.get("R2E_WORKDIR", "/testbed"),
            "mounts": {
                os.environ["HOST_REPO"]: "/testbed",
            },
            "env": {},
            "pytest_cache_root": "/cache/tests",
            "commit_hash": os.environ.get("COMMIT_HASH", "HEAD"),
            "backend": "docker",             # 显式走通用 docker
            "r2e_ds_json": None,
        },
    }

    env = PlannerEnv.from_dict(extra)
    agent = PlannerAgent()

    obs = env.reset()
    done, total_r, step = False, 0.0, 0
    while not done and step < 8:
        msg = agent.step(obs)
        obs, r, done, info = env.step(msg["action_obj"])
        print(f"[A step {step}] action={msg['response']} reward={r} done={done}")
        if "tests" in info:
            print(f"  tests.passed={info['tests']['passed']} rc={info['tests']['rc']}")
        total_r += r; step += 1

    print(f"[A] episode reward={total_r}")
    if env.last_info.get("tests"):
        out = env.last_info["tests"]["stdout"]
        print("\n=== pytest tail (A) ===")
        print("\n".join(out.splitlines()[-20:]))

if __name__ == "__main__":
    main()
