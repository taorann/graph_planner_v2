# /map-vepfs/taoran/graph_planner/scripts/smoke_test_r2e.py
import os
from env.planner_env import PlannerEnv
from agents import PlannerAgent

def main():
    extra = {
        "issue": {"id": "DEMO-B", "title": "r2e-backend-smoke"},
        "sandbox": {
            # 注意：走 r2e 后端时，镜像以 ds_json 为准；这里的 docker_image 只是占位
            "docker_image": os.environ.get("R2E_IMAGE", "python:3.10"),
            "workdir": os.environ.get("R2E_WORKDIR", "/testbed"),
            "mounts": {
                os.environ["HOST_REPO"]: "/testbed",
            },
            "env": {},
            "pytest_cache_root": "/cache/tests",
            "commit_hash": os.environ.get("COMMIT_HASH", "HEAD"),
            "backend": "r2e",  # ✅ 关键：启用 r2e 底座
            "r2e_ds_json": os.environ.get(
                "R2E_DS_JSON",
                "/map-vepfs/taoran/graph_planner/config/r2e_ds_min.json"
            ),
        },
    }

    env = PlannerEnv.from_dict(extra)
    agent = PlannerAgent()

    obs = env.reset()
    done, total_r, step = False, 0.0, 0
    while not done and step < 8:
        msg = agent.step(obs)
        obs, r, done, info = env.step(msg["action_obj"])
        print(f"[B step {step}] action={msg['response']} reward={r} done={done}")
        if "tests" in info:
            print(f"  tests.passed={info['tests']['passed']} rc={info['tests']['rc']}")
        total_r += r; step += 1

    print(f"[B] episode reward={total_r}")
    if env.last_info.get("tests"):
        out = env.last_info["tests"]["stdout"]
        print("\n=== pytest tail (B) ===")
        print("\n".join(out.splitlines()[-20:]))

if __name__ == "__main__":
    main()
