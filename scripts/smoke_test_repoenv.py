# graph_planner/scripts/smoke_test_repoenv.py
from env.planner_env import PlannerEnv
from runtime.sandbox import SandboxConfig
import os, json, pathlib

def main():
    ds_json = os.environ.get("R2E_DS_JSON")
    if not ds_json or not os.path.exists(ds_json):
        raise SystemExit("Please set R2E_DS_JSON to a valid ds json path.")

    extra = {
        "issue": {"id": "DS-1", "title": "repoenv smoke"},
        "sandbox": {
            "docker_image": "unused-for-repoenv",
            "workdir": "/testbed",
            "mounts": {},              # repoenv 不需要挂载
            "env": {},
            "backend": "repoenv",      # 关键：切到 RepoEnv
            "r2e_ds_json": ds_json,
        },
    }
    env = PlannerEnv.from_dict(extra)
    try:
        env.reset()
        # 走你现有回路：先 repair 一步，再 submit
        obs, reward, done, info = env.step({"type": "repair"})
        print("[step 0] done=", done, "reward=", reward)
        obs, reward, done, info = env.step({"type": "submit"})
        print("[step 1] done=", done, "reward=", reward)
        tests = info.get("tests", {})
        print("=== repoenv smoke result ===")
        print("passed:", tests.get("passed"), "rc:", tests.get("rc"))
        print(tests.get("stdout", "")[-800:])
    finally:
        env.close()

if __name__ == "__main__":
    main()
