# graph_planner/scripts/smoke_test_repoenv.py
from env.planner_env import PlannerEnv
from core.actions import RepairAction, SubmitAction
import os

def main():
    ds_json = os.environ.get("R2E_DS_JSON")
    if not ds_json:
        raise SystemExit("Please set R2E_DS_JSON to a valid ds json path.")
    ds_json = os.path.expanduser(ds_json)
    if not os.path.isabs(ds_json):
        ds_json = os.path.abspath(ds_json)
    if not os.path.exists(ds_json):
        raise SystemExit(f"R2E_DS_JSON not found: {ds_json}")

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
        repair = RepairAction(apply=False, issue=extra["issue"])
        obs, reward, done, info = env.step(repair)
        print("[step 0] done=", done, "reward=", reward)
        submit = SubmitAction()
        obs, reward, done, info = env.step(submit)
        print("[step 1] done=", done, "reward=", reward)
        tests = info.get("tests", {})
        print("=== repoenv smoke result ===")
        print("passed:", tests.get("passed"), "rc:", tests.get("rc"))
        print(tests.get("stdout", "")[-800:])
    finally:
        env.close()

if __name__ == "__main__":
    main()
