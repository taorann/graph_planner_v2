# header
import json, subprocess, sys, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

def call(req):
    env = dict(os.environ)
    # 兜底：即使没改 aci_cli.py，设置 PYTHONPATH=项目根也能跑
    env["PYTHONPATH"] = str(ROOT)

    p = subprocess.run(
        [sys.executable, "bin/aci_cli.py"],
        input=json.dumps(req),
        text=True,
        capture_output=True,
        cwd=str(ROOT),
        env=env,
    )
    if p.returncode != 0:
        print("STDERR:\n", p.stderr)
        print("STDOUT:\n", p.stdout)
        raise SystemExit(f"aci_cli.py exited with {p.returncode}")
    if not p.stdout.strip():
        print("STDERR:\n", p.stderr)
        raise SystemExit("aci_cli.py produced empty stdout")
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        print("STDERR:\n", p.stderr)
        print("STDOUT(raw):\n", p.stdout)
        raise


repo = pathlib.Path(".")
target = next(repo.rglob("**/*.py"))

# 1) view
resp = call({"tool":"view_file","args":{"path":str(target),"start":1,"end":40}})
assert resp["ok"]

# 2) search
resp = call({"tool":"search","args":{"query":"def ", "glob":"**/*.py","max_hits":10,"context":1}})
assert resp["ok"]

# 3) edit + lint
resp = call({"tool":"edit_lines","args":{"path":str(target),"start":1,"end":1,"replacement":"# header\n"}})
assert resp["ok"]
resp = call({"tool":"lint_check","args":{"path":str(target)}})
assert resp["ok"]

# 4) run tests（仓库需自带 pytest 配置；否则先跳过）
