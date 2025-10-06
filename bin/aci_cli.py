# bin/aci_cli.py
import sys, os, json

# === 关键：把项目根目录加入 sys.path ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aci.runner import execute  # 现在可以正常导入了

def main():
    data = sys.stdin.read()
    if not data.strip():
        print(json.dumps({"ok": False, "summary": "empty stdin"}), ensure_ascii=False)
        return
    req = json.loads(data)
    resp = execute(req)
    print(json.dumps(resp, ensure_ascii=False))

if __name__ == "__main__":
    main()
