#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 memory layer demo (bin script)

功能：
- 可选 --init：在项目根下创建一个最小“模拟仓库” __aci_step2_demo_repo 与 repo_graph.jsonl
- 运行 demo：加载 repo_graph.jsonl，按锚点加入/扩展工作子图，并线性化输出片段
- 所有导入都通过把“项目根”自动加入 sys.path，避免手动设置 PYTHONPATH

用法：
  python bin/step2_memory_demo.py --init              # 首次：搭建模拟仓库与图
  python bin/step2_memory_demo.py                     # 在模拟仓库里跑一次演示
  python bin/step2_memory_demo.py --demo-dir myrepo   # 指定其它目录（需含 repo_graph.jsonl）
"""

import os
import sys
import argparse
from typing import Dict, Any

# --- 把项目根加入 sys.path，确保能 import memory/*  ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def init_demo_repo(demo_dir: str) -> None:
    """创建最小模拟仓库与 repo_graph.jsonl（幂等）。"""
    os.makedirs(os.path.join(demo_dir, "pkg"), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "app"), exist_ok=True)

    files: Dict[str, str] = {
        os.path.join(demo_dir, "pkg", "__init__.py"): "# package init\n",
        os.path.join(demo_dir, "pkg", "util.py"): (
            "def helper(x):\n"
            "    return x * 2\n\n"
            "class Greeter:\n"
            "    def hi(self, name):\n"
            "        return f\"hi, {name}\"\n"
        ),
        os.path.join(demo_dir, "app", "main.py"): (
            "from pkg.util import helper, Greeter\n\n"
            "def main():\n"
            "    g = Greeter()\n"
            "    print(helper(21))\n"
            "    print(g.hi(\"Zoe\"))\n\n"
            "if __name__ == \"__main__\":\n"
            "    main()\n"
        ),
    }
    for p, content in files.items():
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.isfile(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)

    graph_path = os.path.join(demo_dir, "repo_graph.jsonl")
    if not os.path.isfile(graph_path):
        with open(graph_path, "w", encoding="utf-8") as f:
            f.write('{"type":"node","data":{"id":"file:pkg/util.py","kind":"file","path":"pkg/util.py","name":""}}\n')
            f.write('{"type":"node","data":{"id":"file:app/main.py","kind":"file","path":"app/main.py","name":""}}\n')
            f.write('{"type":"node","data":{"id":"file:pkg/__init__.py","kind":"file","path":"pkg/__init__.py","name":""}}\n')
            f.write('\n')
            f.write('{"type":"node","data":{"id":"func:pkg/util.py#helper@1","kind":"function","path":"pkg/util.py","name":"helper","span":{"start":1,"end":2}}}\n')
            f.write('{"type":"node","data":{"id":"class:pkg/util.py#Greeter@4","kind":"class","path":"pkg/util.py","name":"Greeter","span":{"start":4,"end":6}}}\n')
            f.write('{"type":"node","data":{"id":"func:app/main.py#main@3","kind":"function","path":"app/main.py","name":"main","span":{"start":3,"end":6}}}\n')
            f.write('\n')
            f.write('{"type":"edge","data":{"src":"file:pkg/util.py","dst":"func:pkg/util.py#helper@1","etype":"contains"}}\n')
            f.write('{"type":"edge","data":{"src":"file:pkg/util.py","dst":"class:pkg/util.py#Greeter@4","etype":"contains"}}\n')
            f.write('{"type":"edge","data":{"src":"file:app/main.py","dst":"func:app/main.py#main@3","etype":"contains"}}\n')
            f.write('{"type":"edge","data":{"src":"file:app/main.py","dst":"file:pkg/util.py","etype":"imports"}}\n')
            f.write('{"type":"edge","data":{"src":"func:app/main.py#main@3","dst":"func:pkg/util.py#helper@1","etype":"calls"}}\n')
            f.write('{"type":"edge","data":{"src":"func:app/main.py#main@3","dst":"class:pkg/util.py#Greeter@4","etype":"ref"}}\n')

    print(f"[init] demo repo ready at: {os.path.abspath(demo_dir)}")


def run_demo(demo_dir: str) -> None:
    """在指定目录下运行 Step 2 演示：connect → anchors → 1-hop → linearize → save."""
    # 切换到 demo 目录，让 memory 层以它作为 repo_root()
    prev = os.getcwd()
    os.chdir(demo_dir)
    try:
        from memory import graph_adapter, subgraph_store  # noqa: E402

        gh = graph_adapter.connect("repo_graph.jsonl")
        sg = subgraph_store.load(issue_id="demo-issue-2")

        anchors = [
            {"kind": "function", "text": "helper"},
            {"kind": "file", "text": "pkg/util.py"},
        ]
        seeds = []
        for a in anchors:
            seeds.extend(graph_adapter.find_nodes_by_anchor(gh, a))
        subgraph_store.add_nodes(sg, seeds)

        cands = graph_adapter.one_hop_expand(gh, sg, anchors=anchors, max_nodes=20)
        subgraph_store.add_nodes(sg, cands)

        print("STATS:", subgraph_store.stats(sg))

        chunks = subgraph_store.linearize(sg, mode="wsd")
        print("CHUNKS_N:", len(chunks))
        for ch in chunks[:2]:
            print("CHUNK:", ch["path"], ch["start"], ch["end"])
            print(ch["text"])

        subgraph_store.save("demo-issue-2", sg)
        print("Saved:", os.path.join(".aci", "subgraphs", "demo-issue-2.json"))
    finally:
        os.chdir(prev)


def main():
    ap = argparse.ArgumentParser(description="Step 2 memory demo")
    ap.add_argument("--demo-dir", default="__aci_step2_demo_repo", help="模拟仓库目录（默认：__aci_step2_demo_repo）")
    ap.add_argument("--init", action="store_true", help="初始化模拟仓库与 repo_graph.jsonl")
    args = ap.parse_args()

    if args.__dict__.get("init"):
        init_demo_repo(args.demo_dir)

    # 若目录不存在但未传 --init，给出友好提示
    if not os.path.isdir(args.demo_dir):
        print(f"[error] demo dir not found: {args.demo_dir}. Run with --init first.")
        sys.exit(1)

    run_demo(args.demo_dir)


if __name__ == "__main__":
    main()
