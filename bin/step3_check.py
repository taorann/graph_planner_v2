# -*- coding: utf-8 -*-
"""
Step3 CHECK: 三项轻量校验合一（无需 pytest）
  1) 线性化前检查 + 线性化统计
  2) 候选→MemOps→应用子图（不报错即可）
  3) 配额/比例保护（t-file 比例与总 cap）
返回码：0 通过；非 0 失败

python bin/step3_check.py
"""
import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from memory import subgraph_store
from memory.mem_candidates import build_mem_candidates
from memory.mem_ops_head import suggest as memops_suggest
from memory.memory_bank import apply_ops, ApplyPolicy
from orchestrator.loop import _pre_linearize_checks, _linearize_for_actor

class _Cfg:
    mode = "wsd"
    subgraph_total_cap = 800
    context_budget_tokens = 40000
    dir_diversity_k = 3

def _mk_subgraph_min():
    sg = subgraph_store.new()  
    subgraph_store.add_nodes(sg, [
        {"id":"file:a.py","kind":"file","path":"a.py","name":"","degree":0},
        {"id":"func:a.py#f@3","kind":"function","path":"a.py","name":"f",
         "span":{"start":3,"end":10},"degree":3},
        {"id":"file:tests/test_a.py","kind":"t-file","path":"tests/test_a.py","name":"","degree":0},
    ])
    return sg

def check_precheck_and_linearize():
    cfg = _Cfg()
    sg = _mk_subgraph_min()
    pc = _pre_linearize_checks(sg, cfg)
    ok = pc.get("nodes_total", 0) >= 3
    chunks, meta = _linearize_for_actor(sg, cfg)
    ok = ok and isinstance(chunks, list) and meta.get("chunks", 0) >= 1
    print(f"[CHECK1] nodes_total={pc.get('nodes_total')}, chunks={meta.get('chunks')} -> {'PASS' if ok else 'FAIL'}")
    return ok

def check_candidates_and_memops_flow():
    cfg = _Cfg()
    sg = _mk_subgraph_min()
    cands = build_mem_candidates(
        subgraph=sg,
        anchors=[{"kind":"file","text":"a.py"}],
        max_nodes_per_anchor=10, total_limit=20, dir_diversity_k=2
    )
    ops = memops_suggest(candidates=cands, context=None, subgraph=sg)
    # 应用（不抛错即可）
    summary = apply_ops(ops=ops, subgraph=sg, policy=ApplyPolicy())
    ok = "applied" in summary and "after" in summary
    print(f"[CHECK2] memops={len(ops)}, after_nodes={summary.get('after',{}).get('nodes')} -> {'PASS' if ok else 'FAIL'}")
    return ok

def check_caps_and_guards():
    sg = {"nodes": {}, "edges": []}
    ops = [{"op":"ADD","id":f"file:tests/t{i}.py","kind":"t-file","path":f"tests/t{i}.py","confidence":0.9}
           for i in range(200)]
    summary = apply_ops(ops=ops, subgraph=sg,
                        policy=ApplyPolicy(total_node_cap=50, max_tfile_fraction=0.4))
    after_nodes = summary.get("after", {}).get("nodes", 0)
    ok = after_nodes <= 50
    print(f"[CHECK3] after_nodes={after_nodes} (cap<=50) -> {'PASS' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    ok1 = check_precheck_and_linearize()
    ok2 = check_candidates_and_memops_flow()
    ok3 = check_caps_and_guards()
    all_ok = ok1 and ok2 and ok3
    if all_ok:
        print("[ALL] PASS")
        sys.exit(0)
    else:
        print("[ALL] FAIL")
        sys.exit(1)
