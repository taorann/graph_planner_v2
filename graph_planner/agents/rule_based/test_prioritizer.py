# graph_planner/agents/rule_based/test_prioritizer.py
# -*- coding: utf-8 -*-
"""
Step 5.5 测试优先级
-------------------
对外 API:
    prioritize_tests(
        observation_pack: dict,
        subgraph: object | None = None,
        top_k: int = 8,
        prefer_tfile: bool = True
    ) -> dict

返回:
    {
      "priority_tests": [str, ...],  # 相对 repo-root 的选择器或具体路径
      "why": {...}
    }

启发式来源：失败路径、同目录 tests/*、同名 test_*、子图中的 t-file 节点。
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
import os

class FailureFrame(TypedDict, total=False):
    path: str
    line: int | None
    func: str | None

class ObservationPack(TypedDict, total=False):
    failure_frame: FailureFrame
    issue: str
    top_assert: str | None

def _push_unique(lst: List[str], item: Optional[str]) -> None:
    if not item:
        return
    if item not in lst:
        lst.append(item)

def _dir_of(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else ""

def _is_testy_name(name: str) -> bool:
    n = name.lower()
    return n.startswith("test_") or n.endswith("_test.py") or "test" in n

def _collect_tfiles_from_subgraph(subgraph: Any, cap: int = 16) -> List[str]:
    """从运行时子图对象里抓取 t-file 或路径里含 test 的文件节点（若可用）。"""
    out: List[str] = []
    try:
        nodes = getattr(subgraph, "nodes", None) or subgraph.get("nodes", {})
        for _id, n in (nodes or {}).items():
            path = n.get("path")
            kind = (n.get("kind") or "").lower()
            if not path:
                continue
            if kind == "t-file" or "test" in path.lower():
                _push_unique(out, path)
                if len(out) >= cap:
                    break
    except Exception:
        pass
    return out

def _same_basename_test_candidates(path: str) -> List[str]:
    """给定源文件 pkg/foo/bar.py，猜测 tests 里可能对应的测试文件名。"""
    bn = os.path.basename(path)
    stem = os.path.splitext(bn)[0]
    candidates = [
        os.path.join(_dir_of(path), f"test_{bn}"),
        os.path.join(_dir_of(path), f"{stem}_test.py"),
        os.path.join("tests", f"test_{bn}"),
        os.path.join("tests", f"{stem}_test.py"),
    ]
    return candidates

def prioritize_tests(
    observation_pack: Dict[str, Any],
    subgraph: Any = None,
    top_k: int = 8,
    prefer_tfile: bool = True
) -> Dict[str, Any]:
    fail = (observation_pack or {}).get("failure_frame") or {}
    fpath: Optional[str] = fail.get("path")
    tests: List[str] = []
    why: Dict[str, Any] = {"rules": []}

    # 1) 失败路径是 test 本身
    if fpath and _is_testy_name(os.path.basename(fpath)):
        _push_unique(tests, fpath)
        why["rules"].append("failure_is_test_file")

    # 2) 同目录 tests/*
    if fpath:
        d = _dir_of(fpath)
        if d:
            _push_unique(tests, os.path.join(d, "test_*"))
            why["rules"].append("same_dir_test_glob")

    # 3) 同名猜测
    if fpath:
        for cand in _same_basename_test_candidates(fpath):
            _push_unique(tests, cand)
        why["rules"].append("basename_guess")

    # 4) 子图中的 t-file / 含 test 的文件
    if prefer_tfile and subgraph is not None:
        for tpath in _collect_tfiles_from_subgraph(subgraph, cap=top_k):
            _push_unique(tests, tpath)
        why["rules"].append("subgraph_tfiles")

    # 截断
    tests = tests[:max(1, int(top_k))]
    return {"priority_tests": tests, "why": why}
