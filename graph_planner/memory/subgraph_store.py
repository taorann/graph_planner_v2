# -*- coding: utf-8 -*-
from __future__ import annotations
"""
memory/subgraph_store.py

工作子图存取与线性化（WSD/FULL），标准化为带 .node_ids 的对象：
  - WorkingSubgraph: .nodes(dict)、.edges(list)、.node_ids(set)
  - new()/wrap() 便于创建/迁移
  - load()/save() 读写到 .aci/subgraphs/<issue>.json（nodes 按 list 存）
  - 仍提供函数式 add_nodes/update_node/remove_nodes/stats/linearize，内部都走对象方法

这样，orchestrator/memory/actors 都可以稳定调用：
  - subgraph.iter_node_ids() / subgraph.get_node()
  - 或使用本模块函数式 API（向后兼容）
"""

from typing import Dict, Any, Iterable, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import os
import json

try:
    from aci._utils import repo_root  # 项目已有
except Exception:
    repo_root = os.getcwd  # 兜底


# ======================= 基础类 =======================

@dataclass
class WorkingSubgraph:
    nodes: Dict[str, Dict[str, Any]]
    edges: List[Dict[str, Any]]

    def __post_init__(self):
        # 常驻集合：加速包含/迭代
        self.node_ids: set[str] = set(self.nodes.keys())

    # --- 统一接口 ---
    def iter_node_ids(self) -> Iterable[str]:
        return iter(self.node_ids)

    def get_node(self, node_id: str) -> Dict[str, Any]:
        return self.nodes.get(node_id, {})

    # --- 变更操作 ---
    def add_nodes(self, nodes: Iterable[Dict[str, Any]]) -> None:
        for n in nodes or []:
            nid = n.get("id")
            if not nid:
                continue
            self.nodes[nid] = dict(n)
            self.node_ids.add(nid)

    def update_node(self, node_id: str, **patch) -> None:
        n = self.nodes.get(node_id)
        if not n:
            return
        for k, v in patch.items():
            n[k] = v

    def remove_nodes(self, node_ids: Iterable[str]) -> None:
        ids = set(node_ids or [])
        if not ids:
            return
        for nid in list(ids):
            self.nodes.pop(nid, None)
            self.node_ids.discard(nid)
        if isinstance(self.edges, list):
            keep = []
            for e in self.edges:
                sid = e.get("src")
                did = e.get("dst")
                if sid in ids or did in ids:
                    continue
                keep.append(e)
            self.edges[:] = keep

    # --- 序列化 ---
    def to_json_obj(self) -> Dict[str, Any]:
        return {"nodes": list(self.nodes.values()), "edges": self.edges}


# ======================= 工厂/迁移 =======================

def new() -> WorkingSubgraph:
    return WorkingSubgraph(nodes={}, edges=[])

def wrap(obj) -> WorkingSubgraph:
    """把 dict 或已有对象规范成 WorkingSubgraph。"""
    if isinstance(obj, WorkingSubgraph):
        return obj
    if isinstance(obj, dict):
        nodes_store = obj.get("nodes", {})
        # 允许 list 或 dict 两种形态
        if isinstance(nodes_store, list):
            nodes = {n["id"]: n for n in nodes_store if isinstance(n, dict) and "id" in n}
        elif isinstance(nodes_store, dict):
            nodes = {k: dict(v) for k, v in nodes_store.items()}
        else:
            nodes = {}
        edges = [e for e in obj.get("edges", []) if isinstance(e, dict)]
        return WorkingSubgraph(nodes=nodes, edges=edges)
    # 极端兜底
    return WorkingSubgraph(nodes={}, edges=[])


# ======================= I/O =======================

def _store_dir() -> str:
    root = repo_root() if callable(repo_root) else repo_root
    d = os.path.join(root, ".aci", "subgraphs")
    os.makedirs(d, exist_ok=True)
    return d

def _issue_path(issue_id: str) -> str:
    return os.path.join(_store_dir(), f"{issue_id}.json")

def load(issue_id: str) -> WorkingSubgraph:
    p = _issue_path(issue_id)
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
    return wrap(data)

def save(issue_id: str, subgraph) -> None:
    sg = wrap(subgraph)
    with open(_issue_path(issue_id), "w", encoding="utf-8") as f:
        json.dump(sg.to_json_obj(), f, ensure_ascii=False, indent=2)


# ======================= 变更操作（函数式封装） =======================

def add_nodes(subgraph, nodes: Iterable[Dict[str, Any]]) -> None:
    wrap(subgraph).add_nodes(nodes)

def update_node(subgraph, node_id: str, **patch) -> None:
    wrap(subgraph).update_node(node_id, **patch)

def remove_nodes(subgraph, node_ids: Iterable[str]) -> None:
    wrap(subgraph).remove_nodes(node_ids)


# ======================= 统计 =======================

def stats(subgraph) -> Dict[str, Any]:
    sg = wrap(subgraph)
    kinds: Dict[str, int] = defaultdict(int)
    files: Dict[str, None] = {}
    for n in sg.nodes.values():
        k = (n.get("kind") or "").lower()
        kinds[k] += 1
        p = n.get("path")
        if p:
            files[p] = None
    return {
        "nodes": len(sg.nodes),
        "edges": len(sg.edges),
        "files": len(files),
        "funcs": kinds.get("function", 0),
        "classes": kinds.get("class", 0),
        "kinds": dict(kinds),
    }


# ======================= 线性化 =======================

def _is_tfile_path(path: str) -> bool:
    p = (path or "").lower()
    return ("test" in p) or ("/tests/" in p) or p.endswith("_test.py") or p.endswith("test.py")

def _read_file_lines(rel_path: str) -> List[str]:
    root = repo_root() if callable(repo_root) else repo_root
    abspath = os.path.join(root, rel_path)
    try:
        with open(abspath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().splitlines()
    except Exception:
        return []

def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ls, le = merged[-1]
        if s <= le + 1:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def linearize(subgraph, mode: str = "wsd") -> List[Dict[str, Any]]:
    """
    WSD：以 span 的 function/class/symbol 为主，左右各 ±2 行并合并；无 span 的文件给兜底片段
    FULL：整文件（全文件上限 800 行）
    """
    sg = wrap(subgraph)
    mode = (mode or "wsd").lower()
    FULL_LIMIT = 800
    WSD_PAD = 2
    WSD_FILE_CAP = 100
    WSD_TFILE_CAP = 200

    # 以 path 分组
    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for n in sg.nodes.values():
        p = n.get("path")
        if p:
            by_file[p].append(n)

    chunks: List[Dict[str, Any]] = []

    for path, nodes in by_file.items():
        lines = _read_file_lines(path)
        n_lines = len(lines)
        if n_lines == 0:
            continue

        if mode == "full":
            end = min(n_lines, FULL_LIMIT)
            text = "\n".join(lines[:end])
            chunks.append({"path": path, "start": 1, "end": end, "text": text})
            continue

        # WSD：收集所有 span 窗口并扩展
        spans: List[Tuple[int, int]] = []
        for n in nodes:
            span = n.get("span")
            if not span:
                continue
            s = max(1, int(span.get("start", 1)) - WSD_PAD)
            e = min(n_lines, int(span.get("end", 1)) + WSD_PAD)
            if s <= e:
                spans.append((s, e))

        if spans:
            for s, e in _merge_ranges(spans):
                text = "\n".join(lines[s-1:e])
                chunks.append({"path": path, "start": s, "end": e, "text": text})
        else:
            # 兜底片段：按文件/测试文件限制
            cap = WSD_TFILE_CAP if _is_tfile_path(path) else WSD_FILE_CAP
            end = min(n_lines, cap)
            text = "\n".join(lines[:end])
            chunks.append({"path": path, "start": 1, "end": end, "text": text})

    return chunks
