# -*- coding: utf-8 -*-
from __future__ import annotations
# 2025-10-22 memory hardening
"""
memory/graph_adapter.py  —— 完整版

GraphAdapter:
- connect(handle_or_path) -> GraphHandle
- get_node_by_id(node_id) -> Node | None          # 模块级包装
- find_nodes_by_anchor(anchor) -> list[Node]      # 模块级包装
- one_hop_expand(subgraph, anchors, max_nodes) -> list[Node]  # 模块级包装

内部也保留带 GraphHandle 的下划线版本：
- _get_node_by_id(graph, node_id) -> Node | None
- _find_nodes_by_anchor(graph, anchor) -> list[Node]
- _one_hop_expand(graph, subgraph, anchors, max_nodes) -> list[Node]

优先读取外部 JSONL 图（repo_graph.jsonl / graph.jsonl / code_graph.jsonl），
否则做 Python 友好的轻量本地构图（file/func/class + imports）。

约定：
- path 一律 repo-root 相对
- kind 统一小写并做常见别名映射（func/method→function、tfile/testfile→t-file、package→pkg等）
- span: 1-based, 闭区间
"""

from typing import List, Dict, Any, Optional, Tuple, Iterable
import os
import json
import ast
import re
from collections import defaultdict

from aci._utils import repo_root, list_text_files, safe_read_text
from .types import Node, Edge, Anchor, DocChunk


def _norm_posix(path: str | None) -> str:
    return (path or "").replace("\\", "/")


# ---------------- Graph Handle ----------------

class GraphHandle:
    def __init__(self, root: str):
        self.root = root
        self.nodes: Dict[str, Node] = {}          # id -> Node
        self.adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # id -> [(neighbor_id, etype)]
        self.rev: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add_node(self, node: Node):
        if "path" in node:
            node = dict(node)
            node["path"] = _norm_posix(node.get("path"))
        self.nodes[node["id"]] = node

    def add_edge(self, src: str, dst: str, etype: str):
        self.adj[src].append((dst, etype))
        self.rev[dst].append((src, etype))

    def get(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        return self.adj.get(node_id, [])

    def degree(self, node_id: str) -> int:
        return len(self.adj.get(node_id, [])) + len(self.rev.get(node_id, []))

    def neighbors_undirected(self, node_id: str) -> List[Tuple[str, str]]:
        out = []
        out.extend(self.adj.get(node_id, []))
        out.extend(self.rev.get(node_id, []))
        return out


# ---------- Module-level handle shim ----------
_GH: Optional["GraphHandle"] = None

def _require_handle() -> "GraphHandle":
    """
    返回一个可用的全局 GraphHandle。
    若尚未 connect()，则用默认参数初始化一次（基于当前仓库）。
    """
    global _GH
    if _GH is None:
        connect()  # 会在 connect 里设置 _GH
    return _GH


# ---------------- Public API（外部调用用这些） ----------------

def connect(handle_or_path: Optional[str] = None) -> GraphHandle:
    """
    尝试连接外部图（JSONL）。若未找到则构建轻量本地图。
    支持的外部文件名：handle_or_path（若为文件），或 repo 根下：
      - repo_graph.jsonl
      - graph.jsonl
      - code_graph.jsonl
    JSONL 每行：
      {"type": "node", "data": {...}} / {"type": "edge", "data": {...}}
    node 至少包含：id, kind, path?, name?, span?
    edge：src, dst, etype
    """
    root = repo_root()
    gh = GraphHandle(root=root)

    # Try external
    candidates = []
    if handle_or_path and os.path.isfile(handle_or_path):
        candidates.append(handle_or_path)
    else:
        for fname in ("repo_graph.jsonl", "graph.jsonl", "code_graph.jsonl"):
            p = os.path.join(root, fname)
            if os.path.isfile(p):
                candidates.append(p)
                break

    if candidates:
        _load_jsonl_graph(candidates[0], gh)
    else:
        _build_light_graph(gh)

    # cache degree
    for nid, n in gh.nodes.items():
        n["degree"] = gh.degree(nid)

    # 缓存到模块级句柄
    global _GH
    _GH = gh

    return gh


def get_node_by_id(node_id: str) -> Optional[Node]:
    gh = _require_handle()
    return gh.get(node_id)


def find_nodes_by_anchor(anchor: Anchor) -> List[Node]:
    gh = _require_handle()
    return _find_nodes_by_anchor(gh, anchor)


def one_hop_expand(subgraph: "SubgraphProxy",
                   anchors: List[Anchor],
                   max_nodes: int = 50) -> List[Node]:
    gh = _require_handle()
    return _one_hop_expand(gh, subgraph, anchors, max_nodes)


# ---------------- 带 GraphHandle 参数的内部实现 ----------------

def _get_node_by_id(graph: GraphHandle, node_id: str) -> Optional[Node]:
    return graph.get(node_id)


def _find_nodes_by_anchor(graph: GraphHandle, anchor: Anchor) -> List[Node]:
    """
    解析锚点：
    - 若提供 anchor.id -> 精确匹配
    - kind="file"+text -> 路径包含的模糊匹配
    - kind in {"function","symbol","class"}+text -> 按 name 匹配；精确优先
    回退：按文件基名包含 text；再差则空。
    """
    # id 精确
    if anchor.get("id"):
        n = graph.get(anchor["id"])
        return [n] if n else []

    kind = (anchor.get("kind") or "symbol").lower()
    text = (anchor.get("text") or "").strip()
    text_lower = text.lower()
    if not text:
        return []

    # file：按 path 片段匹配
    out: List[Node] = []
    if kind == "file":
        for n in graph.nodes.values():
            if n["kind"] != "file":
                continue
            path = _norm_posix(n.get("path"))
            if text_lower in path.lower():
                out.append(n)
        out.sort(key=lambda n: len(n.get("path", "")))  # 路径短优先
        return out[:50]

    # function/class/symbol：按 name 匹配
    def score(n: Node) -> Tuple[int, int]:
        name = n.get("name") or ""
        if name.lower() == text_lower:
            return (0, len(name))
        if text_lower in name.lower():
            return (1, len(name))
        return (9, len(name))

    for n in graph.nodes.values():
        if n["kind"] in ("function", "class", "symbol"):
            nm = (n.get("name") or "")
            if nm.lower() == text_lower or text_lower in nm.lower():
                out.append(n)

    if out:
        out.sort(key=score)
        return out[:50]

    # 回退：按文件基名匹配
    for n in graph.nodes.values():
        if n["kind"] != "file":
            continue
        path = _norm_posix(n.get("path"))
        if text_lower in os.path.basename(path).lower():
            out.append(n)
    out.sort(key=lambda n: len(n.get("path", "")))
    return out[:50]


def _one_hop_expand(graph: GraphHandle,
                    subgraph: "SubgraphProxy",
                    anchors: List[Anchor],
                    max_nodes: int = 50) -> List[Node]:
    """
    以 anchors 指向的节点为中心，返回 1-hop 邻居候选（不去重，由上层去重并配额）。
    邻居优先级：
      1) contains/imports/ref/calls 边邻居（无向）
      2) 同文件内的其它 func/class（通过 file->contains）
      3) 同目录下文件（轻度）
    """
    results: List[Node] = []
    seen: set[str] = set()

    # resolve anchor nodes
    anchor_nodes: List[Node] = []
    for a in anchors:
        cand = _find_nodes_by_anchor(graph, a)
        anchor_nodes.extend(cand)

    # 1) direct graph neighbors (无向)
    for n in anchor_nodes:
        for nb, et in graph.neighbors_undirected(n["id"]):
            if nb in seen:
                continue
            node = graph.get(nb)
            if not node:
                continue
            results.append(node)
            seen.add(nb)
            if len(results) >= max_nodes:
                return results

    # 2) 同文件兄弟节点
    for n in anchor_nodes:
        path = _norm_posix(n.get("path"))
        if not path:
            continue
        file_node_id = _file_node_id(path)
        for dst, et in graph.neighbors(file_node_id):
            if et != "contains":
                continue
            if dst in seen:
                continue
            node = graph.get(dst)
            if not node or node["id"] == n["id"]:
                continue
            results.append(node)
            seen.add(dst)
            if len(results) >= max_nodes:
                return results

    # 3) 同目录下其它文件（轻度）
    for n in anchor_nodes:
        path = _norm_posix(n.get("path"))
        if not path:
            continue
        d = os.path.dirname(path)
        for nn in graph.nodes.values():
            if nn["kind"] == "file" and os.path.dirname(_norm_posix(nn.get("path"))) == d:
                if nn["id"] in seen:
                    continue
                results.append(nn)
                seen.add(nn["id"])
                if len(results) >= max_nodes:
                    return results

    return results[:max_nodes]


# ---------------- Internal: load or build graph ----------------

def _load_jsonl_graph(path: str, gh: GraphHandle) -> None:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            typ = obj.get("type")
            data = obj.get("data") or {}
            if typ == "node":
                n = _normalize_node(data, gh.root)
                gh.add_node(n)
            elif typ == "edge":
                src, dst, et = data.get("src"), data.get("dst"), data.get("etype") or "rel"
                if src and dst:
                    gh.add_edge(src, dst, et)

    # ensure file->contains computed degree even if edges missing
    for nid, node in list(gh.nodes.items()):
        if node["kind"] == "file":
            # if no contains recorded, try AST
            has_contains = any(et == "contains" for _, et in gh.neighbors(nid))
            if not has_contains and node.get("path", "").endswith(".py"):
                _add_python_contains(gh, node["path"])


def _build_light_graph(gh: GraphHandle) -> None:
    """
    Lightweight graph:
      - file nodes for text files (<=5MB)
      - python files: ast parse to add function/class (contains edges)
      - python imports: file -> imported file (imports edges, best-effort)
    """
    root = gh.root
    # add file nodes
    files = list_text_files(root, include_exts=[".py"])
    for f in files:
        rel = _norm_posix(os.path.relpath(f, root))
        nid = _file_node_id(rel)
        gh.add_node({"id": nid, "kind": "file", "path": rel, "name": "", "degree": 0})
    # python-specific enrich
    for f in files:
        rel = _norm_posix(os.path.relpath(f, root))
        _add_python_contains(gh, rel)
        _add_python_imports(gh, rel)


def _file_node_id(rel_path: str) -> str:
    return f"file:{_norm_posix(rel_path)}"


def _func_node_id(rel_path: str, name: str, lineno: int) -> str:
    return f"func:{_norm_posix(rel_path)}#{name}@{lineno}"


def _class_node_id(rel_path: str, name: str, lineno: int) -> str:
    return f"class:{_norm_posix(rel_path)}#{name}@{lineno}"


def _normalize_node(data: Dict[str, Any], root: str) -> Node:
    nid = data.get("id") or ""
    raw_kind = (data.get("kind") or "symbol")
    kind = str(raw_kind).lower()                # 统一小写
    # 常见别名归一
    alias = {
        "func": "function",
        "method": "function",
        "tfile": "t-file",
        "testfile": "t-file",
        "pkg": "pkg",
        "package": "pkg",
        "module": "module",
    }
    kind = alias.get(kind, kind)

    path = data.get("path") or ""
    if path and os.path.isabs(path):
        path = os.path.relpath(path, root)
    path = _norm_posix(path)

    node: Node = {
        "id": nid,
        "kind": kind,
        "path": path,
        "name": data.get("name") or "",
        "span": data.get("span") or {},
        "degree": 0,
    }
    return node


def _add_python_contains(gh: GraphHandle, rel_path: str) -> None:
    """
    Scan a python file and add function/class nodes with contains edges from file.
    """
    abs_path = os.path.join(gh.root, rel_path)
    try:
        src = safe_read_text(abs_path)
    except Exception:
        return
    try:
        tree = ast.parse(src)
    except Exception:
        return
    file_nid = _file_node_id(rel_path)
    # walk
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            nid = _func_node_id(rel_path, node.name, node.lineno)
            gh.add_node({"id": nid, "kind": "function", "path": rel_path, "name": node.name,
                         "span": {"start": node.lineno, "end": getattr(node, "end_lineno", node.lineno)},
                         "degree": 0})
            gh.add_edge(file_nid, nid, "contains")
        elif isinstance(node, ast.ClassDef):
            nid = _class_node_id(rel_path, node.name, node.lineno)
            gh.add_node({"id": nid, "kind": "class", "path": rel_path, "name": node.name,
                         "span": {"start": node.lineno, "end": getattr(node, "end_lineno", node.lineno)},
                         "degree": 0})
            gh.add_edge(file_nid, nid, "contains")


def _add_python_imports(gh: GraphHandle, rel_path: str) -> None:
    """
    Best-effort: add file -> file imports edges by resolving simple relative imports.
    """
    abs_path = os.path.join(gh.root, rel_path)
    try:
        src = safe_read_text(abs_path)
    except Exception:
        return
    try:
        tree = ast.parse(src)
    except Exception:
        return
    file_nid = _file_node_id(rel_path)
    pkg_dir = os.path.dirname(rel_path)
    for node in ast.walk(tree):
        target_rel: Optional[str] = None
        if isinstance(node, ast.Import):
            # import x.y -> x/y.py 或 x/y/__init__.py（近似）
            for alias in node.names:
                target_rel = _mod_to_path(alias.name)
                if target_rel:
                    _link_if_exists(gh, file_nid, target_rel, "imports")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            level = node.level or 0
            target_rel = _resolve_from(pkg_dir, mod, level)
            if target_rel:
                _link_if_exists(gh, file_nid, target_rel, "imports")


def _mod_to_path(mod: str) -> Optional[str]:
    # naive: a.b.c -> a/b/c.py
    p = mod.replace(".", "/") + ".py"
    return _norm_posix(p)


def _resolve_from(pkg_dir: str, mod: str, level: int) -> Optional[str]:
    # relative levels: 1 means current pkg
    base = pkg_dir
    for _ in range(max(0, level - 1)):
        base = os.path.dirname(base)
    if mod:
        p = os.path.join(base, mod.replace(".", "/") + ".py")
    else:
        p = os.path.join(base, "__init__.py")
    norm = os.path.normpath(p)
    return _norm_posix(norm)


def _link_if_exists(gh: GraphHandle, src_nid: str, target_rel: str, etype: str):
    target_rel = _norm_posix(os.path.normpath(target_rel))
    candidate_files = [target_rel]
    if target_rel.endswith(".py"):
        candidate_files.append(_norm_posix(os.path.join(target_rel[:-3], "__init__.py")))
    for rel in candidate_files:
        nid = _file_node_id(rel)
        if nid in gh.nodes:
            gh.add_edge(src_nid, nid, etype)
            return
