# 2025-10-22 memory hardening
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
memory/memory_bank.py

✅ 一体化“完整版”：
1) Step 2 的轻量日志本（MemoryBank）：仅记录 memops/feedback，不修改子图
2) Step 3.3 的子图应用入口（顶层函数 apply_ops + ApplyPolicy）：真正对工作子图执行 ADD/UPDATE/DELETE，
   带配额、目录多样性、t-file 比例等治理，并返回 summary（给 events/训练用）

orchestrator 侧可直接：
  from memory.memory_bank import apply_ops as apply_memops, ApplyPolicy
  from memory.memory_bank import MemoryBank  # 用于追加日志

依赖：
  - aci._utils.repo_root
  - memory.subgraph_store: add_nodes / update_node / remove_nodes / stats
"""

from typing import List, Dict, Any, Optional, Iterable
from dataclasses import dataclass
from collections import defaultdict, deque
from pathlib import PurePosixPath
import os
import json
import time

from aci._utils import repo_root
from . import subgraph_store

__all__ = ["MemoryItem", "MemoryBank", "ApplyPolicy", "apply_ops"]


# =============================================================================
# Part A: Step 2 - 轻量日志/便笺本（不改子图）
# =============================================================================

class MemoryItem(dict):
    """自由结构的记录条目；可包含 {id, ts, kind, payload...}。"""


class MemoryBank:
    """
    Step 2：把“记忆操作/反馈”等落到 .aci/memlog.json，便于调试/训练数据抽取。
    不对工作子图做任何修改（真正的子图维护见顶层 apply_ops）。
    """
    def __init__(self):
        self.root = repo_root()
        self.dir = os.path.join(self.root, ".aci")
        os.makedirs(self.dir, exist_ok=True)
        # 新文件名，避免与 Step3 概念混淆
        self.path_new = os.path.join(self.dir, "memlog.json")
        # 兼容旧文件名（若存在则并入）
        self.path_old = os.path.join(self.dir, "memory_bank.json")
        self.items: List[MemoryItem] = []
        self._load()

    def _load(self):
        arr: List[dict] = []
        # 先读新
        if os.path.isfile(self.path_new):
            try:
                with open(self.path_new, "r", encoding="utf-8") as f:
                    arr = json.load(f) or []
            except Exception:
                arr = []
        # 再读旧（合并一次即可）
        if os.path.isfile(self.path_old):
            try:
                with open(self.path_old, "r", encoding="utf-8") as f:
                    legacy = json.load(f) or []
                arr.extend(legacy)
                # 可选：合并后把旧文件重命名/备份
                # os.replace(self.path_old, self.path_old + ".bak")
            except Exception:
                pass
        # 规范化
        if isinstance(arr, list):
            self.items = [MemoryItem(x) for x in arr]

    def _save(self):
        tmp = self.path_new + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path_new)

    # ---- 兼容旧 API：仅记录，不改子图 ----
    def apply_ops(self, ops: List[Dict[str, Any]]) -> None:
        """
        兼容旧 API：本地记录 ops，但不修改子图。
        真正“应用到工作子图”，请调用模块顶层的 `apply_ops()`（见下）。
        """
        if not ops:
            return
        ts = time.time()
        self.items.append(MemoryItem({
            "id": f"ops::{len(self.items)+1}",
            "ts": ts,
            "kind": "memops_log",
            "deprecated": True,   # 提醒这不是在改子图
            "payload": {"ops": ops},
        }))
        self._save()

    def record_memops(self, issue_id: Optional[str], ops: List[Dict[str, Any]],
                      summary: Optional[Dict[str, Any]] = None) -> None:
        """
        推荐使用：记录 3.2 的建议与可选摘要（例如 3.3 返回的 summary）。
        """
        ts = time.time()
        self.items.append(MemoryItem({
            "id": f"memops::{len(self.items)+1}",
            "ts": ts,
            "kind": "memops_log",
            "issue_id": issue_id,
            "payload": {
                "ops": ops,
                "summary": summary or {}
            }
        }))
        self._save()

    def list_items(self) -> List[MemoryItem]:
        return list(self.items)

    def write_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        追加一条反馈记录（例如 ACI 回执 / 测试结果摘要 / 统计等）。
        """
        ts = time.time()
        self.items.append(MemoryItem({
            "id": f"fb::{len(self.items)+1}",
            "ts": ts,
            "kind": "feedback",
            "payload": feedback
        }))
        self._save()


# =============================================================================
# Part B: Step 3.3 - 真正应用到子图（配额/多样性/t-file 比例治理）
# =============================================================================

@dataclass(frozen=True)
class ApplyPolicy:
    # 总量与各类配额
    total_node_cap: int = 800
    add_limit: int = 40
    delete_limit: int = 20
    update_limit: int = 20

    # 目录多样性与每目录上限
    dir_diversity_k: int = 2
    per_dir_cap: int = 120

    # 测试文件偏好与比例控制
    prefer_test_files: bool = True
    max_tfile_fraction: float = 0.60

    # 安全策略
    forbid_delete_tfile: bool = True
    forbid_pure_delete: bool = True


def _dirname(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        return str(PurePosixPath(path).parent)
    except Exception:
        return ""


def _is_tfile(node_like: Dict[str, Any]) -> bool:
    kind = (node_like.get("kind") or "").lower()
    path = (node_like.get("path") or "").lower()
    if kind == "t-file":
        return True
    if not path:
        return False
    return ("test" in path) or ("/tests/" in path) or path.endswith("_test.py") or path.endswith("test.py")


def _iter_node_ids(subgraph: Any) -> Iterable[str]:
    """尽量通用地拿到当前子图中的节点 id 集。"""
    if hasattr(subgraph, "iter_node_ids"):
        return subgraph.iter_node_ids()  # type: ignore[attr-defined]
    nodes = (getattr(subgraph, "get", lambda k, d=None: d)("nodes", {}) or {})
    if isinstance(nodes, dict):
        return nodes.keys()  # type: ignore[return-value]
    return []


def _get_node(subgraph: Any, node_id: str) -> Dict[str, Any]:
    """尽量通用地读取单个节点。"""
    if hasattr(subgraph, "get_node"):
        return subgraph.get_node(node_id)  # type: ignore[attr-defined]
    nodes = (getattr(subgraph, "get", lambda k, d=None: d)("nodes", {}) or {})
    return nodes.get(node_id, {}) if isinstance(nodes, dict) else {}


def apply_ops(
    *,
    ops: List[Dict[str, Any]],
    subgraph: Any,
    policy: ApplyPolicy = ApplyPolicy(),
) -> Dict[str, Any]:
    """
    顶层函数：把 3.2 的 MemOp 列表应用到工作子图（ADD/UPDATE/DELETE），并返回 summary。
    - 提供 `apply_memops` 等别名，兼容旧版流水线的直接调用
    - 强制治理：配额/目录多样性/t-file 比例/安全开关
    """
    # 读取当前子图信息
    current_ids = set(_iter_node_ids(subgraph))
    current_stats = subgraph_store.stats(subgraph) or {}
    current_total = int(current_stats.get("nodes", len(current_ids)))

    # 按优先级去重：UPDATE > ADD > DELETE > KEEP > NOOP
    prio = {"UPDATE": 3, "ADD": 2, "DELETE": 1, "KEEP": 0, "NOOP": -1}
    by_id: Dict[str, Dict[str, Any]] = {}
    for o in sorted(
        ops or [],
        key=lambda x: (
            prio.get(x.get("op"), -1),
            float(x.get("score") or 0.0),
            float(x.get("confidence") or 0.0),
        ),
        reverse=True,
    ):
        nid = o.get("id")
        if not nid:
            continue
        if (nid not in by_id) or (prio.get(o.get("op"), -1) > prio.get(by_id[nid].get("op"), -1)):
            by_id[nid] = o
    ops = list(by_id.values())

    # 三类操作
    adds = [o for o in ops if o.get("op") == "ADD"]
    updates = [o for o in ops if o.get("op") == "UPDATE"]
    deletes = [o for o in ops if o.get("op") == "DELETE"]

    # 基础过滤
    adds = [o for o in adds if o.get("id") not in current_ids]
    updates = [o for o in updates if o.get("id") in current_ids]
    deletes = [o for o in deletes if o.get("id") in current_ids]

    updates.sort(
        key=lambda o: (
            float(o.get("score") or 0.0),
            float(o.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    deletes.sort(
        key=lambda o: (
            float(o.get("score") or 0.0),
            float(o.get("confidence") or 0.0),
        ),
        reverse=True,
    )

    # 安全：禁止删除 t-file（可通过 policy 放开）
    if policy.forbid_delete_tfile:
        deletes = [o for o in deletes if not _is_tfile(o)]

    # 预算裁剪
    remaining_cap = max(0, policy.total_node_cap - current_total)
    add_cap = min(policy.add_limit, remaining_cap)
    updates = updates[:policy.update_limit]
    deletes = deletes[:policy.delete_limit]

    # 目录多样性 + t-file 轻度偏好：选择 ADD
    for a in adds:
        base = float(a.get("confidence") or 0.6)
        base += float(a.get("score") or 0.0)
        if policy.prefer_test_files and _is_tfile(a):
            base += 0.12
        a["__sel_score"] = base

    by_dir: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in sorted(adds, key=lambda x: float(x.get("__sel_score") or 0.0), reverse=True):
        by_dir[_dirname(a.get("path"))].append(a)

    selected_adds: List[Dict[str, Any]] = []
    rr = {d: deque(lst[:policy.dir_diversity_k]) for d, lst in by_dir.items()}
    while rr and len(selected_adds) < add_cap:
        for d in list(rr.keys()):
            if rr[d]:
                selected_adds.append(rr[d].popleft())
                if len(selected_adds) >= add_cap:
                    break
            else:
                rr.pop(d, None)

    if len(selected_adds) < add_cap:
        picked = {o.get("id") for o in selected_adds}
        rest = [o for o in sorted(adds, key=lambda x: float(x.get("__sel_score") or 0.0), reverse=True)
                if o.get("id") not in picked]
        # 每目录上限
        per_dir_counter = defaultdict(int)
        for sel in selected_adds:
            per_dir_counter[_dirname(sel.get("path"))] += 1
        for o in rest:
            d = _dirname(o.get("path"))
            if per_dir_counter[d] >= policy.per_dir_cap:
                continue
            selected_adds.append(o)
            per_dir_counter[d] += 1
            if len(selected_adds) >= add_cap:
                break

    # t-file 比例保护
    def _count_tfiles(ids: Iterable[str]) -> int:
        c = 0
        for nid in ids:
            node = _get_node(subgraph, nid)
            if _is_tfile(node):
                c += 1
        return c

    current_tfiles = _count_tfiles(current_ids)
    max_allowed_tfiles = int(policy.max_tfile_fraction * max(1, current_total + len(selected_adds)))
    tmp: List[Dict[str, Any]] = []
    t_added = 0
    for o in selected_adds:
        if _is_tfile(o):
            if current_tfiles + t_added + 1 <= max_allowed_tfiles:
                tmp.append(o)
                t_added += 1
        else:
            tmp.append(o)
    selected_adds = tmp

    # 不允许“纯删除回合”（更稳健）
    if policy.forbid_pure_delete and (not selected_adds) and (not updates) and deletes:
        deletes = []

    # 执行写操作
    before_stats = dict(current_stats)
    applied = {"ADD": [], "UPDATE": [], "DELETE": [], "SKIPPED": []}

    # ADD：把 MemOp 映射为最小 Node 字段
    if selected_adds:
        nodes_to_add: List[Dict[str, Any]] = []
        seen = set()
        for a in selected_adds:
            nid = a.get("id")
            if not nid or nid in seen:
                continue
            seen.add(nid)
            nodes_to_add.append({
                "id": nid,
                "kind": (a.get("kind") or ""),
                "path": a.get("path"),
                "name": a.get("name"),
                "span": a.get("span"),
                "degree": int(a.get("degree") or 0),
            })
        if nodes_to_add:
            try:
                subgraph_store.add_nodes(subgraph, nodes_to_add)
                applied["ADD"] = [n["id"] for n in nodes_to_add]
            except Exception as e:
                applied["SKIPPED"].append({"op": "ADD", "reason": f"add_nodes_failed: {e}"})

    # UPDATE：只更新元信息
    for u in updates:
        nid = u.get("id")
        if not nid:
            continue
        try:
            patch: Dict[str, Any] = {}
            for k in ("path", "name", "span", "degree", "kind"):
                if k in u and u.get(k) is not None:
                    patch[k] = u.get(k)
            if patch:
                subgraph_store.update_node(subgraph, nid, **patch)
            applied["UPDATE"].append(nid)
        except Exception as e:
            applied["SKIPPED"].append({"op": "UPDATE", "id": nid, "reason": f"update_node_failed: {e}"})

    # DELETE
    if deletes:
        try:
            subgraph_store.remove_nodes(subgraph, [d.get("id") for d in deletes if d.get("id")])
            applied["DELETE"] = [d.get("id") for d in deletes if d.get("id")]
        except Exception as e:
            applied["SKIPPED"].append({"op": "DELETE", "reason": f"remove_nodes_failed: {e}"})

    final_stats = subgraph_store.stats(subgraph) or {}
    return {
        "applied": applied,
        "policy": {
            "total_node_cap": policy.total_node_cap,
            "add_limit": policy.add_limit,
            "delete_limit": policy.delete_limit,
            "update_limit": policy.update_limit,
            "dir_diversity_k": policy.dir_diversity_k,
            "per_dir_cap": policy.per_dir_cap,
            "prefer_test_files": policy.prefer_test_files,
            "max_tfile_fraction": policy.max_tfile_fraction,
            "forbid_delete_tfile": policy.forbid_delete_tfile,
            "forbid_pure_delete": policy.forbid_pure_delete,
        },
        "before": before_stats,
        "after": final_stats,
    }
