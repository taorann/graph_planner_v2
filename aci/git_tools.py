# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Git 操作：创建分支、提交、回滚、查看 diff。
Git helpers with safe fallbacks and unified AciResp.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

from .schema import AciResp, validate_aci_resp
from ._utils import now_iso, run_cmd, repo_root


def _resp(success: bool, message: str, data: Dict[str, Any], logs: List[str], t0: float) -> AciResp:
    resp: AciResp = {
        "success": success,
        "message": message,
        "data": data,
        "logs": logs,
        "metrics": data.get("metrics", {}),
        "ts": now_iso(),
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
    validate_aci_resp(resp)
    return resp


def _git(cmd: List[str], cwd: Optional[str] = None, timeout: int = 600):
    return run_cmd(["git"] + cmd, cwd=cwd or repo_root(), timeout=timeout)


def create_branch(name: str) -> AciResp:
    """
    创建并切换到新分支（若已存在则直接切换）。
    Prefer `git switch -c` then fallback to `git checkout -b`.
    """
    t0 = time.time()
    logs: List[str] = []

    root = repo_root()
    # Check if branch exists
    rc, out, err = _git(["rev-parse", "--verify", name], cwd=root)
    exists = (rc == 0)
    logs.append(f"branch exists? rc={rc} ({'yes' if exists else 'no'})")

    if exists:
        rc, out, err = _git(["switch", name], cwd=root)
        if rc != 0:
            logs.append(err.strip())
            rc, out, err = _git(["checkout", name], cwd=root)
    else:
        rc, out, err = _git(["switch", "-c", name], cwd=root)
        if rc != 0:
            logs.append(err.strip())
            rc, out, err = _git(["checkout", "-b", name], cwd=root)

    if err:
        logs.append(err.strip())

    if rc == 0:
        return _resp(True, f"On branch {name}", {"branch": name}, logs, t0)
    return _resp(False, f"Failed to switch/create branch {name}", {"branch": name}, logs, t0)


def commit_patch(msg: str) -> AciResp:
    """
    提交当前工作区的改动；若无改动则返回成功且注明 nothing_to_commit。
    Stage & commit. If nothing to commit, return success with flag.
    """
    t0 = time.time()
    logs: List[str] = []
    root = repo_root()

    # Stage all
    rc, out, err = _git(["add", "-A"], cwd=root)
    if err:
        logs.append(err.strip())

    # Check staged changes
    rc, out, err = _git(["diff", "--staged", "--name-only"], cwd=root)
    if err:
        logs.append(err.strip())
    changed_files = [l.strip() for l in out.splitlines() if l.strip()]
    if not changed_files:
        data = {"nothing_to_commit": True, "changed_files": []}
        return _resp(True, "Nothing to commit", data, logs, t0)

    # Commit
    rc, out, err = _git(["commit", "-m", msg], cwd=root)
    if err:
        logs.append(err.strip())
    if rc != 0:
        return _resp(False, "git commit failed", {"changed_files": changed_files}, logs, t0)

    # Get last commit sha
    rc, out, err = _git(["rev-parse", "HEAD"], cwd=root)
    sha = out.strip() if rc == 0 else ""
    data = {"commit": sha, "changed_files": changed_files, "metrics": {"files_changed": len(changed_files)}}
    return _resp(True, f"Committed {len(changed_files)} files", data, logs, t0)


def revert_last() -> AciResp:
    """
    回滚最近一次提交（优先使用 `git revert --no-edit HEAD`；失败则危险回退 `git reset --hard HEAD~1`）。
    Revert last commit safely; fallback to destructive reset.
    """
    t0 = time.time()
    logs: List[str] = []
    root = repo_root()

    # Try revert (creates a new commit)
    rc, out, err = _git(["revert", "--no-edit", "HEAD"], cwd=root)
    if err:
        logs.append(err.strip())
    if rc == 0:
        rc2, sha, _ = _git(["rev-parse", "HEAD"], cwd=root)
        return _resp(True, "Reverted last commit (safe)", {"commit": sha.strip(), "destructive": False}, logs, t0)

    # Fallback: hard reset (destructive)
    logs.append("safe revert failed; falling back to hard reset")
    rc, out, err = _git(["reset", "--hard", "HEAD~1"], cwd=root)
    if err:
        logs.append(err.strip())
    if rc != 0:
        return _resp(False, "Failed to revert/reset last commit", {}, logs, t0)

    rc2, sha, _ = _git(["rev-parse", "HEAD"], cwd=root)
    return _resp(True, "Reset to HEAD~1 (destructive)", {"commit": sha.strip(), "destructive": True}, logs, t0)


def diff_head() -> AciResp:
    """
    查看工作区相对 HEAD 的 diff（unified=3），并返回变更统计。
    Show diff vs HEAD and basic stats.
    """
    t0 = time.time()
    logs: List[str] = []
    root = repo_root()

    # Stats via numstat
    rc, out, err = _git(["diff", "--numstat"], cwd=root)
    if err:
        logs.append(err.strip())
    added = 0
    deleted = 0
    files: List[str] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            try:
                a = 0 if parts[0] == "-" else int(parts[0])
                d = 0 if parts[1] == "-" else int(parts[1])
                added += a
                deleted += d
                files.append(parts[2])
            except Exception:
                continue

    # Patch
    rc2, patch, err2 = _git(["diff", "--unified=3"], cwd=root)
    if err2:
        logs.append(err2.strip())

    data = {
        "files": files,
        "added": added,
        "deleted": deleted,
        "patch": patch,
        "metrics": {"files": len(files), "added": added, "deleted": deleted},
    }
    ok = len(files) > 0
    msg = "Working tree is clean" if not ok else f"{len(files)} file(s) changed"
    return _resp(ok, msg, data, logs, t0)
