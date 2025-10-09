# -*- coding: utf-8 -*-
from __future__ import annotations

"""
内部工具：命令执行、仓库根、时间、备份、可执行检测。
Internal helpers used by ACI modules.
"""

import os
import re
import json
import time
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple


def now_iso() -> str:
    # Always use Z to avoid tz ambiguity in logs.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_cmd(
    cmd: List[str],
    cwd: Optional[str] = None,
    timeout: int = 600,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    """
    统一命令执行（捕获 stdout/stderr），不抛异常。
    Unified command runner with timeout; returns (rc, out, err).
    """
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", e.stderr or f"Timeout after {timeout}s"
    except FileNotFoundError:
        return 127, "", f"Executable not found: {cmd[0]}"
    except Exception as e:
        return 1, "", f"Command error: {e}"


def repo_root(cwd: Optional[str] = None) -> str:
    """
    获取 git 仓库根；若非 git 仓库，返回绝对 cwd。
    Get git root or absolute cwd if not a git repo.
    """
    cwd = cwd or os.getcwd()
    rc, out, _ = run_cmd(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
    if rc == 0 and out.strip():
        return out.strip()
    return os.path.abspath(cwd)


def choose_executable(candidates: List[List[str]]) -> Optional[List[str]]:
    """
    从候选可执行（含参数）里选择第一个可用的命令（通过 which 检测）。
    Pick the first available executable (with args) from candidates.
    """
    for cmd in candidates:
        exe = cmd[0]
        if shutil.which(exe):
            return cmd
    return None


def ensure_backup(path: str) -> str:
    """
    为被编辑的文件创建时间戳备份：.aci/backups/<relpath>.<ts>.bak
    Create timestamped backup under .aci/backups preserving relative tree.
    """
    root = repo_root()
    rel = os.path.relpath(os.path.abspath(path), root)
    backup_dir = os.path.join(root, ".aci", "backups", os.path.dirname(rel))
    os.makedirs(backup_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    suffix = f".{ts}.bak"
    backup_path = os.path.join(backup_dir, os.path.basename(rel) + suffix)
    shutil.copy2(path, backup_path)
    return backup_path


def safe_read_text(path: str, max_bytes: int = 10 * 1024 * 1024) -> str:
    """
    尝试 utf-8，失败回退 latin-1，限制最大读取字节。
    Try utf-8 then latin-1; cap read size.
    """
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"File too large ({size} bytes > {max_bytes}) to read safely: {path}")
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def safe_write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def list_text_files(root: str, include_exts: Optional[List[str]] = None) -> List[str]:
    """
    简单的文本文件遍历（可按扩展过滤）。
    """
    results: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        # Skip VCS / build artifacts
        if any(part.startswith(".git") for part in dirpath.split(os.sep)):
            continue
        for fn in filenames:
            if include_exts and not any(fn.endswith(ext) for ext in include_exts):
                continue
            full = os.path.join(dirpath, fn)
            # Heuristic: small/binary skip
            try:
                if os.path.getsize(full) > 5 * 1024 * 1024:
                    continue
                with open(full, "rb") as f:
                    blob = f.read(2048)
                if b"\x00" in blob:
                    continue
                results.append(full)
            except Exception:
                continue
    return results
