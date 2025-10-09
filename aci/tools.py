# -*- coding: utf-8 -*-
from __future__ import annotations

"""
ACI 基础工具：查看文件、搜索、编辑行、静态检查、运行测试
Core ACI tools: view/search/edit/lint/test (unified AciResp).
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

from .schema import AciResp, validate_aci_resp
from ._utils import (
    now_iso, run_cmd, repo_root, choose_executable,
    ensure_backup, safe_read_text, safe_write_text, list_text_files
)


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


# ---------- view_file ----------

def view_file(path: str) -> AciResp:
    """
    查看文件内容（utf-8 优先，回退 latin-1），限制 10MB。
    View file content with safe decodes; 10MB cap.
    """
    t0 = time.time()
    logs: List[str] = []
    try:
        abspath = os.path.abspath(path)
        if not os.path.isfile(abspath):
            return _resp(False, f"Not a file: {path}", {"path": path}, logs, t0)
        content = safe_read_text(abspath)
        data = {
            "path": path,
            "size": os.path.getsize(abspath),
            "content": content,
        }
        return _resp(True, "OK", data, logs, t0)
    except Exception as e:
        return _resp(False, f"view_file error: {e}", {"path": path}, logs + [str(e)], t0)


# ---------- search ----------

def search(query: str, filters: Optional[Dict[str, Any]] = None) -> AciResp:
    """
    代码搜索（优先 rg 再 git grep，再纯 Python 全文搜索）。
    filters:
      - roots: List[str]   # 限定根目录（默认仓库根）
      - exts:  List[str]   # 仅匹配扩展，如 [".py", ".ts"]
      - max_hits: int
    """
    t0 = time.time()
    logs: List[str] = []
    filters = filters or {}
    roots = filters.get("roots") or [repo_root()]
    exts = filters.get("exts") or []
    max_hits = int(filters.get("max_hits") or 500)

    # Try ripgrep JSON
    cmd = choose_executable([["rg", "-n", "--json", query]])
    results: List[Dict[str, Any]] = []

    try:
        if cmd:
            for root in roots:
                rc, out, err = run_cmd(cmd + ["."], cwd=root, timeout=120)
                logs.append(f"rg rc={rc}")
                if err:
                    logs.append(err.strip())
                if rc == 0 or rc == 2:  # rg returns 2 when some files were skipped
                    # Parse JSON lines, only "match" events
                    for line in out.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = __import__("json").loads(line)
                            if obj.get("type") != "match":
                                continue
                            data = obj["data"]["path"]["text"], obj["data"]["lines"]["text"], obj["data"]["line_number"]
                            path, text, lineno = data
                            if exts and not any(path.endswith(ext) for ext in exts):
                                continue
                            results.append({
                                "path": os.path.join(root, path),
                                "line": int(lineno),
                                "text": text.rstrip("\n"),
                            })
                            if len(results) >= max_hits:
                                break
                        except Exception:
                            continue
                # Stop when enough
                if len(results) >= max_hits:
                    break

        # Fallback: git grep
        if not results:
            cmd = choose_executable([["git", "grep", "-n", query]])
            if cmd:
                for root in roots:
                    rc, out, err = run_cmd(cmd, cwd=root, timeout=120)
                    logs.append(f"git grep rc={rc}")
                    if err:
                        logs.append(err.strip())
                    if rc == 0 and out:
                        for line in out.splitlines():
                            try:
                                path, rest = line.split(":", 1)
                                lineno_str, text = rest.split(":", 1)
                                lineno = int(lineno_str)
                                if exts and not any(path.endswith(ext) for ext in exts):
                                    continue
                                results.append({
                                    "path": os.path.join(root, path),
                                    "line": lineno,
                                    "text": text.rstrip("\n"),
                                })
                                if len(results) >= max_hits:
                                    break
                            except Exception:
                                continue
                    if len(results) >= max_hits:
                        break

        # Fallback: Python scan
        if not results:
            for root in roots:
                files = list_text_files(root, include_exts=exts or None)
                pat = re.compile(re.escape(query))
                for f in files:
                    try:
                        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                            for i, line in enumerate(fh, start=1):
                                if pat.search(line):
                                    results.append({"path": f, "line": i, "text": line.rstrip("\n")})
                                    if len(results) >= max_hits:
                                        break
                    except Exception:
                        continue
                    if len(results) >= max_hits:
                        break

        msg = f"{len(results)} hits"
        data = {"query": query, "results": results[:max_hits], "metrics": {"hits": len(results)}}
        return _resp(True, msg, data, logs, t0)
    except Exception as e:
        return _resp(False, f"search error: {e}", {"query": query}, logs + [str(e)], t0)


# ---------- edit_lines ----------

def edit_lines(path: str, start: int, end: int, new_text: str) -> AciResp:
    """
    以 1-based 包含区间替换文件的行范围；自动备份；保持换行一致。
    Replace [start, end] (inclusive, 1-based) with new_text; creates backup.
    """
    t0 = time.time()
    logs: List[str] = []
    try:
        if start < 1 or end < start:
            return _resp(False, f"invalid range: {start}-{end}", {"path": path}, logs, t0)

        abspath = os.path.abspath(path)
        if not os.path.isfile(abspath):
            return _resp(False, f"Not a file: {path}", {"path": path}, logs, t0)

        backup_path = ensure_backup(abspath)
        logs.append(f"backup: {backup_path}")

        original = safe_read_text(abspath)
        lines = original.splitlines(keepends=True)

        if end > len(lines):
            return _resp(False, f"end line {end} > total {len(lines)}", {"path": path}, logs, t0)

        # Normalize new_text to end with newline if the replaced block ended with newline
        replace_block = lines[start - 1:end]
        replaced_had_trailing_nl = replace_block[-1].endswith("\n") if replace_block else True
        new_block = new_text
        if replaced_had_trailing_nl and not new_block.endswith("\n"):
            new_block += "\n"

        updated = lines[:start - 1] + [new_block] + lines[end:]
        new_content = "".join(updated)

        safe_write_text(abspath, new_content)

        changed = end - start + 1
        data = {
            "path": path,
            "changed_lines": changed,
            "range": [start, end],
            "backup": backup_path,
            "metrics": {"changed_lines": changed},
        }
        return _resp(True, "Edited lines", data, logs, t0)
    except Exception as e:
        return _resp(False, f"edit_lines error: {e}", {"path": path}, logs + [str(e)], t0)


# ---------- lint_check ----------

def lint_check(paths: Optional[List[str]] = None) -> AciResp:
    """
    运行静态检查（优先 ruff / flake8 / pylint，回退 pyflakes）。
    Run linter with graceful fallback.
    """
    t0 = time.time()
    logs: List[str] = []
    root = repo_root()
    targets = paths or [root]

    # Candidate linters (first available wins)
    candidates = [
        ["ruff", "check", "--quiet"],
        ["flake8"],
        ["pylint", "--output-format=text"],  # heavy, may warn a lot
        ["python", "-m", "pyflakes"],
    ]
    cmd = choose_executable([candidates[0], candidates[1], candidates[2], candidates[3]])
    if not cmd:
        return _resp(False, "No linter found (ruff/flake8/pylint/pyflakes)", {"targets": targets}, logs, t0)

    rc, out, err = run_cmd(cmd + targets, cwd=root, timeout=600)
    logs.extend([f"cmd: {' '.join(cmd + targets)}", f"rc: {rc}"])
    if err:
        logs.append(err.strip())

    # Parse basic summary
    text = (out or "") + ("\n" + err if err else "")
    issues = 0
    for line in text.splitlines():
        # ruff/flake8/pyflakes formats are similar: file:line:col: code message
        if re.match(r".+:\d+:\d+:", line):
            issues += 1

    data = {
        "targets": targets,
        "report": out.strip(),
        "issues": issues,
        "rc": rc,
        "metrics": {"issues": issues},
    }
    ok = (rc == 0)
    msg = "Lint clean" if ok else f"Lint found {issues} issue(s)"
    return _resp(ok, msg, data, logs, t0)


# ---------- run_tests ----------

def run_tests(selectors: Optional[List[str]] = None) -> AciResp:
    """
    运行测试（优先 pytest，其次 unittest discover）。
    selectors 例如：["tests/test_foo.py::TestFoo::test_bar"]
    """
    t0 = time.time()
    logs: List[str] = []
    root = repo_root()

    selectors = selectors or []
    # Prefer pytest
    cmd = choose_executable([["pytest", "-q"], ["python", "-m", "pytest", "-q"]])
    used = None
    if cmd:
        used = cmd + selectors
        rc, out, err = run_cmd(used, cwd=root, timeout=3600)
        logs.extend([f"cmd: {' '.join(used)}", f"rc: {rc}"])
        if err:
            logs.append(err.strip())
        report_text = (out or "") + ("\n" + err if err else "")
        passed = _extract_int(report_text, r"(\d+)\s+passed")
        failed = _extract_int(report_text, r"(\d+)\s+failed")
        errors = _extract_int(report_text, r"(\d+)\s+error")
        skipped = _extract_int(report_text, r"(\d+)\s+skipped")
        data = {
            "framework": "pytest",
            "rc": rc,
            "report": report_text.strip(),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "metrics": {"passed": passed, "failed": failed, "errors": errors, "skipped": skipped},
        }
        ok = (rc == 0 and failed == 0 and errors == 0)
        msg = "Tests passed" if ok else "Tests have failures"
        return _resp(ok, msg, data, logs, t0)

    # Fallback: unittest discover
    used = ["python", "-m", "unittest", "discover", "-v"]
    rc, out, err = run_cmd(used, cwd=root, timeout=3600)
    logs.extend([f"cmd: {' '.join(used)}", f"rc: {rc}"])
    if err:
        logs.append(err.strip())
    report_text = (out or "") + ("\n" + err if err else "")

    # Heuristic parse
    # Ran X tests in Ys
    ran = _extract_int(report_text, r"Ran\s+(\d+)\s+tests?")
    failed = 0
    errors = 0
    m = re.search(r"FAILED\s+\(failures=(\d+)(?:,\s+errors=(\d+))?", report_text, re.I)
    if m:
        failed = int(m.group(1))
        if m.group(2):
            errors = int(m.group(2))
    ok = (rc == 0 and failed == 0 and errors == 0)
    data = {
        "framework": "unittest",
        "rc": rc,
        "report": report_text.strip(),
        "ran": ran,
        "failed": failed,
        "errors": errors,
        "metrics": {"ran": ran, "failed": failed, "errors": errors},
    }
    msg = "Tests passed" if ok else "Tests have failures"
    return _resp(ok, msg, data, logs, t0)


def _extract_int(text: str, pattern: str) -> int:
    m = re.search(pattern, text, re.I)
    return int(m.group(1)) if m else 0
