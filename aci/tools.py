import os, re, time, json, subprocess, tempfile, py_compile
from pathlib import Path
from typing import Dict, Any, List
from .schema import ACIResponse
from .hunkmap import compute_unified_diff

MAX_HITS_DEFAULT = 50

def view_file(args:Dict[str,Any])->ACIResponse:
    t0=time.time()
    p = Path(args["path"]); start=int(args["start"]); end=int(args["end"])
    text = p.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    start = max(1, start); end = min(len(lines), end)
    window = [f"{i:>6}: {lines[i-1]}" for i in range(start, end+1)]
    obs = {
        "snippets":[{"path":str(p), "start":start, "end":end, "text":"\n".join(window)}],
        "file_meta":{"total_lines":len(lines), "path":str(p)}
    }
    return ACIResponse(True, f"Viewed {p} [{start}:{end}]",
                       observations=obs,
                       metrics={"elapsed_ms":int(1000*(time.time()-t0))})

def search(args:Dict[str,Any])->ACIResponse:
    t0=time.time()
    query = args["query"]; glob = args.get("glob","**/*.py")
    max_hits = int(args.get("max_hits", MAX_HITS_DEFAULT))
    context = int(args.get("context", 3))
    root = Path(args.get("root","."))

    hits=[]
    for path in root.rglob(glob):
        if not path.is_file(): continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception: 
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            if re.search(query, line):
                s=max(1, i-context); e=min(len(lines), i+context)
                snippet="\n".join(f"{k:>6}: {l}" for k,l in enumerate(lines[s-1:e], start=s))
                hits.append({"path":str(path.relative_to(root)), "line":i, "text":snippet})
                if len(hits)>=max_hits: break
        if len(hits)>=max_hits: break

    ok = len(hits)>0
    limits = {"truncated": len(hits)>=max_hits}
    return ACIResponse(ok, f"Search '{query}' hits={len(hits)}",
                       observations={"search_hits":hits},
                       limits=limits,
                       metrics={"elapsed_ms":int(1000*(time.time()-t0))})

def lint_check(args:Dict[str,Any])->ACIResponse:
    t0=time.time()
    path = Path(args["path"])
    try:
        if path.suffix==".py":
            py_compile.compile(str(path), doraise=True)
        ok=True; summary=f"Syntax OK: {path}"
    except py_compile.PyCompileError as e:
        return ACIResponse(False, f"Syntax Error: {path}",
                           observations={"lint_error":str(e)},
                           metrics={"elapsed_ms":int(1000*(time.time()-t0))})
    return ACIResponse(True, summary, metrics={"elapsed_ms":int(1000*(time.time()-t0))})

def edit_lines(args:Dict[str,Any])->ACIResponse:
    t0=time.time()
    p = Path(args["path"]); start=int(args["start"]); end=int(args["end"])
    replacement = args["replacement"]
    original = p.read_text(encoding="utf-8", errors="ignore")
    lines = original.splitlines(True)  # keep \n
    start = max(1, start); end = min(len(lines), end)

    repl_text = replacement if replacement.endswith("\n") else replacement+"\n"
    new_lines = lines[:start-1] + [repl_text] + lines[end:]
    new_text = "".join(new_lines)

    if p.suffix==".py":
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(new_text); tmp.flush()
            try:
                py_compile.compile(tmp.name, doraise=True)
            except py_compile.PyCompileError as e:
                os.unlink(tmp.name)
                return ACIResponse(False, "Edit rejected by syntax check",
                                   observations={"lint_error":str(e)},
                                   metrics={"elapsed_ms":int(1000*(time.time()-t0))})
            os.unlink(tmp.name)

    p.write_text(new_text, encoding="utf-8")
    diff = compute_unified_diff(original, new_text, str(p), n_context=3)
    lines_changed = list(range(start, start+len(replacement.splitlines())))
    effects = {"files_touched":[str(p)], "lines_changed":lines_changed, "diff_summary":diff}
    obs = {"snippets":[{"path":str(p), "start":start, "end":start+len(replacement.splitlines())-1,
                        "text": replacement}]}
    return ACIResponse(True, f"Edited {p} [{start}:{end}]",
                       effects=effects, observations=obs,
                       metrics={"elapsed_ms":int(1000*(time.time()-t0))})

def run_tests(args:Dict[str,Any])->ACIResponse:
    t0=time.time()
    targets:List[str] = args.get("targets", ["-q"])
    timeout = int(args.get("timeout_s", 180))
    cmd = ["pytest"] + targets
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=timeout, text=True)
        out = cp.stdout
    except subprocess.TimeoutExpired:
        return ACIResponse(False, "pytest timeout",
                           observations={"test_report":{"timeout":True}},
                           metrics={"elapsed_ms":int(1000*(time.time()-t0))})

    passed = 0
    m_pass = re.search(r"=+\s*(\d+)\s+passed", out)
    if m_pass: passed = int(m_pass.group(1))
    failed = 1 if "FAILED" in out else 0

    # 解析首个失败栈帧（简版）
    frame = None
    m_frame = re.search(r"^([^\n:]+\.py):(\d+):\s+in\s+([A-Za-z0-9_<>]+)", out, re.M)
    if m_frame:
        frame = {"path": m_frame.group(1), "line": int(m_frame.group(2)), "func": m_frame.group(3)}

    head = re.search(r"(AssertionError:.*|E\s+.*)", out)
    top_assert = head.group(1)[:300] if head else None

    obs = {"test_report":{"passed":passed, "failed":failed, 
                          "top_assert":top_assert, "first_failure_frame":frame}}
    limits={"truncated": True, "omitted_tokens": max(0, len(out)-1500)}
    return ACIResponse(failed==0, "tests passed" if failed==0 else "tests failed",
                       observations=obs, limits=limits,
                       metrics={"elapsed_ms":int(1000*(time.time()-t0))})
