# -*- coding: utf-8 -*-
from __future__ import annotations
"""
actor/cgm_adapter.py

Step 4.2（本地路线）：
- generate(subgraph_linearized, plan, constraints) -> Patch
  在每个 PlanTarget 的行窗内做一次“最小可见修改”（末行追加注释标记），
  以便跑通 Guard/ACI/Lint/Test 的端到端闭环。

特点：
- 仅修改 PlanTarget 的末行（start..end 内），严格遵守 Guard 的行窗约束
- 语言适配注释前缀（.py/.js/.ts/.go/.rs/.java/.c/.cpp/.sh/.sql/.toml/.ini/.yaml/.yml/.md 等）
- 幂等保护：若该行已包含标记，不再重复编辑
- 失败容错：文件无法读取时兜底写入简单标记行

你可以 later 将此文件升级为真实 CGM（大模型）调用；对 orchestrator 的接口不变。
"""

from typing import List, Dict, Any, Optional, Tuple
import os

from aci.schema import Patch, PatchEdit, Plan, PlanTarget

_MARKER = "CGM-LOCAL"


# ---------------- helpers ----------------

def _detect_comment_prefix(path: str) -> str:
    """根据扩展名返回单行注释前缀；未知类型用 '#'。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".py", ".rb", ".pl"}:
        return "#"
    if ext in {".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".hpp", ".cpp", ".cc", ".go", ".rs", ".kt", ".scala"}:
        return "//"
    if ext in {".sh", ".bash", ".zsh", ".fish", ".dockerfile"}:
        return "#"
    if ext in {".sql"}:
        return "--"
    if ext in {".toml", ".ini", ".cfg"}:
        return "#"
    if ext in {".yaml", ".yml"}:
        return "#"
    if ext in {".md", ".rst"}:
        # Markdown/RST 没有统一注释，改用 HTML 风格；为“追加到行尾”安全起见仍用 '<!-- -->'
        return "<!--"
    return "#"


def _normalize_path_for_read(path: str) -> str:
    """兼容绝对/相对路径：相对路径拼 cwd。"""
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path)


def _read_lines_safe(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().splitlines(True)  # 保留换行符
    except Exception:
        return []


def _append_marker_line(line: str, path: str) -> str:
    """在一行尾部追加标记注释，尽量不破坏原行尾换行（由调用方补）。"""
    pref = _detect_comment_prefix(path)
    # Markdown 特殊：用 <!-- ... --> 包裹
    if pref == "<!--":
        # 避免破坏行内代码块，采用尾部空格+注释块
        suffix = " <!-- {} -->".format(_MARKER)
    else:
        suffix = f" {pref} {_MARKER}"
    # 去掉行尾换行，在外层统一补回
    core = line.rstrip("\r\n")
    return core + suffix


def _build_single_edit_for_target(t: PlanTarget) -> Optional[PatchEdit]:
    """
    仅编辑目标窗口的末行（end），在行尾追加注释标记。
    - 若该行已包含标记则跳过（返回 None）
    - 若文件读不到，则兜底：把末行替换为单行标记（尽量不破坏 guard 约束）
    """
    rpath = t.path
    abspath = _normalize_path_for_read(rpath)
    lines = _read_lines_safe(abspath)

    # 计算可用的行号（1-based）
    if not lines:
        # 兜底：给出一个最小 new_text，仍将 start=end 指向 t.end
        return PatchEdit(path=rpath, start=int(t.end), end=int(t.end),
                         new_text=f"# {_MARKER}\n")

    end_idx_1b = max(1, min(int(t.end), len(lines)))
    line = lines[end_idx_1b - 1]
    # 幂等保护
    if _MARKER in line:
        return None

    new_line = _append_marker_line(line, rpath) + "\n"
    return PatchEdit(path=rpath, start=end_idx_1b, end=end_idx_1b, new_text=new_line)


def _summarize_patch(edits: List[PatchEdit]) -> str:
    if not edits:
        return "local-cgm: no-op (no eligible lines to edit)"
    paths = sorted({e["path"] for e in edits})
    return f"local-cgm: edits={len(edits)} files={len(paths)} marker={_MARKER}"


# ---------------- main API ----------------

def generate(subgraph_linearized: Optional[List[Dict[str, Any]]],
             plan: Plan,
             constraints: Optional[Dict[str, Any]] = None) -> Patch:
    """
    本地 CGM 占位实现：
    - 遍历 plan.targets
    - 对每个 target 的末行进行单行替换（追加注释标记）
    - 最多编辑 N 个 target（可由 constraints['max_edits'] 控制，默认 3）
    """
    max_edits = int((constraints or {}).get("max_edits", 3))
    edits: List[PatchEdit] = []

    for t in plan.targets:
        if len(edits) >= max_edits:
            break
        e = _build_single_edit_for_target(t)
        if e:
            edits.append(e)

    patch: Patch = {"edits": edits, "summary": _summarize_patch(edits)}
    return patch
