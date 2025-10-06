# actor/cgm_adapter.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---- 依赖你已有的类型 ----
# Plan/Target: memory.types.Plan
# ContextPack: memory.context_builder.ContextPack
# DocChunk   : memory.context_builder.DocChunk

# ========= Patch Guard 异常 =========

class PatchGuardError(ValueError):
    """Raised when a model patch violates allowed files/windows or size limits."""
    pass


# ========= 可替换的模型客户端接口 =========

class BaseCGMClient:
    """
    真实 CGM 的最小接口。你需要用自己的模型实现 infer()。
    约定：
      - graph_inputs: dict（子图编码），可按需使用或忽略
      - text_prompt : str（已拼装好的提示）
      - max_new_tokens: int
    返回：
      - 统一 diff 字符串（git-style）
    """
    def infer(self, graph_inputs: Dict[str, Any], text_prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError("Please implement CGMClient.infer(...) for your model.")


# ========= 适配器配置 =========

@dataclass
class CGMAdapterConfig:
    max_new_tokens: int = 2048
    max_patch_lines: int = 200     # 规模阈值（+/- 总数）
    allow_suffix_match: bool = False
    """当 diff 中文件路径与 Plan.targets 不完全相同时，是否允许用 endswith 方式匹配（谨慎开启）"""


# ========= 主类：CGMAdapter =========

class CGMAdapter:
    def __init__(self, cfg: Optional[CGMAdapterConfig] = None, client: Optional[BaseCGMClient] = None):
        self.cfg = cfg or CGMAdapterConfig()
        self.client = client or BaseCGMClient()  # 若未注入真实模型，会在 infer 时报错

    # ---------- 外部调用入口 ----------
    def generate(self, subgraph_nodes: List[str], plan, context_pack) -> str:
        """
        产出统一 diff（字符串）。
        - subgraph_nodes: RepoGraph 子图节点列表（文件路径等）
        - plan: memory.types.Plan
        - context_pack: memory.context_builder.ContextPack
        """
        graph_inputs = self._pack_graph_inputs(subgraph_nodes)
        prompt = self._build_prompt(plan, context_pack)

        raw = self.client.infer(
            graph_inputs=graph_inputs,
            text_prompt=prompt,
            max_new_tokens=self.cfg.max_new_tokens,
        )
        diff = self._sanitize_model_output_to_diff(raw)
        self._patch_guard(diff, plan)  # 违规直接抛 PatchGuardError
        return diff

    # ---------- 模型输入打包 ----------
    def _pack_graph_inputs(self, subgraph_nodes: List[str]) -> Dict[str, Any]:
        """
        子图 → 模型图模态的最小占位打包。
        如你的 CGM 有专用编码器，这里替换为相应格式（edge_index/node_attrs 等）。
        """
        return {
            "nodes": subgraph_nodes,
            # "edge_index": ...,
            # "node_attrs": ...,
        }

    # ---------- Prompt 拼装 ----------
    def _build_prompt(self, plan, context_pack) -> str:
        header_sys = (
            "You are a code patch generator. Follow ALL rules strictly:\n"
            "1) Only edit the specified file(s) and line windows.\n"
            "2) Output ONLY a valid unified diff (git-style).\n"
            "3) Minimal patch: no reformatting or unrelated changes.\n"
            "4) If a change is outside allowed windows, DO NOT produce it.\n"
        )

        # WINDOWS 列表
        windows_lines = []
        for t in getattr(plan, "targets", []):
            windows_lines.append(f"- {t.path}: [{int(t.start)},{int(t.end)}]")

        header_user = "ISSUE (rewritten substep):\n"
        issue_text = getattr(plan, "why", None) or ""  # 如有更好的子步描述，从外层传入/附加到 plan

        windows_block = "WINDOWS (allowed edit ranges):\n" + "\n".join(windows_lines) + "\n"

        # CONTEXT
        ctx_lines = []
        if context_pack.mode == "full":
            ctx_lines.append("CONTEXT (FULL FILES):")
        else:
            ctx_lines.append("CONTEXT (chunked, within token budget):")

        for i, ch in enumerate(context_pack.chunks, 1):
            if not ch.text:
                continue
            if ch.kind == "full_file":
                ctx_lines.append(f"=== BEGIN [{ch.kind}] {ch.path} ===")
            else:
                ctx_lines.append(f"=== BEGIN [{ch.kind}] {ch.path}:{ch.start}-{ch.end} ===")
            ctx_lines.append(ch.text.rstrip("\n"))
            ctx_lines.append("=== END ===")

        tail = (
            "\nOUTPUT FORMAT:\n"
            "Return ONLY the unified diff. Example:\n"
            "--- a/pkg/mod.py\n"
            "+++ b/pkg/mod.py\n"
            "@@ -310,7 +310,8 @@\n"
            "- old line\n"
            "+ new line\n"
        )

        parts = [
            header_sys,
            header_user,
            issue_text.strip(),
            "\n",
            windows_block,
            "\n".join(ctx_lines),
            "\n",
            tail,
        ]
        return "\n".join(parts)

    # ---------- 将模型原文规整为 diff ----------
    def _sanitize_model_output_to_diff(self, raw: str) -> str:
        """
        有些模型可能额外输出解释或包裹代码块。尽量抽取统一 diff 主体。
        规则：
          - 优先找到第一处 '--- a/' 与 '+++ b/' 开头的行；
          - 截取到文本末尾；
          - 去除 markdown 代码围栏；
        """
        s = raw.strip()

        # 去掉 ```diff/``` 等围栏
        if s.startswith("```"):
            # 删除首行 ```xxx
            s = "\n".join(s.splitlines()[1:])
            # 删除末尾围栏 ```
            if s.rstrip().endswith("```"):
                s = "\n".join(s.splitlines()[:-1])

        # 从第一处文件头开始截取
        start_idx = None
        lines = s.splitlines()
        for idx, ln in enumerate(lines):
            if ln.startswith("--- a/") or ln.startswith("+++ b/"):
                start_idx = idx
                break

        if start_idx is None:
            # 容忍模型把文件头放在下一行：尝试从第一处 hunk 开始（不推荐，但兜底）
            for idx, ln in enumerate(lines):
                if _HUNK_HEADER.match(ln):
                    start_idx = max(0, idx - 2)  # 盲插两行，可能缺文件头，会被 Guard 拒
                    break

        if start_idx is not None:
            s = "\n".join(lines[start_idx:]).strip()

        return s

    # ---------- Patch Guard ----------
    def _patch_guard(self, diff: str, plan) -> None:
        """
        核心约束：
          1) 只允许修改 Plan.targets 中的文件；
          2) 每个 hunk 必须完全落在对应文件的行窗内（新/旧行范围至少一个匹配）；
          3) 规模阈值：新增+删除行数 <= cfg.max_patch_lines；
        违规直接抛 PatchGuardError。
        """
        if not diff or ("@@ " not in diff):
            raise PatchGuardError("Invalid patch: empty or missing hunks.")

        allowed_files = set(t.path for t in getattr(plan, "targets", []))
        windows_by_file = _targets_windows_map(plan)

        # 1) 校验涉及文件
        files_in_diff = _parse_diff_files(diff)
        if not files_in_diff:
            raise PatchGuardError("Invalid patch: no file headers found (---/+++).")

        if self.cfg.allow_suffix_match:
            # 放宽匹配：a/b/c.py 允许匹配 plan 的 c.py（谨慎使用）
            def ok_file(fh: str) -> bool:
                return any(fh == af or fh.endswith("/" + af) for af in allowed_files)
        else:
            def ok_file(fh: str) -> bool:
                return fh in allowed_files

        for f in files_in_diff:
            if not ok_file(f):
                raise PatchGuardError(f"Patch touches a disallowed file: {f}")

        # 2) 校验所有 hunk 在窗口内
        current_file: Optional[str] = None
        for ln in diff.splitlines():
            if ln.startswith('--- a/'):
                current_file = ln[6:].strip()
            elif ln.startswith('+++ b/'):
                current_file = ln[6:].strip()
            else:
                hm = _HUNK_HEADER.match(ln)
                if hm and current_file:
                    old_l = int(hm.group(1)); old_n = int(hm.group(2) or "1")
                    new_l = int(hm.group(3)); new_n = int(hm.group(4) or "1")
                    old_start, old_end = old_l, old_l + max(old_n, 1) - 1
                    new_start, new_end = new_l, new_l + max(new_n, 1) - 1

                    wins = windows_by_file.get(current_file, [])
                    if self.cfg.allow_suffix_match and not wins:
                        # 尝试用后缀匹配找到目标窗口
                        match_key = _best_suffix_match(current_file, windows_by_file.keys())
                        if match_key:
                            wins = windows_by_file.get(match_key, [])

                    if not wins:
                        raise PatchGuardError(f"No allowed windows registered for file: {current_file}")

                    ok_old = _range_in_any_window(old_start, old_end, wins)
                    ok_new = _range_in_any_window(new_start, new_end, wins)
                    if not (ok_old or ok_new):
                        raise PatchGuardError(
                            f"Hunk out of allowed windows in {current_file}: "
                            f"old[{old_start},{old_end}] new[{new_start},{new_end}]"
                        )

        # 3) 规模阈值
        added = sum(1 for l in diff.splitlines() if l.startswith('+') and not l.startswith('+++ b/'))
        removed = sum(1 for l in diff.splitlines() if l.startswith('-') and not l.startswith('--- a/'))
        if (added + removed) > self.cfg.max_patch_lines:
            raise PatchGuardError(
                f"Patch too large ({added}+{removed} lines) > limit {self.cfg.max_patch_lines}"
            )


# ========= Diff/Window 解析辅助 =========

_FILE_HEADER = re.compile(r"^(--- a/(?P<a>.+))|(\+\+\+ b/(?P<b>.+))")
_HUNK_HEADER = re.compile(r"^@@ \-(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")  # old_l,old_n,new_l,new_n

def _parse_diff_files(diff: str) -> List[str]:
    files: List[str] = []
    seen = set()
    for line in diff.splitlines():
        m = _FILE_HEADER.match(line)
        if not m:
            continue
        fn = m.group("a") or m.group("b")
        if fn and fn not in seen:
            seen.add(fn)
            files.append(fn)
    return files

def _targets_windows_map(plan) -> Dict[str, List[Tuple[int, int]]]:
    m: Dict[str, List[Tuple[int, int]]] = {}
    for t in getattr(plan, "targets", []):
        m.setdefault(t.path, []).append((int(t.start), int(t.end)))
    # 合并重叠窗口
    out: Dict[str, List[Tuple[int, int]]] = {}
    for p, wins in m.items():
        wins = sorted(wins)
        merged: List[Tuple[int, int]] = []
        cur_s, cur_e = wins[0]
        for s, e in wins[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        out[p] = merged
    return out

def _range_in_any_window(a: int, b: int, windows: List[Tuple[int, int]]) -> bool:
    for s, e in windows:
        if a >= s and b <= e:
            return True
    return False

def _best_suffix_match(name: str, candidates: List[str]) -> Optional[str]:
    # 返回与 name 后缀匹配度最高的候选（简化版）
    best = None
    best_len = 0
    for c in candidates:
        if name.endswith("/" + c) and len(c) > best_len:
            best, best_len = c, len(c)
    return best
