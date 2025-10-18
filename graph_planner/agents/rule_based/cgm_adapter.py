# -*- coding: utf-8 -*-
from __future__ import annotations
"""
agents/rule_based/cgm_adapter.py

规则驱动的补丁生成器：
- generate(subgraph_linearized, plan, constraints, snippets) -> Patch
  根据 PlannerEnv 返回的片段数据，对每个 PlanTarget 的末行附加注释标记，
  以保证补丁在 Guard/测试流程中的可见性，同时避免依赖容器外的文件系统。
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import os

from aci.schema import Patch, PatchEdit, Plan, PlanTarget
from ...infra import telemetry
from ...infra.config import load as load_config
from ...integrations.codefuse_cgm import (
    CGMExample,
    CGMGenerationConfig,
    CodeFuseCGMClient,
    CodeFuseCGMGenerator,
    ConversationEncoder,
    GraphLinearizer,
    SnippetFormatter,
)
from ..common.chat import extract_json_payload

_MARKER = "CGM-LOCAL"
_LOCAL_INSTRUCTION = (
    "Generate a JSON object with a `patch` field. The `patch.edits` array must"
    " list objects containing `path`, `start`, `end`, and `new_text`. Ensure"
    " `new_text` ends with a newline."
)


# ---------------- helpers ----------------

def _detect_comment_prefix(path: str) -> str:
    """根据扩展名返回单行注释前缀；未知类型用 '#'."""

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
        return "<!--"
    return "#"


def _append_marker_line(line: str, path: str) -> str:
    """在一行尾部追加标记注释，保持原缩进。"""

    pref = _detect_comment_prefix(path)
    if pref == "<!--":
        suffix = f" <!-- {_MARKER} -->"
    else:
        suffix = f" {pref} {_MARKER}"
    return line.rstrip("\r\n") + suffix


def _parse_snippet_lines(snippet: Dict[str, Any]) -> Dict[int, str]:
    lines: Dict[int, str] = {}
    for raw in snippet.get("snippet") or []:
        if ":" not in raw:
            continue
        prefix, remainder = raw.split(":", 1)
        try:
            idx = int(prefix)
        except ValueError:
            continue
        text = remainder[1:] if remainder.startswith(" ") else remainder
        lines[idx] = text.rstrip("\r\n")
    return lines


def _index_snippets(snippets: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[int, str]]:
    index: Dict[str, Dict[int, str]] = {}
    for snip in snippets or []:
        path = snip.get("path")
        if not path:
            continue
        lines = _parse_snippet_lines(snip)
        if not lines:
            continue
        index.setdefault(path, {}).update(lines)
        abs_path = snip.get("abs_path")
        if abs_path:
            index.setdefault(abs_path, {}).update(lines)
    return index


def _read_line_from_fs(path: str, lineno: int) -> Optional[str]:
    abspath = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    try:
        with open(abspath, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, 1):
                if idx == lineno:
                    return line.rstrip("\r\n")
    except Exception:
        return None
    return None


def _build_single_edit_for_target(
    target: PlanTarget,
    snippet_index: Dict[str, Dict[int, str]],
) -> Optional[PatchEdit]:
    path = target.path
    line_no = int(getattr(target, "end", 1))
    line_map = snippet_index.get(path) or snippet_index.get(os.path.abspath(path))
    line_text: Optional[str] = None
    if line_map:
        line_text = line_map.get(line_no) or line_map.get(int(getattr(target, "start", line_no)))
    if not line_text:
        line_text = _read_line_from_fs(path, line_no)
    if not line_text:
        return None
    if _MARKER in line_text:
        return None
    new_line = _append_marker_line(line_text, path) + "\n"
    return PatchEdit(path=path, start=line_no, end=line_no, new_text=new_line)


def _summarize_patch(edits: List[PatchEdit]) -> str:
    if not edits:
        return "local-cgm: no-op (no eligible lines to edit)"
    paths = sorted({e["path"] for e in edits})
    return f"local-cgm: edits={len(edits)} files={len(paths)} marker={_MARKER}"


# ---------------- main API ----------------

_CLIENT_CACHE: Optional[CodeFuseCGMClient] = None
_CLIENT_FINGERPRINT: Optional[tuple] = None
_LOCAL_RUNTIME_CACHE: Optional["_LocalCGMRuntime"] = None
_LOCAL_RUNTIME_FINGERPRINT: Optional[tuple] = None


def _get_client() -> Optional[CodeFuseCGMClient]:
    global _CLIENT_CACHE, _CLIENT_FINGERPRINT
    cfg = load_config()
    cgm_cfg = getattr(cfg, "cgm", None)
    if not cgm_cfg or not getattr(cgm_cfg, "enabled", False):
        return None
    endpoint = getattr(cgm_cfg, "endpoint", None)
    if not endpoint:
        return None
    api_key = None
    api_key_env = getattr(cgm_cfg, "api_key_env", None)
    if api_key_env:
        api_key = os.environ.get(api_key_env)
    fingerprint = (
        endpoint,
        api_key,
        getattr(cgm_cfg, "model", None),
        getattr(cgm_cfg, "temperature", None),
        getattr(cgm_cfg, "max_tokens", None),
        getattr(cgm_cfg, "timeout_s", 60),
    )
    if _CLIENT_CACHE and _CLIENT_FINGERPRINT == fingerprint:
        return _CLIENT_CACHE
    _CLIENT_CACHE = CodeFuseCGMClient(
        endpoint=endpoint,
        api_key=api_key,
        model=getattr(cgm_cfg, "model", None),
        temperature=getattr(cgm_cfg, "temperature", None),
        max_tokens=getattr(cgm_cfg, "max_tokens", None),
        timeout_s=int(getattr(cgm_cfg, "timeout_s", 60)),
    )
    _CLIENT_FINGERPRINT = fingerprint
    return _CLIENT_CACHE


@dataclass
class _LocalCGMRuntime:
    generator: CodeFuseCGMGenerator
    instruction: str = _LOCAL_INSTRUCTION

    def generate_patch(
        self,
        *,
        issue: Optional[Mapping[str, Any]],
        plan: Plan,
        plan_text: Optional[str],
        subgraph_linearized: Optional[Iterable[Mapping[str, Any]]],
        snippets: Optional[Iterable[Mapping[str, Any]]],
        constraints: Optional[Mapping[str, Any]] = None,
    ) -> Patch:
        example = self._build_example(
            plan=plan_text,
            issue=issue,
            snippets=snippets,
            subgraph_linearized=subgraph_linearized,
        )
        sequences = self.generator.generate(example)

        max_edits = None
        if constraints:
            try:
                max_edits = int(constraints.get("max_edits", 0)) or None
            except Exception:
                max_edits = None

        summary_fallback = plan_text or self.instruction
        for seq in sequences:
            patch = self._parse_sequence(seq, max_edits=max_edits, summary=summary_fallback)
            if patch:
                patch.setdefault("summary", summary_fallback)
                return patch
        raise RuntimeError("local CGM produced no valid patch")

    def _build_example(
        self,
        *,
        plan: Optional[str],
        issue: Optional[Mapping[str, Any]],
        snippets: Optional[Iterable[Mapping[str, Any]]],
        subgraph_linearized: Optional[Iterable[Mapping[str, Any]]],
    ) -> CGMExample:
        graph_obj = self._chunks_to_graph(subgraph_linearized)
        snippet_seq: Sequence[Mapping[str, Any]] = [dict(s) for s in (snippets or []) if isinstance(s, Mapping)]
        return CGMExample(
            prompt=self.instruction,
            response="",
            graph=graph_obj,
            plan=plan,
            issue=dict(issue or {}),
            snippets=snippet_seq,
            metadata={"source": "graph_planner"},
        )

    def _chunks_to_graph(
        self, chunks: Optional[Iterable[Mapping[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        if not chunks:
            return None
        nodes: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                continue
            text = chunk.get("text")
            snippet = chunk.get("snippet") or chunk.get("lines")
            if text is None and isinstance(snippet, Sequence):
                text = "\n".join(str(line) for line in snippet)
            if text is None:
                continue
            path = chunk.get("path") or chunk.get("file") or f"chunk-{idx}"
            node = {
                "id": str(chunk.get("id") or chunk.get("node_id") or f"chunk-{idx}"),
                "name": str(path),
                "text": text,
            }
            summary = chunk.get("summary")
            if isinstance(summary, str) and summary.strip():
                node["summary"] = summary.strip()
            anchors = chunk.get("anchors") or chunk.get("keywords")
            if anchors:
                node["anchors"] = list(anchors)
            start = chunk.get("start", chunk.get("line"))
            end = chunk.get("end", start)
            try:
                if start is not None:
                    node["span"] = {
                        "start": int(start),
                        "end": int(end if end is not None else start),
                    }
            except Exception:
                pass
            nodes.append(node)
        if not nodes:
            return None
        return {"nodes": nodes, "edges": []}

    def _parse_sequence(
        self,
        text: str,
        *,
        max_edits: Optional[int],
        summary: Optional[str],
    ) -> Optional[Patch]:
        payload = extract_json_payload(text)
        if payload is None:
            try:
                payload = json.loads(text)
            except Exception:
                return None
        if not isinstance(payload, Mapping):
            return None

        patch_obj: Optional[Mapping[str, Any]] = None
        candidate = payload.get("patch")
        if isinstance(candidate, Mapping):
            patch_obj = candidate
        elif isinstance(payload.get("edits"), Sequence):
            patch_obj = payload
        if patch_obj is None:
            return None

        edits_raw = patch_obj.get("edits")
        if not isinstance(edits_raw, Sequence):
            return None

        edits: List[PatchEdit] = []
        for entry in edits_raw:
            if not isinstance(entry, Mapping):
                continue
            path = entry.get("path")
            if not path:
                continue
            new_text = entry.get("new_text") or entry.get("text") or entry.get("diff")
            if new_text is None:
                continue
            start = entry.get("start", entry.get("line", 1))
            end = entry.get("end", start)
            try:
                start_i = int(start)
                end_i = int(end if end is not None else start_i)
            except Exception:
                continue
            text_val = str(new_text)
            if not text_val.endswith("\n"):
                text_val += "\n"
            edits.append(
                {
                    "path": str(path),
                    "start": start_i,
                    "end": end_i,
                    "new_text": text_val,
                }
            )
            if max_edits and max_edits > 0 and len(edits) >= max_edits:
                break

        if not edits:
            return None

        summary_text = (
            patch_obj.get("summary")
            or payload.get("summary")
            or summary
            or "cgm-local"
        )
        return {"edits": edits, "summary": str(summary_text)}


def _get_local_runtime() -> Optional[_LocalCGMRuntime]:
    global _LOCAL_RUNTIME_CACHE, _LOCAL_RUNTIME_FINGERPRINT
    cfg = load_config()
    cgm_cfg = getattr(cfg, "cgm", None)
    if not cgm_cfg or not getattr(cgm_cfg, "enabled", False):
        return None
    model_path = getattr(cgm_cfg, "model_path", None)
    if not model_path:
        return None

    fingerprint = (
        model_path,
        getattr(cgm_cfg, "tokenizer_path", None),
        getattr(cgm_cfg, "max_tokens", None),
        getattr(cgm_cfg, "temperature", None),
        getattr(cgm_cfg, "top_p", None),
        getattr(cgm_cfg, "max_input_tokens", None),
        getattr(cgm_cfg, "device", None),
    )
    if _LOCAL_RUNTIME_CACHE and _LOCAL_RUNTIME_FINGERPRINT == fingerprint:
        return _LOCAL_RUNTIME_CACHE

    generation_cfg = CGMGenerationConfig(
        model_name_or_path=model_path,
        tokenizer_name_or_path=getattr(cgm_cfg, "tokenizer_path", None),
        max_length=int(getattr(cgm_cfg, "max_input_tokens", 8192)),
        max_new_tokens=int(getattr(cgm_cfg, "max_tokens", 2048)),
        temperature=float(getattr(cgm_cfg, "temperature", 0.0)),
        top_p=float(getattr(cgm_cfg, "top_p", 0.9)),
        do_sample=float(getattr(cgm_cfg, "temperature", 0.0)) > 0,
        device=getattr(cgm_cfg, "device", None),
    )
    generator = CodeFuseCGMGenerator(generation_cfg)
    _LOCAL_RUNTIME_CACHE = _LocalCGMRuntime(generator=generator)
    _LOCAL_RUNTIME_FINGERPRINT = fingerprint
    return _LOCAL_RUNTIME_CACHE


def _generate_local_patch(
    plan: Plan,
    constraints: Optional[Dict[str, Any]],
    snippets: Optional[List[Dict[str, Any]]],
) -> Patch:
    max_edits = int((constraints or {}).get("max_edits", 3))
    snippet_index = _index_snippets(snippets)
    edits: List[PatchEdit] = []

    for target in plan.targets:
        if len(edits) >= max_edits:
            break
        edit = _build_single_edit_for_target(target, snippet_index)
        if edit:
            edits.append(edit)

    patch: Patch = {"edits": edits, "summary": _summarize_patch(edits)}
    return patch


def generate(
    subgraph_linearized: Optional[List[Dict[str, Any]]],
    plan: Plan,
    constraints: Optional[Dict[str, Any]] = None,
    snippets: Optional[List[Dict[str, Any]]] = None,
    plan_text: Optional[str] = None,
    issue: Optional[Dict[str, Any]] = None,
) -> Patch:
    """生成补丁：优先调用 CodeFuse CGM，失败时回退到本地标记方案。"""

    client = _get_client()
    runtime = _get_local_runtime()
    if client:
        try:
            return client.generate_patch(
                issue=issue,
                plan=plan,
                plan_text=plan_text,
                subgraph_linearized=subgraph_linearized,
                snippets=snippets,
                metadata={"constraints": constraints or {}},
            )
        except Exception as exc:  # pragma: no cover - graceful fallback
            telemetry.log_event(
                {
                    "kind": "cgm",
                    "ok": False,
                    "error": str(exc),
                    "endpoint": getattr(client, "endpoint", ""),
                }
            )

    if runtime:
        try:
            patch = runtime.generate_patch(
                issue=issue,
                plan=plan,
                plan_text=plan_text,
                subgraph_linearized=subgraph_linearized,
                snippets=snippets,
                constraints=constraints,
            )
            return patch
        except Exception as exc:  # pragma: no cover - graceful fallback
            telemetry.log_event(
                {
                    "kind": "cgm-local",
                    "ok": False,
                    "error": str(exc),
                    "model": getattr(runtime.generator.config, "model_name_or_path", ""),
                }
            )

    patch = _generate_local_patch(plan, constraints, snippets)
    if client or runtime:
        base_summary = patch.get("summary") or "local-cgm"
        suffix_parts = []
        if client:
            suffix_parts.append("remote-fallback")
        if runtime:
            suffix_parts.append("local-runtime")
        suffix = "+".join(suffix_parts) if suffix_parts else "fallback"
        patch["summary"] = f"{base_summary} | {suffix}"
    return patch
