"""Contracts and validators for planner and CGM interactions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ...core.actions import (
    ActionUnion,
    ExploreAction,
    MemoryAction,
    NoopAction,
    RepairAction,
    SubmitAction,
)

__all__ = [
    "PlannerContract",
    "CGMContract",
    "ProtocolError",
    "PLANNER_CONTRACT",
    "CGM_CONTRACT",
    "PLANNER_SYSTEM_PROMPT",
    "CGM_SYSTEM_PROMPT",
    "CGM_PATCH_INSTRUCTION",
    "parse_action_block",
    "validate_planner_action",
    "validate_cgm_patch",
    "normalize_newlines",
]


class PlannerErrorCode(str, Enum):
    INVALID_MULTI_BLOCK = "invalid-multi-block"
    MISSING_FUNCTION_TAG = "missing-function-tag"
    UNKNOWN_ACTION = "unknown-action"
    DUPLICATE_PARAM = "duplicate-param"
    UNKNOWN_PARAM = "unknown-param"
    EXTRA_TEXT = "extra-text"
    INVALID_JSON_PARAM = "invalid-json-param"
    MISSING_REQUIRED_PARAM = "missing-required-param"


class CGMPatchErrorCode(str, Enum):
    INVALID_PATCH_SCHEMA = "invalid-patch-schema"
    MULTI_FILE_DIFF = "multi-file-diff"
    NEWLINE_MISSING = "newline-missing"
    RANGE_INVALID = "range-invalid"
    PATH_MISSING = "path-missing"
    INVALID_UNIFIED_DIFF = "invalid-unified-diff"
    HUNK_MISMATCH = "hunk-mismatch"
    ENCODING_UNSUPPORTED = "encoding-unsupported"
    DIRTY_WORKSPACE = "dirty-workspace"
    DUPLICATE_PATCH = "duplicate-patch"


class ProtocolError(ValueError):
    """Exception raised when planner or CGM output violates the contract."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class PlannerContract:
    """Single source of truth for planner prompts and schema."""

    SYSTEM_PROMPT: str
    ACTIONS: Tuple[str, ...]
    allowed_params: Mapping[str, Set[str]]
    required_params: Mapping[str, Set[str]] = field(default_factory=dict)
    errors: Tuple[str, ...] = (
        PlannerErrorCode.INVALID_MULTI_BLOCK.value,
        PlannerErrorCode.MISSING_FUNCTION_TAG.value,
        PlannerErrorCode.UNKNOWN_ACTION.value,
        PlannerErrorCode.DUPLICATE_PARAM.value,
        PlannerErrorCode.UNKNOWN_PARAM.value,
        PlannerErrorCode.EXTRA_TEXT.value,
        PlannerErrorCode.INVALID_JSON_PARAM.value,
        PlannerErrorCode.MISSING_REQUIRED_PARAM.value,
    )

    def normalise_action(self, name: str) -> str:
        action = (name or "").strip().lower()
        if action not in self.ACTIONS:
            raise ProtocolError(
                PlannerErrorCode.UNKNOWN_ACTION.value,
                f"action '{name}' is not supported; expected one of {sorted(self.ACTIONS)}",
            )
        return action


@dataclass(frozen=True)
class CGMContract:
    """Single source of truth for CGM prompts and patch schema."""

    SYSTEM_PROMPT: str
    schema: Mapping[str, Any]
    constraints: Mapping[str, Any]


@dataclass(frozen=True)
class CGMPatch:
    """Normalised CGM patch guaranteed to touch exactly one file."""

    path: str
    edits: List[Dict[str, Any]]
    summary: Optional[str] = None


PLANNER_SYSTEM_PROMPT = (
    "You are the Graph Planner decision model.\n"
    "Every reply MUST contain exactly one text-trajectory block in the format:\n\n"
    "<function=ACTION_NAME>\n"
    "  <param name=\"thought\"><![CDATA[free-form reasoning]]></param>\n"
    "  <param name=\"k\">JSON or text values</param>\n"
    "</function>\n\n"
    "Replace ACTION_NAME with one of: explore, memory, repair, submit, noop.\n"
    "Do not emit any other text outside the block.\n\n"
    "For each action:\n"
    "- explore: params may include op (find|expand|read), anchors (list), nodes (list), hop (int), limit (int).\n"
    "- memory: provide target (explore|observation), scope (turn|session), intent (commit|delete) and optional selector (\"latest\" or specific id).\n"
    "- repair: set subplan (multi-line steps, wrap in CDATA), optional focus_ids (list of graph node ids) and apply (true/false).\n"
    "- submit: no extra params besides thought.\n"
    "- noop: empty operation when no action fits.\n\n"
    "Encode lists/dicts as JSON. Use CDATA for multi-line text."
)


PLANNER_CONTRACT = PlannerContract(
    SYSTEM_PROMPT=PLANNER_SYSTEM_PROMPT,
    ACTIONS=("explore", "memory", "repair", "submit", "noop"),
    allowed_params={
        "explore": {"thought", "op", "anchors", "nodes", "hop", "limit"},
        "memory": {"thought", "target", "scope", "intent", "selector"},
        "repair": {"thought", "subplan", "focus_ids", "apply"},
        "submit": {"thought"},
        "noop": {"thought"},
    },
    required_params={
        "repair": {"subplan"},
        "memory": {"target", "intent"},
        "explore": {"anchors"},
    },
)


CGM_SYSTEM_PROMPT = (
    "You are CodeFuse-CGM, a graph-aware assistant that generates precise code patches. "
    "Use the issue description, planner plan, graph context and snippets to derive the necessary edits. "
    "Reply with a JSON object containing a top-level \"patch\" field. The patch must include an \"edits\" array listing objects with \"path\", \"start\", \"end\" and \"new_text\" fields. Ensure new_text entries end with a newline."
)

CGM_PATCH_INSTRUCTION = CGM_SYSTEM_PROMPT

CGM_CONTRACT = CGMContract(
    SYSTEM_PROMPT=CGM_SYSTEM_PROMPT,
    schema={
        "patch": {
            "type": "object",
            "required": ["edits"],
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["path", "start", "end", "new_text"],
                    },
                    "minItems": 1,
                }
            },
        },
        "summary": {"type": "string"},
    },
    constraints={"one_file_per_patch": True, "newline_required": True},
)


_BLOCK_RE = re.compile(r"<function\s*=\s*([a-zA-Z0-9_.-]+)\s*>", re.IGNORECASE)
_END_RE = re.compile(r"</function>", re.IGNORECASE)
_PARAM_RE = re.compile(r"<param\s+name=\"([^\"]+)\">(.*?)</param>", re.DOTALL | re.IGNORECASE)
_CDATA_START = "<![CDATA["


def _looks_like_json(text: str) -> bool:
    if not text:
        return False
    first = text[0]
    if first in '{["' or first in '-0123456789':
        return True
    lowered = text.lower()
    return lowered in {"true", "false", "null"}


def normalize_newlines(text: str) -> str:
    """Return text with CRLF/CR normalised to LF."""

    if not isinstance(text, str):
        return str(text)
    return text.replace('\r\n', '\n').replace('\r', '\n')

def _normalise_json_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if not _looks_like_json(text):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            PlannerErrorCode.INVALID_JSON_PARAM.value,
            f"unable to parse JSON value: {exc}"
        ) from exc


def parse_action_block(text: str) -> Dict[str, Any]:
    """Parse a planner action block and enforce structural guarantees."""

    if not isinstance(text, str):
        raise ProtocolError(PlannerErrorCode.MISSING_FUNCTION_TAG.value, "planner response must be a string")

    matches = list(_BLOCK_RE.finditer(text))
    if not matches:
        raise ProtocolError(PlannerErrorCode.MISSING_FUNCTION_TAG.value, "response does not contain <function=...>")
    if len(matches) > 1:
        raise ProtocolError(PlannerErrorCode.INVALID_MULTI_BLOCK.value, "response must contain exactly one function block")

    match = matches[0]
    prefix = text[: match.start()]
    if prefix.strip():
        raise ProtocolError(PlannerErrorCode.EXTRA_TEXT.value, "unexpected text before function block")

    end_match = _END_RE.search(text, match.end())
    if not end_match:
        raise ProtocolError(PlannerErrorCode.MISSING_FUNCTION_TAG.value, "missing </function> terminator")

    suffix = text[end_match.end() :]
    if suffix.strip():
        raise ProtocolError(PlannerErrorCode.EXTRA_TEXT.value, "unexpected text after function block")

    action_name = PLANNER_CONTRACT.normalise_action(match.group(1))

    inner = text[match.end() : end_match.start()]
    params: Dict[str, Any] = {}
    last_end = 0
    for param_match in _PARAM_RE.finditer(inner):
        start, end = param_match.span()
        if inner[last_end:start].strip():
            raise ProtocolError(PlannerErrorCode.EXTRA_TEXT.value, "unexpected text between <param> elements")
        key = param_match.group(1).strip()
        allowed = PLANNER_CONTRACT.allowed_params.get(action_name, set())
        if key not in allowed:
            raise ProtocolError(PlannerErrorCode.UNKNOWN_PARAM.value, f"parameter '{key}' is not allowed for action '{action_name}'")
        if key in params:
            raise ProtocolError(PlannerErrorCode.DUPLICATE_PARAM.value, f"duplicate parameter '{key}'")
        raw_value = param_match.group(2).strip()
        if raw_value.startswith(_CDATA_START) and raw_value.endswith("]]>"):
            value = raw_value[len(_CDATA_START) : -3]
        else:
            try:
                value = _normalise_json_value(raw_value)
            except ProtocolError as exc:
                if exc.code == PlannerErrorCode.INVALID_JSON_PARAM.value:
                    raise ProtocolError(exc.code, f"parameter '{key}' {exc.detail}") from exc
                raise
        params[key] = value
        last_end = end
    if inner[last_end:].strip():
        raise ProtocolError(PlannerErrorCode.EXTRA_TEXT.value, "unexpected trailing text inside function block")

    return {"name": action_name, "params": params}


def _coerce_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _ensure_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    if isinstance(value, (str, int, float)):
        text = str(value)
        return [text] if text else []
    return []


def _ensure_dict_list(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _attach_meta(action: ActionUnion, meta: Dict[str, Any]) -> ActionUnion:
    object.__setattr__(action, "_meta", dict(meta))
    return action


def validate_planner_action(result: Mapping[str, Any]) -> ActionUnion:
    """Validate planner parameters and convert to a typed action."""

    action_name = PLANNER_CONTRACT.normalise_action(result.get("name"))
    params = dict(result.get("params") or {})
    meta: Dict[str, Any] = {}

    required = PLANNER_CONTRACT.required_params.get(action_name, set())
    missing = [key for key in required if key not in params]
    if missing:
        raise ProtocolError(
            PlannerErrorCode.MISSING_REQUIRED_PARAM.value,
            f"action '{action_name}' missing required params: {', '.join(sorted(missing))}",
        )

    if action_name == "explore":
        op = str(params.get("op", "expand")).lower()
        anchors = _ensure_dict_list(params.get("anchors"))
        if not anchors:
            raise ProtocolError(
                PlannerErrorCode.MISSING_REQUIRED_PARAM.value,
                "explore action requires at least one anchor",
            )
        nodes = _ensure_str_list(params.get("nodes"))
        try:
            hop_raw = int(params.get("hop", 1))
        except Exception:
            hop_raw = 1
        try:
            limit_raw = int(params.get("limit", 50))
        except Exception:
            limit_raw = 50

        hop = max(0, min(2, hop_raw))
        limit = max(1, min(100, limit_raw))
        capped_fields: Dict[str, Any] = {}
        if hop != hop_raw:
            capped_fields["hop"] = hop_raw
        if limit != limit_raw:
            capped_fields["limit"] = limit_raw
        if capped_fields:
            meta.setdefault("warnings", []).append("value-capped")
            meta["capped"] = True
            meta["capped_fields"] = capped_fields

        return _attach_meta(ExploreAction(op=op, anchors=anchors, nodes=nodes, hop=hop, limit=limit), meta)

    if action_name == "memory":
        target = str(params.get("target", "explore"))
        scope = str(params.get("scope", "turn"))
        intent = str(params.get("intent", "commit"))
        selector = params.get("selector")
        if isinstance(selector, (dict, list)):
            selector = json.dumps(selector, ensure_ascii=False)
        return _attach_meta(MemoryAction(target=target, scope=scope, intent=intent, selector=selector), meta)

    if action_name == "repair":
        subplan = params.get("subplan")
        if not isinstance(subplan, str) or not subplan.strip():
            raise ProtocolError(PlannerErrorCode.MISSING_REQUIRED_PARAM.value, "repair action requires non-empty subplan")
        focus_ids = _ensure_str_list(params.get("focus_ids"))
        apply_flag = _coerce_bool(params.get("apply"), default=True)
        plan_targets = [
            {"id": fid, "why": "planner-focus"}
            for fid in focus_ids
            if fid
        ]
        return _attach_meta(
            RepairAction(apply=apply_flag, issue={}, plan=subplan.strip(), plan_targets=plan_targets, patch=None),
            meta,
        )

    if action_name == "submit":
        return _attach_meta(SubmitAction(), meta)

    if action_name == "noop":
        return _attach_meta(NoopAction(), meta)

    raise ProtocolError(PlannerErrorCode.UNKNOWN_ACTION.value, f"unsupported action '{action_name}'")


def validate_cgm_patch(obj: Mapping[str, Any]) -> CGMPatch:
    """Validate CGM patch structure and ensure a single target file."""

    if not isinstance(obj, Mapping):
        raise ProtocolError(CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value, "CGM output must be a mapping")
    patch = obj.get("patch")
    if not isinstance(patch, Mapping):
        raise ProtocolError(CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value, "'patch' field missing or not an object")
    edits_raw = patch.get("edits")
    if not isinstance(edits_raw, Sequence) or not edits_raw:
        raise ProtocolError(CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value, "'patch.edits' must be a non-empty list")

    edits: List[Dict[str, Any]] = []
    paths: Set[str] = set()
    for idx, item in enumerate(edits_raw):
        if not isinstance(item, Mapping):
            raise ProtocolError(CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value, f"edit #{idx} is not an object")
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ProtocolError(CGMPatchErrorCode.PATH_MISSING.value, f"edit #{idx} missing file path")
        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except Exception as exc:
            raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, f"edit #{idx} has non-integer range") from exc
        if start <= 0 or end < start:
            raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, f"edit #{idx} has invalid span {start}->{end}")
        new_text = item.get("new_text")
        if not isinstance(new_text, str):
            raise ProtocolError(CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value, f"edit #{idx} missing new_text string")
        if CGM_CONTRACT.constraints.get("newline_required") and not new_text.endswith("\n"):
            raise ProtocolError(CGMPatchErrorCode.NEWLINE_MISSING.value, f"edit #{idx} new_text must end with newline")
        normalized = {
            "path": path,
            "start": start,
            "end": end,
            "new_text": new_text,
        }
        edits.append(normalized)
        paths.add(path)

    if len(paths) != 1 and CGM_CONTRACT.constraints.get("one_file_per_patch"):
        raise ProtocolError(CGMPatchErrorCode.MULTI_FILE_DIFF.value, "patch must touch exactly one file")

    path = next(iter(paths))
    summary = obj.get("summary")
    if summary is not None and not isinstance(summary, str):
        summary = str(summary)
    return CGMPatch(path=path, edits=edits, summary=summary)
