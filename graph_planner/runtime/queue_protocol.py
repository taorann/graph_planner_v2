from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Literal

RequestType = Literal["exec", "put", "get", "cleanup", "noop"]


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    runtime_sec: float
    ok: bool
    error: Optional[str] = None


@dataclass
class QueueRequest:
    req_id: str
    runner_id: int
    run_id: str
    type: RequestType
    image: Optional[str] = None
    sif_path: Optional[str] = None
    cmd: Optional[List[str]] = None
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    timeout_sec: Optional[float] = None
    src: Optional[str] = None
    dst: Optional[str] = None
    meta: Optional[Dict[str, str]] = None


@dataclass
class QueueResponse:
    req_id: str
    runner_id: int
    run_id: str
    type: RequestType
    ok: bool
    returncode: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    runtime_sec: Optional[float] = None
    error: Optional[str] = None
    meta: Optional[Dict[str, str]] = None


INBOX_NAME = "in"
OUTBOX_NAME = "out"


def runner_inbox(root: Path, rid: int) -> Path:
    return root / f"runner-{rid}" / INBOX_NAME


def runner_outbox(root: Path, rid: int) -> Path:
    return root / f"runner-{rid}" / OUTBOX_NAME

