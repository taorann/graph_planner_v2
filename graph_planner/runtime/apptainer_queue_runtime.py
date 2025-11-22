from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Mapping

from .queue_protocol import (
    QueueRequest,
    QueueResponse,
    ExecResult,
    runner_inbox,
    runner_outbox,
)


class ApptainerQueueRuntime:
    def __init__(
        self,
        queue_root: Path,
        sif_dir: Path,
        num_runners: int,
        *,
        default_timeout_sec: float = 900.0,
        poll_interval_sec: float = 0.5,
        max_stdout_bytes: int = 512_000,
    ) -> None:
        self.queue_root = Path(queue_root)
        self.sif_dir = Path(sif_dir)
        self.num_runners = int(num_runners)
        self.default_timeout_sec = float(default_timeout_sec)
        self.poll_interval_sec = float(poll_interval_sec)
        self.max_stdout_bytes = int(max_stdout_bytes)
        self._run_to_runner: Dict[str, int] = {}

    def exec(
        self,
        *,
        run_id: str,
        docker_image: str,
        cmd: List[str],
        cwd: Path,
        env: Optional[Mapping[str, str]] = None,
        timeout_sec: Optional[float] = None,
        meta: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        runner_id = self._choose_runner(run_id)
        sif_path = self._image_to_sif(docker_image)
        req = QueueRequest(
            req_id=self._new_req_id(),
            runner_id=runner_id,
            run_id=run_id,
            type="exec",
            image=docker_image,
            sif_path=str(sif_path),
            cmd=list(cmd),
            cwd=str(cwd),
            env=dict(env or {}),
            timeout_sec=float(timeout_sec or self.default_timeout_sec),
            meta=meta or {},
        )
        resp = self._roundtrip(req)
        return self._resp_to_exec_result(resp)

    def put_file(self, *, run_id: str, src: Path, dst: Path, meta: Optional[Dict[str, str]] = None) -> None:
        runner_id = self._choose_runner(run_id)
        req = QueueRequest(
            req_id=self._new_req_id(),
            runner_id=runner_id,
            run_id=run_id,
            type="put",
            src=str(src),
            dst=str(dst),
            meta=meta or {},
        )
        self._roundtrip(req)

    def get_file(self, *, run_id: str, src: Path, dst: Path, meta: Optional[Dict[str, str]] = None) -> None:
        runner_id = self._choose_runner(run_id)
        req = QueueRequest(
            req_id=self._new_req_id(),
            runner_id=runner_id,
            run_id=run_id,
            type="get",
            src=str(src),
            dst=str(dst),
            meta=meta or {},
        )
        self._roundtrip(req)

    def cleanup_run(self, run_id: str) -> None:
        runner_id = self._choose_runner(run_id)
        req = QueueRequest(
            req_id=self._new_req_id(),
            runner_id=runner_id,
            run_id=run_id,
            type="cleanup",
        )
        self._roundtrip(req)

    def _choose_runner(self, run_id: str) -> int:
        rid = self._run_to_runner.get(run_id)
        if rid is None:
            rid = hash(run_id) % max(self.num_runners, 1)
            self._run_to_runner[run_id] = rid
        return rid

    def _new_req_id(self) -> str:
        return uuid.uuid4().hex

    def _image_to_sif(self, docker_image: str) -> Path:
        normalized = (
            docker_image.replace("/", "-")
            .replace(":", "-")
            .replace("@", "-")
        )
        return self.sif_dir / f"{normalized}.sif"

    def _roundtrip(self, req: QueueRequest) -> QueueResponse:
        root = self.queue_root
        inbox = runner_inbox(root, req.runner_id)
        outbox = runner_outbox(root, req.runner_id)
        inbox.mkdir(parents=True, exist_ok=True)
        outbox.mkdir(parents=True, exist_ok=True)

        req_path = inbox / f"{req.req_id}.json"
        with req_path.open("w", encoding="utf-8") as f:
            json.dump(req.__dict__, f, ensure_ascii=False)

        resp_path = outbox / f"{req.req_id}.json"
        deadline = time.time() + (req.timeout_sec or self.default_timeout_sec) * 1.5
        while True:
            if resp_path.exists():
                with resp_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return QueueResponse(**data)
            if time.time() > deadline:
                raise TimeoutError(f"Timeout waiting for response for {req.req_id}")
            time.sleep(self.poll_interval_sec)

    def _resp_to_exec_result(self, resp: QueueResponse) -> ExecResult:
        stdout = (resp.stdout or "")[: self.max_stdout_bytes]
        stderr = (resp.stderr or "")[: self.max_stdout_bytes]
        if not resp.ok:
            return ExecResult(
                returncode=resp.returncode if resp.returncode is not None else -1,
                stdout=stdout,
                stderr=stderr,
                runtime_sec=resp.runtime_sec or 0.0,
                ok=False,
                error=resp.error or "ApptainerQueueRuntime request failed",
            )
        return ExecResult(
            returncode=resp.returncode or 0,
            stdout=stdout,
            stderr=stderr,
            runtime_sec=resp.runtime_sec or 0.0,
            ok=True,
            error=None,
        )

