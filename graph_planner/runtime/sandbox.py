# graph_planner/runtime/sandbox.py
import os, json, random, string, time, shutil, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import docker

from ..agents.common.contracts import CGM_CONTRACT, CGMPatchErrorCode, ProtocolError, normalize_newlines

# 遥测
from ..infra import telemetry as telemetry_mod

# R2E 组件（可选）
try:
    from r2egym.agenthub.runtime.docker import DockerRuntime as R2EDockerRuntime
    from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
    _HAS_R2E = True
except Exception:
    R2EDockerRuntime = None
    EnvArgs = None
    RepoEnv = None
    _HAS_R2E = False

def _rand_name(prefix="gp"):
    import string as _s, random as _r
    return f"{prefix}-" + "".join(_r.choices(_s.ascii_lowercase + _s.digits, k=8))

def _dbg(msg: str):
    if os.environ.get("DEBUG"):
        print(f"[sandbox] {msg}")

@dataclass
class SandboxConfig:
    docker_image: str
    workdir: str
    mounts: Dict[str, str]
    env: Dict[str, str]
    pytest_cache_root: Optional[str] = None
    commit_hash: Optional[str] = None
    # 统一后端切换：
    backend: str = "auto"            # "auto" | "r2e" | "repoenv" | "docker"
    r2e_ds_json: Optional[str] = None  # 指向一个 JSON 文件，内容是 r2e 期望的 ds dict
    requires_build: bool = False       # SWE-bench 元数据：提示容器是否需要 build
    swebench_spec: Optional[Dict[str, Any]] = None  # 透传 swe-bench 的构建脚本信息
    force_docker_backend: bool = False
    port_forwards: Optional[List[Dict[str, Any]]] = None

class SandboxRuntime:
    """
    统一接口：
      run / apply_patch / get_patch / lint / test / reset_soft / close
    后端：
      - "repoenv"  : RepoEnv(EnvArgs(ds)) → 官方评测最友好
      - "r2e"      : R2E DockerRuntime(ds)（我们自己掌控容器，但仍用 R2E 底座）
      - "docker"   : 纯 docker-py（最自由）
      - "auto"     : 有 ds 用 "repoenv"，否则 "docker"
    """
    def __init__(self, cfg: SandboxConfig, force_backend: Optional[str] = None):
        self.cfg = cfg
        preferred_mode = force_backend or cfg.backend or "docker"
        if cfg.force_docker_backend:
            preferred_mode = "docker"
        if preferred_mode == "auto":
            preferred_mode = (
                "repoenv" if (_HAS_R2E and cfg.r2e_ds_json and os.path.exists(cfg.r2e_ds_json)) else "docker"
            )
        self._mode = preferred_mode

        self._env = None  # only populated when using RepoEnv as the backend

        if self._mode == "repoenv":
            try:
                self._init_repoenv_backend()
            except Exception as e:
                _dbg(f"repoenv init failed: {e!r}; falling back to docker")
                self._mode = "docker"
                self._init_docker_backend()
        elif self._mode == "r2e":
            self._init_r2e_backend()
        else:
            self._init_docker_backend()
        self._exposed_ports: List[Dict[str, Any]] = getattr(self, "_exposed_ports", [])

    # ---------- backend: RepoEnv ----------
    def _init_repoenv_backend(self):
        if not _HAS_R2E:
            raise RuntimeError("r2egym is not available but backend='repoenv' was requested.")
        ds_path = self.cfg.r2e_ds_json
        if ds_path:
            ds_path = os.path.expanduser(ds_path)
            if not os.path.isabs(ds_path):
                ds_path = os.path.abspath(ds_path)
        if not (ds_path and os.path.exists(ds_path)):
            raise ValueError(f"r2e ds json not found: {self.cfg.r2e_ds_json}")

        with open(ds_path, "r") as f:
            ds = json.load(f)

        env_args = EnvArgs(ds=ds)
        env = RepoEnv(env_args)
        self._env = env
        self._rt = env.runtime  # r2e 的 DockerRuntime
        _dbg("repoenv initialized")

        # --- 关键保底：先用根目录作为 workdir 创建 repo_path，避免 chdir 失败 ---
        repo_path = getattr(self._rt, "repo_path", "/testbed")
        try:
            # 直接用 docker-py 在容器 root workdir 执行 mkdir，绕开 /testbed 不存在的问题
            self._rt.container.exec_run("bash -lc 'mkdir -p {}'".format(repo_path), workdir="/")
        except Exception:
            pass

        # 基本工具 + git 安全目录（现在 chdir 到 repo_path 已不会报 126）
        self._rt.run("python -m pip -q install --upgrade pip >/dev/null 2>&1 || true", timeout=180)
        self._rt.run("python -m pip -q install pytest >/dev/null 2>&1 || true", timeout=300)
        self._rt.run(f"git config --global --add safe.directory {repo_path} || true", timeout=30)

        self.repo = None

    # ---------- backend: R2E DockerRuntime（仍保留，训练期灵活） ----------
    def _init_r2e_backend(self):
        if not _HAS_R2E:
            raise RuntimeError("r2egym is not available but backend='r2e' was requested.")
        ds_path = self.cfg.r2e_ds_json
        if ds_path:
            ds_path = os.path.expanduser(ds_path)
            if not os.path.isabs(ds_path):
                ds_path = os.path.abspath(ds_path)
        if not (ds_path and os.path.exists(ds_path)):
            raise ValueError(f"r2e ds json not found: {self.cfg.r2e_ds_json}")

        with open(ds_path, "r") as f:
            ds = json.load(f)

        # 宿主挂载（只为把你的代码带进容器；真正工作目录在 /work）
        volumes = {}
        for host, container in (self.cfg.mounts or {}).items():
            if not os.path.isabs(host):
                raise ValueError(f"HOST mount path must be absolute: {host}")
            volumes[os.path.abspath(host)] = {"bind": container, "mode": "rw"}

        repo_src = next(iter((self.cfg.mounts or {}).values()), "/testbed")
        repo_path = "/work"

        self._rt = R2EDockerRuntime(
            ds=ds,
            repo_path=repo_path,
            command="/bin/bash",
            working_dir=repo_path,
            volumes=volumes,
            environment=self.cfg.env or {},
        )
        # 拷贝到 /work，避开 root_squash
        self._rt.run("mkdir -p /root/.local/bin /work", timeout=60)
        self._rt.run(f"rsync -a --delete {repo_src}/ {repo_path}/ || cp -a {repo_src}/. {repo_path}/", timeout=600)
        self._rt.run(f"git config --global --add safe.directory {repo_path} || true", timeout=30)
        self._rt.run("python -m pip -q install pytest >/dev/null 2>&1 || true", timeout=300)
        self.repo = None

    # ---------- backend: docker-py（自管容器） ----------
    def _init_docker_backend(self):
        self.client = docker.from_env(timeout=120)
        volumes = {}
        for host, container in (self.cfg.mounts or {}).items():
            if not os.path.isabs(host):
                raise ValueError(f"HOST mount path must be absolute: {host}")
            volumes[os.path.abspath(host)] = {"bind": container, "mode": "rw"}
        if self.cfg.workdir:
            workdir = self.cfg.workdir
        elif "/testbed" in (self.cfg.mounts or {}).values():
            workdir = "/testbed"
        else:
            workdir = next(iter(self.cfg.mounts.values()), "/") if self.cfg.mounts else "/"
        self.workdir = workdir

        ports_arg = None
        if self.cfg.port_forwards:
            ports_arg = {}
            for spec in self.cfg.port_forwards:
                if not isinstance(spec, Mapping):
                    continue
                container_port = spec.get("container_port")
                if container_port is None:
                    continue
                try:
                    container_port = int(container_port)
                except Exception:
                    continue
                protocol = str(spec.get("protocol", "tcp")).lower()
                host_ip = spec.get("host_ip")
                host_port = spec.get("host_port")
                binding: Any
                if host_port is None or host_port == "":
                    binding = None
                else:
                    try:
                        host_port = int(host_port)
                    except Exception:
                        continue
                    if host_ip:
                        binding = (str(host_ip), host_port)
                    else:
                        binding = host_port
                key = f"{container_port}/{protocol}" if protocol else int(container_port)
                ports_arg[key] = binding
        self.container = self.client.containers.run(
            image=self.cfg.docker_image,
            command="/bin/bash",
            name=_rand_name("gp"),
            environment=self.cfg.env or {},
            working_dir=self.workdir,
            tty=True,
            stdin_open=True,
            detach=True,
            volumes=volumes,
            ports=ports_arg,
        )
        # git 安全目录兜底
        self._exec(f"git config --global --add safe.directory {self.workdir} || true")
        self.repo = None
        self._exposed_ports = []
        if ports_arg:
            try:
                self.container.reload()
                ports_info = (
                    self.container.attrs.get("NetworkSettings", {}).get("Ports", {})
                )
            except Exception:
                ports_info = {}
            for key, bindings in (ports_info or {}).items():
                if not bindings:
                    # None 表示 Docker 仍然会随机分配主机端口
                    self._exposed_ports.append(
                        {
                            "container_port": key,
                            "host_ip": None,
                            "host_port": None,
                        }
                    )
                    continue
                for binding in bindings:
                    self._exposed_ports.append(
                        {
                            "container_port": key,
                            "host_ip": binding.get("HostIp"),
                            "host_port": int(binding.get("HostPort"))
                            if binding.get("HostPort")
                            else None,
                        }
                    )

    # ---------- 通用执行 ----------
    def _exec(self, cmd: str, timeout: int = 900) -> Tuple[str, int]:
        if self._mode in ("r2e", "repoenv"):
            out, rc = self._rt.run(cmd, timeout=timeout)
            try:
                rc_int = int(rc)
            except (TypeError, ValueError):
                rc_int = 0 if str(rc).strip() == "" else 1
            return out, rc_int
        # docker-py
        q = "'" + cmd.replace("'", "'\"'\"'") + "'"
        exec_cmd = f"bash -lc {q}"
        res = self.container.exec_run(exec_cmd, demux=True)
        rc = res.exit_code if hasattr(res, "exit_code") else res[0]
        out, err = res.output if hasattr(res, "output") else res[1]
        out = (out or b"") + (err or b"")
        return out.decode("utf-8", errors="ignore"), rc

    # ---------- ACI 接口 ----------
    def run(self, cmd: str, timeout: int = 900) -> Tuple[str, int]:
        return self._exec(cmd, timeout)

    def apply_patch(self, unified_diff: str) -> bool:
        if self._mode in ("r2e", "repoenv"):
            return self._rt.apply_patch(unified_diff)
        # docker-py 路径
        heredoc = f"cat >/tmp/graph_planner.patch <<'EOF'\n{unified_diff}\nEOF"
        _, rc1 = self._exec(heredoc)
        if rc1 != 0: return False
        _, rc2 = self._exec("git apply --reject --whitespace=fix /tmp/graph_planner.patch")
        return rc2 == 0

    def get_patch(self) -> str:
        if self._mode in ("r2e", "repoenv"):
            return self._rt.get_patch()
        out, _ = self._exec("git diff")
        return out

    def lint(self) -> bool:
        _, rc = self._exec(
            "ruff --version >/dev/null 2>&1 || true; "
            "black --version >/dev/null 2>&1 || true; "
            "ruff check . || true; black --check . || true"
        )
        return rc == 0

    def test(self, selector: Optional[List[str]] = None, timeout: int = 1800) -> Dict:
        selector_tuple: Tuple[str, ...] = tuple(selector or ())
        sel = " ".join(selector_tuple)
        # RepoEnv / R2E：尝试官方脚本 → 失败再回退 pytest
        if self._mode in ("repoenv", "r2e"):
            # 探测常见官方入口
            probes = [
                "test -x /testbed/run_tests.sh",
                "test -x /work/run_tests.sh",
                "test -d /r2e_tests",
            ]
            if any(self._exec(p)[1] == 0 for p in probes):
                # 先尝试 /testbed 下的脚本；没有就 /work；再没有就 /r2e_tests
                for candidate in ("/testbed/run_tests.sh", "/work/run_tests.sh"):
                    cmd = f"bash {candidate} --json /tmp/_r2e_eval.json"
                    start = time.time()
                    out, rc = self._exec(cmd, timeout=timeout)
                    duration = time.time() - start
                    if rc == 0 or "No such file" not in out:
                        dump, _ = self._exec("cat /tmp/_r2e_eval.json || true")
                        passed = False
                        try:
                            data = json.loads(dump) if dump.strip() else {}
                            passed = bool(data.get("passed", False))
                        except Exception:
                            # 退化关键词匹配
                            passed = ("PASSED" in out) and ("FAILED" not in out)
                        result = {"mode": "r2e", "passed": passed, "rc": 0 if passed else 1, "stdout": out}
                        return self._finalize_test_result(
                            result,
                            command=cmd,
                            selector=selector_tuple,
                            duration=duration,
                        )
                # r2e_tests 目录的自定义入口（按需定制）
                cmd = "bash /r2e_tests/run.sh || true"
                start = time.time()
                out, rc = self._exec(cmd, timeout=timeout)
                duration = time.time() - start
                passed = ("PASSED" in out) and ("FAILED" not in out)
                result = {"mode": "r2e", "passed": passed, "rc": 0 if passed else 1, "stdout": out}
                return self._finalize_test_result(
                    result,
                    command=cmd,
                    selector=selector_tuple,
                    duration=duration,
                )

        # 回退 pytest（禁用 --cache-dir，统一用 python -m pytest）
        cmd = f"python -m pytest -q {sel}".strip()
        _dbg(f"pytest cmd: {cmd}")
        start = time.time()
        out, rc = self._exec(cmd, timeout=timeout)
        duration = time.time() - start
        result = {"mode": "pytest", "passed": rc == 0, "rc": rc, "stdout": out}
        return self._finalize_test_result(
            result,
            command=cmd,
            selector=selector_tuple,
            duration=duration,
        )

    def reset_soft(self) -> None:
        if self._mode in ("r2e", "repoenv"):
            self._rt.soft_git_reset()
        else:
            self._exec("git reset --hard HEAD && git clean -fd")

    def _finalize_test_result(
        self,
        result: Dict[str, Any],
        *,
        command: str,
        selector: Tuple[str, ...],
        duration: float,
    ) -> Dict[str, Any]:
        payload = {
            "kind": "test_run",
            "backend": self._mode,
            "command": command,
            "selector": list(selector),
            "duration_sec": round(duration, 3),
            "result": {
                "mode": result.get("mode"),
                "rc": result.get("rc"),
                "passed": bool(result.get("passed")),
            },
            "stdout": result.get("stdout", ""),
        }
        if self.cfg.workdir:
            payload["workdir"] = self.cfg.workdir
        if self._mode in ("repoenv", "r2e") and self.cfg.r2e_ds_json:
            payload["dataset_json"] = self.cfg.r2e_ds_json
        try:
            telemetry_mod.log_test_result(payload)
        except Exception:
            pass
        return result

    def close(self):
        if self._mode in ("r2e", "repoenv"):
            try:
                close = getattr(self._rt, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            if self._env is not None:
                try:
                    self._env.runtime = None
                except Exception:
                    pass
                self._env = None
            self._rt = None
        else:
            try: self.container.stop(timeout=5)
            except Exception: pass
            try: self.container.remove(force=True)
            except Exception: pass

    # ---------- 调试辅助 ----------
    @property
    def exposed_ports(self) -> List[Dict[str, Any]]:
        return list(self._exposed_ports)


class PatchApplier:
    """Apply unified diffs in a temporary workspace before committing changes."""

    def __init__(self) -> None:
        self._applied: Dict[str, set[str]] = {}

    def apply_in_temp_then_commit(
        self,
        repo_root: Path,
        patch_text: str,
        path: str,
        run_tests: Callable[[Path], Mapping[str, Any]],
        run_lint: Optional[Callable[[Path], Mapping[str, Any]]] = None,
        patch_id: Optional[str] = None,
        *,
        new_content: Optional[str] = None,
        stats: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        """Validate, trial, and commit a patch atomically.

        Parameters
        ----------
        repo_root:
            Root directory of the working repository on the host filesystem.
        patch_text:
            Unified diff text generated from the CGM candidate.
        path:
            Relative file path within ``repo_root`` touched by the patch.
        run_tests / run_lint:
            Callbacks executed inside the temporary copy. They must accept a
            ``Path`` argument pointing to the trial workspace and return a
            mapping containing status flags (``passed``/``ok``) and optional
            logs. ``run_lint`` is optional.
        patch_id:
            Optional deterministic identifier used to detect duplicate
            applications.
        new_content:
            Optional fully materialised file contents. When omitted the diff is
            applied to the current workspace to derive the new text.
        stats:
            Optional telemetry dictionary with ``n_hunks``/``added_lines``/
            ``removed_lines`` counters.
        """

        repo_root = Path(repo_root)
        if not repo_root.exists():
            raise ProtocolError(CGMPatchErrorCode.PATH_MISSING.value, f"repo root '{repo_root}' does not exist")

        normalized_path = path.strip()
        if not normalized_path:
            raise ProtocolError(CGMPatchErrorCode.PATH_MISSING.value, "patch path is empty")

        if patch_id:
            applied = self._applied.setdefault(normalized_path, set())
            if patch_id in applied:
                raise ProtocolError(CGMPatchErrorCode.DUPLICATE_PATCH.value, f"patch {patch_id} already applied to {normalized_path}")

        source_file = repo_root.joinpath(normalized_path)
        if not source_file.exists():
            raise ProtocolError(CGMPatchErrorCode.PATH_MISSING.value, f"target file '{normalized_path}' not found")

        try:
            original_text = normalize_newlines(source_file.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            error = ProtocolError(
                CGMPatchErrorCode.ENCODING_UNSUPPORTED.value,
                f"file '{normalized_path}' is not UTF-8 encoded: {exc}",
            )
            error.__cause__ = exc
            raise error

        if new_content is None:
            analysis = _analyse_diff_fallback(original_text, patch_text, normalized_path)
            new_text = analysis.new_text
            computed_stats = {
                "n_hunks": analysis.n_hunks,
                "added_lines": analysis.added_lines,
                "removed_lines": analysis.removed_lines,
            }
        else:
            new_text = normalize_newlines(new_content)
            computed_stats = {
                "n_hunks": int((stats or {}).get("n_hunks", 0)),
                "added_lines": int((stats or {}).get("added_lines", 0)),
                "removed_lines": int((stats or {}).get("removed_lines", 0)),
            }

        if CGM_CONTRACT.constraints.get("newline_required") and not new_text.endswith("\n"):
            raise ProtocolError(CGMPatchErrorCode.NEWLINE_MISSING.value, f"resulting file '{normalized_path}' must end with newline")

        temp_dir = Path(tempfile.mkdtemp(prefix="gp-patch-", dir=str(repo_root.parent)))
        try:
            trial_root = temp_dir
            shutil.copytree(repo_root, trial_root, dirs_exist_ok=True)
            trial_file = trial_root.joinpath(normalized_path)
            trial_file.parent.mkdir(parents=True, exist_ok=True)
            trial_file.write_text(new_text, encoding="utf-8")

            lint_result = run_lint(trial_root) if run_lint else {"ok": True, "stdout": ""}
            lint_ok = bool(lint_result.get("ok"))
            if not lint_ok:
                error = ProtocolError("lint-failed", lint_result.get("stdout", "lint failed"))
                error.temp_path = temp_dir.name  # type: ignore[attr-defined]
                error.n_hunks = computed_stats["n_hunks"]  # type: ignore[attr-defined]
                error.added_lines = computed_stats["added_lines"]  # type: ignore[attr-defined]
                error.removed_lines = computed_stats["removed_lines"]  # type: ignore[attr-defined]
                raise error

            tests_result = run_tests(trial_root)
            tests_passed = bool(tests_result.get("passed"))
            if not tests_passed:
                error = ProtocolError("build-failed", tests_result.get("stdout", "tests failed"))
                error.temp_path = temp_dir.name  # type: ignore[attr-defined]
                error.n_hunks = computed_stats["n_hunks"]  # type: ignore[attr-defined]
                error.added_lines = computed_stats["added_lines"]  # type: ignore[attr-defined]
                error.removed_lines = computed_stats["removed_lines"]  # type: ignore[attr-defined]
                raise error

            temp_target = trial_file
            final_path = source_file
            final_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = final_path.parent.joinpath(f".{final_path.name}.gp_tmp")
            tmp_file.write_text(new_text, encoding="utf-8")
            with tmp_file.open("rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp_file, final_path)
            if patch_id:
                self._applied.setdefault(normalized_path, set()).add(patch_id)
            return {
                "ok": True,
                "applied": True,
                "path": normalized_path,
                "tests_passed": tests_passed,
                "lint_ok": lint_ok,
                "tests": tests_result,
                "lint": lint_result,
                "n_hunks": computed_stats["n_hunks"],
                "added_lines": computed_stats["added_lines"],
                "removed_lines": computed_stats["removed_lines"],
                "temp_path": temp_dir.name,
            }
        except ProtocolError as exc:
            if not hasattr(exc, "temp_path"):
                exc.temp_path = temp_dir.name  # type: ignore[attr-defined]
                exc.n_hunks = computed_stats["n_hunks"]  # type: ignore[attr-defined]
                exc.added_lines = computed_stats["added_lines"]  # type: ignore[attr-defined]
                exc.removed_lines = computed_stats["removed_lines"]  # type: ignore[attr-defined]
            raise
        except Exception as exc:  # pragma: no cover - defensive
            error = ProtocolError(CGMPatchErrorCode.DIRTY_WORKSPACE.value, str(exc))
            error.temp_path = temp_dir.name  # type: ignore[attr-defined]
            error.n_hunks = computed_stats["n_hunks"]  # type: ignore[attr-defined]
            error.added_lines = computed_stats["added_lines"]  # type: ignore[attr-defined]
            error.removed_lines = computed_stats["removed_lines"]  # type: ignore[attr-defined]
            raise error
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _analyse_diff_fallback(original_text: str, diff_text: str, path: str) -> "DiffAnalysis":
    """Fallback diff analysis shared with :class:`PatchApplier`."""

    diff_text = normalize_newlines(diff_text)
    lines = diff_text.splitlines()
    added = removed = n_hunks = 0
    new_lines: List[str] = []
    original_lines = original_text.splitlines()
    idx = 0
    current_orig = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("@@ "):
            n_hunks += 1
            idx += 1
            while idx < len(lines):
                segment = lines[idx]
                if segment.startswith("@@ ") or segment.startswith("diff --git") or segment.startswith("--- ") or segment.startswith("+++ "):
                    break
                if segment.startswith(" "):
                    if current_orig < len(original_lines):
                        new_lines.append(original_lines[current_orig])
                        current_orig += 1
                elif segment.startswith("-"):
                    removed += 1
                    current_orig += 1
                elif segment.startswith("+"):
                    added += 1
                    new_lines.append(segment[1:])
                idx += 1
            continue
        idx += 1
    new_lines.extend(original_lines[current_orig:])
    new_text = "\n".join(new_lines) + "\n"
    class DiffAnalysis:  # local alias to avoid importing from agents
        def __init__(self, new_text: str, n_hunks: int, added_lines: int, removed_lines: int) -> None:
            self.new_text = new_text
            self.n_hunks = n_hunks
            self.added_lines = added_lines
            self.removed_lines = removed_lines
    return DiffAnalysis(new_text, n_hunks, added, removed)
