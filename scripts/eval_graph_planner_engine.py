"""Evaluate Graph Planner tasks using rLLM's :class:`AgentExecutionEngine`."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml
from transformers import AutoTokenizer

from graph_planner.datasets.prepare import ensure_directory, sanitize_identifier

_REPO_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("PYTHONPATH", str(_REPO_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOGGER = logging.getLogger("graph_planner.eval.engine")


def _ensure_rllm_components():
    from graph_planner.infra.vendor import ensure_rllm_importable

    if not ensure_rllm_importable():
        raise RuntimeError("Unable to import vendored rLLM modules")

    from graph_planner.integrations.rllm import (
        GRAPH_PLANNER_DATASET_NAME,
        GraphPlannerRLLMAgent,
        GraphPlannerRLLMEnv,
        ensure_dataset_registered,
    )
    from graph_planner.integrations.rllm.dataset import load_task_entries

    from rllm.engine.agent_execution_engine import AgentExecutionEngine
    from rllm.utils import compute_pass_at_k

    return (
        GRAPH_PLANNER_DATASET_NAME,
        GraphPlannerRLLMAgent,
        GraphPlannerRLLMEnv,
        ensure_dataset_registered,
        load_task_entries,
        AgentExecutionEngine,
        compute_pass_at_k,
    )


def _load_config_defaults(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    data: Any
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format for {path}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got {type(data)!r}")

    defaults: Dict[str, Any] = {}
    for key, value in data.items():
        normalised = key.replace("-", "_")
        defaults[normalised] = value
    return defaults


def _parse_port_forward_spec(spec: str) -> Dict[str, Any]:
    raw = str(spec).strip()
    if not raw:
        raise ValueError("empty port forwarding spec")

    protocol = "tcp"
    base = raw
    if "/" in raw:
        base, protocol = raw.rsplit("/", 1)
        protocol = protocol.lower()
        if not protocol:
            protocol = "tcp"

    parts = base.split(":")
    host_ip: str | None = None
    host_port_raw: str
    container_port_raw: str

    if len(parts) == 3:
        host_ip, host_port_raw, container_port_raw = parts
    elif len(parts) == 2:
        host_port_raw, container_port_raw = parts
        host_ip = None
    elif len(parts) == 1:
        host_port_raw = ""
        container_port_raw = parts[0]
        host_ip = None
    else:
        raise ValueError(f"invalid port forwarding spec: {spec!r}")

    try:
        container_port = int(str(container_port_raw).strip())
    except Exception as exc:
        raise ValueError(f"invalid container port in spec: {spec!r}") from exc

    host_port: int | None = None
    if str(host_port_raw).strip():
        try:
            host_port = int(str(host_port_raw).strip())
        except Exception as exc:
            raise ValueError(f"invalid host port in spec: {spec!r}") from exc

    payload: Dict[str, Any] = {
        "container_port": container_port,
        "protocol": protocol,
    }
    if host_ip:
        payload["host_ip"] = host_ip
    if host_port is not None:
        payload["host_port"] = host_port
    return payload


def _build_sandbox_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    port_specs: List[Dict[str, Any]] = getattr(args, "sandbox_port_forwardings", []) or []
    if port_specs:
        overrides["port_forwards"] = port_specs
        overrides["force_docker_backend"] = True
        overrides.setdefault("backend", "docker")
    if getattr(args, "sandbox_force_docker_backend", False):
        overrides["force_docker_backend"] = True
        overrides.setdefault("backend", "docker")
    return overrides


def _resolve_path(
    value: Path,
    *,
    config_path: Path | None,
    must_exist: bool,
) -> Path:
    """Resolve ``value`` against reasonable lookup roots.

    Preference order:

    1. Absolute paths are returned as-is.
    2. Paths relative to the config file's parent directory.
    3. Paths relative to the repository root.
    4. Paths relative to the current working directory.

    If ``must_exist`` is ``True`` we require at least one of the candidates to
    exist; otherwise we fall back to the repository-root resolution.
    """

    if value.is_absolute():
        return value

    candidates: list[Path] = []
    if config_path is not None:
        candidates.append((config_path.parent / value).resolve())
    candidates.append((_REPO_ROOT / value).resolve())
    candidates.append((Path.cwd() / value).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if must_exist:
        raise FileNotFoundError(
            f"Unable to resolve path {value!s}; tried: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    # Default to the repository-root resolution for non-existent paths.
    return candidates[0]


def _build_models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/models"


def _planner_endpoint_alive(base_url: str, *, timeout: float = 2.0) -> bool:
    try:
        request = Request(_build_models_url(base_url))
        with urlopen(request, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except URLError:
        return False
    except Exception:  # noqa: BLE001 - best effort probe
        return False


def _is_local_host(hostname: str | None) -> bool:
    if hostname is None:
        return True
    hostname = hostname.lower()
    if hostname in {"localhost", "0.0.0.0", "127.0.0.1", "::1"}:
        return True
    return hostname.startswith("127.")


def _parse_device_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        cleaned: list[int] = []
        for item in value:
            try:
                cleaned.append(int(item))
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Ignoring non-integer GPU identifier in planner_service_gpus: %s",
                    item,
                )
                return None
        return cleaned or None

    devices: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            devices.append(int(part))
        except ValueError:
            LOGGER.warning(
                "Ignoring non-integer GPU identifier in planner_service_gpus: %s",
                part,
            )
            return None

    return devices or None


def _query_gpu_memory(indices: Sequence[int] | None) -> list[tuple[float, float]] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except FileNotFoundError:
        LOGGER.warning(
            "nvidia-smi not available; unable to probe GPU memory for planner auto-launch"
        )
        return None
    except subprocess.CalledProcessError as exc:
        LOGGER.warning(
            "nvidia-smi command failed (returncode=%s); skipping GPU memory probe",
            exc.returncode,
        )
        return None

    rows: list[tuple[float, float]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [segment.strip() for segment in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            free = float(parts[0])
            total = float(parts[1])
        except ValueError:
            continue
        rows.append((free, total))

    if not rows:
        return None

    if indices is None:
        return rows

    selected: list[tuple[float, float]] = []
    for index in indices:
        if index < 0 or index >= len(rows):
            LOGGER.warning(
                "GPU index %s requested for planner auto-launch is unavailable (total GPUs: %s)",
                index,
                len(rows),
            )
            return None
        selected.append(rows[index])

    return selected


def _query_gpu_uuid_map() -> dict[str, int] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    mapping: dict[str, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [segment.strip() for segment in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        mapping[parts[1]] = index

    return mapping or None


def _collect_gpu_process_snapshot(
    indices: Sequence[int] | None,
) -> list[tuple[int, int, str, float]] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None

    cleaned_lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not cleaned_lines or cleaned_lines == ["No running processes found"]:
        return []

    uuid_map = _query_gpu_uuid_map()
    if uuid_map is None:
        return None

    allowed = set(indices) if indices is not None else None
    entries: list[tuple[int, int, str, float]] = []
    for line in cleaned_lines:
        parts = [segment.strip() for segment in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        process_name = parts[1]
        uuid = parts[2]
        try:
            memory_mib = float(parts[3])
        except ValueError:
            continue

        index = uuid_map.get(uuid)
        if index is None:
            continue
        if allowed is not None and index not in allowed:
            continue
        entries.append((index, pid, process_name, memory_mib))

    return entries


def _log_gpu_memory_snapshot(
    stats: Sequence[tuple[float, float]],
    *,
    indices: Sequence[int] | None,
    headroom: float,
    requested: float | None,
    adjusted: float | None,
) -> None:
    if not stats:
        return

    logical_indices = list(indices) if indices is not None else list(range(len(stats)))
    details: list[str] = []
    for logical_index, (free, total) in zip(logical_indices, stats):
        ratio = (free / total) if total else 0.0
        used = max(total - free, 0.0)
        details.append(
            f"GPU{logical_index}: free={free:.2f} GiB used={used:.2f} GiB total={total:.2f} GiB (free_ratio={ratio:.3f})"
        )

    ratio_info = "; ".join(details)
    requested_repr = f"{requested:.3f}" if requested is not None else "None"
    adjusted_repr = f"{adjusted:.3f}" if adjusted is not None else "None"
    LOGGER.info(
        "Planner GPU memory probe -> %s; requested=%s headroom=%.3f adjusted=%s",
        ratio_info,
        requested_repr,
        headroom,
        adjusted_repr,
    )

    process_entries = _collect_gpu_process_snapshot(logical_indices)
    if process_entries is None:
        return
    if not process_entries:
        LOGGER.info("Planner GPU memory active processes: none reported by nvidia-smi")
        return

    formatted = []
    for gpu_index, pid, name, memory_mib in process_entries:
        formatted.append(
            f"GPU{gpu_index}: pid={pid} name={name} memory={memory_mib:.0f} MiB"
        )
    LOGGER.info(
        "Planner GPU memory active processes: %s",
        "; ".join(formatted),
    )


def _compute_gpu_memory_utilization(
    requested: float | None,
    *,
    devices: Sequence[int] | None,
    headroom: float,
) -> float | None:
    if requested is None and headroom <= 0:
        return None

    stats = _query_gpu_memory(devices)
    if not stats:
        return requested

    ratios = [free / total for free, total in stats if total > 0]
    if not ratios:
        return requested
    max_available_ratio = min(ratios)
    if headroom > 0:
        max_available_ratio = max(0.0, max_available_ratio - headroom)

    if requested is None:
        adjusted = max_available_ratio if max_available_ratio > 0 else None
        _log_gpu_memory_snapshot(
            stats,
            indices=devices,
            headroom=headroom,
            requested=requested,
            adjusted=adjusted,
        )
        return adjusted

    adjusted = min(requested, max_available_ratio)
    if adjusted < 0:
        adjusted = 0.0

    if adjusted < requested - 1e-6:
        LOGGER.info(
            "Reducing planner service gpu_memory_utilization from %.3f to %.3f based on available memory",
            requested,
            adjusted,
        )

    _log_gpu_memory_snapshot(
        stats,
        indices=devices,
        headroom=headroom,
        requested=requested,
        adjusted=adjusted,
    )

    return adjusted


@contextlib.contextmanager
def _auto_launch_planner_service(args: argparse.Namespace):
    if not getattr(args, "auto_launch_planner_service", False):
        yield None
        return

    if args.engine_name != "openai":
        LOGGER.info("Skipping auto-launch: engine_name=%s", args.engine_name)
        yield None
        return

    if args.planner_model_path is None:
        raise RuntimeError(
            "--auto-launch-planner-service requires --planner-model-path to point to a local checkpoint"
        )

    parsed = urlparse(args.planner_base_url)
    if not _is_local_host(parsed.hostname):
        LOGGER.info(
            "Skipping auto-launch: planner_base_url=%s is not a localhost endpoint",
            args.planner_base_url,
        )
        yield None
        return

    if not args.planner_model_path.exists():
        raise FileNotFoundError(
            f"Planner checkpoint not found: {args.planner_model_path}"
        )

    if _planner_endpoint_alive(args.planner_base_url):
        LOGGER.info(
            "Planner endpoint already responding at %s; auto-launch skipped",
            args.planner_base_url,
        )
        yield None
        return

    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    cmd: list[str] = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.planner_model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        str(args.planner_model),
        "--trust-remote-code",
    ]

    if args.planner_tokenizer:
        cmd.extend(["--tokenizer", str(args.planner_tokenizer)])

    gpu_devices = _parse_device_indices(
        getattr(args, "planner_service_gpus", None)
    )

    tensor_parallel_size = getattr(args, "planner_service_tensor_parallel_size", None)
    if tensor_parallel_size is None and gpu_devices:
        tensor_parallel_size = len(gpu_devices)
    if tensor_parallel_size:
        cmd.extend([
            "--tensor-parallel-size",
            str(tensor_parallel_size),
        ])

    if args.planner_max_input_tokens:
        cmd.extend(["--max-model-len", str(args.planner_max_input_tokens)])

    gpu_memory_utilization = getattr(
        args, "planner_service_gpu_memory_utilization", None
    )
    headroom_ratio = float(
        getattr(args, "planner_service_gpu_memory_headroom", 0.05)
    )
    adjusted_gpu_memory_utilization = _compute_gpu_memory_utilization(
        gpu_memory_utilization,
        devices=gpu_devices,
        headroom=headroom_ratio,
    )
    if adjusted_gpu_memory_utilization is not None:
        cmd.extend(
            [
                "--gpu-memory-utilization",
                str(adjusted_gpu_memory_utilization),
            ]
        )

    LOGGER.info("Auto-launching planner service with command: %s", shlex.join(cmd))

    env = os.environ.copy()
    if gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_devices)

    process = subprocess.Popen(cmd, env=env)
    ready = False
    deadline = time.time() + float(getattr(args, "planner_service_startup_timeout", 300.0))
    poll_interval = 2.0

    try:
        while time.time() < deadline:
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"Planner service exited early with code {retcode}"
                )
            if _planner_endpoint_alive(args.planner_base_url):
                ready = True
                LOGGER.info(
                    "Planner service is ready at %s",
                    args.planner_base_url,
                )
                break
            time.sleep(poll_interval)

        if not ready:
            raise TimeoutError(
                "Planner service did not become ready before timeout"
            )

        yield process
    finally:
        if process.poll() is None:
            LOGGER.info("Shutting down auto-launched planner service")
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        else:
            LOGGER.info("Planner service process ended with code %s", process.returncode)


def _format_netloc(host: str, port: int | None, scheme: str) -> str:
    if port is None:
        return host
    default_port = 80 if scheme == "http" else 443 if scheme == "https" else None
    if default_port is not None and port == default_port:
        return host
    return f"{host}:{port}"


def _cgm_health_url(
    parsed: "urllib.parse.ParseResult", *, port_override: int | None = None
) -> str:
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = port_override if port_override is not None else parsed.port
    netloc = _format_netloc(host, port, scheme)
    return f"{scheme}://{netloc}/healthz"


def _cgm_service_alive(url: str, *, timeout: float = 2.0) -> bool:
    try:
        request = Request(url)
        with urlopen(request, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except URLError:
        return False
    except Exception:  # noqa: BLE001 - best effort probe
        return False


@contextlib.contextmanager
def _auto_launch_cgm_service(args: argparse.Namespace):
    if not getattr(args, "auto_launch_cgm_service", False):
        yield None
        return

    if args.disable_cgm_synthesis:
        LOGGER.info("Skipping CGM auto-launch because synthesis is disabled")
        yield None
        return

    if args.cgm_endpoint is None:
        raise RuntimeError(
            "--auto-launch-cgm-service requires --cgm-endpoint to point to the local service"
        )

    if args.cgm_model_path is None:
        raise RuntimeError(
            "--auto-launch-cgm-service requires --cgm-model-path to provide a local checkpoint"
        )

    parsed = urlparse(args.cgm_endpoint)
    scheme = parsed.scheme or "http"
    if scheme not in {"http", "https", ""}:
        raise RuntimeError(
            f"Unsupported scheme for CGM endpoint: {scheme}; expected http or https"
        )

    if not _is_local_host(parsed.hostname):
        LOGGER.info(
            "Skipping CGM auto-launch: endpoint %s is not localhost",
            args.cgm_endpoint,
        )
        yield None
        return

    if not args.cgm_model_path.exists():
        raise FileNotFoundError(
            f"CGM checkpoint not found: {args.cgm_model_path}"
        )

    route = parsed.path or "/generate"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 30001)
    health_url = _cgm_health_url(parsed, port_override=port)

    if _cgm_service_alive(health_url):
        LOGGER.info("CGM endpoint already responding at %s; auto-launch skipped", args.cgm_endpoint)
        yield None
        return

    cmd: list[str] = [
        "python",
        "-m",
        "graph_planner.integrations.codefuse_cgm.service",
        "--model",
        str(args.cgm_model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--route",
        route or "/generate",
        "--max-input-tokens",
        str(args.cgm_max_input_tokens),
        "--max-new-tokens",
        str(args.cgm_max_output_tokens),
        "--temperature",
        str(args.cgm_temperature),
        "--top-p",
        str(args.cgm_top_p),
        "--log-level",
        str(getattr(args, "cgm_service_log_level", "info")),
    ]

    if args.cgm_tokenizer_path:
        cmd.extend(["--tokenizer", str(args.cgm_tokenizer_path)])
    if args.cgm_device:
        cmd.extend(["--device", str(args.cgm_device)])
    if args.cgm_device_map:
        cmd.extend(["--device-map", str(args.cgm_device_map)])

    LOGGER.info("Auto-launching CGM service with command: %s", shlex.join(cmd))

    env = os.environ.copy()
    service_gpus = getattr(args, "cgm_service_gpus", None)
    parsed_service_gpus = _parse_device_indices(service_gpus)
    if parsed_service_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in parsed_service_gpus
        )

    process = subprocess.Popen(cmd, env=env)
    ready = False
    deadline = time.time() + float(getattr(args, "cgm_service_startup_timeout", 300.0))
    poll_interval = 2.0

    try:
        while time.time() < deadline:
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"CGM service exited early with code {retcode}"
                )
            if _cgm_service_alive(health_url):
                ready = True
                LOGGER.info("CGM service is ready at %s", args.cgm_endpoint)
                break
            time.sleep(poll_interval)

        if not ready:
            raise TimeoutError("CGM service did not become ready before timeout")

        yield process
    finally:
        if process.poll() is None:
            LOGGER.info("Shutting down auto-launched CGM service")
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        else:
            LOGGER.info("CGM service process ended with code %s", process.returncode)


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON config containing default CLI values",
    )

    config_args, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    config_defaults: Dict[str, Any] | None = None
    if config_args.config is not None:
        config_defaults = _load_config_defaults(config_args.config)

    parser.add_argument("--dataset", type=Path, default=None, help="Path to the task dataset (JSON/JSONL/Parquet)")
    parser.add_argument("--planner-model", default=None, help="Planner model identifier passed to the rollout engine")
    parser.add_argument(
        "--planner-tokenizer",
        type=Path,
        default=None,
        help="Optional tokenizer path used to build the HF tokenizer",
    )
    parser.add_argument("--planner-model-path", type=Path, default=None, help="Optional local HF checkpoint for the planner")
    parser.add_argument(
        "--planner-system-prompt",
        type=Path,
        default=None,
        help="Optional file containing a custom system prompt for the planner",
    )
    parser.add_argument(
        "--planner-base-url",
        default="http://localhost:30000/v1",
        help="OpenAI-compatible planner endpoint",
    )
    parser.add_argument("--planner-api-key", default=None, help="Optional API key for the planner endpoint")
    parser.add_argument("--planner-api-key-env", default=None, help="Environment variable used by the planner runtime for API key lookup")
    parser.add_argument("--planner-temperature", type=float, default=0.0)
    parser.add_argument("--planner-top-p", type=float, default=0.95)
    parser.add_argument("--planner-timeout", type=float, default=120.0, help="Timeout for planner responses in seconds")
    parser.add_argument("--planner-max-input-tokens", type=int, default=None)
    parser.add_argument("--planner-device", default=None)
    parser.add_argument("--planner-device-map", default=None)
    parser.set_defaults(auto_launch_planner_service=False)
    parser.add_argument(
        "--auto-launch-planner-service",
        dest="auto_launch_planner_service",
        action="store_true",
        help="Automatically launch a local vLLM service when planner_base_url targets localhost",
    )
    parser.add_argument(
        "--no-auto-launch-planner-service",
        dest="auto_launch_planner_service",
        action="store_false",
        help="Disable automatic planner service startup",
    )
    parser.add_argument(
        "--planner-service-gpus",
        default=None,
        help="CUDA_VISIBLE_DEVICES value for an auto-launched planner service",
    )
    parser.add_argument(
        "--planner-service-tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel degree passed to the auto-launched planner service",
    )
    parser.add_argument(
        "--planner-service-startup-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for the planner service to report ready",
    )
    parser.add_argument(
        "--planner-service-gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional gpu_memory_utilization override for an auto-launched planner service",
    )
    parser.add_argument(
        "--planner-service-gpu-memory-headroom",
        type=float,
        default=0.05,
        help=(
            "Fractional headroom subtracted from detected free memory ratios before "
            "launching the planner service to avoid oversubscribing GPUs"
        ),
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-response-tokens", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=8, help="Upper bound on planner interactions per task")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel agent/environment pairs")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of tasks to evaluate")
    parser.add_argument("--results-path", type=Path, default=None, help="Optional JSON file to store trajectory outputs")
    parser.add_argument("--cgm-model-path", type=Path, default=None, help="Path to the CGM model checkpoint")
    parser.add_argument("--cgm-tokenizer-path", type=Path, default=None, help="Tokenizer path for the CGM model")
    parser.add_argument("--cgm-endpoint", default=None, help="Optional OpenAI-compatible endpoint for CGM remote inference")
    parser.add_argument("--cgm-model", default=None, help="Model identifier when using a remote CGM endpoint")
    parser.add_argument("--cgm-api-key", default=None, help="Optional API key for the CGM endpoint")
    parser.add_argument("--cgm-api-key-env", default=None, help="Environment variable to expose the CGM API key")
    parser.add_argument("--cgm-timeout", type=float, default=None, help="Timeout for CGM responses in seconds")
    parser.add_argument("--cgm-max-input-tokens", type=int, default=8192)
    parser.add_argument("--cgm-max-output-tokens", type=int, default=1024)
    parser.add_argument("--cgm-temperature", type=float, default=0.0)
    parser.add_argument("--cgm-top-p", type=float, default=0.9)
    parser.add_argument("--cgm-device", default=None)
    parser.add_argument("--cgm-device-map", default=None)
    parser.set_defaults(auto_launch_cgm_service=False)
    parser.add_argument(
        "--auto-launch-cgm-service",
        dest="auto_launch_cgm_service",
        action="store_true",
        help="Automatically launch a local CGM service when cgm_endpoint targets localhost",
    )
    parser.add_argument(
        "--no-auto-launch-cgm-service",
        dest="auto_launch_cgm_service",
        action="store_false",
        help="Disable automatic CGM service startup",
    )
    parser.add_argument(
        "--cgm-service-gpus",
        default=None,
        help="CUDA_VISIBLE_DEVICES value for an auto-launched CGM service",
    )
    parser.add_argument(
        "--cgm-service-startup-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for the CGM service health endpoint",
    )
    parser.add_argument(
        "--cgm-service-log-level",
        default="info",
        help="Log level forwarded to the CGM service",
    )
    parser.add_argument("--gamma", type=float, default=0.2, help="Discount factor for Monte Carlo return aggregation")
    parser.add_argument("--retry-limit", type=int, default=3, help="Maximum number of retries for failed model calls")
    parser.add_argument("--api-retries", type=int, default=3, help="Retry attempts within the OpenAI-compatible client")
    parser.add_argument("--trajectory-timeout", type=float, default=None, help="Optional timeout per trajectory in seconds")
    parser.add_argument("--max-workers", type=int, default=64, help="Thread pool size for environment RPCs")
    parser.add_argument("--enforce-max-prompt-length", action="store_true")
    parser.add_argument("--overlong-filter", action="store_true")
    parser.add_argument("--engine-name", choices=["openai", "verl"], default="openai")
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--failure-penalty", type=float, default=None)
    parser.add_argument("--step-penalty", type=float, default=None)
    parser.add_argument("--timeout-penalty", type=float, default=None)
    parser.add_argument("--repo-op-limit", type=int, default=None, help="Optional limit on repo operations per episode")
    parser.add_argument("--disable-cgm-synthesis", action="store_true")
    parser.add_argument("--synthesis-strategy", default=None, help="Override the planner CGM synthesis strategy")
    parser.add_argument("--agent-system-prompt", default=None, help="Override the planner agent system prompt text")
    parser.add_argument("--agent-system-prompt-path", type=Path, default=None, help="Path to a file containing a custom system prompt")
    parser.add_argument("--disable-rule-fallback", action="store_true", help="Disable rule-based fallback actions inside the agent")
    parser.add_argument(
        "--sandbox-port-forward",
        action="append",
        dest="sandbox_port_forward",
        default=None,
        help=(
            "Expose container ports when using the docker backend. Accepts "
            "[HOST_IP:]HOST_PORT:CONTAINER_PORT or HOST_PORT:CONTAINER_PORT. Repeatable."
        ),
    )
    parser.add_argument(
        "--sandbox-force-docker-backend",
        action="store_true",
        help=(
            "Force SandboxRuntime to use the docker backend, ignoring RepoEnv manifests. "
            "Useful when manually exposing containers for external agents."
        ),
    )
    if config_defaults:
        valid_dests = {action.dest for action in parser._actions}
        for key in list(config_defaults):
            if key not in valid_dests:
                LOGGER.warning("Ignoring unknown config key: %s", key)
                config_defaults.pop(key)
        parser.set_defaults(**config_defaults)

    args = parser.parse_args(remaining)
    if args.config is None:
        args.config = config_args.config

    config_path: Path | None = None
    if args.config is not None:
        config_path = Path(args.config).resolve()
        args.config = config_path

    # Normalise path-like arguments when defaults originate from config files.
    path_fields = [
        "dataset",
        "planner_model_path",
        "planner_system_prompt",
        "results_path",
        "cgm_model_path",
        "cgm_tokenizer_path",
        "agent_system_prompt_path",
    ]
    for field in path_fields:
        value = getattr(args, field, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, field, Path(value))

    must_exist_fields = {
        "dataset",
        "planner_system_prompt",
        "agent_system_prompt_path",
    }
    for field in path_fields:
        value = getattr(args, field, None)
        if value is None:
            continue
        must_exist = field in must_exist_fields
        if field == "planner_model_path" and args.auto_launch_planner_service:
            must_exist = True
        if field == "cgm_model_path" and args.auto_launch_cgm_service:
            must_exist = True
        if (
            field == "cgm_tokenizer_path"
            and args.auto_launch_cgm_service
            and value is not None
        ):
            must_exist = True
        resolved = _resolve_path(
            value,
            config_path=config_path,
            must_exist=must_exist,
        )
        setattr(args, field, resolved)

    raw_port_specs: List[str] = []
    if getattr(args, "sandbox_port_forward", None):
        for item in args.sandbox_port_forward or []:
            if isinstance(item, (list, tuple)):
                raw_port_specs.extend(str(elem) for elem in item)
            else:
                raw_port_specs.append(str(item))
    port_forwardings: List[Dict[str, Any]] = []
    for spec in raw_port_specs:
        try:
            port_forwardings.append(_parse_port_forward_spec(spec))
        except ValueError as exc:
            raise SystemExit(f"Invalid --sandbox-port-forward value {spec!r}: {exc}") from exc
    args.sandbox_port_forwardings = port_forwardings

    missing = [field for field in ("dataset", "planner_model", "cgm_model_path") if getattr(args, field) is None]
    if missing:
        parser.error(
            "The following arguments are required (supply via CLI or config): "
            + ", ".join(f"--{field.replace('_', '-')}" for field in missing)
        )

    return args


def _resolve_api_key(
    explicit: str | None,
    env_name: str | None,
    *,
    default_env: str,
    fallback_envs: Iterable[str] = (),
) -> tuple[str | None, str | None]:
    """Return an API key and the environment variable that should expose it."""

    candidates: list[str] = []
    if env_name:
        candidates.append(env_name)
    if default_env not in candidates:
        candidates.append(default_env)
    for fallback in fallback_envs:
        if fallback and fallback not in candidates:
            candidates.append(fallback)

    if explicit:
        return str(explicit), candidates[0] if candidates else None

    for candidate in candidates:
        if candidate and os.environ.get(candidate):
            return os.environ[candidate], candidate

    return None, candidates[0] if candidates else None


def _configure_runtime_env(
    args: argparse.Namespace,
    *,
    planner_api_key: str | None,
    planner_api_key_env: str | None,
    cgm_api_key: str | None,
    cgm_api_key_env: str | None,
) -> None:
    os.environ["PLANNER_MODEL_ENABLED"] = "1"
    os.environ["PLANNER_MODEL_ENDPOINT"] = str(args.planner_base_url)
    os.environ["PLANNER_MODEL_MODEL"] = str(args.planner_model)
    os.environ["PLANNER_MODEL_TEMPERATURE"] = str(args.planner_temperature)
    os.environ["PLANNER_MODEL_TOP_P"] = str(args.planner_top_p)
    os.environ["PLANNER_MODEL_MAX_TOKENS"] = str(args.max_response_tokens)
    os.environ["PLANNER_MODEL_TIMEOUT_S"] = str(int(args.planner_timeout))
    if args.planner_tokenizer:
        os.environ["PLANNER_MODEL_TOKENIZER_PATH"] = str(args.planner_tokenizer)
    if args.planner_model_path:
        os.environ["PLANNER_MODEL_PATH"] = str(args.planner_model_path)
    if args.planner_system_prompt:
        prompt_text = Path(args.planner_system_prompt).read_text(encoding="utf-8")
        os.environ["PLANNER_MODEL_SYSTEM_PROMPT"] = prompt_text
    if planner_api_key_env:
        if planner_api_key:
            os.environ[planner_api_key_env] = planner_api_key
        os.environ["PLANNER_MODEL_API_KEY_ENV"] = planner_api_key_env
    elif planner_api_key:
        os.environ["PLANNER_MODEL_API_KEY_ENV"] = "PLANNER_MODEL_API_KEY"
        os.environ["PLANNER_MODEL_API_KEY"] = planner_api_key
    if args.planner_max_input_tokens is not None:
        os.environ["PLANNER_MODEL_MAX_INPUT_TOKENS"] = str(args.planner_max_input_tokens)
    if args.planner_device:
        os.environ["PLANNER_MODEL_DEVICE"] = str(args.planner_device)
    if args.planner_device_map:
        os.environ["PLANNER_MODEL_DEVICE_MAP"] = str(args.planner_device_map)

    os.environ["CGM_ENABLED"] = "1"
    os.environ["CGM_MODEL_PATH"] = str(args.cgm_model_path)
    if args.cgm_tokenizer_path:
        os.environ["CGM_TOKENIZER_PATH"] = str(args.cgm_tokenizer_path)
    if args.cgm_endpoint:
        os.environ["CGM_ENDPOINT"] = str(args.cgm_endpoint)
    if args.cgm_model:
        os.environ["CGM_MODEL"] = str(args.cgm_model)
    if cgm_api_key_env:
        if cgm_api_key:
            os.environ[cgm_api_key_env] = cgm_api_key
        os.environ["CGM_API_KEY_ENV"] = cgm_api_key_env
    elif cgm_api_key:
        os.environ["CGM_API_KEY_ENV"] = "CGM_API_KEY"
        os.environ["CGM_API_KEY"] = cgm_api_key
    if args.cgm_timeout is not None:
        os.environ["CGM_TIMEOUT_S"] = str(int(args.cgm_timeout))
    os.environ["CGM_MAX_INPUT_TOKENS"] = str(args.cgm_max_input_tokens)
    os.environ["CGM_MAX_TOKENS"] = str(args.cgm_max_output_tokens)
    os.environ["CGM_TEMPERATURE"] = str(args.cgm_temperature)
    os.environ["CGM_TOP_P"] = str(args.cgm_top_p)
    if args.cgm_device:
        os.environ["CGM_DEVICE"] = str(args.cgm_device)
    if args.cgm_device_map:
        os.environ["CGM_DEVICE_MAP"] = str(args.cgm_device_map)


def _ensure_repoenv_manifest(
    entry: Dict[str, Any],
    *,
    dataset_path: Path,
) -> Dict[str, Any]:
    """Ensure RepoEnv sandbox manifests exist locally.

    Some legacy manifests point to absolute paths from the machine that originally
    generated the dataset (e.g. ``/root/private_data/...``). We rewrite these to the
    current checkout and, when necessary, synthesise a minimal manifest so RepoEnv can
    boot the SWE-bench container.
    """

    sandbox = entry.get("sandbox")
    if not isinstance(sandbox, dict):
        return entry

    backend = sandbox.get("backend", "repoenv")
    if backend not in {"repoenv", "r2e", "auto"}:
        return entry

    ds_hint = sandbox.get("r2e_ds_json")
    dataset_root = dataset_path.parent.resolve()
    candidates: list[Path] = []

    if isinstance(ds_hint, str) and ds_hint.strip():
        hint_path = Path(ds_hint.strip())
        if not hint_path.is_absolute():
            hint_path = (dataset_root / hint_path).resolve()
        candidates.append(hint_path)

    task_id = entry.get("task_id") or sandbox.get("issue_id")
    if isinstance(task_id, str) and task_id.strip():
        safe_id = sanitize_identifier(task_id.strip())
        candidates.append(dataset_root / "instances" / f"{safe_id}.json")

    resolved: Path | None = None
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            resolved = candidate
            break

    if resolved is None:
        # Attempt to reconstruct a minimal manifest. Prefer explicit ``instance`` data.
        manifest: Dict[str, Any]
        raw_instance = entry.get("instance")
        if isinstance(raw_instance, dict):
            manifest = dict(raw_instance)
        else:
            manifest = {}

        if "instance_id" not in manifest and isinstance(task_id, str):
            manifest["instance_id"] = task_id.strip()

        docker_image = manifest.get("docker_image") or sandbox.get("docker_image")
        if not isinstance(docker_image, str) or not docker_image.strip():
            LOGGER.warning(
                "Task %s is missing docker image information; cannot synthesise RepoEnv manifest",
                task_id,
            )
            return entry

        manifest["docker_image"] = docker_image.strip()

        if "repo" not in manifest and isinstance(entry.get("repo"), str):
            manifest["repo"] = entry["repo"]

        if sandbox.get("swebench_spec"):
            manifest.setdefault("swebench_spec", sandbox["swebench_spec"])

        if sandbox.get("requires_build") is not None:
            manifest.setdefault("requires_build", sandbox["requires_build"])

        if sandbox.get("mounts"):
            manifest.setdefault("mounts", sandbox["mounts"])
        if sandbox.get("env"):
            manifest.setdefault("env", sandbox["env"])
        if sandbox.get("workdir"):
            manifest.setdefault("workdir", sandbox["workdir"])

        if "instance_id" not in manifest:
            manifest["instance_id"] = sanitize_identifier(docker_image.strip())

        resolved = (dataset_root / "instances" / f"{sanitize_identifier(manifest['instance_id'])}.json").resolve()
        ensure_directory(resolved.parent)
        resolved.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Synthesised RepoEnv manifest for %s at %s", task_id, resolved)

    sandbox["r2e_ds_json"] = str(resolved)
    entry["sandbox"] = sandbox
    return entry


def _load_tasks(
    dataset_path: Path,
    *,
    ensure_dataset_registered,
    load_task_entries,
    dataset_name: str,
    limit: int | None,
    sandbox_overrides: Dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    registered = ensure_dataset_registered(name=dataset_name, path=str(dataset_path))
    LOGGER.info(
        "Dataset registered -> name=%s split=%s rows=%d parquet=%s",
        registered.name,
        registered.split,
        registered.num_rows,
        registered.out_parquet,
    )

    entries = load_task_entries(str(dataset_path))
    tasks: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if limit is not None and len(tasks) >= limit:
            break

        entry = _ensure_repoenv_manifest(entry, dataset_path=dataset_path)
        if sandbox_overrides:
            sandbox_payload = entry.get("sandbox")
            if isinstance(sandbox_payload, dict):
                merged = dict(sandbox_payload)
                for key, value in sandbox_overrides.items():
                    if key == "port_forwards" and not value:
                        continue
                    merged[key] = value
                entry = dict(entry)
                entry["sandbox"] = merged

        task_id = entry.get("task_id") or entry.get("issue_id") or f"{registered.split}-{index}"
        task_payload = {
            "task_id": task_id,
            "raw_entry_json": json.dumps(entry, ensure_ascii=False),
        }
        if entry.get("repo"):
            task_payload["repo"] = entry["repo"]
        if entry.get("issue"):
            task_payload["issue"] = entry["issue"]
        tasks.append(task_payload)
    if not tasks:
        raise ValueError(f"No tasks could be loaded from dataset: {dataset_path}")
    LOGGER.info("Loaded %d task entries for evaluation", len(tasks))
    return tasks


def _build_tokenizer(args: argparse.Namespace):
    tokenizer_ref = args.planner_tokenizer or args.planner_model
    if not tokenizer_ref:
        LOGGER.warning(
            "No tokenizer reference supplied; remote planner rollouts will disable local token accounting",
        )
        return None

    tokenizer_path = str(tokenizer_ref)
    LOGGER.info("Loading tokenizer from %s", tokenizer_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - best-effort remote fallback
        LOGGER.warning(
            "Failed to load tokenizer from %s (%s). Prompt/token length enforcement will be disabled.",
            tokenizer_path,
            exc,
        )
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _serialise_trajectories(results: Iterable[Any]) -> list[Dict[str, Any]]:
    payload: list[Dict[str, Any]] = []
    for traj in results:
        record = {
            "task": traj.task,
            "reward": float(getattr(traj, "reward", 0.0)),
            "steps": [asdict(step) for step in getattr(traj, "steps", [])],
        }
        payload.append(record)
    return payload


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    (
        dataset_name,
        agent_cls,
        env_cls,
        ensure_dataset_registered,
        load_task_entries,
        AgentExecutionEngine,
        compute_pass_at_k,
    ) = _ensure_rllm_components()

    planner_api_key, planner_api_key_env = _resolve_api_key(
        args.planner_api_key,
        args.planner_api_key_env,
        default_env="PLANNER_MODEL_API_KEY",
        fallback_envs=("OPENAI_API_KEY",),
    )
    if args.engine_name == "openai" and not planner_api_key:
        raise RuntimeError(
            "Planner API key missing. Provide --planner-api-key or set one of "
            "PLANNER_MODEL_API_KEY, the value of --planner-api-key-env, or OPENAI_API_KEY."
        )

    cgm_api_key, cgm_api_key_env = _resolve_api_key(
        args.cgm_api_key,
        args.cgm_api_key_env,
        default_env="CGM_API_KEY",
    )

    _configure_runtime_env(
        args,
        planner_api_key=planner_api_key,
        planner_api_key_env=planner_api_key_env,
        cgm_api_key=cgm_api_key,
        cgm_api_key_env=cgm_api_key_env,
    )

    if args.engine_name == "openai":
        LOGGER.info(
            "Planner rollouts will call an OpenAI-compatible endpoint at %s using model=%s; "
            "start the service (e.g. vLLM api_server) or supply a hosted provider before running.",
            args.planner_base_url,
            args.planner_model,
        )
        if args.planner_model_path:
            LOGGER.info(
                "planner_model_path=%s is exported for Graph Planner's local fallback clients, "
                "but the rLLM engine still relies on the HTTP endpoint above.",
                args.planner_model_path,
            )

    sandbox_overrides = _build_sandbox_overrides(args)

    tasks = _load_tasks(
        args.dataset,
        ensure_dataset_registered=ensure_dataset_registered,
        load_task_entries=load_task_entries,
        dataset_name=dataset_name,
        limit=args.limit,
        sandbox_overrides=sandbox_overrides,
    )

    tokenizer = _build_tokenizer(args)
    enforce_max_prompt_length = args.enforce_max_prompt_length
    if tokenizer is None and enforce_max_prompt_length:
        LOGGER.warning(
            "Planner tokenizer unavailable; disabling --enforce-max-prompt-length",
        )
        enforce_max_prompt_length = False
    sampling_params = {
        "model": args.planner_model,
        "temperature": args.planner_temperature,
        "top_p": args.planner_top_p,
        "max_tokens": args.max_response_tokens,
    }

    rollout_args = {"base_url": args.planner_base_url}
    if planner_api_key:
        rollout_args["api_key"] = planner_api_key
    if args.planner_timeout:
        rollout_args["timeout"] = args.planner_timeout

    agent_args: Dict[str, Any] = {
        "use_rule_fallback": not args.disable_rule_fallback,
    }
    system_prompt_override: str | None = None
    if args.agent_system_prompt_path:
        system_prompt_override = Path(args.agent_system_prompt_path).read_text(encoding="utf-8")
    elif args.agent_system_prompt is not None:
        system_prompt_override = str(args.agent_system_prompt)
    if system_prompt_override is not None:
        agent_args["system_prompt"] = system_prompt_override

    env_args: Dict[str, Any] = {"max_steps": args.max_steps}
    if args.reward_scale is not None:
        env_args["reward_scale"] = args.reward_scale
    if args.failure_penalty is not None:
        env_args["failure_penalty"] = args.failure_penalty
    if args.step_penalty is not None:
        env_args["step_penalty"] = args.step_penalty
    if args.timeout_penalty is not None:
        env_args["timeout_penalty"] = args.timeout_penalty
    if args.repo_op_limit is not None:
        env_args["repo_operation_limit"] = args.repo_op_limit
    env_args["enable_cgm_synthesis"] = not args.disable_cgm_synthesis
    if args.synthesis_strategy:
        env_args["synthesis_strategy"] = args.synthesis_strategy

    trajectory_timeout = args.trajectory_timeout if args.trajectory_timeout is not None else args.planner_timeout

    with contextlib.ExitStack() as stack:
        stack.enter_context(_auto_launch_planner_service(args))
        stack.enter_context(_auto_launch_cgm_service(args))

        engine = AgentExecutionEngine(
            agent_class=agent_cls,
            env_class=env_cls,
            agent_args=agent_args,
            env_args=env_args,
            engine_name=args.engine_name,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            rollout_engine_args=rollout_args,
            n_parallel_agents=args.parallel,
            max_response_length=args.max_response_tokens,
            max_prompt_length=args.max_prompt_tokens,
            trajectory_timeout=trajectory_timeout,
            gamma=args.gamma,
            retry_limit=args.retry_limit,
            api_retries=args.api_retries,
            max_workers=args.max_workers,
            enforce_max_prompt_length=enforce_max_prompt_length,
            overlong_filter=args.overlong_filter,
        )

        LOGGER.info(
            "Starting evaluation: tasks=%d parallel=%d planner_model=%s",
            len(tasks),
            args.parallel,
            args.planner_model,
        )

        results = asyncio.run(engine.execute_tasks(tasks))
    success = sum(1 for traj in results if getattr(traj, "reward", 0.0) > 0)
    LOGGER.info("Completed %d tasks with %d successes", len(results), success)
    compute_pass_at_k(results)

    if args.results_path:
        serialised = _serialise_trajectories(results)
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        args.results_path.write_text(json.dumps(serialised, ensure_ascii=False, indent=2))
        LOGGER.info("Trajectory details written to %s", args.results_path)


if __name__ == "__main__":
    main()

