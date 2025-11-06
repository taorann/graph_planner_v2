"""Evaluate Graph Planner tasks using rLLM's :class:`AgentExecutionEngine`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

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

    # Normalise path-like arguments when defaults originate from config files.
    path_fields = [
        "dataset",
        "planner_tokenizer",
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

    missing = [field for field in ("dataset", "planner_model", "cgm_model_path") if getattr(args, field) is None]
    if missing:
        parser.error(
            "The following arguments are required (supply via CLI or config): "
            + ", ".join(f"--{field.replace('_', '-')}" for field in missing)
        )

    return args


def _configure_runtime_env(args: argparse.Namespace) -> None:
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
    if args.planner_api_key:
        env_name = str(args.planner_api_key_env or "PLANNER_MODEL_API_KEY")
        os.environ[env_name] = str(args.planner_api_key)
        os.environ["PLANNER_MODEL_API_KEY_ENV"] = env_name
    elif args.planner_api_key_env:
        os.environ["PLANNER_MODEL_API_KEY_ENV"] = str(args.planner_api_key_env)
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
    if args.cgm_api_key:
        env_name = str(args.cgm_api_key_env or "CGM_API_KEY")
        os.environ[env_name] = str(args.cgm_api_key)
        os.environ["CGM_API_KEY_ENV"] = env_name
    elif args.cgm_api_key_env:
        os.environ["CGM_API_KEY_ENV"] = str(args.cgm_api_key_env)
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
    tokenizer_path = str(args.planner_tokenizer or args.planner_model)
    LOGGER.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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

    _configure_runtime_env(args)

    tasks = _load_tasks(
        args.dataset,
        ensure_dataset_registered=ensure_dataset_registered,
        load_task_entries=load_task_entries,
        dataset_name=dataset_name,
        limit=args.limit,
    )

    tokenizer = _build_tokenizer(args)
    sampling_params = {
        "model": args.planner_model,
        "temperature": args.planner_temperature,
        "top_p": args.planner_top_p,
        "max_tokens": args.max_response_tokens,
    }

    rollout_args = {"base_url": args.planner_base_url}
    if args.planner_api_key:
        rollout_args["api_key"] = args.planner_api_key
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
        enforce_max_prompt_length=args.enforce_max_prompt_length,
        overlong_filter=args.overlong_filter,
    )

    LOGGER.info(
        "Starting evaluation: tasks=%d parallel=%d planner_model=%s", len(tasks), args.parallel, args.planner_model
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
