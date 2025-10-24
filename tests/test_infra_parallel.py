from __future__ import annotations

import pytest

from graph_planner.infra.parallel import ParallelConfig, preflight_check, resolve_parallel


def test_resolve_parallel_defaults():
    cfg = resolve_parallel({})
    assert cfg.tensor_parallel_planner == 1
    assert cfg.tensor_parallel_cgm == 1
    assert cfg.replicas == 1
    assert cfg.parallel_agents == 1
    assert cfg.rollout_workers == 1
    assert cfg.workflow_parallel == 1
    assert cfg.num_gpus == 1
    assert cfg.num_nodes == 1
    assert cfg.ray_num_gpus == 1
    assert cfg.ray_num_cpus == 4


def test_resolve_parallel_honours_yaml_overrides():
    yaml_cfg = {
        "parallel": {
            "tensor_parallel_planner": 3,
            "tensor_parallel_cgm": 2,
            "replicas": 4,
            "parallel_agents": 5,
            "rollout_workers": 6,
            "workflow_parallel": 7,
        },
        "resources": {
            "num_gpus": 8,
            "num_nodes": 9,
            "ray_num_gpus": 10,
            "ray_num_cpus": 44,
        },
    }

    cfg = resolve_parallel(yaml_cfg)
    assert cfg.tensor_parallel_planner == 3
    assert cfg.tensor_parallel_cgm == 2
    assert cfg.replicas == 4
    assert cfg.parallel_agents == 5
    assert cfg.rollout_workers == 6
    assert cfg.workflow_parallel == 7
    assert cfg.num_gpus == 8
    assert cfg.num_nodes == 9
    assert cfg.ray_num_gpus == 10
    assert cfg.ray_num_cpus == 44


def test_resolve_parallel_custom_values():
    merged = {
        "parallel": {
            "tensor_parallel_planner": 4,
            "tensor_parallel_cgm": 2,
            "replicas": 3,
            "parallel_agents": 8,
            "rollout_workers": 6,
            "workflow_parallel": 10,
        },
        "resources": {
            "num_gpus": 16,
            "num_nodes": 2,
            "ray_num_gpus": 12,
            "ray_num_cpus": 128,
        },
    }
    cfg = resolve_parallel(merged)
    assert cfg.tensor_parallel_planner == 4
    assert cfg.tensor_parallel_cgm == 2
    assert cfg.replicas == 3
    assert cfg.parallel_agents == 8
    assert cfg.rollout_workers == 6
    assert cfg.workflow_parallel == 10
    assert cfg.num_gpus == 16
    assert cfg.num_nodes == 2
    assert cfg.ray_num_gpus == 12
    assert cfg.ray_num_cpus == 128


def test_preflight_check_rejects_invalid_config():
    bad_cfg = ParallelConfig(
        tensor_parallel_planner=4,
        tensor_parallel_cgm=4,
        replicas=2,
        parallel_agents=4,
        rollout_workers=4,
        workflow_parallel=3,
        num_gpus=4,
        num_nodes=1,
        ray_num_gpus=1,
        ray_num_cpus=8,
    )
    with pytest.raises(ValueError) as exc:
        preflight_check(bad_cfg)
    message = str(exc.value)
    assert "tensor parallel" in message
    assert "workflow_parallel" in message
    assert "Ray GPU budget" in message
    assert "Ray CPU budget" in message


def test_preflight_check_accepts_valid_config():
    good_cfg = ParallelConfig(
        tensor_parallel_planner=4,
        tensor_parallel_cgm=2,
        replicas=1,
        parallel_agents=4,
        rollout_workers=4,
        workflow_parallel=6,
        num_gpus=8,
        num_nodes=1,
        ray_num_gpus=4,
        ray_num_cpus=32,
    )
    preflight_check(good_cfg)
