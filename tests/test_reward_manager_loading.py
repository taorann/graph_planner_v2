import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1] / "rllm"))

try:  # pragma: no cover - optional dependency gate
    from rllm.trainer.verl import train_agent_ppo
except ImportError as exc:  # pragma: no cover - exercised when Verl deps missing
    pytest.skip(f"rllm.trainer.verl import unavailable: {exc}", allow_module_level=True)


def test_maybe_load_reward_managers_noop_without_config(monkeypatch):
    calls = []

    def _spy(*args, **kwargs):  # pragma: no cover - helper
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(train_agent_ppo, "load_reward_manager", _spy)

    config = OmegaConf.create({"data": {}})
    reward_fn, val_reward_fn = train_agent_ppo._maybe_load_reward_managers(config, tokenizer=None)

    assert reward_fn is None
    assert val_reward_fn is None
    assert calls == []


def test_maybe_load_reward_managers_invokes_loader(monkeypatch):
    calls = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return kwargs["num_examine"]

    monkeypatch.setattr(train_agent_ppo, "load_reward_manager", _spy)

    config = OmegaConf.create(
        {
            "data": {"reward_fn_key": "dummy"},
            "reward_model": {"reward_kwargs": {"alpha": 0.1}},
        }
    )

    reward_fn, val_reward_fn = train_agent_ppo._maybe_load_reward_managers(config, tokenizer="tok")

    assert reward_fn == 0
    assert val_reward_fn == 1
    assert [call[1]["num_examine"] for call in calls] == [0, 1]
