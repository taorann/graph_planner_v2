# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging
import os
from typing import Callable

import torch

logger = logging.getLogger(__name__)

_CUDA_VISIBLE_OVERRIDE_ENV = "VERL_CUDA_VISIBLE_DEVICES_OVERRIDE"


def _apply_cuda_visible_override() -> None:
    """Re-apply CUDA visibility when Ray resets it for zero-GPU actors.

    Ray clears ``CUDA_VISIBLE_DEVICES`` for actors that do not request GPU
    resources.  When we rely on topology-provided ``worker_env_overrides``
    to pin GPUs, this clearing happens *after* the runtime environment is
    applied, leading to a process that believes no CUDA device is available.

    To defend against this, we stash the desired visibility in
    ``VERL_CUDA_VISIBLE_DEVICES_OVERRIDE`` and restore it here before any
    CUDA availability checks run.
    """

    override = os.environ.get(_CUDA_VISIBLE_OVERRIDE_ENV)
    if override:
        os.environ["CUDA_VISIBLE_DEVICES"] = override


_apply_cuda_visible_override()


class _LazyAvailability:
    """Boolean-like helper that evaluates hardware availability lazily."""

    def __init__(self, getter: Callable[[], bool]) -> None:
        self._getter = getter

    def __bool__(self) -> bool:  # pragma: no cover - trivial wrapper
        try:
            return bool(self._getter())
        except Exception:  # pragma: no cover - defensive guard
            return False

    def __call__(self) -> bool:
        return bool(self)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}({bool(self)})"


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


def _is_cuda_available() -> bool:
    return torch.cuda.is_available()


is_cuda_available = _LazyAvailability(_is_cuda_available)
is_npu_available = _LazyAvailability(is_torch_npu_available)


def get_visible_devices_keyword() -> str:
    """Function that gets visible devices keyword name.
    Returns:
        'CUDA_VISIBLE_DEVICES' or `ASCEND_RT_VISIBLE_DEVICES`
    """
    return "CUDA_VISIBLE_DEVICES" if is_cuda_available() else "ASCEND_RT_VISIBLE_DEVICES"


def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently only supports CPU, CUDA, NPU.
    Returns:
        device
    """
    if is_cuda_available():
        device = "cuda"
    elif is_npu_available():
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device() -> any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    """Return current device id based on the device type.
    Returns:
        device index
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Return nccl backend type based on the device type.
    Returns:
        nccl backend type string.
    """
    if is_cuda_available():
        return "nccl"
    elif is_npu_available():
        return "hccl"
    else:
        raise RuntimeError(f"No available nccl backend found on device type {get_device_name()}.")
