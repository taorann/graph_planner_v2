import multiprocessing.reduction as _mp_reduction

try:
    import torch.multiprocessing.reductions as _torch_mp_reductions

    if not hasattr(_torch_mp_reductions, "ForkingPickler"):
        _torch_mp_reductions.ForkingPickler = _mp_reduction.ForkingPickler  # type: ignore[attr-defined]
except Exception:
    pass
