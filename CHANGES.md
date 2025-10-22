# Changelog

## Unreleased
- Harden planner/CGM contract enforcement with newline normalisation, unified-diff validation, and deterministic patch hashing.
- Introduce `runtime.sandbox.PatchApplier` for atomic apply + telemetry (`patch_id`, `n_hunks`, `temp_path`) and add duplicate/dirty workspace safeguards.
- Add CLI helpers (`scripts/validate_patches.py`) and regression tests covering diff validation, atomic apply, and observation telemetry.
- Fix README quick-start instructions to reference the actual R2E-Gym installation workflow and remove the obsolete `requirements-dev.txt` step.
- Tune the 16×A800 training profile (`gp_full_73b14b_16g.yaml`) for higher throughput and document the updated parallelism hints in the runbook.
- Add `scripts/prepare_datasets.py` plus dataset converters so R2E-Gym train/val and SWE-bench test splits can be downloaded into `datasets/` and consumed by the rLLM pipeline.
