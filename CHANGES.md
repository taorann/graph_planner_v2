# Changelog

## Unreleased
- Harden planner/CGM contract enforcement with newline normalisation, unified-diff validation, and deterministic patch hashing.
- Introduce `runtime.sandbox.PatchApplier` for atomic apply + telemetry (`patch_id`, `n_hunks`, `temp_path`) and add duplicate/dirty workspace safeguards.
- Add CLI helpers (`scripts/validate_patches.py`) and regression tests covering diff validation, atomic apply, and observation telemetry.
