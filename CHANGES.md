# Changelog

## Unreleased
- Remove obsolete configs, scripts, and tests now that `train_planner_grpo.py` is the sole training entry point.
- Refresh README、文档与样例数据集说明，突出 `prepare_datasets.py`、统一的 GRPO 配置以及新的 RepoEnv sample。
- 精简 `models/`、`datasets/` 等目录结构，仅保留 Planner/CGM 需要的占位目录与最新样例。
- 更新 `tests/test_dataset_preparation.py` 以覆盖新的数据准备流程并删除失效的 rLLM 集成测试。
