# 数据集路径说明 / Dataset layout

- `r2e_gym/train.jsonl`：Graph Planner 默认的 R2E-Gym 训练集。运行 `python scripts/prepare_datasets.py` 会用 Hugging Face 上的真实数据覆盖并扩充该文件。
- `r2e_gym/val.jsonl`：Graph Planner 默认的 R2E-Gym 验证集，来源同上。
- `graphplanner_repoenv_sample.jsonl`：保留的历史示例，便于快速回归旧版脚本。
- `swebench/`：通过 `scripts/prepare_datasets.py --skip-r2e` 下载并生成的 SWE-bench 测试集。

将下载好的 R2E-Gym 任务（或自定义任务）放入上述路径即可直接运行训练/评估命令。
