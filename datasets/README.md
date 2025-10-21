# 数据集路径说明 / Dataset layout

- `r2e_gym/graphplanner_repoenv_train.jsonl`：基于 R2E-Gym 训练数据格式整理的 Planner 训练任务列表，供 rLLM PPO 训练脚本默认使用。
- `r2e_gym/graphplanner_repoenv_val.jsonl`：与训练集同源的验证任务列表，供评估脚本演示。
- `graphplanner_repoenv_sample.jsonl`：保留的历史示例，便于快速回归旧版脚本。

将下载好的 R2E-Gym 任务（或自定义任务）放入上述路径即可直接运行训练/评估命令。
