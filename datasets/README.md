# 数据集路径说明 / Dataset layout

- `r2e_gym/train.jsonl`：Graph Planner 默认的 R2E-Gym 训练集。运行 `python scripts/prepare_training_datasets.py`
  会先从 Hugging Face 下载原始 parquet 分片，再转换成 JSON/JSONL 写入该目录。转换过程中脚本会为每条任务
  整理出三个关键信息：

  * **`task_id`**：容器运行与训练日志依赖的唯一标识。原始 R2E 任务散落在不同字段中（`task_id`、`instance_id`、
    `ds.task.task_id` 等），脚本会按顺序回退，必要时基于数据集名 + 下标生成稳定 ID，确保后续缓存/断点恢复
    能够复用同一任务。
  * **`docker_image`**：RepoEnv 启动容器所需镜像名称。如果缺少该字段，任务无法执行，因此脚本会直接跳过并在
    日志中统计 `skipped` 数量。转换时还会生成 `docker_images.txt` manifest，列出所有唯一镜像，供训练脚本加载。
  * **`instance` JSON**：包含 issue 描述、环境变量、挂载配置、最大步数等结构化数据（见下文）。环境运行和
    rLLM 数据注册都依赖这一结构，因此统一写成 JSON 便于快速重放任务。

- `r2e_gym/val.jsonl`：Graph Planner 默认的 R2E-Gym 验证集，生成流程同上。
- `graphplanner_repoenv_sample.jsonl`：保留的历史示例，便于快速回归旧版脚本。
- `swebench/`：通过 `scripts/prepare_swebench_validation.py` 下载或解析的 SWE-bench 验证/测试集。Verified 分支不再
  提供现成的容器镜像，脚本会在 `instances/*.json` 与 JSONL 中额外写入 `requires_build=true` 以及从官方
  `swebench.harness.test_spec` 解析出的 `swebench_spec`（仓库、安装脚本、评测脚本等）。若 manifest 中只有
  build-only 项，`--prepull-containers` 会给出提示，需要先运行 `python -m swebench.harness.prepare_images`
  或仓库的容器构建工具把 `sweb.eval.*` 镜像在本机构建完成，之后训练/评测即可直接消费这些任务。

单条 JSONL 记录的结构如下（字段经过扁平化，方便后续写入 Verl parquet）：

```json
{
  "task_id": "R2E-Gym/R2E-Gym-Lite:train:00042",
  "docker_image": "registry.example.com/r2e/python:3.10",
  "repo": "pallets/flask",
  "max_steps": 40,
  "issue": {"title": "Fix bad request handling", "body": "..."},
  "instance": {
    "mounts": ["repo:/workspace"],
    "commands": {"build": "pytest", "test": "pytest -k failing"},
    "env": {"PYTHONPATH": "."}
  }
}
```

生成完 JSONL 后，训练脚本会按以下步骤消费：

1. `scripts/train_graphplanner_rllm.py` 调用 `ensure_dataset_registered` 读取 JSONL，将上述字段再度压平成 Verl 期望的
   parquet schema，并把 `task_id`、`docker_image`、`instance` 等信息保存到 Ray 可访问的缓存目录。
2. rLLM 在回放数据时会把 JSON 反序列化，交给 `GraphPlannerRLLMEnv.reset` 恢复 RepoEnv 容器。此时 `docker_image` 确定
   运行基础镜像，`instance` 字段提供初始化挂载、工作目录、测试命令等。若 manifest 存在，训练/评估脚本会在启动前读取
   `docker_images.txt`（或通过 `--prepull-containers` 预拉容器），减少首次 rollout 时的镜像拉取等待。
3. 进入强化学习循环后，Planner 依据 issue/子图/历史 observation 产出 `<function=...>`；当触发 `repair` 动作时，环境
   会拼装 CGM payload（含最新代码上下文、子图、记忆）并调用模型生成补丁。

因此，将下载好的 R2E-Gym 任务（或自定义任务）放入上述路径即可直接运行训练/评估命令，数据会沿着上述链路自动
被模型消费。
