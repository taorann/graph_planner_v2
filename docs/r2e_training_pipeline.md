# R2E-Gym 训练数据准备与容器链路说明

本文档汇总了如何在本地准备 R2E-Gym 的训练数据、启动并部署容器、以及 Graph Planner 在强化学习训练过程中调用容器的全链路。

## 1. 准备环境

在开始之前，请确保满足以下条件：

- 已克隆本仓库并初始化 `R2E-Gym` 子模块。
- Python 3.10+，并在虚拟环境中安装本仓库和 R2E-Gym 所需依赖：
  ```bash
  uv venv
  source .venv/bin/activate
  uv pip install -e .
  uv pip install -e R2E-Gym
  ```
- 安装 Hugging Face `datasets`（R2E-Gym 通过该库分发训练环境）：
  ```bash
  uv pip install datasets
  ```
- 已安装 Docker，并确保当前用户可以访问 Docker daemon（例如 `sudo systemctl start docker`）。

## 2. 下载并整理 R2E-Gym 数据

1. **从 Hugging Face 拉取原始任务：**
   ```python
   from datasets import load_dataset
   ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split="train")
   ```
2. **选择需要的任务条目，并生成 Graph Planner 所需的 JSONL 文件：**
   ```python
   import json
   from pathlib import Path

   target = Path("datasets/graphplanner_repoenv_train.jsonl")
   with target.open("w", encoding="utf-8") as handle:
       for row in ds:
           entry = {
               "task_id": row["task_id"],
               "max_steps": 6,
               "issue": row["issue"],
               "sandbox": {
                   "backend": "repoenv",
                   "docker_image": row["docker_image"],
                   "workdir": row.get("workdir", "/repo"),
                   "mounts": {},
                   "env": {},
                   "r2e_ds_json": "config/r2e_ds_repoenv_sample.json"
               }
           }
           handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
   ```
   - 其中 `sandbox.r2e_ds_json` 字段指向 R2E-Gym 提供的 repo 信息，示例配置可参照 `config/r2e_ds_repoenv_sample.json`。
   - 如需自定义任务，可参考仓库提供的样例 `datasets/graphplanner_repoenv_sample.jsonl`。

3. **注册数据集供 rLLM 使用：**
   ```bash
   PYTHONPATH=. python scripts/register_graphplanner_dataset.py datasets/graphplanner_repoenv_train.jsonl
   ```
   该脚本会将 JSONL 转换为 rLLM 的 parquet 数据格式，并写入 rLLM 的数据注册表，供训练脚本读取。注册逻辑详见 `graph_planner/integrations/rllm/dataset.py`。 

## 3. 准备容器镜像

- 数据集中的每个任务都包含 `sandbox.docker_image` 和 `sandbox.r2e_ds_json` 字段。
- `r2e_ds_repoenv_sample.json` 描述了 RepoEnv 所需的镜像、仓库名称和基准提交信息，可作为自定义数据集的模板。【F:config/r2e_ds_repoenv_sample.json†L1-L5】
- 在训练机器上提前拉取所有相关镜像，例如：
  ```bash
  docker pull graph-planner/repoenv-sample:latest
  ```
- 如需根据 R2E-Gym 数据生成 `r2e_ds_json` 文件，可直接复用原始条目中的 `repo_name`、`docker_image` 等字段生成与示例结构一致的 JSON。

## 4. 启动与验证容器

Graph Planner 通过 `SandboxRuntime` 统一管理容器后端：

1. 训练脚本读取数据集后，将任务条目交给 `GraphPlannerRLLMEnv`。
2. `GraphPlannerRLLMEnv` 在 `reset()` 时会生成 `SandboxConfig` 并实例化 `PlannerEnv`。【F:graph_planner/integrations/rllm/env.py†L45-L101】
3. `PlannerEnv` 在构造过程中立即创建 `SandboxRuntime`，该运行时会根据 `backend` 字段选择 RepoEnv、R2E DockerRuntime 或 docker-py 后端。【F:graph_planner/env/planner_env.py†L36-L63】【F:graph_planner/runtime/sandbox.py†L32-L123】
4. 当后端为 `repoenv` 时，`SandboxRuntime` 会读取 `r2e_ds_json` 并通过 R2E-Gym 的 `RepoEnv` 启动目标镜像，再在容器内准备测试工具链。【F:graph_planner/runtime/sandbox.py†L60-L123】

可在本地使用规则代理先运行一次冒烟测试，确认容器可以正常启动：
```bash
PYTHONPATH=. python scripts/run_rule_agent.py \
  --backend repoenv \
  --ds-json config/r2e_ds_repoenv_sample.json \
  --max-steps 6 \
  --report smoke_report.json
```
如果容器启动成功，轨迹日志会保存至 `logs/test_runs.jsonl`，便于核对补丁和命令历史。【F:logs/test_runs.jsonl†L1-L1】

## 5. 训练过程中调用容器的链路

完整的训练链路如下：

1. `scripts/train_graphplanner_rllm.py` 注册数据集并加载 rLLM 的 PPO 配置，然后通过 Ray 启动 `train_agent` 远程任务。【F:scripts/train_graphplanner_rllm.py†L1-L114】
2. rLLM 在每个 episode 中调用 `GraphPlannerRLLMAgent`，将模型输出解析为 Explore/Memory/Repair/Submit 动作；若解析失败则回退至规则策略。【F:graph_planner/integrations/rllm/agent.py†L1-L131】
3. 对应的 `GraphPlannerRLLMEnv` 为每条任务条目实例化一个 `PlannerEnv`，并将动作转交给 `SandboxRuntime` 执行。【F:graph_planner/integrations/rllm/env.py†L45-L101】
4. `SandboxRuntime` 根据配置选择 RepoEnv→R2E DockerRuntime→docker-py 的优先级启动容器并执行命令、应用补丁、运行测试。【F:graph_planner/runtime/sandbox.py†L32-L214】
5. 容器中的命令结果与测试反馈经由 `PlannerEnv` 返回给 Agent，Agent 依据奖励更新策略，Ray 负责聚合梯度并保存模型权重。【F:graph_planner/env/planner_env.py†L36-L107】

## 6. 常见问题

- **找不到 Docker daemon：** 请确认宿主机已启动 Docker 服务，并在运行脚本的用户权限下可访问 `/var/run/docker.sock`。
- **缺少 rLLM 或 R2E 依赖：** 确认已经执行 `uv pip install -e R2E-Gym` 且 rLLM 子模块的依赖已按 `infra/vendor.ensure_rllm_importable()` 的要求加入 `PYTHONPATH`。
- **数据集中无 `r2e_ds_json`：** 训练阶段需要该字段来加载 RepoEnv 元数据，请使用示例 JSON 作为模板补齐。

完成以上步骤后，即可在具备 Docker 的主机上运行 `scripts/train_graphplanner_rllm.py`，使用本地部署的决策模型与 CGM 进行强化学习训练。
