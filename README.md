# Graph Planner

Graph Planner 是一个面向代码修复任务的双智能体系统，通过**规划模型**与**CGM 补丁模型**协作，在容器化沙箱中定位问题并生成修复补丁。项目的主要目标如下：

- 复现论文 CGM 架构（Planner + CGM）在真实代码仓库上的工作流，并提供规则策略作为回退。
- 基于图检索的记忆维护与观察压缩，让模型能够在大规模仓库上进行局部推理。
- 借助 [R2E-Gym](R2E-Gym/README.md) 的 RepoEnv / DockerRuntime 运行环境完成端到端强化学习训练。
- 保留本地部署接口，使 Planner 模型与 CGM 可以在离线环境下无缝接入。

更多现状、缺失信息与后续计划详见 [`docs/project_status.md`](docs/project_status.md)。

## 目录结构概览

核心代码已经收敛到单一的 `graph_planner/` 包，其他目录仅保留必要的配置、脚本与文档：

```
graph_planner/
  agents/          # 规则策略、本地 LLM 代理与共享对话协议
  core/            # Explore / Memory / Repair / Submit 动作数据模型
  env/             # PlannerEnv：把动作映射到 SandboxRuntime
  infra/           # 配置加载、遥测日志路径等运行期开关
  integrations/    # 本地 LLM 客户端与 rLLM 训练对接层
  memory/          # 代码图子图维护与线性化工具
  runtime/         # SandboxRuntime，封装 RepoEnv / R2E / docker 三种后端
scripts/           # 运行代理、注册数据集、启动训练的 CLI
tests/             # rLLM 辅助测试（例如奖励管理器加载路径）
```

各目录的职责与详细说明可参考：

- [`docs/graph_planner_architecture_pipeline.md`](docs/graph_planner_architecture_pipeline.md)：架构分层、容器运行流以及 CGM / rLLM 训练流水线的统一参考。
- [`docs/scripts_and_tests_guide.md`](docs/scripts_and_tests_guide.md)：脚本与测试入口、ACI/Git/Lint/Test 的实现来源。
- [`docs/pain-points.md`](docs/pain-points.md)：记录 Contract-as-Code、补丁落盘与训练集成的痛点与解决方案。

## 快速上手

1. **安装依赖**
   ```bash
   pip install -e .             # 安装 Graph Planner 自身
   pip install -e ./R2E-Gym     # 安装 RepoEnv / DockerRuntime 依赖
   ```
   R2E-Gym 使用 `pyproject.toml` 管理依赖，`pip install -e ./R2E-Gym` 会自动拉取所需包；如需与官方流程保持一致，可按照 `R2E-Gym/README.md` 中的 `uv sync` 步骤进行高级安装。若需要运行本地 LLM / CGM，请在 `.aci/config.json` 中填写对应的 endpoint、model、API Key 等字段。

2. **准备训练/评测数据集**
   ```bash
   PYTHONPATH=. python scripts/prepare_datasets.py \
     --r2e-dataset R2E-Gym/R2E-Gym-Lite \
     --swebench-dataset princeton-nlp/SWE-bench_Verified
   ```
  `prepare_datasets.py` 会把 Hugging Face 上的 R2E-Gym / SWE-bench 数据集转换成 Graph Planner 所需的 JSON/JSONL 结构，并在 `datasets/` 下生成对应的任务文件、`instances/*.json` 以及 docker manifest。脚本同时支持 `--skip-r2e`、`--skip-swebench`、`--prepull-*` 等参数，便于按需刷新或预拉容器。

3. **运行规则代理冒烟**
   ```bash
   PYTHONPATH=. python scripts/run_rule_agent.py \
     --backend repoenv \
     --ds-json /path/to/your_r2e_dataset.jsonl \
     --max-steps 6 \
     --report smoke_report.json
   ```
   请将 `--ds-json` 指向已经准备好的 R2E-Gym（或自定义）任务 JSONL。脚本会生成 `smoke_report.json` 和 `logs/test_runs.jsonl`，便于分析修复轨迹。

4. **启动强化学习训练（需要 rLLM + Docker 环境）**
   ```bash
   PYTHONPATH=. python scripts/train_planner_grpo.py \
     --config configs/experiments/planner_grpo_4gpu.yaml \
     --print-config
   ```
  当前仓库仅保留 `configs/experiments/planner_grpo_4gpu.yaml` 作为默认 GRPO 配置，脚本会自动解析其中的模型、数据集与环境变量设置。如需调整路径或超参，可使用 `--overrides trainer.total_epochs=10 data.train_batch_size=4` 形式的 OmegaConf dotlist 覆写。添加 `--dry-run` 可以在应用完覆写后立即退出，便于审计配置。更多说明见 [`docs/runbook.md`](docs/runbook.md)。

5. **合同冒烟检查**
   ```bash
   PYTHONPATH=. python scripts/validate_contracts.py
   ```
   该脚本会调用解析器与补丁校验器，以确保 Planner/CGM 的协议未发生漂移。

## 文档索引

- [`docs/graph_planner_architecture_pipeline.md`](docs/graph_planner_architecture_pipeline.md)：端到端架构、RepoEnv 冒烟指引与训练命令速查。
- [`docs/runbook.md`](docs/runbook.md)：rLLM 训练/评估配置、YAML-only 模式、并行预检与 W&B 监控指南。

若需进一步了解 ACI 工具链与 Git 封装的使用方式，请参见 `docs/scripts_and_tests_guide.md` 中的“ACI / Git / Lint / Test 的实现来源”章节。

