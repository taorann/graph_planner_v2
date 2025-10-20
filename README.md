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
tests/             # FakeSandbox 测试与 CGM 适配器回归
```

各目录的职责与详细说明可参考：

- [`docs/graph_planner_architecture_pipeline.md`](docs/graph_planner_architecture_pipeline.md)：架构分层、容器运行流以及 CGM / rLLM 训练流水线的统一参考。
- [`docs/scripts_and_tests_guide.md`](docs/scripts_and_tests_guide.md)：脚本与测试入口、ACI/Git/Lint/Test 的实现来源。

## 快速上手

1. **安装依赖**
   ```bash
   pip install -e R2E-Gym  # 提供 RepoEnv、DockerRuntime 及相关依赖
   pip install -r R2E-Gym/requirements-dev.txt
   ```
   如果需要运行本地 LLM / CGM，请在 `.aci/config.json` 中填写对应的 endpoint、model、API Key 等字段。

2. **运行规则代理冒烟**
   ```bash
   PYTHONPATH=. python scripts/run_rule_agent.py \
     --backend repoenv \
     --ds-json config/r2e_ds_repoenv_sample.json \
     --max-steps 6 \
     --report smoke_report.json
   ```
   该脚本会生成 `smoke_report.json` 和 `logs/test_runs.jsonl`，便于分析修复轨迹。

3. **启动强化学习训练（需要 rLLM + Docker 环境）**
   ```bash
  PYTHONPATH=. python scripts/train_graphplanner_rllm.py \
    --agent planner \
    --dataset datasets/r2e_gym/graphplanner_repoenv_train.jsonl \
    --model-path models/qwen3-14b-instruct \
    --cgm-model-path models/codefuse-cgm \
    --print-config
   ```
   如需联动 CGM，可额外传入 `--cgm-model-path`；命令会在真正启动前打印最终 Hydra 配置，便于核对 `trainer.*` 覆写。详细准备步骤见 `docs/graph_planner_architecture_pipeline.md` 的“4.3 Planner / CGM 强化学习”章节。

## 文档索引

- [`docs/github_update_instructions.md`](docs/github_update_instructions.md)：提交前的自检命令、Git 流程与日志要求。
- [`docs/graph_planner_architecture_pipeline.md`](docs/graph_planner_architecture_pipeline.md)：端到端架构、RepoEnv 冒烟指引与训练命令速查。

若需进一步了解 ACI 工具链与 Git 封装的使用方式，请参见 `docs/scripts_and_tests_guide.md` 中的“ACI / Git / Lint / Test 的实现来源”章节。

