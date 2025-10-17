# 脚本与测试总览

本文档总结 `scripts/` 与 `tests/` 目录的主要文件职责，并说明当前仓库中 ACI 工具链、Git 操作封装、Lint 与 Test 的实现来源，帮助贡献者快速了解可执行入口与回归保障。

## 脚本目录（`scripts/`）

| 文件 | 作用 | 主要依赖 |
| --- | --- | --- |
| `run_rule_agent.py` | 统一的沙箱运行入口。按照命令行参数构建 `PlannerEnv`，选择规则或本地 LLM 规划器，并在 RepoEnv / Docker / FakeSandbox 后端之间切换。支持 `--report` 输出轨迹日志。 | `env/planner_env.py`, `runtime/sandbox`, `agents.common`, `infra.config` |
| `register_graphplanner_dataset.py` | 将 RepoEnv 任务描述注册到 rLLM 数据集仓库，便于训练时引用。 | `integrations.rllm.datasets`, `R2E-Gym` |
| `train_graphplanner_rllm.py` | 使用 rLLM/VERL 的 PPO 管道训练规划器，拼接自定义环境、代理与数据集。 | `integrations.rllm`, `agents.model_based`, `R2E-Gym` |

> 早期的 `smoke_test*.py` 已删除，所有端到端演练统一从 `run_rule_agent.py` 启动。

## 测试目录（`tests/`）

| 文件 | 核心覆盖点 | 说明 |
| --- | --- | --- |
| `test_cgm_adapter.py` | 校验 CGM 打补丁适配层的本地后备与远程调用路径，确保规划器调用 CGM 时能够生成正确补丁。 | 覆盖 `agents.common.cgm_adapter` 的异常兜底、请求参数序列化等逻辑。 |
| `test_rule_agent_pipeline.py` | 驱动 FakeSandbox 模拟完整的计划→记忆→打补丁→提交流程，验证规则代理、遥测与日志写入（`logs/test_runs.jsonl`）的行为。 | 强调对 `repair_trace` 的记录：阅读片段、补丁 diff、命令执行与最终文件内容。 |

## ACI / Git / Lint / Test 的实现来源

- **ACI 工具链（`aci/`）**：
  - `aci/tools.py` 提供查看、搜索、编辑、lint、测试等 CLI 操作的统一封装。优先调用项目内实现，缺省回退到宿主机已有的工具。
  - `aci/git_tools.py` 封装分支、提交、回滚、diff 等 Git 操作，统一返回 `AciResp` 结构，方便在 CLI 与 API 中复用。

- **Git 操作**：仓库未依赖 R2E 提供的 Git 管理，所有交互均通过 `aci/git_tools.py` 调用系统 `git`。

- **Lint 与 Test**：
  - `runtime/sandbox.py` 定义 `SandboxRuntime` 抽象，并在 `run_lint`、`run_tests` 中调用我们的本地实现（如 `ruff`、`pytest`）。
  - 当选择 RepoEnv / R2E 后端时，容器调度由 R2E 组件处理，但实际 lint/test 命令仍出自本仓库，实现与普通文件系统一致。

- **与 R2E 的关系**：
  - RepoEnv / Docker 运行时通过 `runtime/sandbox.repoenv`、`runtime/sandbox.docker_runtime` 对接 R2E-Gym，利用其任务定义和容器封装。
  - 除沙箱后端外，ACI、Git、Lint、Test 逻辑均是仓库自研模块，不依赖 R2E 提供的实现。

## 推荐使用流程

1. **本地端到端演练**：在具备 Docker 的环境下执行
   ```bash
   PYTHONPATH=. python scripts/run_rule_agent.py \
     --backend repoenv \
     --ds-json config/r2e_ds_repoenv_sample.json \
     --max-steps 6 \
     --report smoke_report.json
   ```
   生成的 `smoke_report.json` 与 `logs/test_runs.jsonl` 有助于复盘代理行为。

2. **回归测试**：
   ```bash
   PYTHONPATH=. pytest tests -q
   ```
   若依赖项（如 `pydantic`）缺失，可先安装 `R2E-Gym` 提供的环境。

3. **训练任务**：使用 `scripts/train_graphplanner_rllm.py` 启动 rLLM 强化学习，需提前配置本地部署的规划模型与 CGM 端点。

通过以上梳理，贡献者可以快速理解脚本入口、回归保障与基础设施封装，从而在中文文档中定位具体代码并开展开发。
