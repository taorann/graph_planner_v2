# 脚本与测试总览

本文档总结 `scripts/` 与 `tests/` 目录的主要文件职责，并说明当前仓库中 ACI 工具链、Git 操作封装、Lint 与 Test 的实现来源，帮助贡献者快速了解可执行入口与回归保障。

## 核心包结构（`graph_planner/`）
- **`agents/`**：包含规则策略与本地 LLM 决策器，分别负责状态机驱动的修复流程与模型输出解析，并共享对话协议工具。【F:graph_planner/agents/rule_based/planner.py†L26-L187】【F:graph_planner/agents/model_based/planner.py†L38-L178】【F:graph_planner/agents/common/chat.py†L1-L196】
- **`env/planner_env.py`**：封装 Explore/Memory/Repair/Submit 动作到容器操作的映射，维护奖励、终止条件与工作子图状态。【F:graph_planner/env/planner_env.py†L32-L173】
- **`runtime/sandbox.py`**：统一 RepoEnv、R2E DockerRuntime 与 docker-py 的执行接口，负责拉起容器、运行补丁与记录测试结果。【F:graph_planner/runtime/sandbox.py†L30-L264】
- **`integrations/local_llm` 与 `integrations/rllm`**：前者提供 OpenAI 兼容的本地模型客户端，后者封装 rLLM 的 Agent/Env/Dataset 适配层，供强化学习训练复用。【F:graph_planner/integrations/local_llm/client.py†L15-L152】【F:graph_planner/integrations/rllm/agent.py†L1-L158】【F:graph_planner/integrations/rllm/env.py†L1-L110】
- **`infra/`**：集中配置、遥测日志与其他运行期开关，决定补丁模型、本地 LLM、事件路径等行为。【F:graph_planner/infra/config.py†L24-L176】【F:graph_planner/infra/telemetry.py†L20-L39】

## 脚本目录（`scripts/`）

| 文件 | 作用 | 主要依赖 |
| --- | --- | --- |
| `run_rule_agent.py` | 统一的沙箱运行入口。按照命令行参数构建 `PlannerEnv`，选择规则或本地 LLM 规划器，并在 RepoEnv / Docker / FakeSandbox 后端之间切换。支持 `--report` 输出轨迹日志。 | `graph_planner.env.planner_env`, `graph_planner.runtime.sandbox`, `graph_planner.agents.common.chat`, `graph_planner.infra.config` |
| `register_graphplanner_dataset.py` | 将 RepoEnv 任务描述注册到 rLLM 数据集仓库，便于训练时引用。 | `graph_planner.integrations.rllm.dataset`, `R2E-Gym` |
| `train_graphplanner_rllm.py` | 使用 rLLM/VERL 的 PPO 管道训练规划器，拼接自定义环境、代理与数据集。 | `graph_planner.integrations.rllm`, `graph_planner.agents.model_based`, `R2E-Gym` |

> 早期的 `smoke_test*.py` 已删除，所有端到端演练统一从 `run_rule_agent.py` 启动。

- `run_rule_agent.py` 负责构建 `PlannerEnv`、驱动代理循环并根据配置选择不同容器后端，是规则策略与本地 LLM 共用的 CLI。【F:scripts/run_rule_agent.py†L1-L136】
- `register_graphplanner_dataset.py` 把 RepoEnv 任务注册到 rLLM 数据集，自动规范 `r2e_ds_json`、挂载路径等字段。【F:scripts/register_graphplanner_dataset.py†L1-L47】
- `train_graphplanner_rllm.py` 加载默认 PPO 配置，注入 Graph Planner 专属参数并触发 Ray 训练或配置导出。【F:scripts/train_graphplanner_rllm.py†L1-L147】

## 测试目录（`tests/`）

| 文件 | 核心覆盖点 | 说明 |
| --- | --- | --- |
| `test_cgm_adapter.py` | 校验 CGM 打补丁适配层的本地后备与远程调用路径，确保规划器调用 CGM 时能够生成正确补丁。 | 覆盖 `graph_planner.agents.rule_based.cgm_adapter` 的异常兜底、请求参数序列化等逻辑。 |
| `test_rule_agent_pipeline.py` | 驱动 FakeSandbox 模拟完整的计划→记忆→打补丁→提交流程，验证规则代理、遥测与日志写入（`logs/test_runs.jsonl`）的行为。 | 强调对 `repair_trace` 的记录：阅读片段、补丁 diff、命令执行与最终文件内容。 |

- `test_cgm_adapter.py` 验证 CGM 适配器在本地兜底与远程请求路径下能返回预期补丁并携带正确的 API 参数。【F:tests/test_cgm_adapter.py†L1-L88】
- `test_rule_agent_pipeline.py` 模拟完整容器交互，覆盖补丁应用、测试执行与遥测记录，确保规则策略可在无 Docker 环境下回归。【F:tests/test_rule_agent_pipeline.py†L13-L199】

## ACI / Git / Lint / Test 的实现来源

- **ACI 工具链（`aci/`）**：
  - `aci/tools.py` 提供查看、搜索、编辑、lint、测试等 CLI 操作的统一封装。优先调用项目内实现，缺省回退到宿主机已有的工具。
  - `aci/git_tools.py` 封装分支、提交、回滚、diff 等 Git 操作，统一返回 `AciResp` 结构，方便在 CLI 与 API 中复用。

- **Git 操作**：仓库未依赖 R2E 提供的 Git 管理，所有交互均通过 `aci/git_tools.py` 调用系统 `git`。

- **Lint 与 Test**：
  - `graph_planner/runtime/sandbox.py` 定义 `SandboxRuntime` 抽象，并在 `run_lint`、`run_tests` 中调用我们的本地实现（如 `ruff`、`pytest`）。【F:graph_planner/runtime/sandbox.py†L210-L264】
  - 当选择 RepoEnv / R2E 后端时，容器调度由 R2E 组件处理，但实际 lint/test 命令仍出自本仓库，实现与普通文件系统一致。【F:graph_planner/runtime/sandbox.py†L69-L208】

- **与 R2E 的关系**：
  - RepoEnv / Docker 运行时通过 `graph_planner.runtime.sandbox.SandboxRuntime` 的不同分支（`repoenv`、`r2e`、`docker`）对接 R2E-Gym，利用其任务定义和容器封装。【F:graph_planner/runtime/sandbox.py†L62-L208】
  - 除沙箱后端外，ACI、Git、Lint、Test 逻辑均是仓库自研模块，不依赖 R2E 提供的实现。
  - R2E-Gym 专注于“任务数据集 + 环境调度”两类能力，本身并不提供通用的代码编辑 CLI、Git 自动化或 lint/test 驱动器；这些基础设施在本仓库已有成熟实现，也方便离线或无容器环境下工作，因此继续保留自研方案，减少对额外依赖的耦合。

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
