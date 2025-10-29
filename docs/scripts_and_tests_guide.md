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
| `prepare_datasets.py` | 下载并转换 R2E-Gym / SWE-bench 数据，生成 Graph Planner 兼容的 JSON/JSONL、manifest 与实例文件。 | `graph_planner.datasets`, `graph_planner.runtime.containers` |
| `run_rule_agent.py` | 统一的沙箱运行入口。按照命令行参数构建 `PlannerEnv`，选择规则或本地 LLM 规划器，并在 RepoEnv / Docker / FakeSandbox 后端之间切换。支持 `--report` 输出轨迹日志。 | `graph_planner.env.planner_env`, `graph_planner.runtime.sandbox`, `graph_planner.agents.common.chat`, `graph_planner.infra.config` |
| `train_planner_grpo.py` | 以单一 YAML (`configs/experiments/planner_grpo_4gpu.yaml`) 驱动 Planner-only GRPO 训练，自动注册 JSONL、预拉容器并配置 Ray runtime。 | `graph_planner.integrations.rllm`, `omegaconf`, `ray`, `graph_planner.infra.config` |
| `eval_planner_grpo.py` | 复用训练脚本的配置解析逻辑，加载指定 checkpoint 执行验证流程。 | `graph_planner.integrations.rllm`, `omegaconf`, `ray` |
| `register_graphplanner_dataset.py` | 将 RepoEnv 任务描述注册到 rLLM 数据集仓库，便于训练时引用。 | `graph_planner.integrations.rllm.dataset`, `R2E-Gym` |
| `validate_contracts.py` / `validate_patches.py` | 校验 Planner/CGM 协议与补丁结构，防止输出格式漂移。 | `graph_planner.agents.rule_based`, `graph_planner.aci.guard` |

- `prepare_datasets.py` 支持 `--skip-*` 与 `--prepull-*` 参数，可一次性生成训练/评测所需的 JSONL、实例与 docker manifest。【F:scripts/prepare_datasets.py†L12-L86】【F:scripts/prepare_datasets.py†L135-L214】
- `train_planner_grpo.py` 负责加载 YAML、注册数据、构造 Ray runtime，并将 Planner/CGM 路径注入环境变量后启动 GRPO 训练循环。【F:scripts/train_planner_grpo.py†L322-L468】
- `eval_planner_grpo.py` 复用了 `train_planner_grpo` 的配置处理，额外要求 `--ckpt` 指向待评估的 checkpoint 目录。

## 测试目录（`tests/`）

| 文件 | 核心覆盖点 | 说明 |
| --- | --- | --- |
| `test_cgm_adapter.py` | 校验 CGM 打补丁适配层的本地后备与远程调用路径，确保规划器调用 CGM 时能够生成正确补丁。 | 覆盖 `graph_planner.agents.rule_based.cgm_adapter` 的异常兜底、请求参数序列化等逻辑。 |
| `test_rule_agent_pipeline.py` | 驱动 FakeSandbox 模拟完整的计划→记忆→打补丁→提交流程，验证规则代理、遥测与日志写入（`logs/test_runs.jsonl`）的行为。 | 强调对 `repair_trace` 的记录：阅读片段、补丁 diff、命令执行与最终文件内容。 |

- `test_cgm_adapter.py` 验证 CGM 适配器在本地兜底与远程请求路径下能返回预期补丁并携带正确的 API 参数。【F:tests/test_cgm_adapter.py†L1-L88】
- `test_rule_agent_pipeline.py` 模拟完整容器交互，覆盖补丁应用、测试执行与测记录，确保规则策略可在无 Docker 环境下回归。【F:tests/test_rule_agent_pipeline.py†L13-L199】

## ACI / Git / Lint / Test 的实现来源

- **ACI 工具链（`aci/`）**：
  - `aci/tools.py` 提供查看、搜索、编辑、lint、测试等 CLI 操作的统一封装。优先调用项目内实现，缺省回退到宿主机已有的工具。
  - `aci/git_tools.py` 封装分支、提交、回滚、diff 等 Git 操作，统一返回 `AciResp` 结构，方便在 CLI 与 API 中复用。
  - `aci/guard.py` 负责补丁护栏校验与决策清洗逻辑，被 `PlannerEnv` 与外部代理共同调用，以保持编辑窗口、预算等策略约束一致。

- **Git 操作**：仓库未依赖 R2E 提供的 Git 管理，所有交互均通过 `aci/git_tools.py` 调用系统 `git`。

- **Lint 与 Test**：
  - `graph_planner/runtime/sandbox.py` 定义 `SandboxRuntime` 抽象，并在 `run_lint`、`run_tests` 中调用我们的本地实现（如 `ruff`、`pytest`）。【F:graph_planner/runtime/sandbox.py†L210-L264】
  - 当选择 RepoEnv / R2E 后端时，容器调度由 R2E 组件处理，但实际 lint/test 命令仍出自本仓库，实现与普通文件系统一致。【F:graph_planner/runtime/sandbox.py†L69-L208】

- **与 R2E 的关系**：
  - RepoEnv / Docker 运行时通过 `graph_planner.runtime.sandbox.SandboxRuntime` 的不同分支（`repoenv`、`r2e`、`docker`）对接 R2E-Gym，利用其任务定义和容器封装。【F:graph_planner/runtime/sandbox.py†L62-L208】
  - 除沙箱后端外，ACI、Git、Lint、Test 逻辑均是仓库自研模块，可在离线或无容器环境下工作。

## 推荐使用流程

1. **本地端到端演练**：在具备 Docker 的环境下执行
   ```bash
   PYTHONPATH=. python scripts/run_rule_agent.py \
     --backend repoenv \
     --ds-json /path/to/your_r2e_dataset.jsonl \
     --max-steps 6 \
     --report smoke_report.json
   ```
   生成的 `smoke_report.json` 与 `logs/test_runs.jsonl` 有助于复盘代理行为。

2. **回归测试**：
   ```bash
   PYTHONPATH=. pytest tests -q
   ```
   若依赖项缺失，可先安装 `R2E-Gym` 或使用 `pip install -e ./R2E-Gym` 完成补齐。

3. **训练任务**：使用 `scripts/train_planner_grpo.py --config configs/experiments/planner_grpo_4gpu.yaml` 启动 GRPO 训练；需要覆盖特定路径或超参时，通过 `--overrides key=value` 追加 dotlist，运行前可以配合 `--print-config` / `--dry-run` 进行审计。【F:scripts/train_planner_grpo.py†L371-L424】

通过以上梳理，贡献者可以快速理解脚本入口、回归保障与基础设施封装，并在需要时跳转到架构总览文档获取端到端 pipeline 与命令速查。
