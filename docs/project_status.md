# Graph Planner 项目现状汇总

本文件记录当前代码仓库的整体目标、已实现能力、仍需补齐的信息以及下一步建议，方便后续协作者快速了解进度并补全训练链路。

## 项目实现目标

Graph Planner 致力于复现 CGM + Planner 的双智能体代码修复流程：

- **Planner 决策模型**：负责解析容器状态、维护记忆、检索代码子图，并在需要时调用补丁模型；能够基于 R2E/RepoEnv 任务进行多步决策。
- **CGM 补丁模型**：在 Planner 指示下生成代码补丁；目前仓库保留了规则策略与 CGM 适配器的接口，方便在本地部署 CGM 时直接替换。
- **强化学习训练**：通过 rLLM/VERL 的 PPO 管线对 Planner 模型进行 on-policy/online 训练，训练过程中可以接入真实容器（RepoEnv/R2E 数据集）。
- **Rule-based 串联**：为了便于集成测试，规则策略仍可覆盖全部流程，确保在本地模型尚未接入时依旧能跑通完整链路并记录修复日志。

## 当前已有的核心框架

当前仓库已经落实以下框架与配套能力，确保“Planner + CGM” 双智能体流程可以在本地串联、在 rLLM 上训练，并通过规则策略兜底：

1. **Graph Planner 统一包结构**（`graph_planner/`）
   - `graph_planner.agents`：同时保留规则策略（`rule_based`）与本地模型策略（`model_based`），共享 `agents.common.text_protocol` 定义的 `<function=...>` 文本轨迹协议，方便替换决策模型。
   - `graph_planner.env`：`PlannerEnv` 封装 R2E/RepoEnv 任务生命周期，负责动作解析、奖励计算与遥测输出。
   - `graph_planner.runtime`：`SandboxRuntime` 调度 FakeSandbox、RepoEnv、docker-py 三种后端，并统一将 lint/test 结果写入日志。

2. **rLLM 训练集成**
   - `graph_planner.integrations.rllm.agent.GraphPlannerRLLMAgent` 将环境观测整理成系统提示，解析模型输出的 `<function=...>` 区块，并在解析失败时回退到规则策略。
   - `graph_planner.integrations.rllm.env.GraphPlannerRLLMEnv` 将 `PlannerEnv` 暴露给 rLLM，封装奖励、终止条件与 RepoEnv/Sandbox 初始化逻辑。
   - `graph_planner.integrations.rllm.registry` + `graph_planner.infra.vendor.ensure_rllm_importable` 负责定位子模块 rLLM 并在 Hydra/Verl 注册 Graph Planner 自定义 agent/env。
   - `scripts/train_graphplanner_rllm.py` 读取 rLLM PPO 配置，覆盖数据路径与训练超参，可选择性关闭规则回退。

3. **本地运行与冒烟测试脚手架**
   - `scripts/run_rule_agent.py` 支持在 FakeSandbox、RepoEnv、docker 模式下运行规则或本地 LLM 策略，方便训练前验证模型行为。
   - `tests/test_rule_agent_pipeline.py` 通过 FakeSandbox 模拟完整修复流程，并把 `repair_trace` 写入 `logs/test_runs.jsonl`，用于回归测试和日志示例。

4. **补丁生成与护栏**
   - `graph_planner.agents.rule_based.cgm_adapter` 暴露 CGM 的统一接口，既可调用真实模型，也能在缺席时回退到规则补丁。
   - `aci.guard` 提供补丁护栏校验，确保 Planner 与 CGM 输出的 diff 满足安全约束。
   - `aci` 工具链实现文件查看、编辑、lint/test 等基础能力，供 Sandbox 与训练流程复用。

5. **文档与操作指南**
   - `docs/graph_planner_architecture_pipeline.md` 汇总项目架构、模块职责以及 CGM/rLLM 训练运行 pipeline，其中包含最新的冒烟测试与训练脚本速查命令（参见第 4.3 节）。【F:docs/graph_planner_architecture_pipeline.md†L120-L214】
   - `docs/scripts_and_tests_guide.md` 概述脚本入口、测试回归与 ACI/Git/Lint 实现来源。
   - `README.md` 与本文件提供整体架构、配置方法与现状记录。

## 尚需补齐的关键条件

在当前代码基础上，要真正启动强化学习训练还缺少以下条件或信息：

1. **依赖安装**
   - `rllm`、`ray`、`r2egym` 等可选依赖未随仓库自动安装；若不提前装好，训练脚本会报 `ImportError`。
   - 测试环境缺少 `pydantic` 会导致 `pytest` 失败，需要在真实环境中补齐。

2. **容器运行环境**
   - RepoEnv/R2E 后端需要可访问的 Docker 守护进程，本开发容器无法启用 Docker；实际训练需在具备 Docker 权限的主机上执行。
   - 样例数据集仅包含演示条目，尚未准备大规模训练任务与对应镜像。

3. **本地模型接入信息**
   - `.aci/config.json` 中 `planner_model`、`cgm` 默认均为禁用状态，需要提供实际的本地部署端点、模型名、鉴权 token 等信息。
   - 虽然仓库已预留 `models/Qwen3-14B/` 与 `models/CodeFuse-CGM/` 目录作为默认 checkpoint 路径，但仍需在这些目录内放入真实权重与 tokenizer 才能执行训练或推理。

4. **数据与奖励配置**
   - 尚未定义更多 RepoEnv/R2E 任务的奖励 shaping、终止条件调整等策略；如需与真实训练目标对齐，需要进一步扩充。

## 建议的补齐步骤

1. **环境准备**
   - 在目标机器上安装 `rllm>=0.4`、`ray>=2.9`、`r2egym` 以及仓库所需的 Python 依赖；确认 `pytest` 可以在启用 `pydantic` 后成功运行。
   - 启动 Docker 守护进程，并验证当前用户对 `/var/run/docker.sock` 具备访问权限。

2. **数据与镜像就绪**
   - 根据训练需求编写 RepoEnv/R2E JSONL 任务描述，并在 `config/` 或 `datasets/` 目录下维护。
   - 预先拉取或构建所有任务对应的 Docker 镜像，保证 `scripts/run_rule_agent.py` 可以顺利启动容器。
   - 使用 `scripts/register_graphplanner_dataset.py` 将数据集注册到 rLLM/Verl，确认生成的 parquet 文件可被训练脚本读取。

3. **模型服务配置**
   - 将 Planner LLM、CGM 的本地推理服务端点填入 `.aci/config.json`（或使用环境变量覆盖），并确保接口遵循 OpenAI / CGM 适配器协议。
   - 通过 `scripts/run_rule_agent.py --agent llm` 进行冒烟测试，确认模型输出合法的 `<function=...>` 块，补丁流程可走通。

4. **训练启动**
   - 将基础 checkpoint 拷贝至 `models/Qwen3-14B/`（以及需要的 `models/CodeFuse-CGM/`），脚本会自动使用这些路径作为默认模型目录。
   - 根据实验计划编辑 `configs/experiments/*.yaml`（或使用 `--config-file` 指向自定义 YAML），再运行 `scripts/train_graphplanner_rllm.py`；需要临时覆写时可继续使用 CLI（例如 `--dataset`、`--total-epochs`），若希望完全依赖 YAML 则追加 `--yaml-only`。
   - 在训练过程中关注 `logs/` 与 Ray dashboard，确保奖励、轨迹记录符合预期。

## 缺失信息登记

- Planner 模型及 CGM 的具体部署方式、接口鉴权信息（需项目维护者补充）。
- 真实训练数据集的镜像地址、任务描述文件、奖励设计（目前仅有样例占位）。
- 基础策略的 checkpoint 路径及模型规格（PPO 初始化所需）。

当以上信息补齐后，可直接沿用现有脚本与模块完成端到端训练。
