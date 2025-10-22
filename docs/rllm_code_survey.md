# rLLM 代码调研报告

## 1. 总览

`graph_planner/rllm` 子模块提供了一个围绕 `BaseAgent`/`BaseEnv` 接口构建的强化学习框架，涵盖代理执行引擎、数据注册、Verl PPO 训练器、奖励定义以及大量示例脚本。顶层包导出核心的 `BaseAgent`、`Action`、`Trajectory` 等类型，便于下游项目直接使用这些抽象。【F:rllm/rllm/__init__.py†L1-L13】

下文按功能域划分，说明各目录的职责、在 rLLM 内部的引用关系，以及 Graph Planner 当前的使用情况。

## 2. 核心运行时代码

| 模块 | 功能概述 | 在 rLLM 中的用途 | Graph Planner 使用情况 |
| --- | --- | --- | --- |
| `agents/agent.py` | 定义 `Step`、`Action`、`Trajectory`、`Episode` 数据结构，并给出 `BaseAgent` 抽象接口。【F:rllm/rllm/agents/agent.py†L1-L132】【F:rllm/rllm/agents/agent.py†L132-L213】 | 所有内置代理都继承 `BaseAgent`，执行引擎依赖这些结构读取聊天历史与奖励。【F:rllm/rllm/engine/agent_execution_engine.py†L1-L83】 | Graph Planner 代理封装继承自 `BaseAgent`，复用 `Trajectory` 与 `Step` 记录训练轨迹。【F:graph_planner/integrations/rllm/agent.py†L1-L113】【F:graph_planner/integrations/rllm/agent.py†L113-L202】 |
| `environments/base/base_env.py` | 提供 Gym 风格的 `BaseEnv` 抽象，约定 `reset/step/from_dict` 等接口，支持多线程安全检查。【F:rllm/rllm/environments/base/base_env.py†L1-L66】 | AgentExecutionEngine 通过统一接口与各类环境交互，Workflow 也依赖它进行 rollout。【F:rllm/rllm/engine/agent_execution_engine.py†L1-L83】【F:rllm/rllm/workflows/workflow.py†L1-L78】 | Graph Planner 环境适配层继承 `BaseEnv` 暴露 `PlannerEnv`，并扩展奖励缩放、惩罚与 Repo 操作限制配置。【F:graph_planner/integrations/rllm/env.py†L1-L143】【F:graph_planner/integrations/rllm/env.py†L143-L218】 |
| `engine/agent_execution_engine.py` | 异步代理执行引擎，负责并发创建代理/环境、调用 OpenAI/Verl rollout 引擎、跟踪超时与步长限制。【F:rllm/rllm/engine/agent_execution_engine.py†L1-L140】【F:rllm/rllm/engine/agent_execution_engine.py†L140-L212】 | 训练管线与示例脚本都通过该引擎驱动模型推理，支持多线程环境交互与多候选采样。【F:rllm/rllm/engine/agent_execution_engine.py†L83-L160】 | Graph Planner 通过 Verl PPO 训练器间接使用该引擎执行异步 rollout。【F:rllm/rllm/trainer/verl/agent_ppo_trainer.py†L1-L88】 |
| `engine/rollout/openai_engine.py` | 实现基于 OpenAI API 的 rollout 引擎，支持 tokenizer 约束、工具调用以及推理重试逻辑。【F:rllm/rllm/engine/rollout/openai_engine.py†L1-L120】【F:rllm/rllm/engine/rollout/openai_engine.py†L120-L212】 | 作为默认推理后端，在 AgentExecutionEngine 中按 `engine_name="openai"` 使用。【F:rllm/rllm/engine/agent_execution_engine.py†L99-L140】 | Graph Planner 当前主要使用 Verl 推理通路，但在需要时也可复用该实现。 |
| `workflows/workflow.py` | 定义多回合 Workflow 框架、终止原因、轨迹聚合与奖励整形流程。【F:rllm/rllm/workflows/workflow.py†L1-L120】【F:rllm/rllm/workflows/workflow.py†L120-L198】 | 多种 workflow（单轮、多轮、累积）基于该抽象实现任务调度，Ray 训练器可选使用 Workflow 模式。【F:rllm/rllm/trainer/agent_trainer.py†L1-L60】 | Graph Planner 目前未开启 workflow 模式，若未来需要可通过配置启用。 |
| `parser/__init__.py` & `system_prompts.py` | 聚合聊天模板解析器、工具参数解析器，并维护系统提示/格式化模板常量。【F:rllm/rllm/parser/__init__.py†L1-L23】【F:rllm/rllm/system_prompts.py†L1-L60】 | AgentExecutionEngine 通过 `ChatTemplateParser` 限制 prompt 长度，数据脚本复用系统提示生成任务描述。【F:rllm/rllm/engine/agent_execution_engine.py†L52-L83】【F:rllm/rllm/data/utils.py†L1-L39】 | Graph Planner 自定义了提示协议，但仍可在需要时复用 Qwen/Deepseek 模板解析器。 |

## 3. 数据与训练栈

| 模块 | 功能概述 | 在 rLLM 中的用途 | Graph Planner 使用情况 |
| --- | --- | --- | --- |
| `data/dataset.py` | 提供 `Dataset` 张量接口、Parquet 注册、Verl 后处理与 `DatasetRegistry` 管理。【F:rllm/rllm/data/dataset.py†L1-L97】【F:rllm/rllm/data/dataset.py†L97-L214】 | 训练与评测脚本通过注册表定位数据，Verl 训练器读取 `_verl.parquet` 作为输入。【F:rllm/rllm/trainer/agent_trainer.py†L24-L76】 | Graph Planner 将 R2E-Gym JSONL 正规化后注册到该 Registry，供 PPO/GRPO 使用。【F:graph_planner/integrations/rllm/dataset.py†L1-L116】【F:graph_planner/integrations/rllm/dataset.py†L116-L137】 |
| `data/utils.py` | 加载官方 Math/Code 数据集，拼接系统提示模板。【F:rllm/rllm/data/utils.py†L1-L52】【F:rllm/rllm/data/utils.py†L52-L85】 | 官方脚本用于准备基准任务、示例训练数据。 | Graph Planner 未直接复用，仅参考其格式。 |
| `trainer/agent_trainer.py` | 高层封装，负责初始化 Ray、挂接 Verl PPO 训练任务并传入自定义 agent/env。【F:rllm/rllm/trainer/agent_trainer.py†L1-L79】 | CLI/示例使用该入口统一训练流程，内部调用 Verl `TaskRunner`。 | Graph Planner 训练脚本直接实例化 `AgentTrainer` 并传入自定义 agent/env/config。【F:scripts/train_graphplanner_rllm.py†L600-L619】 |
| `trainer/env_agent_mappings.py` | 提供安全导入及 agent/env/workflow 映射字典，支持配置名到类的绑定。【F:rllm/rllm/trainer/env_agent_mappings.py†L1-L35】 | 默认训练配置依赖该映射将字符串解析为类。 | Graph Planner 在启动时把自定义 agent/env 注入这些映射，便于配置引用。【F:graph_planner/integrations/rllm/registry.py†L1-L38】 |
| `trainer/verl/agent_ppo_trainer.py` | Verl PPO 主体，构造异步执行引擎、并行创建 env/agent、聚合梯度并调用 Verl 核心算法。【F:rllm/rllm/trainer/verl/agent_ppo_trainer.py†L1-L96】【F:rllm/rllm/trainer/verl/agent_ppo_trainer.py†L96-L180】 | rLLM 默认的 PPO/GRPO 训练器，封装 step-level/episode-level 优势计算。 | Graph Planner 通过 `AgentTrainer` 间接使用该类执行 rollout 与训练。 |
| `trainer/verl/ray_runtime_env.py` & `patches/verl_patch_hook.py` | 配置 Ray runtime 环境变量，并在 Verl rollout worker 上打补丁确保 ZeroMQ 端口分配安全。【F:rllm/rllm/trainer/verl/ray_runtime_env.py†L1-L27】【F:rllm/rllm/patches/verl_patch_hook.py†L1-L31】 | 默认训练启动 Ray 时自动注入这些设置。 | Graph Planner 继承这一机制，保持与上游一致的 Ray 运行环境。 |
| `rewards/reward_fn.py` | 定义数学/搜索/代码奖励函数并暴露统一接口。【F:rllm/rllm/rewards/reward_fn.py†L1-L71】 | 示例环境调用这些函数计算奖励。 | Graph Planner 自行定义奖励，但可复用 `RewardOutput` 结构。 |
| `utils.py` | 提供 `compute_pass_at_k`、`save_trajectories` 等评测辅助函数。【F:rllm/rllm/utils.py†L1-L34】 | 官方示例用于统计通过率、保存轨迹。 | 可在 Graph Planner 评测脚本中复用。 |

## 4. 工具、集成与其他目录

| 模块 | 功能概述 | 在 rLLM 中的用途 | Graph Planner 使用情况 |
| --- | --- | --- | --- |
| `tools/` | `tool_base.py` 定义工具调用协议，`tools.utils` 负责把 Python 函数转为 JSON schema。【F:rllm/rllm/tools/tool_base.py†L1-L88】【F:rllm/rllm/tools/tool_base.py†L88-L152】 | ToolAgent 以及 OpenAI Engine 工具调用依赖该接口。【F:rllm/rllm/engine/rollout/openai_engine.py†L1-L60】 | Graph Planner 当前采用自定义文本动作协议，暂未直接引用。 |
| `integrations/` | 提供与终端应用（Terminal、Strands 等）的模型封装与工具注册，供示例脚本调用。【F:rllm/examples/terminal/run_terminus.py†L1-L40】【F:rllm/examples/strands/run_strands.py†L1-L20】 | 示例中演示如何接入外部产品。 | 暂未使用，可作为未来扩展参考。 |
| `patches/` | 存放针对 Verl/vLLM 等依赖的运行时补丁（见上）。【F:rllm/rllm/patches/verl_patch_hook.py†L1-L31】 | 训练时通过 Ray runtime hook 自动生效。 | Graph Planner 随 `ensure_rllm_importable()` 复用这些补丁。 |
| `trajectory_visualizer.py`、`misc.py`、`globals.py` | 提供轨迹可视化、彩色打印、全局常量等辅助工具。【F:rllm/rllm/misc.py†L1-L80】【F:rllm/rllm/trajectory_visualizer.py†L1-L60】 | 供调试与日志输出使用。 | 部分功能（彩色日志）被 Graph Planner 间接引用。 |
| `docs/`、`examples/`、`scripts/`、`tests/` | 官方文档、示例、CLI 与单测，演示如何使用 Agent/Env/Trainer。【F:rllm/examples/math_tool/train_math_with_tool.py†L1-L20】【F:rllm/scripts/data/code_dataset.py†L1-L25】 | 作为教程与回归测试存在。 | Graph Planner 目前仅参考其脚本结构，自研 CLI 替代。 |

## 5. Verl 子模块

rLLM 仓库同时内置 Verl 项目，用于提供 PPO/GRPO 算法实现与分布式执行器。`rllm/rllm/trainer/verl` 直接依赖 Verl 的 `DataProto`、Ray PPO 训练器等组件。【F:rllm/rllm/trainer/verl/agent_ppo_trainer.py†L1-L63】Graph Planner 通过 `ensure_rllm_importable()` 注入 Verl 依赖路径，使训练脚本可以直接访问 Verl 算法。【F:graph_planner/infra/vendor.py†L1-L110】

## 6. Graph Planner 当前的依赖情况

* **代理与环境包装**：`GraphPlannerRLLMAgent`/`GraphPlannerRLLMEnv` 分别继承 rLLM 的 `BaseAgent` 与 `BaseEnv`，实现 Planner 文本协议、CGM 集成与奖励缩放逻辑。【F:graph_planner/integrations/rllm/agent.py†L1-L240】【F:graph_planner/integrations/rllm/env.py†L1-L218】
* **数据注册**：训练脚本通过 `ensure_dataset_registered` 将 R2E-Gym JSONL 注册到 rLLM 的 `DatasetRegistry`，并生成 Verl 兼容的 `_verl.parquet`。【F:graph_planner/integrations/rllm/dataset.py†L1-L137】
* **训练入口**：`train_graphplanner_rllm.py` 导入 `AgentTrainer`，在应用配置覆写与资源检查后触发训练；Ray 地址与资源控制沿用 rLLM 默认实现。【F:scripts/train_graphplanner_rllm.py†L600-L619】
* **组件映射**：`register_rllm_components` 将自定义 agent/env 注入 `env_agent_mappings`，保证 Hydra 配置可以通过名称找到对应类。【F:graph_planner/integrations/rllm/registry.py†L1-L38】【F:rllm/rllm/trainer/env_agent_mappings.py†L1-L35】

## 7. 结论与建议

1. Graph Planner 目前主要复用了 rLLM 的 **核心抽象（BaseAgent/BaseEnv）**、**数据注册与 Verl 训练栈**；其余示例/工具可在需要时按需引入。
2. 当升级 rLLM 子模块时，需要重点回归 `AgentExecutionEngine`、`DatasetRegistry`、`AgentTrainer` 等关键接口，以确保现有封装保持兼容。
3. 若未来想使用 rLLM 内置的工具调用或 Workflow 机制，可直接在 Graph Planner 配置中启用对应 parser/engine，无需重复造轮子。
