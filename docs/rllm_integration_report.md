# rLLM 集成评估报告

本文档聚焦 Graph Planner 与 rLLM/VERL 训练栈的衔接情况，说明核心模块、数据流与调试建议。

## 1. 主要组件
- **代理封装**：`GraphPlannerRLLMAgent` 继承 rLLM `BaseAgent`，使用共享的 `agents/common/chat.py` 工具把环境观测压缩为模型提示，解析模型返回的 JSON 并在失败时回退到规则策略。【F:integrations/rllm/agent.py†L1-L159】【F:agents/common/chat.py†L1-L197】
- **环境包装**：`GraphPlannerRLLMEnv` 将 `PlannerEnv` 暴露为 rLLM `BaseEnv` 接口，负责加载任务条目、拼装 `SandboxConfig` 并在每步代理调用后返回奖励与终止信号。【F:integrations/rllm/env.py†L32-L110】【F:env/planner_env.py†L23-L214】
- **数据集注册**：`integrations/rllm/dataset.py` 提供 `ensure_dataset_registered` 与 JSON/JSONL loader，可将 RepoEnv 任务描述标准化后注册到 rLLM 的 `DatasetRegistry`。【F:integrations/rllm/dataset.py†L19-L97】
- **训练入口**：`scripts/train_graphplanner_rllm.py` 读取 rLLM 默认 PPO 配置，注入 Graph Planner 专属的 agent/env 名称与步长参数，可选输出合并后的 Hydra 配置或直接启动 Ray 训练作业。【F:scripts/train_graphplanner_rllm.py†L1-L147】

## 2. 模型与回退策略
- 本地决策模型复用 `LocalLLMPlannerAgent` 中的解析逻辑，通过 `agents/common/chat.py` 的协议确保 rLLM 推理与本地 CLI 行为一致；当模型响应无效时，代理会调用规则策略保证环境仍可推进。【F:agents/model_based/planner.py†L38-L178】【F:agents/common/chat.py†L138-L197】
- 配置中的 `planner_model` 段允许为训练或评估指定本地/远程推理端点；若禁用该段，rLLM 代理仍可运行，但会完全依赖规则 fallback 生成动作。【F:infra/config.py†L60-L176】

## 3. RepoEnv 容器链路
- 训练任务需要提供 `sandbox.r2e_ds_json`，环境包装层会在 `GraphPlannerRLLMEnv._spawn_planner` 中将路径归一化后交给 `PlannerEnv`，后者再通过 `SandboxRuntime` 初始化 RepoEnv 容器并执行补丁/测试循环。【F:integrations/rllm/env.py†L92-L104】【F:runtime/sandbox.py†L69-L143】
- 任务执行过程中的测试结果通过 `infra.telemetry.log_test_result` 记录 JSONL，可在训练失败时复盘模型动作与容器状态。【F:infra/telemetry.py†L31-L39】

## 4. 数据与任务准备
- 示例任务位于 `datasets/graphplanner_repoenv_sample.jsonl`，可结合 `ensure_dataset_registered` 生成 Verl 需要的 `_verl.parquet` 并注册到 rLLM 数据集中；该流程会自动解析 `mounts`、`r2e_ds_json` 的相对路径，适合在远程训练节点运行。【F:integrations/rllm/dataset.py†L32-L71】【F:integrations/rllm/dataset.py†L74-L97】
- 如需自定义任务，可复用相同格式的 JSON/JSONL，确保包含 `issue`、`sandbox` 字段以及合法的 RepoEnv 数据集描述。

## 5. 调试建议
- 训练前可先在本地运行 `scripts/run_rule_agent.py --agent llm` 验证模型解析是否正确，再切换到 rLLM 训练；CLI 与 rLLM 共享相同的动作协议与 CGM 集成，能提前暴露补丁解析问题。【F:scripts/run_rule_agent.py†L54-L133】【F:agents/rule_based/cgm_adapter.py†L156-L210】
- 若需离线排查模型输出，可启用 `GraphPlannerRLLMAgent` 的规则回退并检查 `trajectory` 日志，确保无效响应被记录到 `fallback_reason` 字段后再做进一步调参。【F:integrations/rllm/agent.py†L81-L159】
