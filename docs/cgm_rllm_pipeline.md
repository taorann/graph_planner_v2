# CGM 与 rLLM 训练/推理架构说明

## 摘要 / Summary

本说明梳理 Graph Planner 仓库中自研的 CGM 集成层以及 rLLM 强化学习链路，帮助团队理解各模块的职责与调用顺序。The document explains how the locally implemented CGM dataset/formatter/inference stack plugs into rLLM-based PPO training so that either the planner agent or the CGM patcher can be fine-tuned without relying on the upstream CodeFuse repository.

## 模块分层

1. **`graph_planner/integrations/codefuse_cgm`** – 提供数据结构与格式化工具：
   - `data.py` 定义 `CGMExample`/`CodeFuseCGMDataset`，负责解析 JSON/JSONL 样本，读取计划、子图、片段等字段。
   - `formatting.py` 内的 `GraphLinearizer`、`SnippetFormatter`、`ConversationEncoder` 将图节点和代码片段线性化，并构造训练/推理所需的聊天模板输入。
   - `training.py` 提供 `CGMBatchCollator` 与 `CodeFuseCGMTrainer`，实现最小化的监督微调循环（动态 padding、梯度累积、余弦调度）。
   - `inference.py` 暴露 `CodeFuseCGMGenerator`，以 Hugging Face checkpoint 执行本地推理。
   - `client.py` 实现 `CodeFuseCGMClient`，在需要远程服务时通过 HTTP 访问官方 CGM。

2. **`graph_planner/integrations/local_llm`** – `HuggingFaceChatClient` 将任意 Causal LM 封装为聊天接口，为 Planner 模型推理提供本地选项。

3. **`graph_planner/integrations/rllm`** – 与 rLLM 强化学习栈交互：
   - `agent.py` 定义 `GraphPlannerRLLMAgent`，在 rLLM 侧处理 Planner 环境的 JSON 观察、补丁解析以及规则回退。
   - `cgm_agent.py` 与 `cgm_env.py` 分别包装 CGM agent 与环境，复用 `GraphLinearizer`/`SnippetFormatter` 构造提示并评估补丁质量。
   - `dataset.py` 将本地任务描述注册到 rLLM 的数据集注册表，解析路径、沙箱挂载与分片信息。
   - `__init__.py` 负责懒加载并注册上述组件，防止在缺失 rLLM 依赖时触发导入错误。

4. **`scripts/train_graphplanner_rllm.py`** – 命令行入口，串联数据集注册、模型路径覆盖、Agent/Env 绑定，最终调用 rLLM 的 `AgentTrainer` 进行 PPO 训练。

## 数据与训练流程

1. **准备数据集**：将 Planner 任务或 CGM 示例整理成 JSON/JSONL，字段包括 `prompt`、`answer`、`plan`、`graph`、`snippets` 等。`CodeFuseCGMDataset` 会自动解析这些字段并生成 `CGMExample`。

2. **上下文编排**：
   - 图节点由 `GraphLinearizer` 依据 `serialize_subgraph.py` 输出的结构生成多段可读文本（名称、类型、摘要、代码片段）。
   - 候选片段使用 `SnippetFormatter` 统一成 `path:start-end` + 正文的块状文本。
   - `ConversationEncoder` 将 Issue/Plan/Graph/Snippets 与用户指令整合成聊天消息，并可选输出标签掩码用于训练。

3. **监督微调**：
   - `CGMBatchCollator` 将批次样本编码成 `input_ids`、`attention_mask`、`labels`，自动完成动态 padding 与 prompt token 屏蔽。
   - `CodeFuseCGMTrainer` 载入 Hugging Face 模型、配置优化器/调度器，执行多 epoch 训练，并可按步数触发评估与 checkpoint 持久化。

4. **本地推理**：`CodeFuseCGMGenerator` 复用编码器与格式化器，对任意 `CGMExample` 生成补丁候选；`HuggingFaceChatClient` 则为 Planner 模型提供类似的本地聊天能力。

5. **强化学习**：
   - 使用 `scripts/train_graphplanner_rllm.py` 注册数据集（`dataset.py`），根据 `--agent` 选择 Planner 或 CGM agent/env 组合，并注入模型路径、温度等采样参数。
   - `GraphPlannerRLLMAgent` 在训练过程中处理环境观察、执行规则 fallback（必要时调用 `cgm_adapter` 生成补丁）并输出 JSON 动作。
   - `CGMRLLMAgent`/`CGMRLLMEnv` 负责将 Planner 环境产出的计划、子图、片段包装成 CGM 训练所需的提示，收集奖励并反馈给 rLLM。

6. **远程调用（可选）**：若需要使用官方在线 CGM，`CodeFuseCGMClient` 会将与本地训练一致的数据结构 POST 至远程端点，获取并标准化补丁。

## 运行步骤概览

1. （可选）运行 `python scripts/train_graphplanner_rllm.py --agent planner --dataset <path> --model-path <policy>` 以使用 rLLM 训练 Planner 代理；`--agent cgm` 可切换到 CGM agent。
2. 在本地验证 CGM 模型时，使用 `CodeFuseCGMGenerator.generate(example)` 获取补丁候选。
3. 若需要微调 CGM 模型，则构造 `CGMTrainingConfig` 并调用 `CodeFuseCGMTrainer.train()`。
4. 通过 `graph_planner.integrations.codefuse_cgm` 提供的统一 API，可在不同阶段复用相同的数据编排逻辑，避免与上游脚本产生耦合。

