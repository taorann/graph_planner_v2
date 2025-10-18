# Graph Planner 架构与训练运行全景

> **Summary (English)**
> This document consolidates the previously scattered architecture notes for Graph Planner
> and details how rule-based utilities, CGM integrations, local LLM adapters, and the
> rLLM PPO stack cooperate. It explains the module boundaries, data flow, and end-to-end
> pipelines for both supervised fine-tuning and reinforcement learning.

## 1. 顶层结构 / Repository layering

```
CLI / Scripts / Tests
└── scripts/
    ├── run_rule_agent.py …… 规则 & 本地 LLM 入口
    └── train_graphplanner_rllm.py …… rLLM PPO 训练入口
└── tests/ …… FakeSandbox & 集成测试
    └── tests/test_cgm_adapter.py …… CGM 本地推理覆盖
    └── tests/test_rllm_integration.py …… rLLM 适配层冒烟
└── graph_planner/
    ├── agents/(rule_based|model_based|common)
    ├── env/planner_env.py …… 环境包装（R2E/RepoEnv）
    ├── runtime/sandbox.py …… 容器运行时抽象
    ├── memory/* …… 子图记忆维护
    └── integrations/
        ├── codefuse_cgm …… 数据编排 & 推理
        ├── local_llm …… HuggingFace/OpenAI 接入
        └── rllm …… PPO 训练适配
└── aci/ …… Guard、文件操作、补丁协议
└── datasets/、config/ …… 任务描述与 RepoEnv 配置
```

- `scripts/run_rule_agent.py` 将配置解析、环境初始化、代理循环封装为单一入口，用于离线调试规则/本地 LLM 决策。【F:scripts/run_rule_agent.py†L1-L136】
- `graph_planner.env.planner_env.PlannerEnv` 负责动作编排、奖励计算、记忆同步，是规则代理、本地 LLM 与 rLLM 环境的共同核心。【F:graph_planner/env/planner_env.py†L32-L173】
- `graph_planner.runtime.sandbox.SandboxRuntime` 根据 `backend` 字段切换 FakeSandbox、RepoEnv、docker-py，并统一命令执行与测试流程。【F:graph_planner/runtime/sandbox.py†L32-L214】
- `graph_planner.integrations` 目录收拢所有外部模型/训练栈：CGM（数据/推理/训练）、Hugging Face 聊天客户端、本地 rLLM 适配层。

## 2. 模块职责一览 / Module responsibilities

| 模块 | 关键类/函数 | 职责摘要 |
| --- | --- | --- |
| `agents/rule_based/planner.py` | `PlannerAgent` | 规则策略状态机，驱动图扩展、记忆维护、补丁触发。【F:graph_planner/agents/rule_based/planner.py†L26-L187】 |
| `agents/model_based/planner.py` | `LocalLLMPlannerAgent` | 调用 `local_llm` 聊天客户端解析模型响应，失败时回退规则策略。【F:graph_planner/agents/model_based/planner.py†L38-L178】 |
| `agents/rule_based/cgm_adapter.py` | `CodeFuseCGMGenerator`、`CodeFuseCGMClient`、本地 fallback | 组合 GraphLinearizer/SnippetFormatter/ConversationEncoder，优先调用本地或远端 CGM，失败时打标记补丁。【F:graph_planner/agents/rule_based/cgm_adapter.py†L20-L200】 |
| `integrations/codefuse_cgm/data.py` | `CGMExample`、`CodeFuseCGMDataset` | 解析训练/推理 JSON，加载图、片段、计划并产出结构化样本。【F:graph_planner/integrations/codefuse_cgm/data.py†L1-L210】 |
| `integrations/codefuse_cgm/formatting.py` | `GraphLinearizer`、`SnippetFormatter`、`ConversationEncoder` | 将 `serialize_subgraph` 结果与候选片段线性化，组合聊天模板用于训练/推理。【F:graph_planner/integrations/codefuse_cgm/formatting.py†L69-L199】 |
| `integrations/codefuse_cgm/training.py` | `CGMBatchCollator`、`CodeFuseCGMTrainer` | 构建监督微调所需的 DataLoader、优化器、调度器与训练循环。【F:graph_planner/integrations/codefuse_cgm/training.py†L1-L250】 |
| `integrations/codefuse_cgm/inference.py` | `CodeFuseCGMGenerator` | 加载 Hugging Face checkpoint，以本地方式生成补丁候选。【F:graph_planner/integrations/codefuse_cgm/inference.py†L1-L160】 |
| `integrations/local_llm/hf.py` | `HuggingFaceChatClient` | 将任意 Causal LM 封装为聊天接口，服务 Planner 决策或调试。【F:graph_planner/integrations/local_llm/hf.py†L1-L120】 |
| `integrations/rllm/agent.py` | `GraphPlannerRLLMAgent` | 继承 rLLM `BaseAgent`，复用聊天协议、CGM fallback，并维护训练轨迹。【F:graph_planner/integrations/rllm/agent.py†L56-L200】 |
| `integrations/rllm/env.py` | `GraphPlannerRLLMEnv` | 将 `PlannerEnv` 暴露为 rLLM `BaseEnv`，规范 reset/step/奖励接口。【F:graph_planner/integrations/rllm/env.py†L36-L114】 |
| `integrations/rllm/cgm_agent.py` & `cgm_env.py` | `CGMRLLMAgent`、`CGMRLLMEnv` | 面向 CGM 的 PPO 训练包装，直接监督补丁生成质量。【F:graph_planner/integrations/rllm/cgm_agent.py†L1-L220】【F:graph_planner/integrations/rllm/cgm_env.py†L1-L260】 |
| `integrations/rllm/dataset.py` | `ensure_dataset_registered` | 将 JSON/JSONL 任务注册为 rLLM 可识别的数据集，解析路径与挂载信息。【F:graph_planner/integrations/rllm/dataset.py†L28-L109】 |
| `scripts/train_graphplanner_rllm.py` | CLI helpers | 注入模型路径、注册数据集、绑定 Agent/Env，并委托 rLLM PPO 训练。【F:scripts/train_graphplanner_rllm.py†L32-L200】 |

### 2.1 rLLM 模块导入路径 / rLLM import resolution

- rLLM 源码以 Git 子模块形式放在仓库根目录下的 `rllm/`，其内部又包含 Python 包目录 `rllm/`（双层目录）。`graph_planner.infra.vendor.ensure_rllm_importable()` 在任何 rLLM 适配模块导入前调用，步骤如下：
  1. 读取可选的环境变量 `GRAPH_PLANNER_RLLM_PATH`，若设置则直接把该路径加入 `sys.path`；
  2. 若环境未指定，先尝试当前仓库根目录内的 `./rllm` 子模块；
  3. 如果有人将 rLLM 独立检出到仓库同级目录，也会检测 `../rllm`；
  4. 最后回退到仓库根本身，以兼容 `pip install -e .` 等开发方式；
  5. 每次插入候选路径后刷新 `importlib` 缓存，并验证 `rllm` 与 `rllm.agents.agent`、`rllm.environments.base.base_env` 是否可解析，只有在确认结构完整后才返回成功。【F:graph_planner/infra/vendor.py†L1-L99】
- 因为路径是明确定义的，所以 `integrations/rllm` 内部直接执行 `from rllm.agents.agent import BaseAgent`、`from rllm.data.dataset import DatasetRegistry` 等标准导入，不再尝试猜测别名或动态包装。IDE 若提示波浪线，通常是尚未执行 `ensure_rllm_importable()`（即缺少 sys.path 注入）导致，此函数在包的 `__init__` 与各子模块文件顶层都会最先运行一次，确保解释器和静态分析都能定位到 vendored rLLM。【F:graph_planner/integrations/rllm/__init__.py†L1-L61】【F:graph_planner/integrations/rllm/dataset.py†L12-L43】

## 3. 数据与上下文流 / Data flow

1. **任务描述**：来自 `datasets/*.jsonl` 或自定义 JSON，包含 Issue 文本、最大步数、容器配置等。`rllm.dataset.load_task_entries` 会解析路径字段并标准化 sandbox 配置。【F:graph_planner/integrations/rllm/dataset.py†L59-L109】
2. **环境初始化**：`GraphPlannerRLLMEnv._spawn_planner` 把归一化后的 sandbox 信息转换为 `SandboxConfig`，再构造 `PlannerEnv` 并在 `reset()` 中生成初始观测。【F:graph_planner/integrations/rllm/env.py†L48-L108】
3. **记忆与子图管理**：`PlannerEnv` 调用 `memory` 模块维护子图；当需要向 CGM 求补丁时，通过 `subgraph_store` 读出线性化输入并交给 `cgm_adapter`。【F:graph_planner/env/planner_env.py†L94-L173】【F:graph_planner/agents/rule_based/cgm_adapter.py†L186-L200】
4. **CGM 上下文编排**：
   - 子图 JSON 通过 `GraphLinearizer.linearize` 转换为带节点摘要的文本块，保留名称、类型、注释、代码片段信息。【F:graph_planner/integrations/codefuse_cgm/formatting.py†L69-L128】
   - 片段候选由 `SnippetFormatter.format` 统一为 `path:start-end` + 正文的块状字符串。【F:graph_planner/integrations/codefuse_cgm/formatting.py†L136-L168】
   - `ConversationEncoder.build_user_message` 合并 Issue、Plan、Graph、Snippets，最终构造聊天消息列表并在训练时输出标签掩码。【F:graph_planner/integrations/codefuse_cgm/formatting.py†L184-L266】
5. **补丁生成**：`CodeFuseCGMGenerator.generate` 或远端 `CodeFuseCGMClient.complete` 获取补丁 JSON；若模型不可用则 `_LocalCGMRuntime` 在目标文件行尾添加 `CGM-LOCAL` 标记构造兜底补丁。【F:graph_planner/integrations/codefuse_cgm/inference.py†L62-L160】【F:graph_planner/agents/rule_based/cgm_adapter.py†L145-L207】
6. **动作流**：Planner 代理（规则或 LLM）依据当前观测和计划决定 Explore/Memory/Repair/Submit 动作，通过 `core.actions` 传递给环境，再由 `SandboxRuntime` 实际执行命令或应用补丁。【F:graph_planner/core/actions.py†L6-L42】【F:graph_planner/runtime/sandbox.py†L60-L214】

## 4. 训练与运行 Pipeline

### 4.1 规则 / 本地推理（离线调试）
1. 选择容器后端并准备 `config/r2e_ds_*.json`、镜像等资源。
2. 执行 `PYTHONPATH=. python scripts/run_rule_agent.py --backend repoenv --ds-json ... --max-steps 6 --agent rule` 触发规则策略；`--agent llm` 将启用 `LocalLLMPlannerAgent` 并在配置启用时调用本地 Hugging Face 模型。【F:scripts/run_rule_agent.py†L54-L136】【F:graph_planner/agents/model_based/planner.py†L38-L178】
3. 运行结果写入 `logs/test_runs.jsonl`，包含动作序列、补丁摘要和测试反馈，便于手动复盘。【F:graph_planner/infra/telemetry.py†L20-L39】

### 4.2 CGM 监督微调
1. 准备包含 `prompt`、`answer`、`plan`、`graph_path`、`snippets` 等字段的 JSON/JSONL 数据集，使用 `CodeFuseCGMDataset` 加载。【F:graph_planner/integrations/codefuse_cgm/data.py†L80-L210】
2. 配置 `CGMTrainingConfig`（模型路径、学习率、batch size、梯度累积等），实例化 `CodeFuseCGMTrainer` 并调用 `train()`。训练过程中 `CGMBatchCollator` 会调用 `ConversationEncoder` 生成输入张量并执行动态 padding。【F:graph_planner/integrations/codefuse_cgm/training.py†L60-L250】
3. 训练结束后，`CodeFuseCGMTrainer` 可保存 checkpoint 并返回评估指标；生成好的模型可交给 `CodeFuseCGMGenerator` 进行本地推理。【F:graph_planner/integrations/codefuse_cgm/inference.py†L62-L160】

### 4.3 Planner / CGM 强化学习（rLLM PPO）
1. 使用 `register_graphplanner_dataset.py` 或直接在训练脚本中调用 `ensure_dataset_registered` 将 RepoEnv 任务 JSONL 注册到 rLLM 的数据集注册表，获得 Verl parquet 路径。【F:graph_planner/integrations/rllm/dataset.py†L85-L109】【F:scripts/train_graphplanner_rllm.py†L123-L145】
2. 运行 `python scripts/train_graphplanner_rllm.py --agent planner --dataset <jsonl> --model-path <checkpoint>`：
   - CLI 解析命令行，注册训练/验证集，按需写入模型路径、温度、TP 等配置。【F:scripts/train_graphplanner_rllm.py†L32-L200】
   - 根据 `--agent` 选择 `GraphPlannerRLLMAgent` + `GraphPlannerRLLMEnv` 或 `CGMRLLMAgent` + `CGMRLLMEnv`，并在 Planner 模式下根据 `--cgm-model-path` 设置 CGM 本地推理参数。【F:scripts/train_graphplanner_rllm.py†L170-L200】
   - 脚本将覆盖写入的 OmegaConf 配置交给 rLLM 的 `AgentPPOTrainer` 执行分布式训练。
3. 训练过程中：
   - `GraphPlannerRLLMAgent.update_from_env` 把环境观测转成聊天消息并维护 PPO 轨迹；`update_from_model` 解析模型输出并在失败时触发规则 fallback。【F:graph_planner/integrations/rllm/agent.py†L101-L200】
   - `GraphPlannerRLLMEnv.step` 代理调用 `PlannerEnv.step` 执行动作，记录奖励与终止信号；`compute_final_reward` 在 episode 结束时检查测试通过情况。【F:graph_planner/integrations/rllm/env.py†L48-L114】
   - 若使用 CGM agent，则 `CGMRLLMEnv` 会直接调用 `_LocalCGMRuntime.generate_patch` 评估补丁质量并反馈奖励。【F:graph_planner/integrations/rllm/cgm_env.py†L140-L220】
4. 训练日志与模型 checkpoint 由 rLLM / Ray 负责存储，可结合 `--print-config` 输出最终 Hydra 配置进行审计。

### 4.4 远端 CGM / 本地 fallback
- 若 `.aci/config.json` 或环境变量启用了 `cgm.endpoint`，`cgm_adapter` 会优先使用 `CodeFuseCGMClient` 向官方服务发起请求；否则回退到本地 `CodeFuseCGMGenerator` 或规则标记补丁，确保补丁流程不会因模型缺失而中断。【F:graph_planner/agents/rule_based/cgm_adapter.py†L145-L207】
- Planner 模型路径可通过环境变量或配置文件注入，使规则策略、本地 LLM、rLLM 训练共用同一组配置读取逻辑。【F:graph_planner/infra/config.py†L60-L176】

## 5. 常用命令 / Recommended commands

| 场景 | 命令 |
| --- | --- |
| 规则策略冒烟 | `PYTHONPATH=. python scripts/run_rule_agent.py --backend repoenv --ds-json config/r2e_ds_repoenv_sample.json --max-steps 6 --agent rule` |
| 本地 LLM 调试 | `PYTHONPATH=. python scripts/run_rule_agent.py --backend repoenv --ds-json config/r2e_ds_repoenv_sample.json --agent llm --planner-model-path <hf-model>` |
| CGM 本地推理 | `python - <<'PY'`<br>`from graph_planner.integrations.codefuse_cgm import CodeFuseCGMGenerator, CGMExample;`<br>`gen = CodeFuseCGMGenerator.from_pretrained("<model>");`<br>`example = CGMExample.from_json_file("sample.json");`<br>`print(gen.generate(example).model_dump())`<br>`PY` |
| CGM 监督微调 | `PYTHONPATH=. python - <<'PY'`<br>`from graph_planner.integrations.codefuse_cgm import CodeFuseCGMTrainer, CGMTrainingConfig;`<br>`cfg = CGMTrainingConfig(model_name="<model>", train_path="train.jsonl", output_dir="runs/cgm");`<br>`CodeFuseCGMTrainer(cfg).train()`<br>`PY` |
| rLLM PPO 训练 | `python scripts/train_graphplanner_rllm.py --agent planner --dataset datasets/graphplanner_repoenv_sample.jsonl --model-path /path/to/policy --max-steps 6` |

## 6. 日志与调试建议 / Logging & troubleshooting

- 训练或推理期间的容器命令、测试结果会写入 `logs/test_runs.jsonl` 与 `logs/telemetry/*.jsonl`，可配合 `infra.telemetry` 工具分析。【F:graph_planner/infra/telemetry.py†L20-L39】
- 若 rLLM 模型输出解析失败，可启用 `--use-fallback` 并检查 `GraphPlannerRLLMAgent` 记录的 `fallback_reason`；这些信息会出现在 PPO 轨迹和日志中。【F:graph_planner/integrations/rllm/agent.py†L156-L200】
- `ensure_rllm_importable()` 在导入 rLLM 前会自动调整 `sys.path` 并验证子模块结构；如遇导入失败可检查环境变量 `GRAPH_PLANNER_RLLM_PATH`。【F:graph_planner/infra/vendor.py†L1-L86】

> **Update history**: 本文整合了原 `cgm_rllm_pipeline.md`、`r2e_gym_code_overview.md`、`r2e_training_pipeline.md`、`rllm_integration_report.md` 的内容，并加入最新的本地 LLM、CGM 训练与 rLLM 配置指引。
