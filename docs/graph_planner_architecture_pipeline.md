# Graph Planner 架构与训练运行全景

> **Summary (English)**
> This document consolidates the previously scattered architecture notes for Graph Planner
> **2025-11-03 模块状态审计**
> - ✅ 当前表格中的 Graph Planner 核心模块均仍在仓库内（agents/memory/infra/integrations 等路径已逐一核对）。
> - ❌ 历史上的 tests/test_toy_mlp.py 已被移除，相关示例仅保留在文档中供参考。


> and details how rule-based utilities, CGM integrations, local LLM adapters, and the
> rLLM PPO stack cooperate. It explains the module boundaries, data flow, and end-to-end
> pipelines for both supervised fine-tuning and reinforcement learning.

⚠️ **2025-11-07 更新**：`scripts/run_rule_agent.py` 与 `scripts/train_planner_grpo.py` 已从仓库移除，以下涉及旧版 CLI 的章节仅作历史记录，最新评测入口请参见 `scripts/run_eval_graph_planner.sh` / `scripts/eval_graph_planner_engine.py`。

## 1. 顶层结构 / Repository layering

```
CLI / Scripts / Tests
└── scripts/
    ├── run_eval_graph_planner.sh …… 评测入口（自动调用 Python 主程序）
    ├── eval_graph_planner_engine.py …… Planner/CGM 评测主程式
    └── [legacy] train_planner_grpo.py …… 旧版 GRPO 训练 CLI，已移除，历史细节见 docs/legacy_materials.md
└── tests/
    └── test_reward_manager_loading.py …… rLLM 奖励管理器加载兜底覆盖
└── rllm/tests/ …… 上游 rLLM 子模块自带的代理/环境/工具单测
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
    └── datasets/ …… 任务描述与 RepoEnv 配置
```

- `scripts/run_rule_agent.py` 曾用于离线调试规则/本地 LLM 决策，现已移除；评测与联调请改用 `scripts/eval_graph_planner_engine.py` 或 Bash 包装脚本。【F:scripts/eval_graph_planner_engine.py†L40-L200】
- `graph_planner.env.planner_env.PlannerEnv` 负责动作编排、奖励计算、记忆同步，是规则代理、本地 LLM 与 rLLM 环境的共同核心。【F:graph_planner/env/planner_env.py†L32-L173】
- `graph_planner.runtime.sandbox.SandboxRuntime` 根据配置切换 RepoEnv 与纯 Docker 路径，并统一命令执行与测试流程。【F:graph_planner/runtime/sandbox.py†L32-L214】
- `graph_planner.integrations` 目录收拢所有外部模型/训练栈：CGM（数据/推理/训练）、Hugging Face 聊天客户端、本地 rLLM 适配层。
- `tests/` 仅保留针对 rLLM 奖励管理器加载路径的冒烟测试，历史上的 FakeSandbox/文本协议等单测已随重构移除；如需这些验证，可参考文档中的命令自行重放流程。【F:tests/test_reward_manager_loading.py†L1-L45】

## 2. 模块职责一览 / Module responsibilities

| 模块 | 关键类/函数 | 职责摘要 |
| --- | --- | --- |
| `graph_planner/agents/rule_based/planner.py` | `PlannerAgent` | 规则策略状态机，驱动图扩展、记忆维护、补丁触发。【F:graph_planner/agents/rule_based/planner.py†L26-L187】 |
| `graph_planner/agents/model_based/planner.py` | `LocalLLMPlannerAgent` | 调用 `local_llm` 聊天客户端解析模型响应，失败时回退规则策略。【F:graph_planner/agents/model_based/planner.py†L38-L178】 |
| `graph_planner/agents/rule_based/cgm_adapter.py` | `CodeFuseCGMGenerator`、`CodeFuseCGMClient`、本地 fallback | 组合 GraphLinearizer/SnippetFormatter/ConversationEncoder，优先调用本地或远端 CGM，失败时打标记补丁。【F:graph_planner/agents/rule_based/cgm_adapter.py†L20-L200】 |
| `actor/cgm_local.py` | `generate` | 纯本地 CGM 兜底实现：优先复用仓库内的 rule-based 适配器，若缺席则返回空补丁，供 `CGMService` 在无 vLLM 时关闭环路。【F:actor/cgm_local.py†L1-L63】 |
| `graph_planner/integrations/codefuse_cgm/data.py` | `CGMExample`、`CodeFuseCGMDataset` | 解析训练/推理 JSON，加载图、片段、计划并产出结构化样本。【F:graph_planner/integrations/codefuse_cgm/data.py†L1-L210】 |
| `rllm/verl/verl/workers/cgm_service.py` | `CGMService` | rLLM/Verl 训练时调度 CGM 推理；检测不到 vLLM 时切换为 `actor.cgm_local.generate` 以保持闭环且无 RPC 递归。【F:rllm/verl/verl/workers/cgm_service.py†L1-L247】【F:actor/cgm_local.py†L1-L63】 |
| `graph_planner/integrations/codefuse_cgm/formatting.py` | `GraphLinearizer`、`SnippetFormatter`、`ConversationEncoder` | 将 `serialize_subgraph` 结果与候选片段线性化，组合聊天模板用于训练/推理。【F:graph_planner/integrations/codefuse_cgm/formatting.py†L69-L199】 |
| `graph_planner/integrations/codefuse_cgm/training.py` | `CGMBatchCollator`、`CodeFuseCGMTrainer` | 构建监督微调所需的 DataLoader、优化器、调度器与训练循环。【F:graph_planner/integrations/codefuse_cgm/training.py†L1-L250】 |
| `graph_planner/integrations/codefuse_cgm/inference.py` | `CodeFuseCGMGenerator` | 加载 Hugging Face checkpoint，以本地方式生成补丁候选，并支持 device map / 量化推理。【F:graph_planner/integrations/codefuse_cgm/inference.py†L18-L152】 |
| `graph_planner/integrations/local_llm/hf.py` | `HuggingFaceChatClient` | 将任意 Causal LM 封装为聊天接口，提供 device map、量化、`torch_dtype` 等多 GPU 选项。【F:graph_planner/integrations/local_llm/hf.py†L19-L186】 |
| `graph_planner/integrations/rllm/agent.py` | `GraphPlannerRLLMAgent` | 继承 rLLM `BaseAgent`，复用聊天协议、CGM fallback，并维护训练轨迹。【F:graph_planner/integrations/rllm/agent.py†L56-L200】 |
| `graph_planner/integrations/rllm/env.py` | `GraphPlannerRLLMEnv` | 将 `PlannerEnv` 暴露为 rLLM `BaseEnv`，并为每个任务生成唯一 `issue_uid` 以支撑并发容器训练。【F:graph_planner/integrations/rllm/env.py†L1-L134】 |
| `graph_planner/integrations/rllm/cgm_agent.py` & `graph_planner/integrations/rllm/cgm_env.py` | `CGMRLLMAgent`、`CGMRLLMEnv` | 面向 CGM 的 PPO 训练包装，封装唯一 issue id 与上下文采集流程。【F:graph_planner/integrations/rllm/cgm_agent.py†L1-L220】【F:graph_planner/integrations/rllm/cgm_env.py†L11-L197】 |
| `graph_planner/integrations/rllm/dataset.py` | `ensure_dataset_registered` | 将 JSON/JSONL 任务注册为 rLLM 可识别的数据集，解析路径与挂载信息。【F:graph_planner/integrations/rllm/dataset.py†L28-L109】 |
| `graph_planner/memory/text_memory.py` | `memory_commit`、`memory_delete`、`TurnState` | 维护 Explore/工具 observation 的记忆提交与删除，执行配额估算与拒绝逻辑，供环境与代理共享最新上下文。【F:graph_planner/memory/text_memory.py†L37-L420】 |
| `graph_planner/infra/config.py` | `load_launch_config`、`merge_launch_config` | 解析内置默认值、YAML 与 CLI 覆写，写入 `resolved_config.yaml`，并返回结构化配置以供训练/评估脚本复用。【F:graph_planner/infra/config.py†L42-L285】 |
| `graph_planner/infra/parallel.py` | `resolve_parallel`、`preflight_check` | 根据最终配置计算张量并行/副本/并发参数，并在启动前校验 GPU、Ray 资源是否匹配，失败时给出修复建议。【F:graph_planner/infra/parallel.py†L1-L134】 |
| `graph_planner/infra/metrics.py` | `init_wandb`、`log_metrics`、`make_gpu_snapshot` | 初始化 W&B（支持 offline）、记录 Planner/CGM/环境指标，并定期采集 GPU 与 Ray 资源心跳。【F:graph_planner/infra/metrics.py†L1-L93】 |
| `[legacy] scripts/train_planner_grpo.py` | CLI helpers（已移除） | 旧版 CLI 负责解析 YAML、注册 JSONL、预拉容器并构建 Ray runtime；该入口已下线，历史说明保留在 `docs/legacy_materials.md`。 |

### 2.1 rLLM 模块导入路径 / rLLM import resolution

- rLLM 源码以 Git 子模块形式放在仓库根目录下的 `rllm/`，其内部又包含 Python 包目录 `rllm/`（双层目录）。`graph_planner.infra.vendor.ensure_rllm_importable()` 在任何 rLLM 适配模块导入前调用，步骤如下：
  1. 读取可选的环境变量 `GRAPH_PLANNER_RLLM_PATH`，若设置则直接把该路径加入 `sys.path`；
  2. 若环境未指定，先尝试当前仓库根目录内的 `./rllm` 子模块；
  3. 如果有人将 rLLM 独立检出到仓库同级目录，也会检测 `../rllm`；
  4. 最后回退到仓库根本身，以兼容 `pip install -e .` 等开发方式；
  5. 每次插入候选路径后刷新 `importlib` 缓存，并验证 `rllm` 与 `rllm.rllm.agents.agent`/`rllm.agents.agent`、`rllm.rllm.environments.base.base_env`/`rllm.environments.base.base_env` 是否可解析，只有在确认结构完整后才返回成功。【F:graph_planner/infra/vendor.py†L1-L112】
- 因为路径是明确定义的，所以 `integrations/rllm` 内部在导入代理类时会优先尝试 `from rllm.agents.agent import BaseAgent`（与执行引擎保持同一来源），若该层级不存在再退回到 `from rllm.rllm.agents.agent import ...`；数据集注册仍以 `rllm.rllm.data.dataset` 为主。IDE 若提示波浪线，通常是尚未执行 `ensure_rllm_importable()`（即缺少 sys.path 注入）导致，此函数在包的 `__init__` 与各子模块文件顶层都会最先运行一次，确保解释器和静态分析都能定位到 vendored rLLM。【F:graph_planner/integrations/rllm/__init__.py†L1-L61】【F:graph_planner/integrations/rllm/dataset.py†L12-L43】

### 2.2 Planner / CGM Prompt & Response Contracts

- `graph_planner.agents.common.contracts` 统一维护系统提示、动作枚举、错误码，以及 `parse_action_block`/`validate_planner_action`/`validate_cgm_patch` 等校验函数，成为 Planner 与 CGM 的唯一事实来源（SSOT）。【F:graph_planner/agents/common/contracts.py†L1-L394】
- `scripts/validate_contracts.py` 提供契约冒烟检查，可在提交前快速验证解析器与补丁校验逻辑，适用于本地或 CI。【F:scripts/validate_contracts.py†L1-L48】
- Planner 代理的系统提示 (`SYSTEM_PROMPT`) 直接引用合同中的指令，保证模型始终输出带 `thought`/`action` 的 JSON，且各动作字段与解析器保持一致。【F:graph_planner/agents/common/chat.py†L1-L83】
- CGM 侧的 `ConversationEncoder` 与本地 fallback 亦复用同一份合同，确保子图、片段拼接后的提示文本与补丁输出格式（`patch.edits`）在远端服务、本地生成器与标注文档之间保持一致。【F:graph_planner/integrations/codefuse_cgm/formatting.py†L69-L266】【F:graph_planner/agents/rule_based/cgm_adapter.py†L1-L207】

#### 2.2.1 Planner 动作协议

- **动作枚举**：Planner 模型必须输出单个 `<function=ACTION>` 区块，`ACTION` 取自 `explore`、`memory`、`repair`、`submit` 或 `noop`。`parse_action_block` 与 `validate_planner_action` 在合同模块中负责校验标签、参数唯一性与必填项，若检测到异常会抛出带错误码的 `ProtocolError` 并触发规则兜底。【F:graph_planner/agents/common/contracts.py†L193-L343】【F:graph_planner/agents/model_based/planner.py†L149-L205】
- **Explore**：可携带 `op`（`find|expand|read`）、`anchors`（节点/锚点列表）、`nodes`、`hop` 与 `limit` 参数，解析后映射到 `ExploreAction`，驱动图扩展与代码片段读取。【F:graph_planner/core/actions.py†L1-L26】【F:graph_planner/agents/model_based/planner.py†L212-L225】
- **Memory**：`target`/`scope`/`intent` 指定是提交最新 `explore` 结果还是其他工具 observation，`MemoryAction` 交由 `text_memory.memory_commit`/`memory_delete` 判断配额并更新子图或文本记忆。【F:graph_planner/core/actions.py†L28-L39】【F:graph_planner/memory/text_memory.py†L151-L319】
- **Repair**：必须提供 `subplan`（支持 `<![CDATA[...]]>` 多行文本），可选 `focus_ids` 与 `apply`。转换后的 `RepairAction` 仅保留计划文本与关注节点，后续由 `PlannerEnv` 的文本轨迹修复管线拉起 CGM；模型不再直接下发 `patch`。【F:graph_planner/agents/model_based/planner.py†L230-L247】【F:graph_planner/env/planner_env.py†L240-L302】
- **Submit/Noop**：`SubmitAction` 仍然终止 episode；`noop` 会映射为不执行记忆操作的空 `MemoryAction`，供模型显式跳过当前回合。【F:graph_planner/agents/model_based/planner.py†L247-L250】
- **Thought 字段**：`<param name="thought">` 保留模型思考过程，`GraphPlannerRLLMAgent` 与本地 Planner 都会把该文本写入轨迹，便于调试与奖励塑形。【F:graph_planner/integrations/rllm/agent.py†L114-L161】【F:graph_planner/agents/model_based/planner.py†L141-L174】

#### 2.2.2 Planner Prompt 布局

| Prompt 区块 | 填充内容 | 关联实现 |
| --- | --- | --- |
| `[Issue]` | issue id/title/body，来自 `PlannerEnv.reset()` 的观测 | `summarise_observation` 将 `issue` 与 `failure_frame` 等字段整理为自然语言摘要。【F:graph_planner/agents/common/chat.py†L20-L72】 |
| `[Instruction]` | 最近一步的环境反馈（失败信息、reward、动作摘要） | 同上，`summarise_observation` 会把 `last_info`、`reward`、`done` 等写入文本，并返回给聊天客户端。 |
| `[Planner memory]` | 可选的历史上下文：子图统计、plan 摘要等 | `PlannerEnv.render_memory_for_llm()` 将记忆模块内容注入用户提示，缺省时留空。【F:graph_planner/env/planner_env.py†L136-L173】 |

- 系统提示明确要求模型“只输出一个 `<function=...>` 区块”，并提醒多行文本需包裹在 CDATA 中。`parse_action_block` 与 `validate_planner_action` 会在解析后抛出带错误码的 `ProtocolError`；`LocalLLMPlannerAgent` 捕获该错误并记录 `fallback_reason` 元数据，用于下游遥测。【F:graph_planner/agents/common/contracts.py†L109-L343】【F:graph_planner/agents/model_based/planner.py†L149-L205】

#### 2.2.3 CGM Prompt / Patch Schema

- **Prompt 区块**：Issue → Instruction → Plan → Subgraph → Snippets，分别注入 Planner 请求、结构化计划、`GraphLinearizer` 线性化后的节点文本，以及 `SnippetFormatter` 拼接的候选代码片段。【F:graph_planner/integrations/codefuse_cgm/formatting.py†L69-L199】
- **系统指令**：要求输出 JSON `{ "patch": { "edits": [...] }, "summary": "..." }`；`CodeFuseCGMGenerator`、`CodeFuseCGMClient` 与 fallback runtime 使用相同 system prompt，保证本地、远端或标注路径一致。【F:graph_planner/agents/common/contracts.py†L145-L170】【F:graph_planner/integrations/codefuse_cgm/inference.py†L18-L152】
- **补丁结构**：`patch.edits` 中的每个对象必须包含 `path`、`start`、`end`、`new_text`，其中 `new_text` 必须以换行结尾。`CGMPatch` 数据结构与 `SandboxRuntime.apply_patch` 都假定这一 schema，缺项会导致补丁被拒绝。【F:graph_planner/core/patches.py†L1-L96】【F:graph_planner/runtime/sandbox.py†L120-L214】
- **输出解析**：`CodeFuseCGMGenerator._parse_patch` 会在 Hugging Face 生成结果上调用合同中的 schema 校验，若缺失必需字段则回退到规则补丁，并在日志中标记原因。【F:graph_planner/integrations/codefuse_cgm/inference.py†L94-L150】

#### 2.2.4 文本轨迹修复管线（Text-trajectory Repair Pipeline）

> **背景**：当 Planner 仅输出“修复计划”而非直接附带 diff 时，环境需要基于文本轨迹协议 `<function=...>` / `<observation for=...>` 自行拼装上下文、调用 CGM 生成 unified diff，并返回规范化的观测。`graph_planner.agents.common.text_protocol` 汇总了这一闭环逻辑。

| 层级 | 关键函数 | 作用 | 关联实现 |
| --- | --- | --- | --- |
| 解析 | `parse_action_block(text)` | 严格解析单个 `<function=...>` 区块，校验动作与参数并返回结构化字典；一旦检测到额外文本、重复块或非法参数即抛出带错误码的 `ProtocolError`，避免 Planner 输出多段内容。【F:graph_planner/agents/common/contracts.py†L193-L256】 |
| 运行时状态 | `RepairRuntimeState` | 汇集 issue 描述、子图缓存、文本记忆、沙箱路径等上下文字段，统一供上下文构建与补丁应用阶段使用，隔离环境内部实现细节。【F:graph_planner/agents/common/text_protocol.py†L72-L129】 |
| 上下文抽取 | `build_cgm_payload(state, subplan, focus_ids)` | 读取图节点/边、`focus_ids`、记忆摘要、相关文件全文与 issue 信息，并把 `subplan` 拆成步骤列表；同时执行 Top-K 裁剪和 token 预算控制，确保请求不会超出 CGM 长度限制。【F:graph_planner/agents/common/text_protocol.py†L148-L170】 |
| CGM 调用 | `call_cgm(payload, k)` / `pick_best_candidate(cands)` | 透传到本地/远端 CGM 客户端生成补丁候选，并依据置信度、测试建议等字段选择最佳补丁；若返回多文件 diff，`handle_planner_repair` 会拆分并逐一尝试。【F:graph_planner/agents/common/text_protocol.py†L174-L188】 |
| 补丁校验 | `validate_unified_diff(patch, path)` | 检查 diff 头部与目标路径匹配、hunk 语法合法、禁止跨文件编辑；违反约束时抛出 `ValueError`，阻止异常补丁污染仓库。【F:graph_planner/agents/common/text_protocol.py†L190-L236】 |
| 应用与测试 | `try_apply_and_test(path, patch)` | 在沙箱临时目录应用 diff，并调用构建/测试/静态检查；将是否应用、hunk 数量、测试与 lint 结果封装为 JSON，供 observation 使用。【F:graph_planner/agents/common/text_protocol.py†L242-L272】 |
| 观察发射 | `emit_observation(name, data)` | 将任意 JSON 编码为 `<observation for="name">…</observation>` 字符串，下一轮直接送回 Planner。【F:graph_planner/agents/common/text_protocol.py†L275-L279】 |
| 管线入口 | `handle_planner_repair(action, state)` | 统筹整个流程：解析 Planner 的 `repair` 参数（`subplan` 必填、`focus_ids`/`apply` 可选），构建 payload、调用 CGM、按需应用补丁并组装最终 observation；失败时会在 `error` 字段标识 `apply-failed`/`build-failed` 等阶段。【F:graph_planner/agents/common/text_protocol.py†L331-L396】 |

- **执行示例 / Worked example**：
  1. Planner 输出：

     ```
     <function=repair>
       <param name="subplan"><![CDATA[
     1) Locate check_bound
     2) Change <= to <
     3) Add regression test for boundary case
     ]]></param>
       <param name="focus_ids">["n_buf"]</param>
       <param name="apply">true</param>
     </function>
     ```

  2. `parse_action_block` 校验只有一个 `<function>` 块，并产出 `{"name": "repair", "params": {"subplan": "…", "focus_ids": ["n_buf"], "apply": true}}`；
  3. `build_cgm_payload` 结合 `RepairRuntimeState` 中的图节点、历史笔记、相关文件与 Issue 文本生成 CGM 请求 JSON，并将 `subplan` 拆分为 `plan` 列表；
  4. `call_cgm` 根据 payload 返回候选补丁，`pick_best_candidate` 按置信度和测试建议挑选最佳项，若出现多文件 diff 会拆分成多个单文件尝试；
  5. `validate_unified_diff` 确认补丁语法与路径一致，`try_apply_and_test` 在沙箱内应用补丁、运行构建/测试/静态检查并统计 hunk 数、测试结果；
  6. `emit_observation("repair", {...})` 产出形如 `<observation for="repair">{"ok":true,"applied":true,"tests_passed":true,...}</observation>` 的字符串发回 Planner。

- **协议对齐 / Contract alignment**：生成的 CGM payload 与回包严格遵循“单文件 unified diff”约束，`constraints.one_file_per_patch=true` 让 CGM 只编辑一个文件。若 CGM 侧仍返回多文件 diff，`handle_planner_repair` 会拆分并串行处理，确保每次只对一个文件运行 `validate_unified_diff` 和 `try_apply_and_test`。

- `PlannerEnv.step()` 在检测到 Planner 动作为 `repair` 且未携带显式 `patch` 时，会实例化 `RepairRuntimeState` 并委托上述管线；若 Planner 请求 `apply=false`，则仅返回候选补丁信息，不会修改仓库。【F:graph_planner/env/planner_env.py†L142-L229】
- 文本协议的端到端流程依托 `text_protocol.handle_planner_repair` 与 `SandboxRuntime` 的联动完成，当前仓库不再附带早期的 e2e 单测，建议按文档命令手动重放以验证自定义改动。【F:graph_planner/agents/common/text_protocol.py†L148-L396】【F:graph_planner/runtime/sandbox.py†L60-L214】

## 3. 数据与上下文流 / Data flow

1. **任务描述**：来自 `datasets/*.jsonl` 或自定义 JSON，包含 Issue 文本、最大步数、容器配置等。`rllm.dataset.load_task_entries` 会解析路径字段并标准化 sandbox 配置。【F:graph_planner/integrations/rllm/dataset.py†L59-L109】
2. **环境初始化**：`GraphPlannerRLLMEnv._spawn_planner` 把归一化后的 sandbox 信息转换为 `SandboxConfig`，并将任务 issue 注入唯一 `issue_uid`，再构造 `PlannerEnv` 并在 `reset()` 中生成初始观测。【F:graph_planner/integrations/rllm/env.py†L46-L134】
3. **记忆与子图管理**：`PlannerEnv` 调用 `memory` 模块维护子图；当需要向 CGM 求补丁时，通过 `subgraph_store` 读出线性化输入并交给 `cgm_adapter`。【F:graph_planner/env/planner_env.py†L94-L173】【F:graph_planner/agents/rule_based/cgm_adapter.py†L186-L200】
4. **CGM 上下文编排**：
   - 子图 JSON 通过 `GraphLinearizer.linearize` 转换为带节点摘要的文本块，保留名称、类型、注释、代码片段信息。【F:graph_planner/integrations/codefuse_cgm/formatting.py†L69-L128】
   - 片段候选由 `SnippetFormatter.format` 统一为 `path:start-end` + 正文的块状字符串。【F:graph_planner/integrations/codefuse_cgm/formatting.py†L136-L168】
   - `ConversationEncoder.build_user_message` 合并 Issue、Plan、Graph、Snippets，最终构造聊天消息列表并在训练时输出标签掩码。【F:graph_planner/integrations/codefuse_cgm/formatting.py†L184-L266】
5. **补丁生成**：`CodeFuseCGMGenerator.generate` 或远端 `CodeFuseCGMClient.complete` 获取补丁 JSON；本地路径支持多 GPU `device_map`、量化或自定义 dtype，若模型不可用则 `_LocalCGMRuntime` 在目标文件行尾添加 `CGM-LOCAL` 标记构造兜底补丁。【F:graph_planner/integrations/codefuse_cgm/inference.py†L18-L152】【F:graph_planner/agents/rule_based/cgm_adapter.py†L145-L207】
6. **动作流**：Planner 代理（规则或 LLM）依据当前观测和计划决定 Explore/Memory/Repair/Submit 动作，通过 `core.actions` 传递给环境，再由 `SandboxRuntime` 实际执行命令或应用补丁。【F:graph_planner/core/actions.py†L6-L42】【F:graph_planner/runtime/sandbox.py†L60-L214】

#### 子图维护与 1-hop 扩展

- **工作子图结构**：`memory.subgraph_store` 将子图标准化为 `WorkingSubgraph`，内部持久化 `nodes`/`edges` 字典并维护 `node_ids` 集合，支持 `iter_node_ids`、`get_node`、`add_nodes`、`remove_nodes` 等操作。子图会被序列化到仓库根目录下的 `.aci/subgraphs/<issue>.json`，从而在 Planner 重启后恢复上下文。【F:graph_planner/memory/subgraph_store.py†L23-L90】【F:graph_planner/memory/subgraph_store.py†L107-L141】
- **统计与线性化**：环境在返回 observation 时调用 `subgraph_store.stats/linearize` 统计节点类型、文件数，并依据需要（WSD/FULL）把节点映射到文本片段，为 CGM、规则策略或 UI 展示提供统一上下文。【F:graph_planner/memory/subgraph_store.py†L143-L204】
- **Explore→Memory 流程**：`PlannerEnv._handle_explore` 的 `expand` 分支会使用 `mem_candidates.build_mem_candidates` 生成 1-hop 候选并附带得分、来源标签；后续若 Planner 选择 `memory` 提交，`text_memory.memory_commit` 会把这些节点写入子图并同步统计信息。【F:graph_planner/env/planner_env.py†L167-L215】【F:graph_planner/memory/text_memory.py†L501-L566】
- **1-hop 扩展实现**：`mem_candidates.build_mem_candidates` 内部调用 `graph_adapter.one_hop_expand`。适配器优先加载外部 `repo_graph.jsonl/graph.jsonl/code_graph.jsonl`，若缺失则按需扫描仓库源文件构建轻量代码图（文件节点、Python 函数/类、导入边）。1-hop 邻居不仅包含原图边，还会补充同文件函数/类及同目录文件，使 Planner 无需手工解析仓库即可围绕锚点扩展上下文。【F:graph_planner/memory/mem_candidates.py†L39-L152】【F:graph_planner/memory/graph_adapter.py†L19-L231】【F:graph_planner/memory/graph_adapter.py†L250-L371】
- **是否需要预先解析代码图？** 不需要额外的前置步骤：若仓库随任务提供序列化图，适配器会直接载入；否则会在首次调用 `graph_adapter.connect()` 时构建本地图句柄，并在 1-hop 扩展时缓存和复用，从而保证 Explore 功能在没有独立图生成服务的情况下也能运行。【F:graph_planner/memory/graph_adapter.py†L52-L133】【F:graph_planner/memory/graph_adapter.py†L308-L371】

### 3.1 Toy MLP 示例（手动验证）

- `graph_planner/models/toy_lm.py` 继续提供 `ToyTokenizer`、`ToyLMForCausalLM` 以及 `create_toy_checkpoint`，可在本地生成 Hugging Face 兼容的玩具模型与权重，便于在无网环境下演练推理与反向传播。【F:graph_planner/models/toy_lm.py†L18-L215】
- 生成的 checkpoint 可交由 `HuggingFaceChatClient` 加载，以聊天接口驱动 Planner 模型链路；该客户端同样负责解析自定义设备、精度与采样参数。【F:graph_planner/integrations/local_llm/hf.py†L71-L186】
- 若需要模拟 CGM 端输入，可结合 `CGMExample`、`GraphLinearizer` 与 `SnippetFormatter` 构造最小样本，再调用本地推理器验证补丁生成逻辑。【F:graph_planner/integrations/codefuse_cgm/data.py†L70-L156】【F:graph_planner/integrations/codefuse_cgm/formatting.py†L73-L172】【F:graph_planner/integrations/codefuse_cgm/inference.py†L18-L152】
- 仓库不再附带旧版 历史上的 tests/test_toy_mlp.py（已移除，不再随仓库提供），需要这些路径时可根据上述模块手动编写脚本重现原有测试行为。

### 3.2 GRPO 测试调用流程（文件 | 函数 | 作用）

| 文件 | 函数 | 作用 |
| --- | --- | --- |
| `graph_planner/models/toy_lm.py` | `create_toy_checkpoint → ToyTokenizer.save_pretrained → ToyLMConfig.save_pretrained → ToyLMForCausalLM.save_pretrained` | 构建字符级分词器、ToyLM 配置与权重并写入磁盘，为后续 `AutoTokenizer`/`AutoModelForCausalLM` 加载做准备。 【F:graph_planner/models/toy_lm.py†L190-L219】 |

## 4. 训练与运行 Pipeline

### 4.1 规则 / 本地推理（离线调试）
1. 选择容器后端并准备包含 RepoEnv 任务描述的 JSONL（例如 `datasets/r2e_gym/train.jsonl`）以及对应的容器镜像。
2. 执行 `PYTHONPATH=. python scripts/run_rule_agent.py --backend repoenv --ds-json ... --max-steps 6 --agent rule` 触发规则策略；`--agent llm` 将启用 `LocalLLMPlannerAgent` 并在配置启用时调用本地 Hugging Face 模型。【F:scripts/run_rule_agent.py†L54-L136】【F:graph_planner/agents/model_based/planner.py†L38-L178】
3. 运行结果写入 `logs/test_runs.jsonl`，包含动作序列、补丁摘要和测试反馈，便于手动复盘。【F:graph_planner/infra/telemetry.py†L20-L39】

### 4.2 CGM 监督微调
1. 准备包含 `prompt`、`answer`、`plan`、`graph_path`、`snippets` 等字段的 JSON/JSONL 数据集，使用 `CodeFuseCGMDataset` 加载。【F:graph_planner/integrations/codefuse_cgm/data.py†L80-L210】
2. 配置 `CGMTrainingConfig`（模型路径、学习率、batch size、梯度累积等），实例化 `CodeFuseCGMTrainer` 并调用 `train()`。训练过程中 `CGMBatchCollator` 会调用 `ConversationEncoder` 生成输入张量并执行动态 padding。【F:graph_planner/integrations/codefuse_cgm/training.py†L60-L250】
3. 训练结束后，`CodeFuseCGMTrainer` 可保存 checkpoint 并返回评估指标；生成好的模型可交给 `CodeFuseCGMGenerator` 进行本地推理。【F:graph_planner/integrations/codefuse_cgm/inference.py†L18-L152】

### 4.3 Planner / CGM 强化学习（rLLM PPO）
1. 使用 `scripts/register_graphplanner_dataset.py` 或直接在训练脚本中调用 `ensure_dataset_registered` 将 RepoEnv 任务 JSONL 注册到 rLLM 的数据集注册表，获得 Verl parquet 路径。【F:graph_planner/integrations/rllm/dataset.py†L85-L109】【F:scripts/train_planner_grpo.py†L212-L235】
2. 运行 `PYTHONPATH=. python scripts/train_planner_grpo.py --config configs/experiments/planner_grpo_4gpu.yaml --print-config`（如需覆写路径或超参，可追加 `--overrides data.train_files=[... ] trainer.total_epochs=200`）：
   - 脚本会加载 YAML 并应用 dotlist 覆写，然后调用 `_maybe_materialize_json_to_verl_parquet` 将 JSONL 转换成 `_verl.parquet`，同时在需要时生成 docker manifest 并触发 `_maybe_prepull_docker_images`。【F:scripts/train_planner_grpo.py†L322-L362】【F:scripts/train_planner_grpo.py†L293-L320】
   - `_resolve_planner_env` / `_resolve_cgm_env` 会把模型路径解析为绝对路径，并写入 Ray runtime 的环境变量；如果 CLI 添加 `--dry-run`，流程会在此阶段退出，方便预检配置而不启动 Ray。【F:scripts/train_planner_grpo.py†L110-L210】【F:scripts/train_planner_grpo.py†L415-L424】
   - `--ray-address` 支持连接既有 Ray 集群，未指定时脚本会自动拉起本地 runtime；日志里会打印批量大小、Ray 环境变量与最终合并后的配置，便于排查资源分配问题。【F:scripts/train_planner_grpo.py†L258-L312】【F:scripts/train_planner_grpo.py†L312-L352】【F:scripts/train_planner_grpo.py†L424-L468】
> **结论**：当前仓库已串联 planner 与 CGM 的本地/远端模型加载逻辑，并在 `scripts/train_planner_grpo.py` 中将回合轨迹直接送入 Verl GRPO 算法，因此只需提供模型与数据路径即可在容器内完成“代码修复 → 经验采样 → GRPO 更新”的闭环训练。【F:scripts/train_planner_grpo.py†L322-L468】

1. **准备模型 checkpoint**
   - Planner：默认使用 Qwen3-14B 权重，需解压到仓库根目录的 `models/Qwen3-14B/`；如需替换，可在 YAML 中覆盖 `paths.planner_model`。CGM 目录同理，放置于 `models/CodeFuse-CGM/` 并通过 `paths.cgm_model` 指定。【F:configs/experiments/planner_grpo_4gpu.yaml†L1-L38】

2. **准备任务数据集**
   - 运行 `scripts/prepare_datasets.py` 可一次性生成 R2E-Gym/SWE-bench JSONL、实例与 docker manifest；脚本支持 `--skip-*`、`--prepull-*` 调整行为。【F:scripts/prepare_datasets.py†L12-L86】【F:scripts/prepare_datasets.py†L135-L214】
   - 训练脚本在启动时会调用 `ensure_dataset_registered` 将 JSONL 转换为 `_verl.parquet` 并写回配置，确保 Verl 可以直接消费。【F:scripts/train_planner_grpo.py†L212-L235】【F:scripts/train_planner_grpo.py†L322-L362】

3. **执行训练命令**
   ```bash
   PYTHONPATH=. python scripts/train_planner_grpo.py      --config configs/experiments/planner_grpo_4gpu.yaml      --overrides data.train_files=[datasets/r2e_gym/train.jsonl]      --print-config
   ```
   - 可通过 `--overrides` 指定新的 JSONL 或调整 epoch/batch 等超参；若仅需校验配置，可结合 `--print-config` 或 `--dry-run` 在不启动 Ray 的情况下退出。【F:scripts/train_planner_grpo.py†L371-L424】
   - 未提供 `--ray-address` 时脚本会自动启动本地 Ray；提供远程地址则复用既有集群，并在日志中打印批量配置与运行时环境变量，便于排查资源问题。【F:scripts/train_planner_grpo.py†L258-L312】【F:scripts/train_planner_grpo.py†L312-L352】【F:scripts/train_planner_grpo.py†L424-L468】

4. **训练期工作流**
   - `GraphPlannerRLLMEnv` 调用 `PlannerEnv` 执行真实仓库操作，并为每条任务分配唯一 `issue_uid` 以支撑并发采样。【F:graph_planner/integrations/rllm/env.py†L46-L134】
   - `GraphPlannerRLLMAgent` 将观测整理成聊天提示，请求 Planner 模型生成 `<function=...>`，并在解析失败时自动回退规则策略。【F:graph_planner/integrations/rllm/agent.py†L101-L210】
   - `graph_planner.agents.common.text_protocol` 负责在 `repair` 动作下拼装 CGM payload、调用补丁模型并记录测试结果，最终 observation 会带着 `apply`/`submit` 反馈返回给代理。【F:graph_planner/agents/common/text_protocol.py†L148-L272】

5. **结果产出与复现**
   - 合并后的配置会写入 `outputs/<run_name>/resolved_config.yaml`，日志中同时打印本地/远端环境变量映射，便于后续复现实验；同一命令再次运行即可复用缓存的 parquet 与 Ray 配置。【F:scripts/train_planner_grpo.py†L258-L352】【F:scripts/train_planner_grpo.py†L415-L468】
#### 4.3.1.1 与「通用 ToolAgent/ToolEnvironment」流程的对比

近期有同事提到一套更贴近原生 rLLM 示例的流程：直接复用 rLLM 自带的 `ToolAgent` / `ToolEnvironment`，通过配置 `agent_args`（工具清单、解析器、system prompt）、`env_args`（奖励函数、工具开关）、`sampling_params`（温度、top-p、模型名），再交给 `AgentExecutionEngine(engine_name="openai", rollout_engine_args={...})`，并由 `n_parallel_agents` 控制并发。我们当前仓库的实现与该描述既有重合也存在明显差异，具体如下表所示：

| 维度 | 通用 ToolAgent/ToolEnvironment 示例流程 | 本仓库现有实现 | 备注 |
| --- | --- | --- | --- |
| Agent 类型 | 使用 rLLM 内置 `ToolAgent`，主要依赖通用工具接口 | 自定义 `GraphPlannerRLLMAgent` / `CGMRLLMAgent`，继承自 `BaseAgent`，封装 planner prompt、CGM fallback 与轨迹字段。【F:graph_planner/integrations/rllm/agent.py†L56-L210】【F:graph_planner/integrations/rllm/cgm_agent.py†L1-L220】 | 为了保证 `<function=...>` 文本协议、补丁调用与本地记忆结构对齐，我们实现了专用 agent，而不是直接复用通用工具 agent。 |
| Environment | 直接使用 rLLM 的 `ToolEnvironment`，依赖工具抽象 | 自定义 `GraphPlannerRLLMEnv` / `CGMRLLMEnv`，在 `reset/step` 内部驱动 `PlannerEnv` 与 RepoEnv 容器，并注入奖励 shaping。【F:graph_planner/integrations/rllm/env.py†L1-L134】【F:graph_planner/integrations/rllm/cgm_env.py†L70-L214】 | 需要与 RepoEnv/CGM 的数据结构深度耦合，故编写了专用环境包装。 |
| Prompt & 输出约定 | 由工具解析器决定，通常是标准的思维链 + 工具指令模板 | 通过 `graph_planner.agents.common.contracts` 定义 planner/CGM 的系统提示、section 布局与响应协议；Planner 使用 `<function=...>` 区块，CGM 仍输出 JSON 补丁。【F:graph_planner/agents/common/contracts.py†L1-L394】【F:graph_planner/agents/common/text_protocol.py†L37-L396】 | 该合同是我们仓库新增的约束，确保模型输出与代码解析器完全一致。 |
| 采样/推理引擎 | `AgentExecutionEngine` 可以配置为 `engine_name="openai"` 并指向 HTTP API | 训练脚本使用本地 Hugging Face 模型（或玩具模型）作为 planner/CGM，推理由 `HuggingFaceChatClient` 和 `CodeFuseCGMGenerator` 实现。【F:graph_planner/integrations/local_llm/hf.py†L19-L186】【F:graph_planner/integrations/codefuse_cgm/inference.py†L18-L152】 | 这样能在离线容器中完成训练，无需外部 API。 |
| 经验采样与更新 | `AgentExecutionEngine` 同时负责 rollout 和工具调用 | 采样由自定义 env/agent 执行，轨迹交给 Verl GRPO / PPO 算法处理；Ray 并发、资源检查、日志等由 `scripts/train_planner_grpo.py` 注入配置。【F:scripts/train_planner_grpo.py†L70-L336】 | 虽然同属 rLLM 生态，但我们直接对接 Verl 的训练器，而非 AgentExecutionEngine 的“一站式”接口。 |
| 配置方式 | 主要靠传入 `agent_args`、`env_args`、`sampling_params` 字典 | 结合 CLI 参数与 OmegaConf 覆写，提供 `--reward-scale`、`--failure-penalty`、`--precision`、`--grad-accum-steps` 等开关，并通过 `_set_if_exists` 写入 Hydra 配置。【F:scripts/train_planner_grpo.py†L180-L336】 | CLI 更贴近项目习惯，也方便与 Ray 资源、dataset 注册流程统一。 |
| 复用点 | 使用 rLLM 的 BaseAgent/BaseEnv 基类、Ray 并发、Verl GRPO 算法 | 同上 | 核心算法与并行设施仍然来自 rLLM/Verl，两边在此保持一致。 |

- **我们直接复用的 rLLM 组件**：
  - `BaseAgent` / `Trajectory` / `Step` 等抽象定义仍来自 `rllm.rllm.agents.agent`，Graph Planner 的自定义 Agent 仅在这些基类上扩展额外字段与解析逻辑。【F:rllm/rllm/agents/agent.py†L1-L120】【F:graph_planner/integrations/rllm/agent.py†L56-L200】
  - 环境包装遵循 `rllm.rllm.environments.base.base_env.BaseEnv` 的 `reset/step/close` 接口，使得我们的 RepoEnv/CGM 环境可以无缝挂载到 rLLM 训练器。【F:rllm/rllm/environments/base/base_env.py†L1-L64】【F:graph_planner/integrations/rllm/env.py†L1-L134】
  - 数据注册依赖 `rllm.rllm.data.dataset.Dataset` 与 `DatasetRegistry`，我们只是在写入 parquet 前做字段扁平化，继续沿用其磁盘布局与 Verl 派生文件命名规则。【F:rllm/rllm/data/dataset.py†L1-L128】【F:graph_planner/integrations/rllm/dataset.py†L28-L109】
  - 训练入口最终仍交给 `rllm.rllm.trainer.agent_trainer.AgentTrainer` 与 Verl 的 `TaskRunner`/GRPO 算法执行，我们的 CLI 只是在调用前构造 OmegaConf 覆写与资源校验。【F:rllm/rllm/trainer/agent_trainer.py†L1-L75】【F:rllm/verl/verl/trainer/ppo/core_algos.py†L246-L313】【F:scripts/train_planner_grpo.py†L200-L336】

#### 4.3.1.2 与 R2E-Gym 默认流程的对比

R2E-Gym 本身提供了 RepoEnv 环境、动作解析与容器运行时等通用组件。我们复用其容器底座，但在动作语义、记忆管理与奖励逻辑上另起炉灶。下表总结了两边的差异：

| 维度 | R2E-Gym 默认实现 | 本仓库现有实现 | 备注 |
| --- | --- | --- | --- |
| 容器运行时 | `RepoEnv` 直接实例化 `DockerRuntime`，根据数据集描述启动容器并暴露 `reset/step` 接口。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L18-L118】【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L40-L124】 | `SandboxRuntime` 根据配置在 `repoenv`、`r2e`、`docker` 三种后端间切换，并在 repoenv 模式下复用 R2E 的 `RepoEnv`/`DockerRuntime`，同时补上挂载、pip 工具安装和 git safe-directory 兜底。【F:graph_planner/runtime/sandbox.py†L1-L139】 | 通过统一运行时接口，我们可以在训练时根据任务选择最合适的后端，而无需修改上层 Agent 逻辑。 |
| 动作/指令格式 | `Action` 采用 XML 风格文本解析，`ParseCommandBash` 将动作映射为容器内命令，再由 `RepoEnv.step` 执行。【F:R2E-Gym/src/r2egym/agenthub/action/action.py†L1-L94】【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L28-L204】 | `core.actions` 使用 Pydantic 数据类描述 Explore/Memory/Repair/Submit 四类动作，`PlannerEnv.step` 根据类型调度记忆维护、图扩展与补丁生成。【F:graph_planner/core/actions.py†L1-L38】【F:graph_planner/env/planner_env.py†L90-L172】 | JSON Schema 化的动作定义方便直接传给 LLM/规则策略，也利于遥测与回放。 |
| 记忆与上下文 | R2E-Gym 主要提供命令执行与文件编辑工具，默认不维护代码子图或记忆结构。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L200-L244】 | `PlannerEnv` 在 `reset` 时挂载子图存储、候选生成与节点阅读逻辑，训练过程中持续更新图记忆与候选片段。【F:graph_planner/env/planner_env.py†L60-L160】 | 我们围绕补丁计划构建了额外的图记忆层，以便 CGM/Planner 共享上下文。 |
| 奖励策略 | `RepoEnv.calculate_reward` 仍返回常数 0，主要依赖后续扩展。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L220-L244】 | `GraphPlannerRLLMEnv` 在 step 结束时依据提交结果、步数、失败类型叠加奖励缩放、失败惩罚与超步惩罚，并在 `compute_final_reward` 中检查测试结果给终局奖励。【F:graph_planner/integrations/rllm/env.py†L24-L134】 | 自定义奖励有助于在 GRPO 中对“修复成功/失败”进行精细化反馈。 |
| 数据/任务描述 | R2E 数据集通过 JSON/Parquet 描述容器镜像、补丁、测试指令，并由 `DatasetRegistry` 注册。【F:rllm/rllm/data/dataset.py†L92-L148】 | `ensure_dataset_registered` 在写入 registry 前会展开 mount/env 字段、生成 Verl 兼容的 parquet 路径，并回传给训练/评测脚本使用。【F:graph_planner/integrations/rllm/dataset.py†L59-L109】 | 任务格式保持兼容，使我们既能复用 R2E 提供的资源，也能扩展自定义字段（奖励系数、CGM 策略开关等）。 |

总结：我们并没有直接照搬「ToolAgent + ToolEnvironment + AgentExecutionEngine」的最小示例，而是在其基础设施之上实现了专用的 Agent/Env，以匹配 Graph Planner 的动作集合、RepoEnv 容器交互和 CGM 契约。不过，两者在奖励算法（Verl GRPO/PPO）、并发管理、OmegaConf 配置体系等底层组件上完全一致。如果后续需要与原生示例互通，可参考本节表格逐项映射所需配置与抽象。

#### 4.3.2 训练脚本速查 / CLI quick start

> 希望“直接跑起来”时，可依次完成以下步骤；更深入的流程说明请参考前文各节。

1. **激活环境与依赖**：
   ```bash
   source .venv/bin/activate   # 或进入自定义虚拟环境
   export PYTHONPATH=$(pwd)
   ```
   - 执行 `pip install -e .` 与 `pip install -e rllm` 以安装 Graph Planner 与 rLLM 依赖。
   - 如 rLLM 子模块位于非默认路径，设置 `export GRAPH_PLANNER_RLLM_PATH=/abs/path/to/rllm`；`ensure_rllm_importable()` 会读取该变量完成导入。【F:graph_planner/infra/vendor.py†L1-L112】

2. **准备数据与模型**：
   - `DATASET_JSONL`：RepoEnv/Repo 任务描述文件；脚本会调用 `ensure_dataset_registered` 自动生成 Verl 可读的 parquet。【F:graph_planner/integrations/rllm/dataset.py†L85-L109】
   - `PLANNER_MODEL`：Planner 的 Hugging Face checkpoint（可以是仓库随附的 Toy 模型或自定义权重），默认放置在 `models/Qwen3-14B/`。
   - （可选）`CGM_MODEL`：若训练 Planner agent 并需要 CGM 输出补丁，可使用默认的 `models/CodeFuse-CGM/` 目录存放模型。

3. **执行训练命令**：
   ```bash
   PYTHONPATH=. python scripts/train_planner_grpo.py \
     --config configs/experiments/planner_grpo_4gpu.yaml \
     --overrides data.train_files=[\"$DATASET_JSONL\"] \
     --print-config
   ```
   - YAML 已包含默认的 batch size、max steps 等字段，可按需在 YAML 中调整；CLI 可通过 `--overrides` 追加 OmegaConf dotlist 覆写（例如自定义训练集路径或迭代总数）。
   - 如果 YAML 中关闭 `graph_planner.cgm` 环境，CGM 将保持禁用；若只训练补丁模型，可在 overrides 中设置 `graph_planner.planner.env.CGM_ENABLED=0` 等选项。
   - `--print-config` 会在真正启动前打印最终配置；结合 `--dry-run` 可以在注册数据与生成 docker manifest 后直接退出。`--ray-address` 用于连接已有集群，未指定时脚本会启动本地 Ray。并行、资源等参数建议在 YAML 或 `--overrides` 中修改。

4. **观察输出**：
   - rLLM 会在当前目录生成 `outputs/`（Hydra 配置、Ray 日志、checkpoint）。
   - Graph Planner 的遥测仍写入 `logs/`，可检查容器步骤、补丁 diff 与测试结果。【F:graph_planner/infra/telemetry.py†L20-L39】

#### 4.3.3 RepoEnv 冒烟测试（规则或本地 Planner）

在具备 Docker 权限的主机上，可通过下列步骤快速验证容器运行链路：

1. **准备环境**
   - 安装 Python 3.10+ 并创建虚拟环境，执行 `pip install -e .` 与 `pip install -e R2E-Gym` 获取 RepoEnv 依赖；
   - 确保 Docker daemon 可访问（本地或远程 socket）。
2. **拉取示例镜像与数据集**
- 样例数据位于 `datasets/r2e_gym/train.jsonl`，可根据需要自定义生成方式或使用 `scripts/prepare_datasets.py` 的输出；
   - 运行 `docker pull graph-planner/repoenv-sample:latest` 预热镜像。
3. **执行冒烟命令**
   ```bash
   PYTHONPATH=. python scripts/run_rule_agent.py \
     --backend repoenv \
     --ds-json /path/to/your_r2e_dataset.jsonl \
     --agent rule \
     --max-steps 6 \
     --report smoke_report.json
   ```
   - 使用 `--agent llm` 可切换到本地 Planner 模型；命令会打印奖励、补丁 diff，并将完整轨迹写入 `smoke_report.json`。
4. **检查日志**
    - `logs/test_runs.jsonl`、`logs/events.jsonl` 会追加遥测记录，包括阅读片段、补丁 diff、命令序列与测试结果；若需要快速回放旧版冒烟流程，可使用最小化 JSONL（仅含一条任务）运行 `scripts/run_rule_agent.py`，复现日志产出而无需额外单测文件。【F:scripts/run_rule_agent.py†L54-L136】

#### 4.3.4 16 GPU 并行配置

针对 16 张 A800 的集群，可以通过新增的 CLI 参数在同一容器内同时扩展模型推理、容器交互和 GRPO 训练：

1. **并行 agent / rollout 数量**：
   ```bash
   --overrides parallel.agents=32 parallel.workers=32 parallel.workflow_parallel=64
   ```
   上述 dotlist 会直接覆写 YAML 中的并发设置；`parallel.agents` 控制 Planner 并行实例数，`parallel.workers` 决定 Verl rollout worker 数量，`parallel.workflow_parallel` 限制 Ray workflow 并发度。【F:configs/experiments/planner_grpo_4gpu.yaml†L87-L124】

2. **Ray 与集群资源**：
   ```bash
   --overrides trainer.n_gpus_per_node=16 trainer.nnodes=1 resources.ray_cpus=256 resources.ray_gpus=16
   ```
   可继续通过 overrides 调整 `resources.*`、`ray.*` 字段，确保 Ray 初始化时正确声明 GPU/CPU 预算。【F:configs/experiments/planner_grpo_4gpu.yaml†L146-L198】

3. **模型并行 / Tensor Parallel**：如需在多卡上切分模型，可在 YAML 或 overrides 中设置 `parallel.tensor_parallel_planner`、`parallel.tensor_parallel_cgm` 以及 `actor_rollout_ref.rollout.tensor_model_parallel_size`；`HuggingFaceChatClient` / `CodeFuseCGMGenerator` 也支持 `device_map='auto'` 或显式列表，以覆盖多 GPU。【F:configs/experiments/planner_grpo_4gpu.yaml†L1-L80】【F:graph_planner/integrations/local_llm/hf.py†L19-L186】

4. **容器隔离**：新的 `issue_uid` 机制为每个并发任务生成独立的子图缓存键，防止 16 个容器同时写 `.aci/subgraphs` 时互相覆盖。【F:graph_planner/integrations/rllm/env.py†L46-L134】【F:graph_planner/integrations/rllm/cgm_env.py†L70-L197】

5. **配置验证**：执行 `--print-config` 或 `--dry-run` 可在启动前打印最终合并后的 OmegaConf 配置并提前注册数据，确认资源字段与并行度满足目标需求。
#### 4.3.5 验证脚本（evaluation-only）

当只需要收集 pass@k / 成功率指标而不做参数更新时，可使用 `scripts/eval_planner_grpo.py`：

1. `--config` 指向训练时使用的 YAML，`--ckpt` 指向待评估的 checkpoint 目录；脚本会复用数据注册与 docker manifest 逻辑，必要时自动 materialize JSONL。【F:scripts/eval_planner_grpo.py†L90-L168】
2. `--overrides`、`--ray-address`、`--print-config` 与 `--dry-run` 的行为与训练脚本一致，可在不启动 Ray 的情况下审计配置或导出最终 YAML。【F:scripts/eval_planner_grpo.py†L48-L136】
3. 运行示例：
   ```bash
   PYTHONPATH=. python scripts/eval_planner_grpo.py \
     --config configs/experiments/planner_grpo_4gpu.yaml \
     --ckpt /path/to/checkpoint \
     --print-config
   ```
   - 如需针对其他验证集或模型，可通过 `--overrides data.val_files=[...] actor_rollout_ref.model.path=...` 等 dotlist 覆写字段。
   - `trainer.val_only: true` 会强制只运行验证阶段；结合 `--dry-run` 可以在注册数据与加载 checkpoint 后立即退出以审计配置。
   - CLI 会沿用训练脚本的并行/资源检查逻辑，确保在进入 Ray/Verl 之前就能发现配置不一致的问题。【F:scripts/eval_planner_grpo.py†L120-L192】

### 4.4 远端 CGM / 本地 fallback
- 若 `.aci/config.json` 或环境变量启用了 `cgm.endpoint`，`cgm_adapter` 会优先使用 `CodeFuseCGMClient` 向官方服务发起请求；否则回退到本地 `CodeFuseCGMGenerator` 或规则标记补丁，确保补丁流程不会因模型缺失而中断。【F:graph_planner/agents/rule_based/cgm_adapter.py†L145-L207】
- 在 rLLM/Verl 训练场景下，`CGMService` 会在检测到 vLLM 缺席时切换到 `actor.cgm_local.generate`，避免再经由 Ray RPC，确保 Planner↔CGM 闭环保持纯本地执行。【F:rllm/verl/verl/workers/cgm_service.py†L74-L123】【F:actor/cgm_local.py†L1-L63】
- Planner 模型路径可通过环境变量或配置文件注入，使规则策略、本地 LLM、rLLM 训练共用同一组配置读取逻辑。【F:graph_planner/infra/config.py†L60-L176】

## 5. 常用命令 / Recommended commands

| 场景 | 命令 |
| --- | --- |
| 规则策略冒烟 | `PYTHONPATH=. python scripts/run_rule_agent.py --backend repoenv --ds-json /path/to/your_r2e_dataset.jsonl --max-steps 6 --agent rule` |
| 本地 LLM 调试 | `PYTHONPATH=. python scripts/run_rule_agent.py --backend repoenv --ds-json /path/to/your_r2e_dataset.jsonl --agent llm --planner-model-path <hf-model>` |
| CGM 本地推理 | `python - <<'PY'`<br>`from graph_planner.integrations.codefuse_cgm import CodeFuseCGMGenerator, CGMExample;`<br>`gen = CodeFuseCGMGenerator.from_pretrained("<model>");`<br>`example = CGMExample.from_json_file("sample.json");`<br>`print(gen.generate(example).model_dump())`<br>`PY` |
| CGM 监督微调 | `PYTHONPATH=. python - <<'PY'`<br>`from graph_planner.integrations.codefuse_cgm import CodeFuseCGMTrainer, CGMTrainingConfig;`<br>`cfg = CGMTrainingConfig(model_name="<model>", train_path="train.jsonl", output_dir="runs/cgm");`<br>`CodeFuseCGMTrainer(cfg).train()`<br>`PY` |
| rLLM PPO 训练 | `PYTHONPATH=. python scripts/train_planner_grpo.py --config configs/experiments/planner_grpo_4gpu.yaml --print-config` |

- `scripts/run_eval_graph_planner.sh` 会把默认配置 `configs/eval/graph_planner_eval_defaults.yaml` 注入评估 CLI；若 `planner_base_url` 指向本机且给出了 `planner_model_path`，脚本会根据配置自动拉起 vLLM OpenAI 服务（默认映射到 GPU `0,1`，tensor parallel 为 2，gpu memory utilization 为 0.9）。启动前脚本会先通过 `nvidia-smi` 检查所选 GPU 的剩余显存，并在必要时按 `planner_service_gpu_memory_headroom` 预留 5% 安全边界后降低 `gpu_memory_utilization`，避免 GPU 已被占用时启动失败。需要禁用或自定义设备时，可在 YAML/CLI 覆写 `auto_launch_planner_service`、`planner_service_gpus`、`planner_service_tensor_parallel_size`、`planner_service_gpu_memory_utilization` 或 `planner_service_gpu_memory_headroom`。【F:scripts/eval_graph_planner_engine.py†L84-L239】【F:scripts/eval_graph_planner_engine.py†L517-L586】【F:configs/eval/graph_planner_eval_defaults.yaml†L12-L17】

## 6. 日志与调试建议 / Logging & troubleshooting

- 训练或推理期间的容器命令、测试结果会写入 `logs/test_runs.jsonl` 与 `logs/telemetry/*.jsonl`，可配合 `infra.telemetry` 工具分析。【F:graph_planner/infra/telemetry.py†L20-L39】
- 若 rLLM 模型输出解析失败，可启用 `--use-fallback` 并检查 `GraphPlannerRLLMAgent` 记录的 `fallback_reason`；这些信息会出现在 PPO 轨迹和日志中。【F:graph_planner/integrations/rllm/agent.py†L156-L200】
- `ensure_rllm_importable()` 在导入 rLLM 前会自动调整 `sys.path` 并验证子模块结构；如遇导入失败可检查环境变量 `GRAPH_PLANNER_RLLM_PATH`。【F:graph_planner/infra/vendor.py†L1-L86】

> **Update history**: 本文整合了原 `cgm_rllm_pipeline.md`、`r2e_gym_code_overview.md`、`r2e_training_pipeline.md`、`rllm_integration_report.md` 的内容，并加入最新的本地 LLM、CGM 训练与 rLLM 配置指引。
