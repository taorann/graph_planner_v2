# 工具与提示协议调研（Graph Planner vs. rLLM / R2E-Gym）

## 1. Graph Planner 本地工具实现

| 组件 | 作用 | 关键实现 | 当前用途 |
| --- | --- | --- | --- |
| `core/actions.py` | 定义 Planner 可以下发的 5 类动作（explore/memory/repair/submit/noop），规定每类字段与默认值，是文本协议与环境之间的结构桥梁。 | `ExploreAction`/`MemoryAction`/`RepairAction`/`SubmitAction`/`NoopAction` 模型约束了工具调用所需的 anchors、memory target/scope/intent、plan 等字段。【F:graph_planner/core/actions.py†L1-L40】 | Planner 环境根据动作类型分派到扩展、记忆维护、CGM 修复、提交或显式空操作。 |
| `agents/rule_based/tool_policy.py` | 规则基线的“第 1 步”决策器，结合 issue 描述、失败栈与 token 预算，生成 anchors/terms/hop 等下一步探索配置并建议 `next_tool`。 | `decide(state, cfg)` 根据子图规模、失败路径与 token 软上限裁剪 anchors/terms，决定是否继续扩展或转入查看/测试。【F:graph_planner/graph_planner/agents/rule_based/tool_policy.py†L1-L200】 | 供规则版 Planner 与 RL 小头共享，确保后续工具调用有统一的输入。 |
| `agents/rule_based/tool_selector.py` | “第 5 步”工具调度器，独立判定下一轮应执行 expand/view/search/edit/test/lint/noop，并附带优先测试列表。 | `choose_next_tool(state)` 按“刚修→test”“lint 失败→lint”“上下文不足→expand”等优先级切换工具；支持 RL 覆盖钩子。【F:graph_planner/graph_planner/agents/rule_based/tool_selector.py†L1-L200】 | 贯穿规则策略、RL 环境与评测脚本，决定 planner 动作序列。 |
| `env/planner_env.py` | 运行时容器交互层，负责解析动作、执行 Sandbox 操作并回传 Observation。 | `_handle_repair` 先尝试直接应用 Planner 自带 patch，否则构造 `RepairRuntimeState` 调用文本协议修复链，最终把结果合并进 observation。【F:graph_planner/graph_planner/env/planner_env.py†L240-L289】 | 训练与评测时唯一的环境实现，向 rLLM 暴露 `BaseEnv` 接口。 |

## 2. 文本工具调用格式

### 2.1 动作执行映射一览

#### 2.1.1 工具实现文件速查

| 动作 | 数据类定义 | 环境入口 | 依赖工具/模块 | 实现状态 |
| --- | --- | --- | --- | --- |
| `explore` | `graph_planner/core/actions.py` → `ExploreAction`【F:graph_planner/core/actions.py†L1-L40】 | `graph_planner/env/planner_env.py` → `_handle_explore`【F:graph_planner/env/planner_env.py†L97-L169】 | `graph_planner/graph/context.py`（子图上下文）、`graph_planner/memory/candidates.py`（候选生成）、`graph_planner/retriever/snippet_reader.py`（代码片段）等 | ✅ 已实现，可根据 anchors 与 budget 选择 `find`/`expand`/`read`。 |
| `memory` | `MemoryAction`【F:graph_planner/core/actions.py†L42-L74】 | `_handle_memory`【F:graph_planner/env/planner_env.py†L171-L233】 | `graph_planner/memory/text_memory.py`（`memory_commit`/`memory_delete`、配额估算）【F:graph_planner/memory/text_memory.py†L1-L407】 | ✅ 已实现，支持 explore/observation 目标、turn/session scope 与 over-budget 拒绝。 |
| `repair` | `RepairAction`【F:graph_planner/core/actions.py†L76-L118】 | `_handle_repair`【F:graph_planner/env/planner_env.py†L235-L327】 | `graph_planner/agents/common/text_protocol.py`（`handle_planner_repair`、CGM payload）【F:graph_planner/agents/common/text_protocol.py†L189-L401】；`graph_planner/agents/rule_based/cgm_adapter.py`（本地 CGM 客户端）【F:graph_planner/agents/rule_based/cgm_adapter.py†L1-L358】 | ✅ 已实现，Planner 只写 subplan，CGM 生成统一 diff 并可自动测试。 |
| `submit` | `SubmitAction`【F:graph_planner/core/actions.py†L120-L150】 | `_handle_submit`【F:graph_planner/env/planner_env.py†L329-L352】 | `graph_planner/tools/testing.py`（`run_all_checks`）、`graph_planner/tools/git_utils.py`（补丁状态）等 | ✅ 已实现，负责最终测试与 episode 结束。 |
| `noop` | `NoopAction`【F:graph_planner/core/actions.py†L152-L170】 | `PlannerEnv.step` 顶层判断【F:graph_planner/env/planner_env.py†L61-L95】 | 无（直接回传空 observation） | ✅ 已实现，允许模型显式跳过操作。 |

| 动作 | 触发位置 | 环境/实现 | 说明 |
| --- | --- | --- | --- |
| `explore` | Planner 输出 `<function=explore>`；`LocalLLMPlannerAgent._action_from_block`/`action_from_payload` 解析为 `ExploreAction`。 | `PlannerEnv._handle_explore` 调用锚点扩展、图检索与 snippet 读取。【F:graph_planner/agents/model_based/planner.py†L224-L247】【F:graph_planner/agents/common/chat.py†L106-L123】【F:graph_planner/env/planner_env.py†L97-L169】 | 已实现；支持 `find`/`expand`/`read` 三种 `op`。 |
| `memory` | `<function=memory>`；解析为 `MemoryAction`。 | `text_memory.memory_commit`/`memory_delete` 根据 target/scope/intent 处理探索候选或工具 observation，`PlannerEnv._handle_memory` 负责调用并持久化子图/文本记忆。【F:graph_planner/memory/text_memory.py†L1-L407】【F:graph_planner/env/planner_env.py†L171-L233】 | 已实现；支持 over-budget 拒绝、commit/delete 版本号递增。 |
| `repair` | `<function=repair>`；解析为 `RepairAction`。 | `_handle_repair` 若带 patch 则直接应用，否则调用 `text_protocol.handle_planner_repair` → CGM 修复链。【F:graph_planner/env/planner_env.py†L235-L327】 | 已实现；遵循统一 diff 与 Sandbox 测试流程。 |
| `submit` | `<function=submit>`；解析为 `SubmitAction`。 | `_handle_submit` 运行测试并返回终局结果。【F:graph_planner/env/planner_env.py†L329-L352】 | 已实现；触发奖励与 episode 结束。 |
| `noop` | `<function=noop>`；解析为 `NoopAction`。【F:graph_planner/agents/model_based/planner.py†L247-L249】【F:graph_planner/agents/common/chat.py†L123-L151】 | `PlannerEnv.step` 识别 `NoopAction` 并返回 {"kind": "noop"}，不上链任何容器操作。【F:graph_planner/env/planner_env.py†L61-L95】 | 新增显式空操作，便于模型放弃本轮动作且保持协议一致。 |
Graph Planner 约定 **单回合仅一个 `<function=...>` 块**，内含多段 `<param>`。`parse_action_block` 会验证起止标签、参数唯一性与合法动作名，不允许额外文本或重复参数，从而把模型输出解成 `{"name": ..., "params": {...}}` 结构；`format_action_block` 用于 fallback 再输出同样的文本协议。【F:graph_planner/graph_planner/agents/common/text_protocol.py†L59-L167】

Planner 将环境 observation 包装成 `<observation for="{name}">{...JSON...}</observation>`，下一轮模型可直接读取；这一封装由 `emit_observation` 统一实现并在单测中确保严格输出 JSON。【F:graph_planner/graph_planner/agents/common/text_protocol.py†L330-L335】【F:graph_planner/tests/test_text_protocol.py†L136-L141】

## 3. 模型提示模板与输出契约

我们用 `PromptContract` 聚合 system prompt、用户分区与响应约定。`PLANNER_CONTRACT` 现要求模型回复单个 `<function=...>` 区块，并给出 `<param name="thought">`、`subplan`（CDATA 包裹）、`focus_ids` 等键；`CGM_CONTRACT` 仍约束补丁响应须包含 `patch.edits[{path,start,end,new_text}]` 等字段。【F:graph_planner/graph_planner/agents/common/contracts.py†L17-L137】

这些常量被 Planner 聊天客户端和 CGM 生成器共享，保证训练/推理与文档一致；若需要向文档输出模板，可直接调用 `formatted_user_template()` 与 `formatted_response_schema()`。【F:graph_planner/graph_planner/agents/common/contracts.py†L43-L58】

## 4. CGM 修复链路契约

当 Planner 触发 `repair`，`handle_planner_repair` 会：

1. 校验 `subplan` 非空，标准化 `focus_ids`、`apply`。【F:graph_planner/graph_planner/agents/common/text_protocol.py†L337-L353】
2. `build_cgm_payload` 裁剪子图节点、记忆与关联文件，生成 `plan` 步骤列表，并附带 `constraints.one_file_per_patch=true`。【F:graph_planner/graph_planner/agents/common/text_protocol.py†L203-L226】
3. 调用 CGM，挑选最高置信度候选。【F:graph_planner/graph_planner/agents/common/text_protocol.py†L229-L243】
4. `validate_unified_diff` 确认 diff 仅涉及单个文件且包含至少一个 hunk。【F:graph_planner/graph_planner/agents/common/text_protocol.py†L245-L291】
5. 如 `apply=true`，`try_apply_and_test` 会调用 Sandbox 补丁/重置/测试/Lint，并封装结果 JSON，失败阶段写入 `error`。【F:graph_planner/graph_planner/agents/common/text_protocol.py†L297-L385】

环境侧在 `_handle_repair` 中构造 `RepairRuntimeState`，并把结果写回信息字典，从而与奖励、观测打通。【F:graph_planner/graph_planner/env/planner_env.py†L269-L289】

## 5. 与 rLLM 工具协议的异同

rLLM 默认的 `ToolAgent` 依赖 `ToolParser` 解析模型输出中的 **函数调用 JSON**：模型需生成 `function` 名称与 `arguments` JSON；解析失败会降级成 `finish` 工具调用。【F:graph_planner/rllm/rllm/agents/tool_agent.py†L17-L147】

`ToolParser` 的具体子类（如 `QwenToolParser`/`R1ToolParser`）识别特殊 token 或 ```json``` 片段，从而还原工具参数结构，与我们的单块 `<function=...>` 协议不同，但同样强调“动作必须合法+参数 JSON 解析”。【F:graph_planner/rllm/rllm/parser/tool_parser.py†L1-L160】

两者共同点：
- 都将合法动作集写死在解析阶段（我们用 `allowed` 集合，rLLM 用 tool schema）。
- 都会在解析失败时快速失败或降级，避免模型输出污染环境。

差异：
- Graph Planner 用单块文本标签，更易人工调试；rLLM 依赖模型内置的函数调用 token，利于 OpenAI/Qwen 等兼容。
- 我们的观测返回 `<observation>` 字符串，而 rLLM `ToolEnvironment` 直接返回 `tool_outputs` dict，由引擎拼成对话消息。【F:graph_planner/rllm/rllm/agents/tool_agent.py†L63-L141】

## 6. 与 R2E-Gym Prompt 的对比

R2E-Gym 的编辑代理配置以 YAML 描述 system/user prompt，并强制“每次回复都要携带函数调用”，强调脚本化 reproduce 步骤与最少修改原则。例如 `edit_fn_calling.yaml` 将 GitHub issue 嵌入 `<github_issue>` 标签，并反复提醒“每轮必须输出 function call”。【F:graph_planner/R2E-Gym/src/r2egym/agenthub/config/r2egym/edit_fn_calling.yaml†L1-L45】

我们的 `PromptContract` 要求 `<function=...>` 文本轨迹，由环境决定是否触发工具；Planner 如果暂时无操作，可输出 `<function=noop></function>` 走空操作。R2E-Gym 更偏向函数调用框架（类似 rLLM ToolAgent），而 Graph Planner 采用轻量文本协议，方便在 CGM 修复管线中插入统一 diff 校验与 Sandbox 回滚。

## 7. 可复用与待留意项

* **可共用理念**：三套体系都把“工具 schema + 严格解析”作为守门人，可互相借鉴；若未来迁移到 rLLM ToolAgent，只需把 `<function>` 协议映射到 `ToolCall` JSON。
* **提示差异**：R2E-Gym/rLLM 都强调多轮 reasoning + function call，我们在 `PLANNER_CONTRACT` 中同样要求 `thought` 字段，可以吸收其“强制工具调用”策略作为训练约束。
* **输出契约**：Graph Planner 通过 `CGM_CONTRACT` 与 `validate_unified_diff` 限制单文件补丁，避免模型生成多文件 diff；这与 R2E-Gym 直接允许多工具编辑的策略不同，需要在后续文档中继续强调。

