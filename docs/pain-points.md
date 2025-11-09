> ⚠️ **2025-11-07 提醒**：`scripts/run_rule_agent.py` 与 `scripts/train_planner_grpo.py` 已移除，本文件保留旧版流程以供参考；请结合 `docs/legacy_materials.md` 与 `scripts/eval_graph_planner_engine.py` 获取最新评测入口。

# Graph Planner Pain Points & Resolutions

> 本文汇总 Graph Planner 在演进过程中遇到的核心痛点，并记录目前的缓解方案与后续行动项。
> **2025-11-03 审核结论**：痛点章节中引用的模块（contracts/sandbox/train_planner_grpo 等）已核实仍在仓库内，对应缓解措施仍适用。

 Graph Planner 在演进过程中遇到的核心痛点，并记录目前的缓解方案与后续行动项。内容覆盖协议校验、补丁落盘、训练集成与遥测几大模块，便于开发者在排查问题时快速定位到唯一事实来源（SSOT）。

## Contract-as-Code：协议作为唯一事实来源

我们把 Planner 与 CGM 的动作枚举、提示词片段、错误码与校验器统一收敛到 [`graph_planner/agents/common/contracts.py`](../graph_planner/agents/common/contracts.py)，由 `PLANNER_CONTRACT` 与 `validate_cgm_patch` 负责解析与验证。【F:graph_planner/agents/common/contracts.py†L18-L206】所有新增动作都必须在该模块声明允许参数、必填字段与 Pydantic 数据模型，并在同文件内补充错误码常量，保持消息解析的确定性。【F:graph_planner/agents/common/contracts.py†L231-L336】

- `validate_planner_action` 会在入站时执行 JSON Schema 校验、参数去重与类型转换，出现未知字段时直接抛出 `ProtocolError`。
- `validate_cgm_patch` 保证补丁 JSON 结构完备，触发 `invalid-patch-schema`、`newline-missing` 等错误码时立即中止流程。
- `scripts/validate_contracts.py` 提供了本地冒烟入口，可独立验证 Planner/CGM 协议是否漂移。【F:scripts/validate_contracts.py†L16-L120】

## 原子化补丁：防止脏工作区

补丁写入统一交给 [`PatchApplier.apply_in_temp_then_commit`](../graph_planner/runtime/sandbox.py) 负责：它先在临时目录重放补丁并运行 lint/测试，通过后才用 `os.replace` 覆盖真实仓库。【F:graph_planner/runtime/sandbox.py†L150-L228】若 diff 头部与原文件不一致、hunk 校验失败或编码异常，`validate_unified_diff` 会抛出 `invalid-unified-diff`、`hunk-mismatch`、`encoding-unsupported` 等错误码，所有失败都会带上 `fallback_reason` 与 `temp_path` 遥测，便于回溯。【F:graph_planner/runtime/sandbox.py†L230-L284】

当前的遥测字段包含 `patch_id`、`n_hunks`、`added_lines` 与 `removed_lines`，用于识别重复补丁或统计 diff 规模；重复提交会触发 `duplicate-patch` 拦截，从而避免污染主工作区。

## rLLM 训练与回放的常见阻力

强化学习流水线聚焦在 `graph_planner.integrations.rllm` 与训练脚本：

- [`scripts/train_planner_grpo.py`](../scripts/train_planner_grpo.py) 解析 YAML、注册数据集并拼装 Ray Runtime，启动时自动注入 Planner/CGM 路径与奖励管理器配置。【F:scripts/train_planner_grpo.py†L322-L468】
- [`graph_planner/integrations/rllm/env.py`](../graph_planner/integrations/rllm/env.py) 将 RepoEnv / DockerRuntime 包装成 rLLM 兼容的环境，暴露 `step`/`reset` 接口供 PPO 训练循环直接复用。【F:graph_planner/integrations/rllm/env.py†L13-L110】
- 奖励模型在 `train_agent_ppo._maybe_load_reward_managers` 中按需加载，缺少配置时会回退为 `None`，防止离线调试时因为奖励依赖导致崩溃。【F:rllm/rllm/trainer/verl/train_agent_ppo.py†L124-L195】

常见痛点包括 YAML 覆写后路径不一致、容器预拉失败或奖励函数缺失。建议配合 `--print-config`、`--dry-run` 与 `scripts/prepare_datasets.py --prepull-*` 逐步排除环境问题。

## 仍在跟踪的事项

- **动作扩展与回归**：新增 Planner 动作时需同时补充 `tests/test_reward_manager_loading.py` 以覆盖奖励管理器路径，以及 rLLM 侧的解析测试，避免协议漂移。【F:tests/test_reward_manager_loading.py†L1-L45】
- **遥测统一**：计划把 `fallback_reason`、补丁规模等字段接入集中日志管道，便于跨任务对比修复瓶颈。
- **容器依赖压测**：`graph_planner/runtime/containers.py` 中的镜像收集与预拉逻辑仍需在大规模数据集上验证性能，后续会补充新的基准测试。

> 若遇到未覆盖的痛点，请在 `docs/pain-points.md` 追加案例，并附带触发场景与缓解建议，方便后续版本收敛。
