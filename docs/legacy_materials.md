# Legacy CLI 与文档清单

> **2025-11-07 记录**：仓库已移除 `scripts/run_rule_agent.py` 与 `scripts/train_planner_grpo.py`，下列文档仅保留历史上下文，帮助未来重建训练入口时参考旧流程。

| 文件 | 当前状态 | 后续建议 |
| --- | --- | --- |
| `docs/runbook.md` | 全文围绕 `scripts/train_planner_grpo.py` 的 YAML/Ray 配置说明，暂无法直接执行。 | 待新的训练 CLI 成型后，按同样结构重写并替换示例命令。 |
| `docs/pain-points.md` | 多处引用 train_planner_grpo 作为缓解措施示例。 | 在新训练链路确定后重跑审计，删除或更新相关小节。 |
| `docs/rllm_code_survey.md` | 训练入口分析以旧 CLI 为核心。 | 重新梳理 rLLM/VERL 调用栈，补充新的入口函数与配置结构。 |
| `docs/planner train eval split plan.md` | 描述旧版 Ray shared actor 启动流程。 | 将 Ray actor 启动说明迁移到新的训练脚本或评测文档，旧内容可归档。 |
| `docs/tool_prompt_contract_survey.md` | 仍引用 `scripts/run_rule_agent.py` 作为合同回放手段。 | 用 `eval_graph_planner_engine.py` 的示例替代，或提供新的最小化回放脚本。 |

如需恢复基于 rLLM 的训练，请以 `scripts/eval_graph_planner_engine.py` 为参考复用容器与模型编排逻辑，再结合 rLLM/VERL 的 `AgentTrainer` 构建新的 GRPO CLI。
