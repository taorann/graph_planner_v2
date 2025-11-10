# Graph Planner GRPO Runbook

本指南说明如何利用仓库现有模块组装 Planner/CGM 的 GRPO 训练流程，涵盖数据准备、配置合并、训练入口以及常见排障要点。新的训练 CLI 正在重构中，但下述步骤已经覆盖 rLLM/Verl 所需的全部组件，便于在自定义脚本或 Notebook 中快速复现实验。

## 0. 数据准备

1. **下载/转换数据集**
   ```bash
   PYTHONPATH=. python scripts/prepare_datasets.py \
     --r2e-dataset R2E-Gym/R2E-Gym-Lite \
     --swebench-dataset princeton-nlp/SWE-bench_Verified
   ```
   脚本会在 `datasets/` 目录下生成标准化的 JSON/JSONL、`instances/*.json` 以及 docker manifest；可以通过 `--skip-r2e`、`--skip-swebench`、`--prepull-*` 精细控制各阶段。【F:scripts/prepare_datasets.py†L12-L214】

2. **注册到 rLLM 数据集仓库**：
   ```bash
   PYTHONPATH=. python scripts/register_graphplanner_dataset.py \
     --name graph_planner_repoenv \
     --split train \
     --jsonl datasets/r2e_gym/train.jsonl
   ```
   或者在训练脚本中直接调用 `ensure_dataset_registered`，该函数会生成 Verl 兼容的 `_verl.parquet` 并写回配置。【F:graph_planner/integrations/rllm/dataset.py†L85-L137】

## 1. YAML 与配置合并

`configs/experiments/planner_grpo_4gpu.yaml` 仍是推荐的起点，定义了模型路径、采样策略、并行度与 Ray 资源。配置合并流程与旧版 CLI 保持一致：

- `graph_planner.infra.config.load_config()` 负责读取 YAML、应用 dotlist 覆写并写出 `resolved_config.yaml`，同时注入模型路径、环境变量等运行期开关。【F:graph_planner/infra/config.py†L42-L285】
- `graph_planner.infra.parallel.resolve_parallel()` 会根据最终配置计算张量并行、梯度累积与世界尺寸，并做 GPU/Ray 资源预检，避免训练期间出现显存或 worker 数量不匹配。【F:graph_planner/infra/parallel.py†L1-L134】
- `graph_planner.infra.metrics.init_wandb()` 可在需要时开启 W&B 或离线日志，保持与评测 CLI 相同的遥测格式。【F:graph_planner/infra/metrics.py†L1-L93】

## 2. 组装训练入口

新的训练脚本可按以下步骤搭建：

1. **确保 rLLM 可导入**：调用 `graph_planner.infra.vendor.ensure_rllm_importable()`，自动把 vendored rLLM/Verl 插入 `sys.path` 并校验结构。【F:graph_planner/infra/vendor.py†L1-L112】
2. **注册自定义组件**：
   ```python
   from graph_planner.integrations.rllm import registry
   registry.register_rllm_components()
   ```
   这样 Hydra/OMEGACONF 配置即可解析 `GraphPlannerRLLMAgent`、`GraphPlannerRLLMEnv` 等自定义类。【F:graph_planner/integrations/rllm/registry.py†L1-L38】【F:graph_planner/integrations/rllm/agent.py†L56-L210】【F:graph_planner/integrations/rllm/env.py†L35-L177】
3. **构建 `AgentTrainer`**：
   ```python
   from graph_planner.infra import config as cfg_mod
   from graph_planner.infra import parallel as parallel_mod
   from rllm.rllm.trainer.agent_trainer import AgentTrainer

   cfg = cfg_mod.load_config("configs/experiments/planner_grpo_4gpu.yaml", overrides=[...])
   parallel_mod.resolve_parallel(cfg)
   trainer = AgentTrainer(cfg)
   trainer.train()
   ```
   `AgentTrainer` 会自动创建执行引擎、Ray runtime、Verl GRPO 算法，并在训练过程中调用前述 agent/env 适配层。【F:rllm/rllm/trainer/agent_trainer.py†L1-L75】【F:rllm/verl/verl/trainer/ppo/core_algos.py†L246-L313】
4. **回调与日志**：`graph_planner.infra.telemetry` 负责写入 `logs/test_runs.jsonl`、`logs/events.jsonl` 等文件，保留和评测 CLI 相同的回放数据，便于调试与对齐指标。【F:graph_planner/infra/telemetry.py†L20-L56】

## 3. 监控与常见检查

- **批量与并行**：`resolve_parallel` 会在日志中输出有效 batch size、梯度累积与世界尺寸；若配置不一致会显式告警，提示调整 `parallel.*` 或 `trainer.*` 字段。【F:graph_planner/infra/parallel.py†L60-L134】
- **Ray 环境变量**：`AgentTrainer` 会打印最终的 `runtime_env`，可确认 Planner/CGM 模型路径、温度与显存限制均已同步到远程 worker。【F:rllm/rllm/trainer/agent_trainer.py†L39-L75】
- **W&B / 本地日志**：若在 YAML 中开启 `metrics.enable_wandb`，`init_wandb` 会自动处理 API Key 缺失、离线模式等场景；关闭时仍会写入本地 JSONL 记录。【F:graph_planner/infra/metrics.py†L25-L93】

## 4. 故障排查建议

1. **数据路径异常**：`ensure_dataset_registered` 会在 JSONL 缺失时抛出 `FileNotFoundError`，请确认路径与仓库根目录一致，或在 overrides 中显式传入绝对路径。【F:graph_planner/integrations/rllm/dataset.py†L85-L137】
2. **FSDP/Tensor Parallel 配置错误**：`resolve_parallel` 会检查 `tensor_parallel_planner`、`tensor_parallel_cgm` 等字段，若不满足约束会提前报错，避免训练时才触发 RuntimeError。【F:graph_planner/infra/parallel.py†L90-L134】
3. **Ray 地址或资源不足**：在 `trainer.runtime` 中将 `ray.address="local"` 可强制启动本地 Ray；结合 `cfg_mod.dump_resolved_config(cfg)` 或 `--dry-run`（待新 CLI 提供）可以在正式启动前检查资源声明。
4. **CGM/Planner 服务**：评测与训练共用的自动部署逻辑位于 `scripts/eval_graph_planner_engine.py`，其中的 `_configure_runtime_env`、`_auto_launch_vllm_service` 可作为参考，在自定义脚本中复用同样的环境变量与服务启动流程。【F:scripts/eval_graph_planner_engine.py†L84-L353】

通过以上步骤，即可在不依赖旧版 CLI 的情况下复现 Graph Planner 的 GRPO 训练：数据准备与注册仍由脚本辅助完成，配置与资源校验交给 `infra` 子模块处理，最后由 `AgentTrainer` 驱动实际的 rLLM/Verl 训练循环。
