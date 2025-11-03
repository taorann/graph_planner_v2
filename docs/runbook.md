# Graph Planner GRPO Runbook

本指南说明如何使用 `scripts/train_planner_grpo.py` 复现实验配置、准备数据以及检查运行环境。
> **2025-11-03 审核结论**：runbook 中引用的脚本路径与配置键均已对照最新版仓库确认有效，训练日志提醒项保持不变。

新版流程只保留了一个公开的 YAML——`configs/experiments/planner_grpo_4gpu.yaml`——其余配置项通过 CLI dotlist 覆写即可。

## 0. 数据准备

1. **下载/转换数据集**：
   ```bash
   PYTHONPATH=. python scripts/prepare_datasets.py \
     --r2e-dataset R2E-Gym/R2E-Gym-Lite \
     --swebench-dataset princeton-nlp/SWE-bench_Verified
   ```
   脚本会在 `datasets/` 目录下生成标准化的 JSON/JSONL 以及 `instances/*.json`，并按需写入 docker manifest；可以通过 `--skip-r2e`、`--skip-swebench`、`--prepull-*` 控制各阶段。

2. **确认训练/验证文件**：`configs/experiments/planner_grpo_4gpu.yaml` 的 `data.train_files` / `data.val_files` 默认指向 `datasets/r2e_gym/{train,val}.jsonl`，如需替换成自定义任务，只需修改 YAML 或在 CLI 中传入 `--overrides data.train_files=[...]`。

3. **预拉容器（可选）**：当 YAML 中启用 `graph_planner.env.prepull_containers=true` 时，训练脚本会扫描 JSONL 任务并调用 `_maybe_prepull_docker_images`，逐个执行 `docker pull` 缓解首次 rollout 的等待时间。【F:scripts/train_planner_grpo.py†L293-L320】

## 1. YAML 字段速览

`planner_grpo_4gpu.yaml` 覆盖了模型、采样、并行、资源与 Ray runtime 设置，重点字段如下：

- `paths.*`：Planner / CGM 权重与 tokenizer 的本地目录。训练脚本会通过 `resolve_repo_path` 把它们转换为绝对路径，并填充到 Ray runtime 的环境变量中。【F:scripts/train_planner_grpo.py†L110-L150】【F:scripts/train_planner_grpo.py†L207-L252】
- `data.*`：训练/验证 JSONL 列表以及 dataloader 相关参数。脚本会在启动前调用 `_maybe_materialize_json_to_verl_parquet` 将 JSONL 转换成 `_verl.parquet` 并写回配置，后续的 GRPO 流程直接消费 parquet。【F:scripts/train_planner_grpo.py†L322-L362】
- `actor_rollout_ref.*`：Planner 模型的 FSDP 策略、学习率和 rollout 配置，默认固定在 4×GPU GRPO 场景。`scripts/train_planner_grpo.py` 会验证 strategy 是否为 FSDP，并保证 tensor parallel 度为 1。【F:scripts/train_planner_grpo.py†L237-L284】
- `trainer.*`、`parallel.*`、`resources.*`：声明梯度累积、保存/评估周期与 Ray 资源，用于构建 Verl 的训练器和 runtime。

## 2. CLI 与覆写

`scripts/train_planner_grpo.py` 的核心参数：

- `--config`：必填，指向 YAML（默认值 `configs/experiments/planner_grpo_4gpu.yaml`）。
- `--overrides`：可选的 OmegaConf dotlist，例如 `--overrides trainer.total_epochs=200 data.train_files=[my/train.jsonl]`。
- `--ray-address`：连接既有集群；`local` 表示强制启动本地 Ray，`auto` 沿用 Ray 默认逻辑。【F:scripts/train_planner_grpo.py†L371-L412】
- `--print-config`：打印最终合并后的 YAML 并退出，常用于检查路径/环境变量。【F:scripts/train_planner_grpo.py†L415-L420】
- `--dry-run`：完成配置解析与数据注册后立刻退出，不会初始化 Ray 或启动训练。【F:scripts/train_planner_grpo.py†L420-L424】

命令示例：
```bash
PYTHONPATH=. python scripts/train_planner_grpo.py \
  --config configs/experiments/planner_grpo_4gpu.yaml \
  --overrides data.train_files=[/abs/path/train.jsonl] \
  --print-config
```

脚本会完成以下步骤：

1. 加载 YAML + dotlist，确保训练/验证文件不为空。【F:scripts/train_planner_grpo.py†L381-L396】
2. 注册 JSONL 到 Verl registry（若尚未 materialize），并根据配置生成 docker manifest。【F:scripts/train_planner_grpo.py†L322-L362】【F:scripts/train_planner_grpo.py†L212-L235】
3. 检查 FSDP/tensor parallel 配置，计算有效 batch size 并输出到日志。【F:scripts/train_planner_grpo.py†L237-L284】【F:scripts/train_planner_grpo.py†L284-L312】
4. 汇总 Planner/CGM 环境变量，必要时注入 Ray runtime env，确保远端 worker 与主进程一致。【F:scripts/train_planner_grpo.py†L110-L210】【F:scripts/train_planner_grpo.py†L312-L352】
5. 根据 `--ray-address` 决定是否启动本地 Ray 实例，然后交给 rLLM 的 `AgentTrainer` 执行 GRPO 训练循环。【F:scripts/train_planner_grpo.py†L424-L468】

## 3. 监控与常见检查

- **批量尺寸**：脚本会在日志中输出 `Batch configuration -> ...`，如果 `data.train_batch_size` 与计算值不符，会额外打印告警提示需要对齐梯度累积或 FSDP world size。【F:scripts/train_planner_grpo.py†L258-L312】
- **Ray 环境变量**：`runtime_env` 会被打印出来，确认模型路径、温度和设备映射均已同步到远程 worker。【F:scripts/train_planner_grpo.py†L312-L352】
- **W&B / 本地日志**：配置文件中的 `trainer.logger` 默认包含 `console` 与 `wandb`，可按需修改或在 dotlist 中禁用。

## 4. 故障排查建议

1. **数据路径异常**：如果 `data.*` 指向的 JSONL 不存在，脚本会在 materialize 阶段抛出 `FileNotFoundError`。请确认路径与仓库根目录的相对/绝对形式一致，或直接在 dotlist 中传入绝对路径。
2. **FSDP 配置错误**：当 `tensor_parallel_planner != 1` 或 actor/ref strategy 不是 `fsdp` 时会直接报错。请确保只使用 FSDP 组合，并在多节点场景手动调整 `trainer.nnodes` 与 `trainer.n_gpus_per_node`。
3. **Ray 地址**：在 `--dry-run` 阶段可以验证路径与环境变量，一旦通过 `--print-config` 或 `--dry-run` 检查无误，再去掉开关启动真实训练；若连接远程集群失败，请检查 `RAY_ADDRESS` 环境变量或 CLI 覆写。

以上步骤可以覆盖从数据准备到训练启动的关键流程，帮助团队在最小配置集下快速复现 Graph Planner 的 GRPO 实验。
