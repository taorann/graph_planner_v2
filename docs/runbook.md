# Graph Planner 训练/评估 Runbook

本指南描述 rLLM 训练/评估脚本的统一配置方式、启动命令与监控要点，解决“多处配置”“指令不一致”等痛点。默认语言为中文，括号内附英文关键词，便于跨团队协作。

## 1. 配置加载优先级

`scripts/train_graphplanner_rllm.py` 与 `scripts/eval_graphplanner_rllm.py` 现在支持三层优先级：

1. **内置默认值**（`default_training_run_config`）
2. **YAML 配置文件**（`--config-file`，使用 `yaml.safe_load`）
3. **CLI 覆盖**（除非加 `--yaml-only`）

当设置 `--yaml-only` 时，仅解析 YAML（以及 `--config-file` / `--yaml-only` / `--wandb-offline` 本身），其余 CLI 参数会被忽略并在日志中提示。

最终合并后的配置会写入 `logging.output_dir/run_name/resolved_config.yaml`，并同步到 W&B `config`（`wandb.config.update`）。

## 2. YAML 字段总览

YAML 顶层字段与含义如下，所有路径均默认为仓库相对路径：

| Section | 字段 | 说明 |
|---------|------|------|
| `experiment` | `name`, `seed`, `notes` | 运行标识与随机种子 |
| `paths` | `dataset_train`, `dataset_val`, `planner_model`, `planner_tokenizer`, `cgm_model`, `cgm_tokenizer` | 数据与模型路径 |
| `backends` | `planner_backend`, `cgm_backend`, `dtype`, `device_map_*`, `max_gpu_memory` | HF / 远程推理后端、精度与显存限制 |
| `planner_sampling` | `temperature`, `top_p`, `top_k`, `max_input_tokens`, `max_new_tokens`, `stop`, `stop_ids`, `repetition_penalty`, `do_sample`, `stop_on_invalid_json` | Planner 采样策略 |
| `cgm_generation` | `temperature`, `top_p`, `top_k`, `max_new_tokens`, `num_return_sequences`, `do_sample` | CGM 生成策略 |
| `training` | `total_epochs`, `train_batch_size`, `grad_accum_steps`, `precision`, `lr`, `weight_decay`, `warmup_steps`, `clip_grad_norm`, `kl_coef`, `entropy_coef`, `value_coef`, `clip_coef`, `target_kl`, `total_steps`, `resume_from`, `gradient_checkpointing` | 训练超参与 Verl/GRPO 相关系数 |
| `env` | `max_steps`, `reward_scale`, `failure_penalty`, `step_penalty`, `timeout_penalty`, `repo_op_limit`, `disable_cgm_synthesis`, `apply_patches` | 环境奖励与工具开关 |
| `parallel` | `tensor_parallel_planner`, `tensor_parallel_cgm`, `replicas`, `parallel_agents`, `rollout_workers`, `workflow_parallel` | 模型并行与 rollout 并发 |
| `resources` | `num_gpus`, `num_nodes`, `ray_num_gpus`, `ray_num_cpus`, `ray_memory`, `ray_object_store_memory` | 物理/ Ray 资源预算 |
| `logging` | `wandb.*`, `log_backend`, `output_dir`, `save_interval`, `eval_interval` | 日志与输出目录；`wandb.watch` 支持 `{enabled, log, log_freq}` |
| `telemetry` | `log_gpu`, `log_ray`, `log_patch_stats`, `log_planner_parse_errors`, `log_cgm_errors` | 监控开关 |
| `verl_overrides` | 任意 Hydra key | 直接透传到 Verl/rLLM 配置 |

> **提示**：`logging.output_dir` 为 run 目录的父路径，实际运行目录 = `output_dir / wandb.run_name`。脚本会将 `resolved_run_dir` 与 `resolved_config_path` 写回配置，便于外部工具查找。

### 示例 YAML（片段）

```yaml
experiment:
  name: planner_grpo
  seed: 1234
paths:
  dataset_train: datasets/r2e_gym/graphplanner_repoenv_train.jsonl
  dataset_val: datasets/r2e_gym/graphplanner_repoenv_val.jsonl
  planner_model: models/qwen3-14b-instruct
  cgm_model: models/codefuse-cgm
training:
  total_epochs: 1
  train_batch_size: 4
  grad_accum_steps: 8
  precision: bf16
parallel:
  tensor_parallel_planner: 4
  tensor_parallel_cgm: 4
  replicas: 1
  parallel_agents: 4
resources:
  num_gpus: 8
  ray_num_gpus: 8
logging:
  output_dir: outputs
  wandb:
    enabled: true
    project: graph-planner
    run_name: planner-grpo-8g
```

## 3. 启动方式

### 3.1 CLI + YAML（默认）

```bash
PYTHONPATH=. python scripts/train_graphplanner_rllm.py \
  --agent planner \
  --config-file configs/experiments/planner_8g.yaml \
  --dataset datasets/r2e_gym/graphplanner_repoenv_train.jsonl \
  --model-path models/qwen3-14b-instruct \
  --cgm-model-path models/codefuse-cgm \
  --print-config
```

CLI 可继续覆盖 YAML 中的任意字段；最终合并配置会打印在屏幕上，并写入 `outputs/<run_name>/resolved_config.yaml`。

### 3.2 YAML-only 模式

```bash
PYTHONPATH=. python scripts/train_graphplanner_rllm.py \
  --config-file configs/experiments/planner_8g.yaml \
  --yaml-only \
  --print-config
```

仅使用 YAML（外加 `--wandb-offline` 的快速切换）；脚本会记录所有被忽略的 CLI 参数名称，方便排查。

### 3.3 冻结配置复现

已生成的 `outputs/<run_name>/resolved_config.yaml` 可直接复用：

```bash
PYTHONPATH=. python scripts/train_graphplanner_rllm.py \
  --config-file outputs/planner-grpo-8g/resolved_config.yaml \
  --yaml-only
```

同理，`scripts/eval_graphplanner_rllm.py` 也支持 `--config-file` 与 `--yaml-only`，默认在 run 目录下生成新的 `resolved_config.yaml`。

## 4. 并行预检

`graph_planner.infra.parallel.preflight_check` 会在启动前验证 GPU/Ray 配置：

- `tensor_parallel_planner + tensor_parallel_cgm <= num_gpus * num_nodes`
- `replicas * max(tensor_parallel) <= num_gpus * num_nodes`
- `workflow_parallel >= max(parallel_agents, rollout_workers)`
- `ray_num_gpus >= replicas`
- `ray_num_cpus >= rollout_workers * 4`

若校验失败，将抛出 `ValueError` 并给出三条修复建议（降低 TP/replicas、降低并发、提升 Ray 资源）。

## 5. W&B 监控

`infra.metrics` 在训练与评估脚本中自动初始化 W&B，并记录：

- **并行状态**：`parallel/tensor_parallel_*`, `parallel/agents`, `parallel/workers`, `resources/num_gpus` 等
- **GPU/Ray 心跳**：通过 `pynvml` 与 `ray.available_resources()` 获取 `gpu/<idx>/util`, `gpu/<idx>/mem_GB`, `ray/cpus_avail`, `ray/gpus_avail`
- **补丁链路指标**：训练期间环境会继续上报 `patch_*`, `planner_parse_error_rate`, `reject/*` 等字段
- **Verl 指标**：rLLM/Verl 原生的 `train/kl`, `train/entropy`, `train/clipfrac`, `train/policy_loss`, `train/value_loss`, `train/grad_norm`, `reward/*`, `lr` 等

设置 `logging.wandb.offline: true` 或 `--wandb-offline` 可在完全离线环境下写入本地 W&B 日志（默认为 `wandb/` 目录）。

### 面板建议

推荐在 W&B Dashboard 中建立如下图表：

1. **Training**：`train/kl`, `train/entropy`, `train/clipfrac`, `lr`
2. **Policy Quality**：`reward/mean`, `reward/std`, 自定义 `patch_success_rate`
3. **Throughput**：`parallel/agents`, `parallel/workers`, `episodes_per_min`
4. **System**：`gpu/*`, `ray/*` 心跳曲线

## 6. GPU 配置示例

| 场景 | 关键字段 | 命令示例 |
|------|----------|---------|
| 单卡调试（Planner + CGM 本地权重） | `tensor_parallel_* = 1`, `parallel_agents = 1`, `rollout_workers = 1`, `num_gpus = 1`, `device_map_* = [0]` | `PYTHONPATH=. python scripts/train_graphplanner_rllm.py --config-file configs/experiments/debug_single_gpu.yaml --yaml-only` |
| 8×A800（Planner 训练 + CGM 推理） | `tensor_parallel_planner = 4`, `tensor_parallel_cgm = 4`, `parallel_agents = 4`, `rollout_workers = 4`, `num_gpus = 8`, `device_map_planner = [0,1,2,3]`, `device_map_cgm = [4,5,6,7]` | `PYTHONPATH=. python scripts/train_graphplanner_rllm.py --config-file configs/experiments/planner_cgm_8g.yaml --yaml-only --print-config` |
| 16×A800（Planner 14B + CGM 73B） | `tensor_parallel_planner = 8`, `tensor_parallel_cgm = 8`, `parallel_agents = 4`, `rollout_workers = 5`, `num_gpus = 16`, `device_map_planner = [0-7]`, `device_map_cgm = [8-15]` | `PYTHONPATH=. python scripts/train_graphplanner_rllm.py --config-file configs/experiments/gp_full_73b14b_16g.yaml --yaml-only --print-config` |

> **注意**：上表中的 YAML 样例需要同时指定 `paths.planner_model: models/qwen3-14b-instruct` 与 `paths.cgm_model: models/codefuse-cgm`，仓库已在 `models/` 目录下预留路径。

## 7. 离线评估与复现

评估脚本使用同一套配置，额外将 `trainer.val_only = True`、`train_batch_size = batch_size`。建议在 YAML 中为评估 run 设定独立 `logging.wandb.run_name`（例如 `planner-grpo-eval`），以便区分训练/验证轨迹。

## 8. 快速排错

- **并行配置失败**：根据异常消息调节 `tensor_parallel_*`、`replicas`、`parallel_agents`、`rollout_workers` 或提高 `resources.ray_*`。
- **W&B 无法连接外网**：设置 `logging.wandb.offline: true` 或 `--wandb-offline`，离线日志仍会写入。
- **YAML-only 忽略参数**：日志中会列出被忽略的 CLI 参数名称，确保 YAML 已覆盖所需字段。

## 9. 相关脚本

- `scripts/train_graphplanner_rllm.py`：训练入口，支持断点恢复、早停、梯度累积等。
- `scripts/eval_graphplanner_rllm.py`：评估入口，仅执行 rollout 与指标统计。
- `scripts/validate_contracts.py` / `scripts/validate_patches.py`：协议与补丁快速校验。

如需了解更底层的协议与动作实现，请参考 `docs/graph_planner_architecture_pipeline.md` 的配套章节。
