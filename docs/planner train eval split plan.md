# DeepSWE Training and Evaluation Launch Paths

## Training entrypoint
1. **Launch script** – Run `python -m rllm.examples.swe.train_deepswe_agent` (or execute the module file directly). Hydra wires the script to the PPO trainer configuration via `@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer")`.【F:rllm/examples/swe/train_deepswe_agent.py†L1-L26】
2. **Dataset resolution** – The script pulls the training and validation splits (`R2E_Gym_Subset/train`, `SWE_Bench_Verified/test`) through the shared `DatasetRegistry`. These objects expose Verl-compatible parquet paths so the downstream trainer can read them.【F:rllm/examples/swe/train_deepswe_agent.py†L11-L20】【F:rllm/rllm/trainer/agent_trainer.py†L46-L49】
3. **AgentTrainer bootstrap** – `AgentTrainer` stores the SWE agent/environment classes and patches the Hydra config with the dataset file locations before invoking `.train()`. On the first call it also initializes Ray with tokenizer/NCCL environment variables to match the distributed rollout assumptions.【F:rllm/rllm/trainer/agent_trainer.py†L39-L55】
4. **Ray remote training task** – `.train()` dispatches `train_agent` as a Ray remote actor defined in `rllm.rllm.trainer.verl.train_agent_ppo`. The remote function resolves the Hydra config, downloads the base checkpoint, builds the tokenizer, and chooses the concrete agent/environment via the `AGENT_CLASS_MAPPING`/`ENV_CLASS_MAPPING` tables (defaults expect `config.agent.name == "sweagent"` and `config.env.name == "swe"`).【F:rllm/rllm/trainer/verl/train_agent_ppo.py†L21-L109】【F:rllm/rllm/trainer/env_agent_mappings.py†L8-L21】
5. **Trainer workers and rollout** – Inside the remote, `AgentPPOTrainer` binds reward managers, rollout/critic workers, and resource pools before calling `init_workers()` and `fit_agent()` to start PPO fine-tuning the SWE agent against the SWE environment scaffold.【F:rllm/rllm/trainer/verl/train_agent_ppo.py†L61-L112】

## Evaluation entrypoint
1. **Launch script** – Run `python -m rllm.examples.swe.run_deepswe`. The script enforces tokenizer parallelism, selects the hosted `agentica-org/DeepSWE-Preview` checkpoint, and loads its tokenizer up front.【F:rllm/examples/swe/run_deepswe.py†L19-L35】
2. **Dataset fetch** – `load_swe_data()` ensures the evaluation split (`SWE_Bench_Verified/test`) exists in the dataset registry and surfaces the raw list of task dictionaries that will seed environments.【F:rllm/examples/swe/run_deepswe.py†L12-L16】
3. **Execution engine wiring** – `AgentExecutionEngine` is constructed with the SWE agent/environment classes, OpenAI-style rollout parameters (base URL, API key, sample temperature), and batching limits for prompt/response length plus 48-way parallelism.【F:rllm/examples/swe/run_deepswe.py†L28-L43】
4. **Environment/agent lifecycle per task** – `execute_tasks()` asynchronously assigns each task entry to an engine slot, instantiating `SWEEnv.from_dict({**task, **env_args})` and `SWEAgent(**agent_args)` before driving the agent/environment loop via `run_agent_trajectory_async`. Finished trajectories are collated in submission order.【F:rllm/rllm/engine/agent_execution_engine.py†L534-L571】
5. **Computing metrics** – The script runs all tasks through the engine (`asyncio.run(engine.execute_tasks(tasks))`) and evaluates the resulting trajectories with `compute_pass_at_k` for SWE-Bench style reporting.【F:rllm/examples/swe/run_deepswe.py†L45-L48】

## How the SWE agent environment is materialised
- **Environment factory** – `SWEEnv.from_dict` builds an environment from each dataset row, handing the entry to the constructor so the agent gets an isolated repo sandbox per task.【F:rllm/rllm/environments/swe/swe.py†L118-L142】
- **Reset behaviour** – When the engine first resets the environment, `RepoEnv` from R2E-Gym spins up the Docker-backed SWE-Bench workspace and loads either the R2E-Gym or SWE-agent command toolchain before returning the instruction string to the agent.【F:rllm/rllm/environments/swe/swe.py†L43-L111】
- **Action loop** – Every agent action (function call or XML tool invocation) is converted into an `Action` object and passed to the underlying `RepoEnv`, which executes the operation, returns observations/rewards, and tracks termination for final reward computation.【F:rllm/rllm/environments/swe/swe.py†L80-L114】

## RepoEnv 接入与 SWE-Bench 交互深挖
### Graph Planner 链路
**评估入口与命令**
- 在仓库根目录运行 `python scripts/eval_graph_planner_engine.py --dataset <DATASET.jsonl> --planner-model <OPENAI_MODEL> --cgm-model-path <PATH>` 即可触发 Graph Planner 评测。CLI 现已覆盖 planner 侧的本地权重、system prompt、自定义 API key ENV、最大输入 token、推理设备，以及 CGM 端点/模型/API key/超时等参数，并新增 `--gamma`、`--retry-limit`、`--api-retries`、`--max-workers` 等执行引擎调优项，确保与 run_deepswe 和 actor_rollout_ref 配置保持一致的可调范围。【F:scripts/eval_graph_planner_engine.py†L54-L121】【F:rllm/rllm/engine/agent_execution_engine.py†L26-L118】
- `scripts/run_eval_graph_planner.sh` 会在没有显式 `--config` 参数时自动加载 `configs/eval/graph_planner_eval_defaults.yaml`，该文件枚举了默认数据集、模型路径、并发度与 RepoEnv 限额，让使用者只需覆写少量参数即可启动评测；默认数据集已经切换为 `datasets/swebench/test.jsonl`，确保评测直接命中 SWE-bench Verified 测试集，如需轻量自检可改回仓库内的 `graphplanner_repoenv_sample.jsonl`。【F:scripts/run_eval_graph_planner.sh†L1-L24】【F:configs/eval/graph_planner_eval_defaults.yaml†L1-L26】【F:rllm/examples/swe/run_deepswe.py†L19-L43】
- CLI 会在缺省 `--planner-api-key` 时自动回落到 `--planner-api-key-env` 指定的变量（默认为 `PLANNER_MODEL_API_KEY`），若该变量仍未设置，则再尝试读取 `OPENAI_API_KEY` 并在必要时复制到前述变量中，避免因 AsyncOpenAI 缺乏凭据而在启动时崩溃。【F:scripts/eval_graph_planner_engine.py†L444-L482】
- 解析到的 CLI 参数会被 `_configure_runtime_env` 写入 `PLANNER_MODEL_*`、`CGM_*` 系列环境变量，随后 `graph_planner.infra.config.load_config()` 在 agent/env 初始化阶段读取这些变量，落地为推理端点、温度、token 限额、设备映射及远端 CGM API 访问设置。【F:scripts/eval_graph_planner_engine.py†L124-L173】【F:graph_planner/infra/config.py†L248-L315】
- CLI 还支持通过 `--agent-system-prompt(--path)` 与 `--disable-rule-fallback` 覆盖 agent 行为，并用 `--reward-scale`、`--failure-penalty`、`--step-penalty`、`--repo-op-limit`、`--synthesis-strategy` 等选项直接传入环境参数；这些值通过 `AgentExecutionEngine` 的 `agent_args` 与 `env_args` 注入，实现与 rLLM 训练配置相同的调优粒度。【F:scripts/eval_graph_planner_engine.py†L360-L417】【F:rllm/rllm/engine/agent_execution_engine.py†L520-L579】
- GPU 拓扑由 `graph_planner.integrations.rllm.shared_actors` 中的 Ray actor 固定声明：`PlannerEngine` 与 `CGMTool` 默认各占用 2 张 GPU（分别绑定 `CUDA_VISIBLE_DEVICES=0,1` 与 `2,3`），共四张卡。如需调整，可修改对应环境变量或在 CLI 中传入 `--cgm-device` / `--cgm-device-map` 覆盖 CGM 侧的设备映射。【F:graph_planner/integrations/rllm/shared_actors.py†L23-L111】【F:scripts/eval_graph_planner_engine.py†L101-L114】

1. **任务条目携带 RepoEnv 配置** – 数据准备脚本会把 R2E-Gym/SWE-Bench 的 `instance` 信息落盘到 JSONL，每条记录的 `sandbox` 字段都含有 `docker_image`、挂载、环境变量以及指向原始 R2E `ds` 的 `r2e_ds_json` 路径，方便后续按需切换后端。【F:datasets/README.md†L1-L59】
2. **评估脚本兜底 manifest** – `eval_graph_planner_engine.py` 在装载任务时会尝试把历史机器写入的绝对路径重写到当前仓库的 `datasets/*/instances/*.json`；若文件缺失，则按任务条目与 `sandbox` 元数据即时合成最小化 manifest 并写回，确保 RepoEnv 初始化时总能找到有效的 `r2e_ds_json`。【F:scripts/eval_graph_planner_engine.py†L26-L143】
3. **rLLM 环境工厂恢复沙箱** – `GraphPlannerRLLMEnv.from_dict` 将 JSON 还原为 `entry`，在 `_spawn_planner` 中读取 `sandbox` 字段并构造 `SandboxConfig`；如果任务没有携带沙箱配置，会直接抛错，保证每个评测都能找到容器元数据。【F:graph_planner/integrations/rllm/env.py†L35-L177】
4. **PlannerEnv 选择 RepoEnv 后端** – `PlannerEnv` 在初始化时创建 `SandboxRuntime`，其默认模式为 `auto`：当检测到 `r2e_ds_json` 存在时自动切换到 `repoenv`，否则回退到 docker-py 自管容器，从而兼容纯离线训练与官方评测容器。【F:graph_planner/env/planner_env.py†L110-L186】【F:graph_planner/runtime/sandbox.py†L33-L111】
5. **RepoEnv 生命周期** – 选择 `repoenv` 后，`SandboxRuntime` 会把 `r2e_ds_json` 反序列化为 R2E `ds` 字典，通过 `EnvArgs`/`RepoEnv` 启动官方 SWE-Bench 容器，同时安装 `pip`/`pytest` 并配置 `safe.directory`，确保后续命令执行环境一致。【F:graph_planner/runtime/sandbox.py†L78-L111】
6. **动作执行与补丁应用** – `PlannerEnv.step` 在解析出 Explore/Repair/Submit 等动作后，调用 `SandboxRuntime.run/apply_patch/test` 与容器交互；这些封装会在 RepoEnv 模式下委托给底层的 R2E DockerRuntime，实现与 DeepSWE 相同的命令执行与奖励获取，再叠加 Graph Planner 自定义的奖励缩放与罚项。【F:graph_planner/env/planner_env.py†L198-L344】【F:graph_planner/runtime/sandbox.py†L181-L218】

### DeepSWE 链路
1. **入口数据同样包含 RepoEnv 元数据** – DeepSWE 直接读取 HuggingFace 上的 R2E-Gym / SWE-Bench 数据集，`SWEEnv` 在缺省情况下会 `load_dataset(DEFAULT_R2E_ENV_ID, split="test")`，这些记录内置容器镜像、挂载和测试脚本，可被 RepoEnv 直接消费。【F:rllm/rllm/environments/swe/swe.py†L1-L64】
2. **Reset 即构造 RepoEnv** – 首次 `reset` 时，`SWEEnv` 把任务条目包装成 `EnvArgs` 并实例化 `RepoEnv`，选择 R2E-Gym 或 SWE-Agent 指令集后返回任务说明；如果同一进程重复评测，则重用容器并调用 `env.reset()`。【F:rllm/rllm/environments/swe/swe.py†L65-L102】
3. **后续交互完全依赖 RepoEnv** – `step` 把模型生成的字符串解析成 `Action` 并交给 `RepoEnv.step`，奖励由容器 runtime 计算；`compute_final_reward` 直接回调 RepoEnv 的聚合逻辑，形成与 Graph Planner 相比更“薄”的环境包装。【F:rllm/rllm/environments/swe/swe.py†L103-L149】

### 关键差异小结
- **RepoEnv 接入方式**：Graph Planner 通过 `SandboxRuntime` 在运行期选择 RepoEnv、R2E DockerRuntime 或纯 Docker，方便训练时脱离官方容器；DeepSWE 始终固定使用 RepoEnv，拓扑更简单。【F:graph_planner/runtime/sandbox.py†L55-L176】【F:rllm/rllm/environments/swe/swe.py†L45-L118】
- **SWE-Bench 任务拉取**：Graph Planner 依赖预生成的 `r2e_ds_json` 清单在本地还原 SWE-Bench 容器；DeepSWE 则依靠 HuggingFace 数据集在线提供的 `RepoEnv` 配置，在 `reset` 时即时初始化容器。【F:datasets/README.md†L1-L59】【F:rllm/rllm/environments/swe/swe.py†L1-L102】
- **动作执行链路**：Graph Planner 在 RepoEnv 输出的基础上叠加图记忆、CGM 修复与奖励 shaping；DeepSWE 直接透传 RepoEnv 的观测与奖励，仅负责命令集切换与 Action 解析。【F:graph_planner/env/planner_env.py†L198-L344】【F:rllm/rllm/environments/swe/swe.py†L74-L149】

### RepoEnv 内部执行链路细化
1. **容器引导** – `SandboxRuntime._init_repoenv_backend` 使用 manifest 创建 `EnvArgs(ds)` 并实例化 `RepoEnv`，底层的 `DockerRuntime` 会解析镜像、拉起容器并根据 SWE-bench spec 设置测试脚本、提交 commit、pip 安装等前置步骤。【F:graph_planner/runtime/sandbox.py†L78-L129】【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L20-L88】【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L73-L180】
2. **动作执行** – Planner 的动作在 `SandboxRuntime.run` 中被转发到 RepoEnv 的 `runtime.run`，随后由 `DockerRuntime.run_action` 将结构化指令映射成 bash 命令，执行后返回 stdout/stderr 以及退出码，Graph Planner 再把这些组装成 Observation。【F:graph_planner/runtime/sandbox.py†L181-L230】【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L89-L170】
3. **奖励与终止** – RepoEnv 根据函数名 (`finish`/`submit`) 与命令执行结果标记 `done` 状态，Graph Planner 则在 `compute_final_reward` 中读取测试通过与否、步数超限等信号，将最终奖励反馈给执行引擎，实现与 DeepSWE 相同的 pass/fail 判定逻辑。【F:graph_planner/integrations/rllm/env.py†L92-L123】【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L130-L170】
