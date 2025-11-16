# 脚本与测试总览

本文档总结 `scripts/` 与 `tests/` 目录的主要文件职责，并说明当前仓库中 ACI 工具链、Git 操作封装、Lint 与 Test 的实现来源，帮助贡献者快速了解可执行入口与回归保障。
> **2025-11-03 审核结论**：列出的脚本与测试路径均已对照仓库确认存在，旧版 FakeSandbox 测试仍未恢复（维持文档说明）。

## 核心包结构（`graph_planner/`）
- **`agents/`**：包含规则策略与本地 LLM 决策器，分别负责状态机驱动的修复流程与模型输出解析，并共享对话协议工具。【F:graph_planner/agents/rule_based/planner.py†L26-L187】【F:graph_planner/agents/model_based/planner.py†L38-L178】【F:graph_planner/agents/common/chat.py†L1-L196】
- **`graph_planner/env/planner_env.py`**：封装 Explore/Memory/Repair/Submit 动作到容器操作的映射，维护奖励、终止条件与工作子图状态。【F:graph_planner/env/planner_env.py†L32-L173】
- **`graph_planner/runtime/sandbox.py`**：统一 RepoEnv、R2E DockerRuntime 与 docker-py 的执行接口，负责拉起容器、运行补丁与记录测试结果。【F:graph_planner/runtime/sandbox.py†L30-L264】
- **`integrations/local_llm` 与 `integrations/rllm`**：前者提供 OpenAI 兼容的本地模型客户端，后者封装 rLLM 的 Agent/Env/Dataset 适配层，供强化学习训练复用。【F:graph_planner/integrations/local_llm/client.py†L15-L152】【F:graph_planner/integrations/rllm/agent.py†L1-L158】【F:graph_planner/integrations/rllm/env.py†L1-L110】
- **`infra/`**：集中配置、遥测日志与其他运行期开关，决定补丁模型、本地 LLM、事件路径等行为。【F:graph_planner/infra/config.py†L24-L176】【F:graph_planner/infra/telemetry.py†L20-L39】

## 脚本目录（`scripts/`）

当前仓库保留的脚本集中在评测、数据准备与协议校验三个方向，核心入口只有 `eval_graph_planner_engine.py` 与其配套的 Shell 包装脚本。

| 文件 | 作用 | 主要依赖 |
| --- | --- | --- |
| `scripts/run_eval_graph_planner.sh` | Bash 包装层，负责解析 CLI/配置文件、导出 `PYTHONPATH`，并调用 `eval_graph_planner_engine.py` 执行端到端评测。 | `scripts/eval_graph_planner_engine.py` |
| `scripts/eval_graph_planner_engine.py` | 评测主程式：加载数据集、探测/拉起 planner & CGM 服务、构建 rLLM 环境并串联 Graph Planner -> RepoEnv -> 结果写出。 | `graph_planner.eval.engine`, `graph_planner.runtime.sandbox`, `graph_planner.integrations` |
| `scripts/prepare_datasets.py` | 下载并转换 R2E-Gym / SWE-bench 数据，生成 Graph Planner 兼容的 JSON/JSONL、manifest 与实例文件。 | `graph_planner.datasets`, `graph_planner.runtime.containers` |
| `scripts/register_graphplanner_dataset.py` | 将 RepoEnv 任务描述注册到 rLLM 数据集仓库，生成 `rllm/rllm/data/datasets/<name>/val_verl.parquet` 等索引文件，可在训练或评测前直接 `DatasetRegistry.get("graph_planner_repoenv")` 复用实例清单。 | `graph_planner.integrations.rllm.dataset`, `datasets` |
| `scripts/build_repoenv_sample.sh` | 构建最小化 RepoEnv 容器样例，帮助验证 docker 构建链路是否可用。 | `docker`, `r2egym` |
| `scripts/validate_contracts.py` / `scripts/validate_patches.py` | 校验 Planner/CGM 协议与补丁结构，防止输出格式漂移。 | `graph_planner.agents.rule_based`, `graph_planner.aci.guard` |

- `scripts/run_eval_graph_planner.sh` 只是薄包装，最终逻辑全部落在 Python 主程序里，适合在集群上通过 CLI/配置切换参数。【F:scripts/run_eval_graph_planner.sh†L1-L31】
- `scripts/eval_graph_planner_engine.py` 包含配置解析、GPU/端点探测、任务加载与 rLLM 推理循环，是评估 Graph Planner 的唯一入口。【F:scripts/eval_graph_planner_engine.py†L40-L200】【F:scripts/eval_graph_planner_engine.py†L1187-L1334】
- `scripts/prepare_datasets.py` 支持 `--skip-*` 与 `--prepull-*` 参数，可一次性生成训练/评测所需的 JSONL、实例与 docker manifest。【F:scripts/prepare_datasets.py†L12-L138】【F:scripts/prepare_datasets.py†L200-L276】
- `scripts/register_graphplanner_dataset.py` 会把 `datasets/<dataset>/instances/*.json` 与 JSONL 元信息注册到 rLLM 的本地数据集仓库（`rllm/rllm/data/datasets/`），写出 Parquet 索引供 `DatasetRegistry` 快速加载；多机复现时只需同步 `datasets/` 与生成的 Parquet 文件即可跳过重复解析。【F:scripts/register_graphplanner_dataset.py†L18-L119】

## 测试目录（`tests/` 与 `rllm/tests/`）

当前仓库的轻量测试集中在两个入口：

| 目录 | 文件 | 核心覆盖点 | 说明 |
| --- | --- | --- | --- |
| `tests/` | `tests/test_reward_manager_loading.py` | 确保 `train_agent_ppo._maybe_load_reward_managers` 在缺省配置与启用奖励时都能正确回退/加载。 | 直接引用 rLLM 训练入口，避免奖励依赖导致的离线调试崩溃。【F:tests/test_reward_manager_loading.py†L1-L45】 |
| `rllm/tests/` | `agents/`, `envs/`, `rewards/`, `tools/` 子目录 | 校验强化学习 Agent、环境包装器、奖励模型与工具函数。 | 运行 `pytest rllm/tests -q` 可覆盖 FrozenLake/AppWorld/ToolAgent 等核心逻辑。【F:rllm/tests/agents/test_tool_agent.py†L1-L151】【F:rllm/tests/envs/test_tool_env.py†L1-L134】 |

## ACI / Git / Lint / Test 的实现来源

- **ACI 工具链（`aci/`）**：
  - `aci/tools.py` 提供查看、搜索、编辑、lint、测试等 CLI 操作的统一封装。优先调用项目内实现，缺省回退到宿主机已有的工具。
  - `aci/git_tools.py` 封装分支、提交、回滚、diff 等 Git 操作，统一返回 `AciResp` 结构，方便在 CLI 与 API 中复用。
  - `aci/guard.py` 负责补丁护栏校验与决策清洗逻辑，被 `PlannerEnv` 与外部代理共同调用，以保持编辑窗口、预算等策略约束一致。

- **Git 操作**：仓库未依赖 R2E 提供的 Git 管理，所有交互均通过 `aci/git_tools.py` 调用系统 `git`。

- **Lint 与 Test**：
  - `graph_planner/runtime/sandbox.py` 定义 `SandboxRuntime` 抽象，并在 `run_lint`、`run_tests` 中调用我们的本地实现（如 `ruff`、`pytest`）。【F:graph_planner/runtime/sandbox.py†L210-L264】
  - 当选择 RepoEnv / R2E 后端时，容器调度由 R2E 组件处理，但实际 lint/test 命令仍出自本仓库，实现与普通文件系统一致。【F:graph_planner/runtime/sandbox.py†L69-L208】

- **与 R2E 的关系**：
  - RepoEnv / Docker 运行时通过 `graph_planner.runtime.sandbox.SandboxRuntime` 的不同分支（`repoenv`、`r2e`、`docker`）对接 R2E-Gym，利用其任务定义和容器封装。【F:graph_planner/runtime/sandbox.py†L62-L208】
  - 除沙箱后端外，ACI、Git、Lint、Test 逻辑均是仓库自研模块，可在离线或无容器环境下工作。

## 推荐使用流程

1. **准备评测数据**：
   ```bash
   PYTHONPATH=. python scripts/prepare_datasets.py \
     --r2e-dataset R2E-Gym/R2E-Gym-Lite \
     --swebench-dataset princeton-nlp/SWE-bench_Verified
   ```
   该脚本会生成 Graph Planner 期望的 JSONL、RepoEnv 实例文件与 Docker manifest，必要时还能批量预拉镜像。【F:scripts/prepare_datasets.py†L12-L138】【F:scripts/prepare_datasets.py†L200-L276】

2. **运行 Graph Planner 评测**：
   ```bash
   bash scripts/run_eval_graph_planner.sh \
     --config configs/eval/graph_planner_eval_defaults.yaml \
     --planner-api-key sk-xxxx
   ```
   Shell 包装脚本会把配置与 CLI 合并后调用 `scripts/eval_graph_planner_engine.py`，自动探测/拉起 planner 与 CGM 服务，并在任务结束后整理结果与日志。【F:scripts/run_eval_graph_planner.sh†L1-L31】【F:scripts/eval_graph_planner_engine.py†L468-L720】【F:scripts/eval_graph_planner_engine.py†L1187-L1334】

3. **回归测试**：
   ```bash
   PYTHONPATH=. pytest tests -q
   PYTHONPATH=. pytest rllm/tests -q
   ```
   若依赖项缺失，可先安装 `R2E-Gym` 或使用 `pip install -e ./R2E-Gym` 完成补齐；当缺少 Verl 依赖时，`tests/test_reward_manager_loading.py` 会自动跳过。【F:tests/test_reward_manager_loading.py†L1-L45】【F:rllm/tests/agents/test_tool_agent.py†L1-L151】

## Graph Planner 与 SWE 容器交互详解

1. **数据准备阶段**：`scripts/prepare_datasets.py` 会把 R2E-Gym / SWE-bench 的条目转换成 Graph Planner 任务 JSONL，并为每个实例写出 RepoEnv 兼容的 `instances/<task>.json`。转换结果在 `sandbox` 字段中预填 `backend="repoenv"`、`docker_image` 与 `r2e_ds_json`，为后续容器拉起提供足够元数据。【F:graph_planner/datasets/prepare.py†L260-L280】

2. **评测入口修正清单路径**：`scripts/eval_graph_planner_engine.py` 加载任务时调用 `_ensure_repoenv_manifest`，优先尝试复用数据集中给出的 `r2e_ds_json`，否则会根据任务 ID 或嵌入的实例描述在当前仓库内重建最小化 manifest，确保 RepoEnv 无论在何处运行都能找到合法的 SWE 容器配置。【F:scripts/eval_graph_planner_engine.py†L1031-L1118】

3. **构造运行环境**：rLLM 封装在每条任务开始前根据 JSON 里的 `sandbox` 字段创建 `SandboxConfig`，并交给 `PlannerEnv`；路径字段会被展开成当前机器的绝对路径，避免跨机器数据集造成的相对路径失效。【F:graph_planner/integrations/rllm/env.py†L221-L235】

4. **容器运行时**：`PlannerEnv` 在初始化时直接实例化 `SandboxRuntime`，该运行时会按需选择 RepoEnv、R2E DockerRuntime 或 docker-py 后端，统一提供 `run`、`apply_patch`、`test` 等接口以便 Graph Planner 对 SWE 容器执行命令、应用补丁及运行测试。【F:graph_planner/env/planner_env.py†L100-L186】【F:graph_planner/runtime/sandbox.py†L47-L220】

5. **动作与结果回传**：`SandboxRuntime` 会在 RepoEnv 模式下安装基础依赖、修正工作目录，并在执行阶段复用 RepoEnv/DockerRuntime 的原生命令，最终把 stdout/stderr 与退出码回传给 `PlannerEnv`，后者再据此更新 observation、奖励与轨迹日志。【F:graph_planner/runtime/sandbox.py†L80-L220】【F:graph_planner/env/planner_env.py†L198-L332】
   - 当后端被强制改为 `docker` 时，运行时会通过 docker-py 的 `containers.run` 以交互式 `/bin/bash` 启动容器，并挂载 manifest 中声明的工作目录、环境变量与端口映射。【F:graph_planner/runtime/sandbox.py†L138-L199】
   - 后续的 `run`/`apply_patch`/`test` 调用都会合成 Shell 字符串传给 `_exec`，该方法把指令封装成 `bash -lc '<cmd>'`，借助 `exec_run(demux=True)` 同时获取 stdout/stderr 以及退出码，再拼装成 `SandboxResult` 返回给上层。【F:graph_planner/runtime/sandbox.py†L200-L260】
   - 这意味着我们可以向容器输送任意命令：读取文件时会把动态生成的 Python heredoc 注入到 Shell；应用补丁时会将 unified diff 写入临时文件再执行 `git apply`；运行测试时则优先调用 SWE 官方脚本，若缺失则回退到 `python -m pytest`。容器完成后会返回命令输出与整数状态码，`PlannerEnv` 依据这些信息决定奖励、是否终止与后续动作选择。【F:graph_planner/env/planner_env.py†L283-L1040】

6. **Action 到容器指令的映射**：`PlannerEnv.step()` 会先用 `validate_planner_action` 校验协议，把外部 JSON 动作恢复成内部的 `Explore`/`Memory`/`Repair`/`Submit` 枚举类，再分派到对应处理函数。【F:graph_planner/env/planner_env.py†L201-L268】 例如 `ExploreAction(op="read")` 会调用 `_read_node_snippet`，动态生成一段 Python heredoc，通过 `SandboxRuntime.run()` 在 SWE 容器里读取目标文件的指定行并把结果回传给观察空间。【F:graph_planner/env/planner_env.py†L283-L325】【F:graph_planner/env/planner_env.py†L561-L610】 `RepairAction` 如果自带补丁，则会进入 `_apply_patch_edits`，按每个 edit 组装 base64 载荷，同样以 heredoc 方式发送到容器执行文件修改；随后触发 `self.box.lint()` / `self.box.test()` 收集 lint 与测试结果。【F:graph_planner/env/planner_env.py†L362-L387】【F:graph_planner/env/planner_env.py†L971-L1040】 当需要 CGM 协助时，`PlannerEnv` 会将计划、子图和读取到的代码片段整理成 prompt，调用 `cgm_adapter.generate` 拿到结构化补丁，再复用同一套 `_apply_patch_edits` 和测试流程。【F:graph_planner/env/planner_env.py†L396-L505】 所有这些容器调用最终都落在 `SandboxRuntime` 的统一接口上——RepoEnv 模式下会直接复用 `RepoEnv.runtime.run/apply_patch/test` 与 SWE 官方脚本，docker 模式则通过 docker-py 的 `exec_run` 执行命令并返回 stdout/退出码。【F:graph_planner/runtime/sandbox.py†L47-L260】

7. **容器生命周期与并发关系**：每个 `PlannerEnv` 在构造时都会绑定一个独立的 `SandboxRuntime`，从而创建并持有单个 RepoEnv/R2E/Docker 容器；环境关闭时会调用 `SandboxRuntime.close()` 终止该容器，因此一个 `PlannerEnv` 即对应一套隔离的容器实例。【F:graph_planner/env/planner_env.py†L115-L196】【F:graph_planner/runtime/sandbox.py†L33-L220】 `GraphPlannerRLLMEnv.reset()` 会先关闭上一轮的 `PlannerEnv`，再通过 `_spawn_planner()` 生成新的 `PlannerEnv`/容器，确保每个任务都在全新沙箱里执行。【F:graph_planner/integrations/rllm/env.py†L110-L174】【F:graph_planner/integrations/rllm/env.py†L221-L234】 并发评测时，`AgentExecutionEngine.execute_tasks()` 为每个并行槽位实例化一个 `GraphPlannerRLLMEnv`，进而派生出独立的 `PlannerEnv`，同时利用线程池驱动 `reset/step/close` 调用，所以活跃容器数量等于 CLI 的 `--parallel` 配置。【F:rllm/rllm/engine/agent_execution_engine.py†L64-L112】【F:rllm/rllm/engine/agent_execution_engine.py†L528-L581】 `PlannerEnv` 内部的 `_get_shared_actor()` 仅在外部已经 `ray.init()` 时复用现有 Ray Actor（例如训练阶段的共享 CGM 工具），评测脚本默认不会拉起 Ray worker，因此容器生命周期与 Ray 没有硬绑定关系。【F:graph_planner/env/planner_env.py†L91-L97】【F:graph_planner/env/planner_env.py†L888-L919】

### 单容器联调与端口暴露

若需要在联调阶段仅启动一个 SWE 容器并把内部端口暴露给外部 Agent，可在 CLI 上组合 `--limit 1 --parallel 1` 与下面两个新增开关：

- `--sandbox-force-docker-backend`：即便数据集中声明 `backend="repoenv"`，也强制改用 docker-py 后端启动容器，方便统一控制端口映射。【F:scripts/eval_graph_planner_engine.py†L928-L946】【F:graph_planner/runtime/sandbox.py†L40-L137】
- `--sandbox-port-forward [HOST_IP:]HOST_PORT:CONTAINER_PORT`：声明需要映射的端口，参数可重复。脚本会在解析 CLI 后把结果注入 `SandboxConfig.port_forwards`，并在容器启动完成后从 Docker API 读取实际绑定的主机端口，将其写入观测的 `last_info.sandbox_ports`，便于日志或下游系统获取公网入口。【F:scripts/eval_graph_planner_engine.py†L64-L129】【F:graph_planner/runtime/sandbox.py†L138-L209】【F:graph_planner/env/planner_env.py†L166-L173】

示例：

```bash
PYTHONPATH=. python scripts/eval_graph_planner_engine.py \
  --config configs/eval/graph_planner_eval_defaults.yaml \
  --limit 1 --parallel 1 \
  --sandbox-force-docker-backend \
  --sandbox-port-forward 0.0.0.0:2222:22 \
  --sandbox-port-forward 0.0.0.0:18080:8080
```

运行后你会得到一个唯一的容器实例，同时在日志与观测中看到 `sandbox_ports` 条目记录了 `22/tcp → 2222`、`8080/tcp → 18080` 等映射，便于把端口转发到互联网供外部 Agent 直接连入。未设置这些参数时，评测脚本仍会沿用默认的 RepoEnv 生命周期，不会额外暴露端口。

### 手动启动本地推理服务

评测脚本会在 `planner_base_url`/`cgm_endpoint` 指向本机且检测到端点尚未就绪时自动拉起推理服务；若你希望在运行 `scripts/run_eval_graph_planner.sh` 之前手动启动，或想复用已经存在的进程，可参考以下命令行模板。

1. **Planner（vLLM OpenAI 端点）** — 复用默认配置中的张量并行与端口：

   ```bash
   CUDA_VISIBLE_DEVICES="0,1" \
   python -m vllm.entrypoints.openai.api_server \
     --model /path/to/graph_planner_v2/models/Qwen3-14B \
     --tokenizer /path/to/graph_planner_v2/models/Qwen3-14B \
     --host localhost \
     --port 30000 \
     --served-model-name models/Qwen3-14B \
     --tensor-parallel-size 2 \
     --gpu-memory-utilization 0.9 \
     --trust-remote-code
  ```

   该服务对外暴露 `http://localhost:30000/v1`，与默认配置的 `planner_base_url`、`planner_model` 一致。【F:configs/eval/graph_planner_eval_defaults.yaml†L6-L17】
   如果你在 YAML/CLI 中把 `planner_service_gpus` 写成列表（如 `[0,1]`），评测脚本会自动规范化为 `CUDA_VISIBLE_DEVICES="0,1"` 并在未显式设置 `planner_service_tensor_parallel_size` 时根据 GPU 数量补上张量并行度，因此无需手动调整命令行即可让 vLLM 在多卡间平均分配权重。【F:scripts/eval_graph_planner_engine.py†L154-L185】【F:scripts/eval_graph_planner_engine.py†L468-L524】

2. **CodeFuse CGM 服务（FastAPI）** — 按默认推理超参启动补丁生成后端：

   ```bash
   CUDA_VISIBLE_DEVICES="2,3" \
   python -m graph_planner.integrations.codefuse_cgm.service \
     --model /path/to/graph_planner_v2/models/CodeFuse-CGM \
     --tokenizer /path/to/graph_planner_v2/models/CodeFuse-CGM \
     --host localhost \
     --port 30001 \
     --route /generate \
     --max-input-tokens 8192 \
     --max-new-tokens 1024 \
     --temperature 0.0 \
     --top-p 0.9 \
     --log-level info
   ```

   该命令会监听 `http://localhost:30001/generate` 并加载本地模型权重，参数来源同一份默认配置的 `cgm_*` 条目。【F:configs/eval/graph_planner_eval_defaults.yaml†L19-L33】

3. **单机一体化示例** — 下列命令假设你已经下载 `models/Qwen3-14B` 与 `models/CodeFuse-CGM`，并且评测 CLI、vLLM、CGM 服务都在同一台机器上运行：

   ```bash
   # 先在独立终端启动 Planner vLLM 服务
   CUDA_VISIBLE_DEVICES="0,1" \
   python -m vllm.entrypoints.openai.api_server \
     --model $(pwd)/models/Qwen3-14B \
     --tokenizer $(pwd)/models/Qwen3-14B \
     --host localhost \
     --port 30000 \
     --served-model-name models/Qwen3-14B \
     --tensor-parallel-size 2 \
     --gpu-memory-utilization 0.9 \
     --max-model-len 8192 \
     --kv-cache-dtype fp8 \
     --trust-remote-code

   # 在另一个终端启动 CGM FastAPI 服务
   CUDA_VISIBLE_DEVICES="2" \
   python -m graph_planner.integrations.codefuse_cgm.service \
     --model $(pwd)/models/CodeFuse-CGM \
     --tokenizer $(pwd)/models/CodeFuse-CGM \
     --host localhost \
     --port 30001 \
     --route /generate \
     --max-input-tokens 8192 \
     --max-new-tokens 1024 \
     --temperature 0.0 \
     --top-p 0.9 \
     --log-level info

   # 全部服务就绪后，在第三个终端运行评测脚本
   export PLANNER_MODEL_API_KEY=dummy-sk
   export CGM_API_KEY=dummy-sk
   bash scripts/run_eval_graph_planner.sh \
     --config configs/eval/graph_planner_eval_defaults.yaml \
     --planner-base-url http://localhost:30000/v1 \
     --cgm-endpoint http://localhost:30001/generate \
     --no-auto-launch-planner-service \
     --no-auto-launch-cgm-service \
     --planner-api-key-env PLANNER_MODEL_API_KEY \
     --cgm-api-key-env CGM_API_KEY
   ```

   评测命令会直接复用你手动拉起的两个服务：通过 `--no-auto-launch-*` 显式禁用自动拉起逻辑，`planner_base_url` 与 `cgm_endpoint` 指向本机端点，再由 `planner_api_key_env`/`cgm_api_key_env` 提供密钥或占位值。若模型与评测脚本均运行在单机环境，以上顺序即可完成“服务启动 → Graph Planner 评测”的完整闭环。

   > ❗️ **说明**：CGM 服务当前通过 Hugging Face 的 `AutoModelForCausalLM` 直接加载权重，不会自动启动 vLLM。多卡场景需要依赖 Transformers 的 `device_map` 或 BitsAndBytes 配置手动切分模型。

    > 💡 **Tokenizer 路径兼容性**：很多 CodeFuse CGM 发布包只在模型目录根部提供 `tokenizer.json`/`tokenizer.model`，缺少 `tokenizer_config.json`。服务现在会自动探测这些常见文件并通过 `tokenizer_file=` 传给 `AutoTokenizer.from_pretrained()`，因此 CLI 里的 `--tokenizer` 只需指向模型根目录即可，无需额外复制配置文件。【F:graph_planner/integrations/codefuse_cgm/inference.py†L47-L66】

   ##### CGM FastAPI 服务如何挑选 GPU / 并行度？

   * `graph_planner.integrations.codefuse_cgm.service` 在 `_build_generator()` 中仅把 CLI 里传入的 `--device`、`--device-map`、`--dtype` 等参数原封不动写进 `CGMGenerationConfig`，然后交给 Hugging Face 的 `CodeFuseCGMGenerator` 去加载模型；如果这些参数为空，Transformers 只会使用首张可见 GPU（或回退到 CPU），**不会**根据本机 GPU 数量自动推断张量并行/切分方案。【F:graph_planner/integrations/codefuse_cgm/service.py†L200-L233】
   * 评测脚本的 `_auto_launch_cgm_service()` 也只是把 `cgm_service_gpus` 解析成 `CUDA_VISIBLE_DEVICES` 并附加到子进程环境，再把 `cgm_device_map` 透传给 `--device-map`。因此当你声明 `"0,1,2"` 时，FastAPI 进程能看到 3 张卡，但是否真的使用取决于 `--device-map` 的值（例如 `balanced`、`auto` 或显式 JSON 映射）。【F:scripts/eval_graph_planner_engine.py†L720-L785】

   换句话说：要利用多卡，只需在配置/CLI 中同时设置 `cgm_service_gpus="0,1,..."` 与 `cgm_device_map=balanced`（或更细粒度的 JSON 字符串）；仓库不会额外帮你推算 tensor parallel，而是完全沿用 Transformers 提供的切分逻辑。

   若需在 **三张 GPU** 上加载模型，可把 `CUDA_VISIBLE_DEVICES` 扩展到三张卡，并把 `--device-map` 设为 `balanced`（或自定义 JSON 映射），例如：

   ```bash
   CUDA_VISIBLE_DEVICES="2,3,4" \
   python -m graph_planner.integrations.codefuse_cgm.service \
     --model /path/to/graph_planner_v2/models/CodeFuse-CGM \
     --tokenizer /path/to/graph_planner_v2/models/CodeFuse-CGM \
     --host localhost \
     --port 30001 \
     --route /generate \
     --max-input-tokens 8192 \
     --max-new-tokens 1024 \
     --temperature 0.0 \
     --top-p 0.9 \
     --device-map balanced \
     --log-level info
   ```

   更细粒度的切分可以把 `--device-map` 替换成 JSON 字符串（例如 `'{"model.embed_tokens":0,"model.layers.0":0,"model.layers.1":1,...}`），或在配置/CLI 中设置 `cgm_device_map` 让评测脚本自动传入。评测脚本同时会读取 `cgm_service_gpus` 并写入 `CUDA_VISIBLE_DEVICES`，因此只需把这两个字段改成 `"2,3,4"` 和期望的 `device_map` 值即可沿用自动拉起逻辑。

   #### 想用 vLLM 拉起 CGM？

   目前仓库内的 CGM HTTP 服务仍基于 Hugging Face 推理堆栈，优势是可以一次性把 Qwen-72B 主体、LoRA/Adapter 权重与 CodeT5 子模型装载到同一个 Python 进程中。【F:graph_planner/integrations/codefuse_cgm/service.py†L1-L206】 如果改成 vLLM，理论上可以依赖 PagedAttention、连续 batch 推理等优化获得更高的吞吐率；不过要真正跑通，需要完成以下工作：

   1. **合并或注册 LoRA/Adapter 权重**：vLLM 的标准 Qwen 执行器并不知道 CGM 的图约束模块，需要像 CodeFuse 官方那样扩展模型定义，加载 LoRA rank、adapter 以及 Code Graph 编码逻辑。可以直接参考 `CodeFuse-CGM/cgm/inference/vllm.py`，其中实现了 `CGMQwen2ForCausalLM`，并在前向传播里注入图结构特征。【F:CodeFuse-CGM/cgm/inference/vllm.py†L1-L200】
   2. **重新编译或打补丁给 vLLM**：把上述自定义执行器注册到 vLLM，确保 `vllm.entrypoints.openai.api_server` 能够通过 `--model <自定义包>` 找到它。最简单的做法是在 vLLM 安装目录里打猴子补丁，或者把自定义模型打包成 Python 模块并通过 `PYTHONPATH` 暴露给 vLLM。
   3. **扩展服务启动命令**：在 `scripts/eval_graph_planner_engine.py` 的 `_auto_launch_cgm_service` 中增加一个 `backend=vllm` 分支，构造 `python -m vllm.entrypoints.openai.api_server ...` 的命令，并把 CGM 专用的 `--gpu-memory-utilization`、`--tensor-parallel-size`、`--max-model-len` 等参数传进去，同时带上自定义模型入口（如 `--served-model-name codefuse-cgm`）。【F:scripts/eval_graph_planner_engine.py†L468-L720】
   4. **重新对接客户端协议**：vLLM 的 OpenAI 端点只接受纯文本 prompt/response；要让 CGM 在推理时读取计划目标、子图、代码片段等结构化上下文，需要在 Planner -> CGM 的调用链中把这些信息序列化成 prompt，或在 vLLM 侧实现类似 FastAPI 的自定义接口。

   在实践中，团队通常先验证 Hugging Face 路径的准确性，再考虑 vLLM 方案，因为任何一步适配失败都会让 CGM 退化成纯文本模型。如果你具备足够的 GPU 显存且主要瓶颈在批量吞吐，可以按照上述步骤逐项替换；否则维持默认 Hugging Face 推理会更稳健。后续如果仓库正式提供 vLLM 适配层，会在该文档同步给出命令模板与配置示例。

脚本在启动前会探测端点是否已可用；若收到有效响应，就会跳过对应的自启动逻辑，因此手动进程无需额外禁用自动拉起。【F:scripts/eval_graph_planner_engine.py†L430-L467】 如需强制禁止脚本启动新进程，可在 CLI 里追加 `--no-auto-launch-planner-service` 或 `--no-auto-launch-cgm-service`，或在自定义 YAML 中将对应开关设为 `false`。【F:scripts/eval_graph_planner_engine.py†L748-L767】

通过以上梳理，贡献者可以快速理解脚本入口、回归保障与基础设施封装，并在需要时跳转到架构总览文档获取端到端 pipeline 与命令速查。
