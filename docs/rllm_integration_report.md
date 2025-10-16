# rLLM 集成评估报告

## 仓库结构总览
- **代理抽象层（`rllm/agents/agent.py`）**：定义 `BaseAgent`、`Step`、`Trajectory` 等核心基类，要求具体代理实现 `reset`、`update_from_env`、`update_from_model`、`chat_completions` 等钩子函数，用于保存交互历史、构建模型提示并将模型输出翻译为环境动作。
- **环境基类（`rllm/envs/base_env.py`）**：约束环境需实现 `from_dict`、`reset`、`step`、`close`、`seed` 等接口；官方示例 `SWEEnv` 通过 RepoEnv 调度容器，展示如何在这些接口内部完成仓库初始化、动作执行与奖励结算。
- **执行引擎（`rllm/execution/engine.py`）**：`AgentExecutionEngine` 负责批量协调「环境 → 代理 → 模型」三者之间的调用顺序，包括收集观察、拼装模型输入、调用 OpenAI/Verl 推理后解析输出并回放到代理。
- **训练器封装（`rllm/trainer/agent_ppo_trainer.py`）**：在 Verl 的 `RayPPOTrainer` 之上扩展，完成环境/代理批量实例化、轨迹采集（`generate_agent_steps`、`generate_agent_trajectories`）以及与 PPO 循环的对接。
- **Hydra 配置与入口（`train_agent_ppo.py`、`configs/ppo_trainer.yaml`）**：通过映射表 `ENV_CLASS_MAPPING`、`AGENT_CLASS_MAPPING` 按名称装载环境与代理，同时配置模型推理端点、奖励模型、并行度与数据集。
- **数据与奖励管线（`rllm/data/dataset.py`、`verl/verl/trainer/ppo/ray_trainer.py`）**：提供数据集注册/加载、优势估计、Ray 资源池管理等支撑功能。

## 与本项目 RepoEnv/R2E 栈的关系
- rLLM **不会替换** 本仓库现有的 RepoEnv 集成：它需要环境子类自行调用 RepoEnv。我们可以仿照 `SWEEnv` 编写 `GraphPlannerEnv(BaseEnv)`，在 `from_dict` 里解析任务描述（包含 `ds` JSON）、构建 `PlannerEnv`，并在 `reset`/`step` 中代理到当前的规则型 Planner 流程。
- 代理层需实现 `GraphPlannerAgent(BaseAgent)`，将 Planner 产生的观察、记忆、动作与 rLLM 的执行引擎对接：
  - `update_from_env` 接收环境返回的观察（代码图、记忆、测试结果等），刷新内部状态并拼装下一轮模型提示；
  - `update_from_model` 解析模型输出的自然语言计划、工具调用参数，转化为对 `PlannerEnv.step` 的结构化 `Action`；
  - `chat_completions` / `get_prompt` 可沿用现有的提示模板以保持记忆和子图选择逻辑。
- 将上述类注册进 `ENV_CLASS_MAPPING["graph_planner"]`、`AGENT_CLASS_MAPPING["graph_planner"]`，即可在 `ppo_trainer.yaml` 中通过 `env.name=graph_planner`、`agent.name=graph_planner` 启用我们的环境/代理。

## 训练流程对接建议
1. **准备任务数据集**：利用现有的 RepoEnv 数据集描述（或本仓库的示例镜像），通过 `DatasetRegistry.register_from_jsonl` 生成 Verl 期望的 `_verl.parquet`，并在 Hydra 配置中设置 `dataset.path`。
2. **环境参数传递**：在 `env.env_args` 中提供 `ds_json` 路径、最大步数、日志路径等，`GraphPlannerEnv.from_dict` 读取这些参数后创建 `PlannerEnv`。
3. **模型执行后端**：在 `model.actor_rollout_ref.rollout` 中配置 OpenAI/vLLM/Verl 推理服务地址，确保 `AgentExecutionEngine` 能获取策略模型输出。
4. **奖励结算**：复用 `PlannerEnv`/`SandboxRuntime` 已有的测试执行与奖励逻辑，环境的 `step` 返回 `(obs, reward, done, info)`；必要时可在 `info` 中附加更细粒度的测试指标，供 Verl 的奖励聚合器使用。
5. **并行与容错**：确认 `GraphPlannerEnv.is_multithread_safe()` 返回 `True`，并在内部对 RepoEnv 容器句柄加锁；确保容器异常时 `close` 方法能够清理残留资源，以避免 PPO 并行训练泄露容器。

## RepoEnv 数据集与本仓库集成现状

### rLLM 训练所需的 RepoEnv 数据集
- rLLM 官方仓库**不自带**容器镜像或任务样本，所有示例环境（包括 `SWEEnv`）都要求用户提供 RepoEnv 数据集描述。核心资产包括：
  - 一份 `ds` JSON：至少包含 `docker_image`、`repo`/`commit`、`tests`、`workdir` 等字段，供 RepoEnv 的 `DockerRuntime` 在 `reset` 时拉取镜像、检出代码并准备测试命令；
  - 可选的 `_verl.parquet` 数据集：通过 `DatasetRegistry` 根据 JSONL/Parquet 原始任务列表生成，Verl 的 DataLoader 会读取该文件，把每条任务的 `extra_info` 传入 `env.from_dict`。
- 训练前需要确保 `ds` 指向的镜像已经存在于 Docker 守护进程可访问的仓库（本地 `docker load` / `docker build`、或远程镜像仓库），否则 `RepoEnv` 无法启动容器。
- 若希望复用 R2E-Gym 官方任务，可从 Hugging Face（`Repair2Earn` 组织）下载公开的 `ds` 与镜像，再在 Hydra 配置的 `env.env_args.ds_json` 中指向下载后的路径。

### 本项目当前的 RepoEnv 集成
- 本仓库的 `runtime/SandboxRuntime` 在检测到 `sandbox.backend="repoenv"` 且传入 `r2e_ds_json` 时，会：
  1. 读取 `ds` JSON 并构造 `EnvArgs`；
  2. 初始化 r2e-gym 的 `RepoEnv`，继而托管容器生命周期、文件同步与命令执行；
  3. 将 `run`、`apply_patch`、`lint`、`test` 等高层调用代理给底层 `DockerRuntime`，让现有的 Planner/记忆系统无需感知后端差异。
- 为了便于联调，我们提供了一套可直接构建的示例资产：
  - **镜像上下文**：`docker/repoenv_sample/`（包含一个带缺陷的算术模块与 `pytest` 用例）；
  - **构建脚本**：`scripts/build_repoenv_sample.sh`（在本地 Docker 环境中构建并打上 `graph-planner/repoenv-sample:latest` 标签）；
  - **数据集描述**：`config/r2e_ds_repoenv_sample.json`（指向上述镜像、仓库路径与测试命令）。
- 运行规则驱动代理或后续的 rLLM 训练时，只需在配置中引用这份 `ds` 描述，即可让环境装载示例容器；若要扩展为真实任务，可仿照该结构新增镜像与 `ds` 文件。

### 本仓库新增的 rLLM 对接实现

- **环境适配层（`integrations/rllm/env.py`）**：`GraphPlannerRLLMEnv` 实现了 rLLM `BaseEnv` 接口，内部创建 `PlannerEnv` 并在 `reset`/`step` 中透传我们现有的 Explore/Memory/Repair/Submit 流程；针对未安装 rLLM 的场景提供占位符，避免破坏原有规则代理。
- **代理封装（`integrations/rllm/agent.py`）**：`GraphPlannerRLLMAgent` 继承 rLLM `BaseAgent`，将 `PlannerEnv` 的观察整理为 JSON 提示，让策略模型输出结构化动作（含 anchors、plan targets 等）；解析失败时可回退到规则策略，保证训练早期或未接入模型时的稳定性。
- **数据集注册工具（`integrations/rllm/dataset.py` + `scripts/register_graphplanner_dataset.py`）**：支持把 JSON/JSONL 任务描述注册到 rLLM 的 `DatasetRegistry`，自动生成 `_verl.parquet`；在解析时会补全 `r2e_ds_json` 的绝对路径以兼容 RepoEnv。
- **示例任务清单（`datasets/graphplanner_repoenv_sample.jsonl`）**：与现有的示例容器配套，提供一条最小化的训练任务，可用于验证从数据加载 → 环境构建 → 容器交互的完整链路。
- **训练脚本（`scripts/train_graphplanner_rllm.py`）**：读取 rLLM 默认 `ppo_trainer.yaml`，注入上述数据集、环境与代理类，再通过 `train_agent.remote` 启动 Verl PPO 训练；脚本保留 `--model-path`、`--ray-address` 等参数以对接未来的策略模型与分布式推理服务。

### 端到端训练流程示例

1. 构建示例镜像并写入 `config/r2e_ds_repoenv_sample.json`（仓库已提供构建脚本与样例）。
2. 执行 `scripts/register_graphplanner_dataset.py datasets/graphplanner_repoenv_sample.jsonl`，生成 `_verl.parquet` 并注册到 rLLM 数据集中。
3. 准备策略模型权重（或先占位），随后运行 `scripts/train_graphplanner_rllm.py --model-path <模型目录> --use-fallback --print-config` 查看合并后的 Hydra 配置；移除 `--print-config` 即可通过 Ray 启动 PPO 训练，训练过程中 `GraphPlannerRLLMAgent` 将自动与 RepoEnv 容器交互、回放奖励并支持并行采样。
4. 如需切换到自定义任务，只要新增 JSON/JSONL 描述并重新注册数据集，再在训练脚本参数中替换 `--dataset`/`--dataset-name` 即可。

## 安装方式与推荐流程说明

- **`pip install` vs `pip install -e`**：普通的 `pip install .` 会将当前仓库打包后复制到虚拟环境的 `site-packages`，安装完成后与源码目录脱钩；而 `pip install -e .`（editable 模式）会在 `site-packages` 写入一个指向源码目录的 `.egg-link`，运行时直接 import 本地源码。这样一来，每次修改 rLLM 源码都能立刻生效，无需重新安装，非常适合需要频繁调试或二次开发的场景。
- **官方安装脚本的动机**：rLLM 官方推荐的步骤先通过 `git clone --recurse-submodules` 获取源码与 Verl 子模块，再在新建的 Conda 环境里执行 `bash scripts/install_verl.sh` 安装底座依赖，最后使用 `pip install -e .` 以可编辑模式安装 rLLM 本体。这一流程确保：
  1. Verl 及其 Ray/FSDP 依赖被正确编译安装；
  2. rLLM 在开发阶段可以直接修改源码并立即反映到训练脚本；
  3. 所有依赖隔离在独立的 `rllm` Conda 环境中，避免与系统 Python 或本项目的其他依赖冲突。
- **为何需要克隆仓库才能运行 `install_verl.sh`**：该脚本存放在 rLLM 源码的 `scripts/` 目录下，它会使用相对路径引用仓库内的 Verl 子模块和补丁文件。因此只有在本地克隆 rLLM 后（并确保 `--recurse-submodules` 拉取了 Verl 子模块）才能执行该脚本；如果仅通过 `pip install rllm` 获取预编译包，虚拟环境中不会包含 `scripts/install_verl.sh` 以及所需的子模块资源。
- **没有克隆仓库时的替代方案**：若你不希望克隆完整源码，可以参考脚本内容手动安装 Verl（例如按照 Verl 官方说明从其仓库执行安装命令），随后再运行 `pip install rllm` 或者 `pip install -e /path/to/rllm`。不过在需要频繁更新 Verl 或自定义其依赖时，仍建议按官方推荐的方式克隆 rLLM 仓库并执行脚本，以减少版本不匹配的风险。
- **在本项目中的实践**：若要沿用官方流程，可在任意目录克隆 rLLM 仓库并按上述命令完成安装；之后我们的集成代码即可通过常规的 `import rllm` 访问其模块，无需将 rLLM 源码复制进本仓库。若暂时未安装 rLLM，相关集成会自动退回到规则代理逻辑，不影响现有功能。
- **是否需要在本项目目录中 `pip install -e`？**：本项目无需把 rLLM 直接放在仓库子目录里；只要所在虚拟环境能找到 `rllm` 模块即可。若你正在调试 rLLM 本身、希望修改其源码并立即生效，可以在任意位置（包括本仓库外部）克隆 rLLM 后执行 `pip install -e /path/to/rllm`，即可获得可编辑安装。如果将 rLLM 复制到本仓库并在此目录下执行 `pip install -e .`，效果与在外部目录相同，但会把两个项目耦合在同一个 Git 仓库中，后续同步上游更新会更繁琐，因此默认仍建议分离管理，仅在确实需要联动开发时再采用这种方式。

