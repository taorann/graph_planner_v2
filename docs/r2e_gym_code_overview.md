# R2E-Gym 代理与容器交互代码总览

本文档从代码结构与关键函数层面梳理 R2E-Gym 的运行链路，帮助快速定位代理（agent）与容器（runtime）之间的接口。

## 1. 环境入口：`RepoEnv`

`RepoEnv` 是 Gym 兼容的环境实现，负责把数据集描述转换成可操作的运行时实例：

- **初始化**：构造时直接实例化 `DockerRuntime`，并把数据集条目（`EnvArgs.ds`）传入，默认通过 `/bin/bash -l` 进入容器。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L28-L60】
- **重置与生命周期**：`reset()` 会关闭现有 runtime 并重新创建一个全新的 `DockerRuntime`；`close()` 则把资源释放回容器侧。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L63-L78】【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L235-L239】
- **工具注入**：`add_commands()` 读取代理侧的 shell/Python 命令定义，复制到容器内 `/usr/local/bin` 并赋予可执行权限，随后缓存允许调用的命令清单。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L79-L130】
- **执行动作**：`step()` 会把代理返回的 `Action` 校验、转换为具体 bash 命令，调用 `DockerRuntime.run()` 执行，并封装成 `Observation` 返回给代理。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L145-L197】
- **奖励计算**：环境可以在任意时刻调用 `compute_reward()`，内部直接代理到 runtime 的 `_calculate_reward()`，与特定任务的测评逻辑解耦。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L221-L233】

## 2. 动作与观测：`Action` / `Observation`

- `Action` 负责把 XML/函数调用格式转换为可执行的 bash 命令，并在 `to_bashcmd()` 中统一加入参数转义逻辑。【F:R2E-Gym/src/r2egym/agenthub/action/action.py†L6-L108】
- `Observation` 把 runtime 的原始输出包装成对话友好的字符串，自动处理空动作、`finish/submit` 终止符以及长输出截断。【F:R2E-Gym/src/r2egym/agenthub/observation/observation.py†L7-L44】

## 3. 命令说明：`ParseCommandBash` 与工具集合

- 代理启动时会通过 `ParseCommandBash` 解析命令脚本，支持 `bash` 函数、带 `@yaml` 注释的脚本以及普通 shell 脚本，从而自动生成命令元数据。【F:R2E-Gym/src/r2egym/agenthub/agent/commands.py†L80-L199】
- `agenthub/tools/__init__.py` 中定义了可暴露给 LLM 的函数工具规范（`file_editor`、`search`、`execute_bash` 等），同时与 `Action.to_bashcmd()` 的命令名称保持一致，确保函数调用结果能够映射到容器命令。【F:R2E-Gym/src/r2egym/agenthub/tools/__init__.py†L5-L200】

## 4. 运行时核心：`DockerRuntime`

`DockerRuntime` 是容器生命周期与文件系统操作的封装，既支持直接 Docker，也兼容 Kubernetes：

- **初始化**：根据数据集条目推断镜像、仓库路径、提交信息，创建 docker/k8s 客户端并立即调用 `start_container()` 与 `setup_env_*` 完成容器准备工作。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L83-L183】
- **启动/销毁**：`start_container()` 会复用同名容器或重新拉起；`stop_container()`、`reset()`、`close()` 负责容器回收与客户端释放。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L350-L385】【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L447-L485】【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L1076-L1085】
- **命令执行**：`run()` 在容器内执行任意命令，统一注入 `timeout`、工作目录与 PATH，并根据返回码格式化错误信息；K8s 模式由 `_run_kubernetes()` 适配。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L600-L784】
- **文件操作**：提供 `copy_to_container()`、`create_file()`、`apply_patch()` 等便捷函数用于同步宿主文件、写入补丁并在容器中应用。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L788-L915】
- **测试与奖励**：根据镜像类别（SWE-Bench、SWESmith、R2E）分别实现 `_calculate_reward_*`，自动运行测试脚本并解析日志，最终通过 `_calculate_reward()` 统一对外输出。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L974-L1074】

## 5. 代理主循环：`Agent.run`

`Agent.run()` 连接 LLM、工具与环境，形成完整的交互闭环：

1. **初始化**：重置环境、注入命令、清空轨迹，并从 `DockerRuntime` 读取任务说明和基准补丁，填充系统/User Prompt。【F:R2E-Gym/src/r2egym/agenthub/agent/agent.py†L304-L388】
2. **循环决策**：每轮循环基于历史消息构造对话，调用 `model_query()`（可带函数调用工具），再使用 `parse_response()`/`custom_parser()` 转成 `Action`。【F:R2E-Gym/src/r2egym/agenthub/agent/agent.py†L389-L448】
3. **执行反馈**：通过 `env.step()` 执行动作并记录 `Observation`，根据是否启用函数调用把运行结果写回对话历史，供下一轮 LLM 决策使用。【F:R2E-Gym/src/r2egym/agenthub/agent/agent.py†L450-L498】
4. **终止条件**：根据动作（`finish/submit`）、步数限制或异常退出设定 `exit_reason`，并在需要时调用 `env.compute_reward()` 获取最终得分。【F:R2E-Gym/src/r2egym/agenthub/agent/agent.py†L499-L520】

整体来看，代理只需掌握 `RepoEnv.step()` 与 `Observation` 格式即可驱动容器环境；绝大多数环境感知与副作用都被 `DockerRuntime` 封装，便于更换后端或扩展新工具。

## 6. 本项目 Agent（Graph Planner）执行链路

下面梳理本项目自带 Agent 从并行启动、动作决策到容器交互的完整回路，聚焦于关键函数。

### 6.1 并行实例启动与环境构造

* **Worker 启动入口**：批量评测或并行采样时，每个进程/线程调用与 `scripts/smoke_test.py` 相同的引导逻辑——创建 `PlannerEnv` 与 `PlannerAgent`，彼此无全局状态，可安全并行。【F:scripts/smoke_test.py†L1-L37】
* **环境装配**：`PlannerEnv.from_dict` 负责把外部注入的 issue/sandbox 配置翻译成 `SandboxConfig`，并立即构造 `SandboxRuntime`；因此无论是单机还是并行 worker，都会在这里完成容器后端选择与初始化。【F:env/planner_env.py†L12-L22】
* **Sandbox 启动**：`SandboxRuntime.__init__` 根据 `backend` 字段选择 `repoenv`/`r2e`/`docker` 等后端；在 RepoEnv 模式下 `_init_repoenv_backend` 会加载 `ds` JSON 并实例化 R2E 的 `RepoEnv`，实现官方容器的热启动。【F:runtime/sandbox.py†L27-L118】

### 6.2 动作决策循环

* **观测生成**：`PlannerEnv.reset` 为 agent 提供初始观测（issue 信息 + 步数），`PlannerEnv._obs` 是统一的观测打包函数，保证每一步都返回同样的字典结构。【F:env/planner_env.py†L24-L57】
* **策略薄壳**：`PlannerAgent.step` 是当前占位策略，实现为 `_decide`（首步 `RepairAction`，其余 `SubmitAction`），并返回包含自然语言响应与真实动作对象的消息体。你可以在 `_decide` 中替换为自己的大模型/策略网络，同时保持返回格式不变。【F:agent/planner_agent.py†L6-L24】
* **环境步进**：`PlannerEnv.step` 根据动作类型路由到 `_do_repair`/`_do_submit` 等函数，统计奖励、终止条件及调试信息，形成与 Gym 兼容的 `(obs, reward, done, info)` 元组。【F:env/planner_env.py†L24-L57】

### 6.3 容器互动与工具调用

* **Repair 路径**：`PlannerEnv._do_repair` 在 `apply=True` 时将来会接入 Collater→CGM→Guard 产出的补丁；当前占位实现展示了如何调用 `SandboxRuntime.apply_patch`/`lint`/`test` 并把结果写入 info。你可以在这里拼接自己的补丁生成逻辑。【F:env/planner_env.py†L43-L53】
* **Submit 评测**：`PlannerEnv._do_submit` 会触发 `SandboxRuntime.test()`，并把最近一次补丁通过 `get_patch()` 带回，用于最终评分或上报。【F:env/planner_env.py†L55-L60】
* **底层命令执行**：无论 Repair/Submit，真正的命令都由 `SandboxRuntime.run`、`apply_patch` 等方法代理到底层运行时；在 RepoEnv 模式下，这些函数直接调用 R2E 的 `DockerRuntime.run` 等 API，保证与官方容器一致。【F:runtime/sandbox.py†L120-L185】

### 6.4 Orchestrator 集成（计划 → 补丁 → 容器）

* **单轮执行**：`orchestrator.loop.run_once` 描述了全局一次修复的流程：生成计划 `_make_plan`、`collate` 上下文、调用 `actor.cgm_adapter.generate` 产出补丁、`enforce_patch_guard` 校验，然后通过 `_apply_patch_with_guard` 按顺序调用 `aci.tools.edit_lines` 写入容器，最后统一触发 `lint_check` 与 `run_tests`。这一串调用都在一个函数内串联，方便在多 worker 场景中直接复用。【F:orchestrator/loop.py†L43-L153】【F:orchestrator/loop.py†L155-L248】
* **工具层桥接**：`aci.tools` 提供 `edit_lines`/`lint_check`/`run_tests` 等统一入口，内部再调度 `SandboxRuntime` 的命令执行。因其纯函数式接口，orchestrator 可在任意线程/进程中调用而无需共享状态。【F:aci/tools.py†L1-L126】【F:aci/tools.py†L192-L268】

通过以上链路，你可以在并行 worker 中复用相同的 Agent 与环境栈：每个 worker 独立初始化 `PlannerEnv`（底层自动连到 RepoEnv 容器），Agent 负责产生动作，Orchestrator 决策补丁并通过 ACI 工具落盘，实现对官方容器的稳定控制。
