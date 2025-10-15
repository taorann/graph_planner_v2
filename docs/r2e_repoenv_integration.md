# R2E-Gym RepoEnv 集成概览

本文档梳理 R2E-Gym 仓库中与容器交互相关的关键层次，并说明如何让本项目的 Agent 复用 RepoEnv 以启动并驱动官方容器环境。

## R2E-Gym 的主要层次

### 1. 数据集描述 (`ds`)
* `RepoEnv` 的初始化依赖一个数据源字典 `ds`，经由 `EnvArgs` 包装后传入环境。`ds` 至少需要包含 `docker_image`、`repo_name`/`repo`、提交元数据等字段。 【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L18-L36】
* 仓库默认会把 `ds` 序列化到 JSON（例如 `config/r2e_ds_min.json`），供运行时加载。 【F:runtime/sandbox.py†L49-L60】【F:config/r2e_ds_min.json†L1-L5】

### 2. 环境层 (`RepoEnv`)
* `RepoEnv` 是一个 gym-like 包装器，内部创建 `DockerRuntime` 并对外暴露 `step/reset/add_commands` 等接口，供上层 Agent 调用。初始化时可以选择后端（默认 docker，也支持 kubernetes）。 【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L24-L87】
* `RepoEnv.step` 会把 Agent 给出的函数调用（动作）转换为 bash 命令，交给 runtime 执行，随后把输出封装成 `Observation`。 【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L104-L152】

### 3. 运行时层 (`DockerRuntime`)
* `DockerRuntime` 负责真正的容器生命周期管理：解析 `ds` 决定镜像、拉起容器/Pod、同步仓库、运行命令、复制文件等。它支持本地 Docker 和 Kubernetes 两种后端。 【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L33-L162】
* 运行时提供 `run/apply_patch/get_repo_state/copy_to_container` 等能力，是 RepoEnv 与容器交互的核心。 【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L214-L400】

### 4. Agent 层
* 官方 `EditAgent` 等实现通过 `run/edit.py` 创建 `EnvArgs → RepoEnv → Agent` 链路，并在 `run_agent_with_restarts` 中循环调用 `env.step`。这说明只要提供兼容的环境对象，自定义 Agent 便可接入。 【F:R2E-Gym/src/r2egym/agenthub/run/edit.py†L214-L284】

## 本项目中的 RepoEnv 适配

项目在 `runtime/sandbox.py` 中封装了统一的 `SandboxRuntime`：
* `runtime/__init__.py` 会在导入时自动把 `R2E-Gym/src` 加入 `sys.path`，让 vendored 的 R2E 代码开箱即用。 【F:runtime/__init__.py†L1-L24】
* `SandboxConfig` 支持 `backend="repoenv"`，并要求提供 `r2e_ds_json`。构造函数会自动判断是否使用 RepoEnv。 【F:runtime/sandbox.py†L27-L68】
* `_init_repoenv_backend` 读取 `ds` JSON，实例化 `EnvArgs`/`RepoEnv`，并把返回的 R2E DockerRuntime 作为底层执行引擎；随后做了一些兜底初始化（创建 `/testbed`、安装 pytest、配置 git safe.directory）。 【F:runtime/sandbox.py†L70-L118】
* 其余方法（`run/apply_patch/get_patch/lint/test` 等）在检测到后端为 RepoEnv 时，直接代理到 R2E runtime，实现与本地 docker-py 相同的接口。 【F:runtime/sandbox.py†L120-L216】

`PlannerEnv` 则是训练/推理时与 Agent 打交道的环境抽象，内部持有 `SandboxRuntime` 实例。Agent 调用 `PlannerEnv.step` 时会触发 `SandboxRuntime` 执行 lint、测试等命令，无需关心底层是 RepoEnv 还是纯 Docker。 【F:env/planner_env.py†L3-L60】

`scripts/smoke_test_repoenv.py` 提供了最小化示例：只要设置 `backend="repoenv"` 并传入 `r2e_ds_json`，现有回路就能连通 RepoEnv。 【F:scripts/smoke_test_repoenv.py†L1-L35】

## 将自有 Agent 连接到 RepoEnv 的步骤

1. **准备 `ds` JSON**：根据目标数据集生成符合 R2E 要求的 `ds` 字典（至少包含镜像、仓库、提交信息），存入某个 JSON 文件，并在运行时通过 `SandboxConfig.r2e_ds_json` 指向它。 【F:runtime/sandbox.py†L62-L83】
2. **启用 RepoEnv 后端**：构造 `SandboxConfig` 时设置 `backend="repoenv"`。`SandboxRuntime` 会自动加载 R2E 的 `RepoEnv` 并保留统一接口，现有 `PlannerEnv`/ACI 工具无需改动。 【F:runtime/sandbox.py†L27-L110】【F:env/planner_env.py†L13-L40】
3. **在 Agent 初始化时传入配置**：本项目的管线通常通过 `PlannerEnv.from_dict` 或类似入口创建环境；确保注入的配置中携带 `r2e_ds_json` 和其他必要字段即可。参考 `scripts/smoke_test_repoenv.py`，可从环境变量或配置文件读取路径。 【F:env/planner_env.py†L13-L23】【F:scripts/smoke_test_repoenv.py†L5-L33】
4. **复用现有动作接口**：`PlannerEnv.step` 内部已经把 `RepairAction`/`SubmitAction` 映射到 `SandboxRuntime` 的 `apply_patch/lint/test`。自定义 Agent 只需按照既定动作契约调用 `PlannerEnv`，便能驱动 RepoEnv 中的容器运行。 【F:env/planner_env.py†L24-L60】
5. **本地验证**：使用 `scripts/smoke_test_repoenv.py` 或你自己的集成测试，确认 Agent 能通过 `SandboxRuntime` 成功在 RepoEnv 容器里执行命令与测试。 【F:scripts/smoke_test_repoenv.py†L1-L40】

通过以上设置，你的 Agent 可以在不改动核心业务逻辑的情况下切换到底层 RepoEnv，实现与官方 R2E 容器环境的对接与激活。
