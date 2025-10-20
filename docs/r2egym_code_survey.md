# R2E-Gym 代码调研报告

## 1. 总览

`graph_planner/R2E-Gym/src/r2egym` 中的代码分为两大类：

* **AgentHub 子目录**：提供代理运行时（容器、环境、工具）、动作/观测协议、轨迹与校验器，是在线运行与评测的核心。
* **非 AgentHub 模块**：包含差异解析、数据集构建、Docker 脚本、安装脚本、SWE-Smith 常量等支撑设施，主要为离线数据准备与运行时提供基础能力。

下文分别梳理各模块的职责、在 R2E-Gym 内部的引用关系，以及在 Graph Planner 项目中的使用状态。

## 2. 非 AgentHub 模块

| 模块 | 功能概述 | 在 R2E-Gym 中的用途 | 在 Graph Planner 中的使用情况 |
| --- | --- | --- | --- |
| `bash_utils.py` | 封装 `subprocess.run`，统一 `/bin/bash` 执行、超时与日志兜底处理。【F:R2E-Gym/src/r2egym/bash_utils.py†L1-L50】 | 被 `repo_analysis/repo_testextract.py` 等脚本调用以运行仓库测试与补丁流程。【F:R2E-Gym/src/r2egym/repo_analysis/repo_testextract.py†L15-L206】 | 未直接引用；Graph Planner 使用自有 `SandboxRuntime` 执行命令。 |
| `commit_models/` | `parse_diff.py` 负责把 Git diff 解析为结构化 `ParsedCommit`；`commit_to_ast.py`、`entity_utils.py` 为实体抽取与 AST 建模，`diff_classes.py` 定义数据模型。【F:R2E-Gym/src/r2egym/commit_models/parse_diff.py†L1-L200】【F:R2E-Gym/src/r2egym/commit_models/entity_utils.py†L1-L120】 | AgentHub 的 `DockerRuntime` 解析提交、构造补丁统计，Repo analysis 脚本也依赖这些模型。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L32-L113】【F:R2E-Gym/src/r2egym/repo_analysis/store_repo_commits.py†L10-L68】 | 当前未在 Graph Planner 内直接使用；未来可复用解析逻辑对真实提交做结构化分析。 |
| `docker_bash_utils/` | 提供 Docker 镜像标签查询脚本及批量清理脚本，辅助环境管理。【F:R2E-Gym/src/r2egym/docker_bash_utils/docker_list_tags.py†L1-L55】 | CLI (`agenthub/run/edit.py`) 启动前用于匹配镜像标签与清理容器。【F:R2E-Gym/src/r2egym/agenthub/run/edit.py†L19-L80】 | Graph Planner 未调用；容器生命周期交由自研 `SandboxRuntime` 控制。 |
| `install_utils/` | 针对不同依赖（如 `numpy`、`tornado`、`datalad`）的安装脚本、pytest runner 定制化补丁。【F:R2E-Gym/src/r2egym/install_utils/process_aiohttp_updateasyncio.py†L1-L120】 | `repo_analysis/repo_testextract.py` 在构建合成任务时，把脚本注入容器以安装依赖或替换测试运行器。【F:R2E-Gym/src/r2egym/repo_analysis/repo_testextract.py†L125-L205】 | 尚未使用；若需要复现 R2E-Gym 的数据生成流程，可直接调用。 |
| `logging.py` | 使用 RichHandler 配置彩色日志，并支持写入文件。【F:R2E-Gym/src/r2egym/logging.py†L1-L34】 | AgentHub CLI 与校验脚本通过 `setup_logging` 统一日志格式。【F:R2E-Gym/src/r2egym/agenthub/run/edit.py†L19-L40】 | Graph Planner 自有日志方案；未直接引用。 |
| `repo_analysis/` | 包含数据集生成脚本（抓取 commit、解析测试、构造 issue）、日志解析与 Docker 校验器。【F:R2E-Gym/src/r2egym/repo_analysis/repo_testextract.py†L15-L205】【F:R2E-Gym/src/r2egym/repo_analysis/execution_result_analysis.py†L6-L120】 | 用于离线生成 R2E-Gym 任务与统计回放，AgentHub 运行时依赖其 `ExecutionResult` 等结构解析测试输出。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L1-L75】 | Graph Planner 未直接复用；我们已内置 R2E-Gym 原始 JSONL 数据集，可在需要时调用这些脚本增量扩展任务。 |
| `swesmith/` | 镜像 SWE-Smith 项目的常量与测试命令选择逻辑；`utils.py` 提供 `get_test_command` 等辅助函数。【F:R2E-Gym/src/r2egym/swesmith/constants.py†L1-L84】【F:R2E-Gym/src/r2egym/swesmith/utils.py†L1-L80】 | 供 `DockerRuntime` 针对 SWE-Smith 镜像设置测试指令与打分策略。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L24-L69】 | Graph Planner 暂未针对 SWE-Smith 做适配，暂不使用。 |
| `__init__.py` | 暴露 AgentHub、repo_analysis、swesmith 等子模块，便于外部 `import r2egym`。【F:R2E-Gym/src/r2egym/__init__.py†L1-L26】 | 供 R2E-Gym 及上层项目做命名空间初始化。 | Graph Planner 依赖 `ensure_rllm_importable()` 时也会解析该命名空间，但无直接逻辑。 |

## 3. AgentHub 子目录调研

AgentHub 目录是我们与 R2E-Gym 对接的核心。下表列出各子模块的职责以及在 Graph Planner 中的使用状态。

| 子目录 | 核心文件/类 | 职责与依赖 | Graph Planner 使用情况 |
| --- | --- | --- | --- |
| `action/` | `Action` 类负责解析 `<function=...>` 文本、序列化成 Bash 命令。【F:R2E-Gym/src/r2egym/agenthub/action/action.py†L1-L120】 | 被 RepoEnv、Agent 调用以在容器内执行动作。 | 未直接引用；我们使用自定义文本协议解析器。 |
| `agent/` | `AgentArgs`、`Agent` 负责加载 Prompt、拼装工具列表、调用 LLM、记录 Trajectory。【F:R2E-Gym/src/r2egym/agenthub/agent/agent.py†L1-L200】 | CLI 启动训练/评测时直接实例化；依赖工具集、环境与 DockerRuntime。 | 未使用；Graph Planner 通过自研 Planner/CGM 模型驱动。 |
| `config/` | 各 scaffolds 的 YAML（如 `edit_fn_calling.yaml`）声明系统提示、工具、环境参数。【F:R2E-Gym/src/r2egym/agenthub/config/r2egym/edit_fn_calling.yaml†L1-L120】 | 供 AgentArgs 读取。 | 未引用。 |
| `environment/` | `EnvArgs`、`RepoEnv` 封装容器执行、命令注入、步进逻辑。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L1-L132】 | CLI 和训练框架通过 `RepoEnv.step` 运行代理，与 DockerRuntime 配合。 | **已使用**：`SandboxRuntime` 在 repoenv 模式下注入 `EnvArgs`/`RepoEnv` 并复用其 `DockerRuntime`。【F:graph_planner/runtime/sandbox.py†L1-L126】 |
| `observation/` | `Observation` 类型记录 action 输出、错误码等元数据。【F:R2E-Gym/src/r2egym/agenthub/observation/observation.py†L1-L80】 | RepoEnv/Agent 交互时构建观测。 | 未直接使用；`SandboxRuntime` 直接访问 `RepoEnv.runtime`。 |
| `run/` | `edit.py` CLI 负责装配 Agent、RepoEnv、DockerRuntime 并执行多任务评测，同时加载工具、日志和 Docker 镜像信息。【F:R2E-Gym/src/r2egym/agenthub/run/edit.py†L15-L260】 | 官方命令行入口。 | 未使用；我们自建训练/评测脚本。 |
| `runtime/` | `DockerRuntime` 负责容器/K8s 生命周期、补丁应用、测试执行；依赖 `ParsedCommit`、SWE-Bench 常量。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L1-L130】 | AgentHub 和 RepoEnv 的底层执行引擎。 | **已使用**：`SandboxRuntime` 通过 RepoEnv 间接复用 DockerRuntime 的 run/apply_patch/test 能力。【F:graph_planner/runtime/sandbox.py†L72-L123】 |
| `tools/` | 定义 `file_editor`、`search`、`execute_bash` 等函数调用描述，用于构建 LLM 工具列表。【F:R2E-Gym/src/r2egym/agenthub/tools/__init__.py†L1-L200】 | Agent/CLI 将这些工具注入到模型提示中。 | 未使用；Graph Planner 自有动作集合。 |
| `trajectory/` | `Trajectory`、`TrajectoryStep` 记录历史步骤，包含思考、动作、观测，并提供 SWE-bench 提交转换工具。【F:R2E-Gym/src/r2egym/agenthub/trajectory/trajectory.py†L1-L120】 | 训练日志、Best-of-N 汇总依赖这些结构。 | 未使用；我们使用自定义记忆/日志格式。 |
| `utils/` | `get_logger`、`match_dockerimage_to_repo`、`get_parsed_commit` 等工具函数供 Runtime/Agent 使用。【F:R2E-Gym/src/r2egym/agenthub/utils/utils.py†L3-L120】 | 支持 DockerRuntime、Agent、CLI 进行镜像匹配和日志初始化。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L1-L60】 | 仅间接使用（随 RepoEnv/DockerRuntime 导入）。 |
| `verifiers/` | 多个脚本将轨迹转换为验证器输入、运行复现/回归测试、聚合 best-of-n 结果。【F:R2E-Gym/src/r2egym/agenthub/verifiers/run_regression_tests.py†L1-L120】【F:R2E-Gym/src/r2egym/agenthub/verifiers/run_reproduction_tests.py†L21-L90】 | AgentHub 评测流程中用于生成外部验证报告。 | 未集成；Graph Planner 的评测通过自有测试流水线。 |

### 3.1 已直接复用的组件

* **RepoEnv + DockerRuntime**：`SandboxRuntime` 在 `repoenv`/`r2e` 模式下会读取 R2E 数据集条目，实例化 `EnvArgs` 与 `RepoEnv`，并透传底层的 `DockerRuntime` 来执行命令、应用补丁和运行测试。【F:graph_planner/runtime/sandbox.py†L60-L158】
* **间接依赖**：当 RepoEnv/DockerRuntime 初始化时，会加载 `commit_models`, `swesmith`, `agenthub.utils` 等辅助模块来解析提交、选择测试命令并打印日志。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L1-L120】

### 3.2 可选但尚未接入的组件

* **Action/Observation/Trajectory**：这些类型封装 R2E-Gym 原生文本协议，适合在需要完全复刻官方代理行为时使用。Graph Planner 已重写动作协议，因此当前未用到。
* **Agent 与工具集**：我们自研 Planner/CGM 模型及工具，因此未引入 R2E 的 Agent/工具描述。如果未来希望直接复用 R2E 的调用栈，可加载 `AgentArgs`+`Agent`。
* **Verifiers & CLI**：当前训练/评测脚本独立实现，未调用 `agenthub/run` 或 `verifiers`。如需生成与 R2E 官方一致的复现/回归报告，可参考这些脚本。 

### 3.3 暂时用不上的模块

* **Config YAML**：由于我们不再走 R2E 的 LLM Prompt/工具模板，这些配置文件暂时无用。
* **工具命令脚本**：`agenthub/tools` 中的函数调用描述与我们自定义的文本协议不兼容，除非迁移到 Tool-Calling 模式。 

## 4. 结论与建议

1. **核心依赖**：Graph Planner 目前仅复用 `RepoEnv` 与 `DockerRuntime`，其余模块保持原样即可随时调用。确保在更新 R2E-Gym 时验证这两个类的接口未发生破坏性变化。
2. **潜在扩展**：如果未来希望重放官方 Agent 或使用 R2E 提供的工具链，可引入 `agenthub/action`、`agenthub/tools`、`agenthub/trajectory` 等模块。
3. **数据管线复现**：`repo_analysis` 与 `install_utils` 负责构建/清洗 R2E 任务数据，若要扩充数据集，可直接执行这些脚本，并把结果写入仓库的 `datasets/r2e_gym/` 目录。

以上调研覆盖了 AgentHub 中的所有子模块及非 AgentHub 支撑代码，为后续定制或裁剪提供依据。
