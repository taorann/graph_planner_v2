# R2E-Gym 代理与容器交互代码总览

本文档以“层次结构 → 关键函数 → 端到端调用链”的顺序梳理当前仓库（Graph Planner）在复用 R2E-Gym 容器栈时的整体结构，帮助快速定位每个模块的职责与衔接点。

## 1. 仓库层次总览

整体目录可以按照“控制面 → 决策面 → 容器面”三大层次划分，核心关系如下：

```text
                ┌───────────────────── 控制面（测试/CLI） ─────────────────────┐
                │            scripts/run_rule_agent.py · aci/tools.py          │
                └──────────────────────────────────────────────────────────────┘
                                   │
                      ┌────────────┴────────────┐
                      ▼                         ▼
        ┌─────────────── 决策面（Agent & Memory） ───────────────┐
        │ agent/planner_agent · planner/* · memory/* · core/actions │
        └──────────────────────────────────────────────────────────┘
                      │                         ▲
                      ▼                         │
            ┌────────────────── 环境面（PlannerEnv） ──────────────────┐
            │ env/planner_env · actor/cgm_adapter · actor/collater    │
            └────────────────────────────────────────────────────────┘
                      │
                      ▼
            ┌─────────── 容器面（SandboxRuntime → RepoEnv） ───────────┐
            │ runtime/sandbox · runtime/__init__ · R2E-Gym/src/*      │
            └────────────────────────────────────────────────────────┘
```

### 1.1 模块职责一览

| 目录/文件 | 职责概述 |
| --- | --- |
| `scripts/run_rule_agent.py` | 规则驱动的端到端入口：拼装 Issue & Sandbox 配置，创建 `PlannerEnv` 与 `PlannerAgent`，并驱动一次完整的修复回合。 【F:scripts/run_rule_agent.py†L1-L128】 |
| `aci/tools.py` | 将 CLI/测试传来的 JSON 请求转成对容器的原子操作（查看、搜索、编辑、lint、测试），内部统一调用 `SandboxRuntime`。 【F:aci/tools.py†L1-L268】 |
| `agent/planner_agent.py` | 规则 Agent 的策略外壳：调度记忆更新、图扩展、CGM 协作，并在必要时生成自然语言计划。 【F:agent/planner_agent.py†L1-L205】 |
| `planner/*` | 规则库：包括节点选择、路径扩展、读写策略等，供 `PlannerAgent` 组合使用。 【F:planner/__init__.py†L1-L20】 |
| `memory/*` | 维护“记忆图”与线性化上下文，包括候选生成、操作策略、持久化、线性化等。 【F:memory/memory_bank.py†L1-L120】 |
| `core/actions.py` | 定义 Explore/Memory/Repair/Submit 等动作的数据结构，是 Agent 与环境之间的统一协议。 【F:core/actions.py†L1-L34】 |
| `env/planner_env.py` | 封装一次环境交互：处理动作、拉取代码片段、调用 CGM、执行 lint/test 并返回奖励。 【F:env/planner_env.py†L1-L436】 |
| `actor/cgm_adapter.py` | 与补丁模型（CGM）通信：组装 prompt、调用模型、解析补丁、生成 Guard 元数据。 【F:actor/cgm_adapter.py†L1-L161】 |
| `runtime/__init__.py` | 导入时自动把 vendored 的 `R2E-Gym/src` 加入 `sys.path`，确保 RepoEnv 代码可直接引用。 【F:runtime/__init__.py†L1-L30】 |
| `runtime/sandbox.py` | 按配置选择 RepoEnv / R2E / docker 后端，并提供统一的 `run/apply_patch/lint/test` 接口。 【F:runtime/sandbox.py†L1-L210】 |
| `R2E-Gym/src/*` | 官方 RepoEnv & DockerRuntime 实现，负责真正的容器生命周期与命令执行。 |

以下章节会在此表的基础上，逐层深入关键函数。

## 2. 环境入口：`RepoEnv`

`RepoEnv` 是 R2E-Gym 提供的 Gym 兼容环境，实现从 `ds` 描述到容器的生命周期管理：

- **初始化**：构造时创建 `DockerRuntime`，并将 `EnvArgs.ds` 转成 runtime 的配置，默认用 `/bin/bash -l` 作为交互 shell。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L28-L60】
- **重置与关闭**：`reset()` 释放旧 runtime 后重建一个全新的容器实例；`close()` 直接关闭当前 runtime，常用于评测结束时的清理。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L63-L78】【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L235-L239】
- **工具注入**：`add_commands()` 会把代理侧定义的工具脚本复制到容器 `/usr/local/bin` 并赋予执行权限，随后缓存命令清单以便校验。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L79-L130】
- **动作执行**：`step()` 校验 `Action` 并转换为 bash 命令，通过 `DockerRuntime.run()` 执行，再把 stdout/stderr/returncode 封装成 `Observation`。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L145-L197】
- **奖励计算**：调用 `compute_reward()` 会直接使用 runtime 的 `_calculate_reward()`，以保持任务评分逻辑的一致性。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L221-L233】

## 3. 动作与观测：`Action` / `Observation`

- `Action` 将函数调用或 XML 格式的命令解析为 shell 字符串，同时在 `to_bashcmd()` 中负责参数转义、环境变量注入等细节。【F:R2E-Gym/src/r2egym/agenthub/action/action.py†L6-L108】
- `Observation` 用统一格式承载 stdout/stderr/终止信号，确保代理能稳定地把容器反馈继续写入对话历史。【F:R2E-Gym/src/r2egym/agenthub/observation/observation.py†L7-L44】

## 4. 命令层：`ParseCommandBash` 与工具集合

- `ParseCommandBash` 负责解析 `@yaml` 注释或 bash 脚本，生成命令元数据，供环境注入容器。【F:R2E-Gym/src/r2egym/agenthub/agent/commands.py†L80-L199】
- `agenthub/tools/__init__.py` 定义了函数工具（`file_editor`、`search`、`execute_bash` 等）的输入输出契约，与 `Action.to_bashcmd()` 的命令名完全一致，保证 LLM 函数调用可以映射到真实命令。【F:R2E-Gym/src/r2egym/agenthub/tools/__init__.py†L5-L200】

## 5. 运行时核心：`DockerRuntime`

`DockerRuntime` 封装了容器生命周期与文件系统操作，支持 Docker 和 Kubernetes：

- **启动流程**：解析 `ds` 字段决定镜像/提交信息，随后创建 Docker/K8s 客户端并调用 `start_container()` 完成工作目录、依赖安装等准备。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L83-L183】
- **生命周期管理**：`start_container()` 复用或新建容器；`stop_container()`、`reset()`、`close()` 用于销毁或重置运行时，确保资源释放。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L350-L385】【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L447-L485】【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L1076-L1085】
- **命令执行**：`run()` 在容器内执行命令并注入 PATH、timeout、工作目录，K8s 模式下由 `_run_kubernetes()` 适配。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L600-L784】
- **文件操作**：提供 `copy_to_container()`、`create_file()`、`apply_patch()` 等便捷函数，用于同步代码与应用补丁。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L788-L915】
- **奖励与测试**：根据任务类型调用 `_calculate_reward_*` 跑测试并解析日志，最终统一通过 `_calculate_reward()` 输出分数。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L974-L1074】

## 6. Graph Planner Agent 执行链路

Graph Planner Agent 在本仓库中通过规则驱动的多阶段流程完成一次修复。其关键步骤如下。

### 6.1 启动与环境构造

- **端到端入口**：`scripts/run_rule_agent.py` 从命令行读取 issue/sandbox 配置，构造 `PlannerEnv` 与 `PlannerAgent`，并驱动单回合交互；批量评测可在此基础上做并行扩展。【F:scripts/run_rule_agent.py†L1-L128】
- **环境装配**：`PlannerEnv.from_dict` 将外部配置转换为 `SandboxConfig`，实例化 `SandboxRuntime`，并加载任务的初始子图/记忆。【F:env/planner_env.py†L12-L47】
- **Sandbox 后端选择**：`SandboxRuntime.__init__` 根据 `backend` 字段选择 RepoEnv/R2E/docker 实现，RepoEnv 模式下 `_init_repoenv_backend` 会加载 `ds` JSON 并实例化 R2E 的 `RepoEnv`。【F:runtime/sandbox.py†L27-L118】

### 6.2 决策循环

- **观测构造**：`PlannerEnv.reset` 初始化 issue 摘要、子图、记忆状态，并通过 `_obs` 输出统一的观测字典（含文本记忆、子图节点、候选上下文）。【F:env/planner_env.py†L48-L136】
- **规则策略**：`PlannerAgent.step` 调用 `_decide` 依次执行“扩展图 → 更新记忆 → 读取代码 → 生成自然语言修复计划 → 调用 CGM → 决定是否提交”，并保持动作契约与 `core.actions` 对齐。【F:agent/planner_agent.py†L35-L205】
- **环境步进**：`PlannerEnv.step` 根据动作类型路由到 `_handle_explore`、`_handle_memory`、`_handle_read`、`_handle_repair`、`_handle_submit` 等函数，累计奖励与终止条件。【F:env/planner_env.py†L138-L370】

### 6.3 容器与工具交互

- **图扩展/记忆维护**：`PlannerEnv._handle_explore` 调用 `memory.graph_adapter.one_hop_expand` 并借助 `memory.mem_candidates` 过滤节点，随后 `memory.mem_ops_head.suggest` 决定记忆增删；`memory.memory_bank.apply_ops` 负责真正更新子图并持久化。 【F:env/planner_env.py†L185-L282】【F:memory/memory_bank.py†L33-L120】
- **上下文拼接**：`actor.collater.build_snippets` 与 `actor.cgm_adapter.linearize_snippets` 把选中节点的代码段整理成 CGM 需要的上下文格式，保持容器路径绝对化，避免脱离 RepoEnv。【F:actor/collater.py†L18-L138】【F:actor/cgm_adapter.py†L82-L150】
- **补丁生成与守卫**：`actor.cgm_adapter.generate` 负责调用 CGM 并返回补丁与 Guard 元数据；`PlannerEnv._apply_patch_guarded` 在容器内调用 `SandboxRuntime.apply_patch`，失败时记录守卫反馈并让 Agent 选择新计划。【F:env/planner_env.py†L284-L356】
- **提交与评测**：`PlannerEnv._handle_submit` 调用 `SandboxRuntime.test()` 运行任务测试，并携带最近一次补丁供上层判定最终奖励。【F:env/planner_env.py†L358-L370】

### 6.4 CLI/工具层的补充路径

- `aci.tools.*` 是 CLI 与测试共享的工具层，所有 `edit_lines`、`lint_check`、`run_tests` 等操作最终都调度 `SandboxRuntime`，确保脚本与 agent 看到一致的副作用。【F:aci/tools.py†L1-L268】

## 7. 记忆与图模块盘点

| 文件 | 主要职责 | 当前用途 |
| --- | --- | --- |
| `memory/graph_adapter.py` | 管理代码图句柄，提供连接、锚点检索、1-hop 扩展能力，并在缺省时构建轻量本地图。 | Graph Planner 在 `_handle_explore` 中依赖它解析锚点并扩展子图，是扩展阶段的核心工具。【F:env/planner_env.py†L185-L236】 |
| `memory/mem_candidates.py` | 根据锚点与当前子图生成候选节点并打分，包含目录多样性权重与测试文件偏好。 | `_handle_explore` 通过它筛选优先级高的节点，决定下一步扩展方向。【F:env/planner_env.py†L201-L236】 |
| `memory/mem_ops_head.py` | 将候选与子图状态转换为记忆操作（Add/Update/Delete/Keep）。 | `_handle_memory` 在无显式动作时调用它生成默认操作，保证子图新鲜度。【F:env/planner_env.py†L238-L282】 |
| `memory/memory_bank.py` | 持久化记忆操作与子图状态，限制容量并记录 `.aci/memlog.json`。 | 环境在应用记忆后调用 `record_memops` 与 `apply_ops`，维持多轮交互的一致性。【F:env/planner_env.py†L246-L282】 |
| `memory/subgraph_store.py` | 定义 `WorkingSubgraph`，支持加载/保存/线性化，并把数据写入 `.aci/subgraphs/`。 | `PlannerEnv.reset` & `PlannerAgent` 读取/写回子图，并为 CGM 线性化上下文。【F:env/planner_env.py†L48-L136】【F:agent/planner_agent.py†L82-L153】 |
| `memory/types.py` | 统一 TypedDict/Protocol 定义，规范候选、锚点、记忆操作等结构。 | 被 Actor、Planner、Env 多处引用，为类型检查提供约束。【F:actor/collater.py†L27-L65】 |
| `memory/anchor_planner.py` | 早期的锚点生成器，占位实现。 | 当前规则 Agent 使用 `planner.anchor_planner`，因此此文件暂未接线，可视为备用。 |

除 `memory/anchor_planner.py` 外，其余文件均在主流程中被直接引用，需要保留。

## 8. 测试/CLI 调用链

当前仓库主要通过规则 Agent 冒烟程序验证端到端流程。执行 `python scripts/run_rule_agent.py --config …` 时，调用链如下：

```text
scripts/run_rule_agent.py
└── load_config() / build_issue() / build_sandbox()
    └── PlannerEnv.from_dict() → SandboxRuntime(...)
        └── PlannerAgent(step) ←→ PlannerEnv.step()
            ├── actor.collater / actor.cgm_adapter
            ├── memory.* / planner.*
            └── SandboxRuntime.apply_patch/lint/test → RepoEnv/DockerRuntime
```

与 CLI 平行的 ACI 工具测试（如自定义 pytest）会沿用以下链路：

```text
pytest / python -m <test>
└── JSON 请求 → aci.tools.*
    └── SandboxRuntime.*
        └── RepoEnv / DockerRuntime
```

两条路径的差异在于：前者跑完整的规则 Agent 工作流，后者用于验证底层工具是否能在容器中执行。

## 9. 测试思路与覆盖范围

当前推荐的验证方式是运行 `scripts/run_rule_agent.py`：

1. **准备配置**：提供 issue 描述、Graph 索引路径、Sandbox 选项（包括 `backend="repoenv"` 与 `r2e_ds_json`）。
2. **启动环境**：脚本调用 `PlannerEnv.from_dict` 创建 RepoEnv-backed 的 `SandboxRuntime`，同时加载历史记忆与子图。
3. **执行规则流程**：`PlannerAgent` 依次完成 1-hop 扩展、记忆维护、片段读取、自然语言计划生成、CGM 调用及补丁应用；`PlannerEnv` 在每一步记录奖励与信息。
4. **运行测试并获取奖励**：修复完成后触发 `SandboxRuntime.test()`，解析返回的测试结果和奖励，再把轨迹写入日志。

该流程覆盖了 Agent 决策、记忆维护、CGM 补丁生成、容器守卫、Lint/Test 直至奖励反馈的所有关键环节，是目前验证“规则驱动 Agent 能否在 RepoEnv 容器中完成一次修复”的标准方法。

如需更细粒度调试，可单独调用 `aci.tools` 里的原子操作，定位补丁生成、守卫或容器命令的具体问题。若需要对接 R2E 官方的 `cesium` 之类任务，需要自备对应的 `ds` 描述并通过 `scripts/run_rule_agent.py` 启动；仓库默认不包含 cesium 的运行记录，执行结果会写入 `logs/events.jsonl` 供后续分析。
