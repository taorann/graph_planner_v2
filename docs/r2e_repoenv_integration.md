# R2E-Gym RepoEnv 集成概览

本文档总结 Graph Planner 项目如何把自身的规则 Agent 与 R2E-Gym 的 RepoEnv 容器栈对接，并补充关键代码层次、文件职责与测试思路。

## 1. R2E-Gym 栈的核心层

```text
数据集描述 (ds.json)
        │
        ▼
EnvArgs → RepoEnv → DockerRuntime
        │            ├─ run / copy / apply_patch
        │            └─ reward / test / reset
        ▼
Action / Observation (agenthub/action, observation)
```

- **数据集描述 (`ds`)**：定义镜像、仓库、提交哈希、入口脚本等元信息，是 RepoEnv 启动容器的唯一输入。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L18-L60】
- **RepoEnv**：负责实例化 `DockerRuntime`、注入命令、转换 `Action`，并将容器输出封装成 `Observation`。【F:R2E-Gym/src/r2egym/agenthub/environment/env.py†L79-L197】
- **DockerRuntime**：真正执行 Docker/Kubernetes 命令，提供 `run`、`apply_patch`、`copy_to_container`、`_calculate_reward` 等能力。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L83-L1085】

## 2. Graph Planner 对接 RepoEnv 的层级

```text
scripts/run_rule_agent.py ─────────────┐
                                       ▼
                              PlannerEnv.from_dict
                                       │
                                       ▼
                               SandboxRuntime (backend=repoenv)
                                       │
                                       ▼
                             RepoEnv / DockerRuntime (R2E-Gym)
```

1. **启动脚本**：`scripts/run_rule_agent.py` 解析命令行，加载 issue/sandbox 配置，并构造 `PlannerEnv` + `PlannerAgent`。【F:scripts/run_rule_agent.py†L1-L128】
2. **环境封装**：`PlannerEnv.from_dict` 根据配置实例化 `SandboxConfig`，随后创建 `SandboxRuntime`，并加载记忆/子图等状态。【F:env/planner_env.py†L12-L91】
3. **运行时适配**：`SandboxRuntime` 在 `backend="repoenv"` 时调用 `_init_repoenv_backend`：
   - 读取 `r2e_ds_json`，构造 `EnvArgs` 并实例化 `RepoEnv`；
   - 将返回的 `RepoEnv.runtime`（即 R2E 的 `DockerRuntime`）保存下来，统一转发 `run/apply_patch/lint/test` 调用；
   - 在容器内补齐工具脚本、git safe.directory 等兜底设置。【F:runtime/sandbox.py†L27-L210】
4. **决策循环**：`PlannerAgent.step` 负责生成 Explore/Memory/Repair/Submit 等动作；`PlannerEnv.step` 根据动作调用 `SandboxRuntime`，并将容器反馈折算成奖励和观测。【F:agent/planner_agent.py†L35-L205】【F:env/planner_env.py†L138-L370】

## 3. 关键文件职责速览

| 文件 | 作用 |
| --- | --- |
| `runtime/__init__.py` | 导入时将 `R2E-Gym/src` 自动加入 `sys.path`，无需手动设置 `PYTHONPATH`。【F:runtime/__init__.py†L1-L30】 |
| `runtime/sandbox.py` | 选择后端（repoenv/r2e/docker），持有 RepoEnv 或 docker-py 客户端，并提供统一的 `run/apply_patch/lint/test/get_patch` 接口。【F:runtime/sandbox.py†L1-L210】 |
| `env/planner_env.py` | 包装 Agent 与运行时的交互逻辑，包括记忆维护、图扩展、CGM 调用、守卫应用和奖励统计。【F:env/planner_env.py†L48-L370】 |
| `actor/cgm_adapter.py` | 以容器内路径为基础构造 CGM Prompt，解析补丁、守卫元数据，并与 PlannerEnv 协作落库。【F:actor/cgm_adapter.py†L82-L150】 |
| `aci/tools.py` | ACI 层工具集合，为 CLI/测试提供 `view_file`、`search`、`edit_lines`、`lint_check`、`run_tests` 等原子操作，内部同样复用 `SandboxRuntime`。【F:aci/tools.py†L1-L268】 |
| `scripts/run_rule_agent.py` | 规则 Agent 冒烟测试脚本，串联环境、Agent、RepoEnv 并输出自然语言计划与补丁结果，是当前端到端验证的推荐入口。【F:scripts/run_rule_agent.py†L1-L128】 |

## 4. 端到端工作流（规则 Agent）

```text
CLI/test harness
    │
    ▼
scripts/run_rule_agent.py
    │  解析配置 / 加载 ds.json / 构造 PlannerEnv
    ▼
PlannerAgent.step
    │  规划：图扩展 → 记忆维护 → 节点阅读 → 生成修复计划 → 调 CGM
    ▼
PlannerEnv.step
    │  通过 SandboxRuntime 调用 RepoEnv
    ▼
RepoEnv.runtime (DockerRuntime)
    │  apply_patch / lint / test / reward
    ▼
返回观察、奖励、日志 → 记录到记忆与轨迹
```

该流程确保：

- 所有文件读写都发生在容器内，路径由 PlannerEnv 保持绝对化；
- CGM 只接收来自 RepoEnv 的片段，生成的补丁再由守卫校验后写回容器；
- `SandboxRuntime` 统一管理 lint/test，保持与 ACI 工具的一致性。

## 5. 层次图：仓库主要模块

```text
Graph Planner Repo
│
├── scripts/
│   └── run_rule_agent.py …… 规则 Agent 启动入口
│
├── agent/
│   └── planner_agent.py …… 决策主循环
│
├── planner/ …… 节点选择、读写策略
│
├── memory/
│   ├── graph_adapter.py …… 图句柄 & 1-hop 扩展
│   ├── mem_candidates.py …… 扩展候选打分
│   ├── mem_ops_head.py …… 记忆操作推荐
│   ├── memory_bank.py …… 子图持久化 & 限额
│   └── subgraph_store.py …… 子图序列化
│
├── actor/
│   └── cgm_adapter.py …… CGM Prompt & 补丁解析
│
├── env/
│   └── planner_env.py …… 动作路由 & 奖励汇总
│
├── aci/
│   └── tools.py …… CLI/测试原子操作
│
├── runtime/
│   ├── __init__.py …… 注入 R2E-Gym 源码路径
│   └── sandbox.py …… 后端选择 + RepoEnv 代理
│
└── R2E-Gym/src/ …… 官方 RepoEnv / DockerRuntime 实现
```

## 6. 测试与验证思路

当前仓库主打的冒烟测试是运行 `python scripts/run_rule_agent.py --config <cfg>`：

1. **配置准备**：提供 issue 文本、图索引、`sandbox.backend="repoenv"`、`sandbox.r2e_ds_json` 等参数。
2. **环境初始化**：脚本调用 `PlannerEnv.from_dict`，该函数读取 `ds`，实例化 `SandboxRuntime` 并连接 RepoEnv。
3. **规则流程执行**：`PlannerAgent` 依次完成图扩展、记忆维护、片段读取、自然语言计划生成，并调用 CGM 产出补丁；失败会触发守卫回退。
4. **容器验证**：`SandboxRuntime.apply_patch` 写入补丁后调用 `lint`/`test`，最终 `SandboxRuntime.test()` 运行任务评测并返回奖励。
5. **结果汇总**：脚本打印自然语言计划、补丁摘要、lint/test 状态与最终奖励，可作为 RepoEnv 集成是否成功的判断依据。

若需更细的单元测试，可直接调用 `aci.tools` 的原子操作，验证补丁生成与容器命令是否按预期执行。评测 R2E 公布的 `cesium` 任务时，需要单独准备其 `ds` 描述并通过 `scripts/run_rule_agent.py` 启动；仓库自身不包含该任务的结果日志，运行产生的事件会写入 `logs/events.jsonl`。
