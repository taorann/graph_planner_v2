# R2E-Gym 集成与图规划代理总览

本文档从“决策代理 → 环境封装 → RepoEnv 运行时”的角度梳理 Graph Planner 项目，帮助在本地或训练环境中快速定位各层代码。

## 1. 系统分层
```
CLI / Tests
└── scripts/run_rule_agent.py …… 统一入口（rule/llm 两种代理）
    └── agents/(rule_based|model_based) …… 决策与补丁生成
        └── env/planner_env.py …… 动作编排、奖励、观测
            └── runtime/sandbox.py …… RepoEnv / R2E / docker 后端
                └── R2E-Gym/src/* …… 官方容器栈
```
- 启动脚本 `run_rule_agent.py` 负责解析配置、创建 `PlannerEnv` 并循环调用代理生成动作，是规则/本地 LLM 共用的端到端入口。【F:scripts/run_rule_agent.py†L1-L136】
- `PlannerEnv` 将 Explore/Memory/Repair/Submit 动作转换为容器调用，维护奖励、步数与最新观测，供两类代理复用。【F:env/planner_env.py†L23-L214】
- `SandboxRuntime` 在 RepoEnv、原生 R2E runtime 与 docker-py 之间切换，统一暴露 `run/apply_patch/test` 等接口以屏蔽后端差异。【F:runtime/sandbox.py†L31-L198】

## 2. 代理实现与补丁管线
- **规则代理**：`PlannerAgent` 以状态机方式驱动图扩展、记忆维护、片段阅读、计划与补丁生成，并将补丁交给环境执行。【F:agents/rule_based/planner.py†L26-L187】
- **本地 LLM 代理**：`LocalLLMPlannerAgent` 通过 `LocalLLMClient` 调用本地部署的 OpenAI 兼容接口，将模型响应解析为结构化动作，解析失败时自动回退到规则策略。【F:agents/model_based/planner.py†L38-L178】【F:integrations/local_llm/client.py†L15-L152】
- **对话解析工具**：`agents/common/chat.py` 统一了系统提示、观测摘要、JSON 解析与动作序列化逻辑，供本地 LLM 代理与 rLLM 适配层共享，确保模型交互协议一致。【F:agents/common/chat.py†L1-L197】
- **CGM 集成**：`agents/rule_based/cgm_adapter.py` 读取线性化子图与片段，优先调用 CodeFuse CGM 生成补丁，失败时落回本地标记方案，同时复用配置中的 endpoint、鉴权等参数。【F:agents/rule_based/cgm_adapter.py†L15-L117】【F:agents/rule_based/cgm_adapter.py†L156-L210】
- **配置开关**：`infra.config.Config` 暴露 `cgm` 与 `planner_model` 两个子配置，可通过环境变量或 `.aci/config.json` 切换本地/远端模型，便于在同一代码路径下替换推理后端。【F:infra/config.py†L49-L176】

## 3. 环境、记忆与观测
- `PlannerEnv.reset` 会连接代码图、加载/初始化工作子图，并在读取容器 `pwd` 后确定仓库根路径；`step` 根据动作类型调用子模块并生成统一的观测结构。【F:env/planner_env.py†L47-L214】
- 记忆组件负责维护代码子图：`mem_candidates.build_mem_candidates` 基于锚点做 1-hop 扩展并打分，`mem_ops_head`（未在此处列出）给出操作建议，`memory_bank.apply_ops` 则执行配额校验与持久化，保证子图更新受限且可追踪。【F:memory/mem_candidates.py†L1-L160】【F:memory/memory_bank.py†L1-L158】
- 线性化上下文来自 `memory.subgraph_store`，其 `wrap/linearize` 函数被代理用于拼装 CGM 输入，同时 `PlannerEnv` 在关闭时会把最新子图写回磁盘以支撑多轮训练。【F:agents/rule_based/planner.py†L163-L186】【F:env/planner_env.py†L47-L105】

## 4. RepoEnv / R2E 运行时对接
- `SandboxRuntime` 在 `backend="repoenv"` 时读取 `r2e_ds_json`，实例化 R2E-Gym 的 `RepoEnv` 与 `DockerRuntime`，自动补齐 `pip install pytest`、git safe.directory 等初始化步骤，确保容器即开即用。【F:runtime/sandbox.py†L69-L107】
- `backend="r2e"` 保留对原生 `R2EDockerRuntime` 的访问，用于训练阶段直接挂载宿主代码；`backend="docker"` 则完全依赖 docker-py，适合最小本地复现或 CI。【F:runtime/sandbox.py†L105-L167】
- 测试执行在 `_finalize_test_result` 里统一落盘，返回模式、stdout、RC，并通过 `infra.telemetry.log_test_result` 写入 JSONL，便于调试容器内的补丁结果。【F:runtime/sandbox.py†L218-L260】【F:infra/telemetry.py†L31-L39】

## 5. 训练与外部集成
- rLLM 对接层为 `integrations/rllm/agent.py` 与 `integrations/rllm/env.py`：分别把模型输出解析为 `ActionUnion` 并把 `PlannerEnv` 暴露为 rLLM `BaseEnv`，缺省情况下还会自动回退到规则策略，确保训练期间模型失效时依然可用。【F:integrations/rllm/agent.py†L1-L159】【F:integrations/rllm/env.py†L32-L110】
- 本地部署模型可通过配置中的 `planner_model.enabled/endpoint/model` 切换，无缝复用规则代理的观测结构；CGM 与本地模型可以同时开启，实现“模型决策 + CGM 打补丁”的链路，满足强化学习训练需求。【F:infra/config.py†L49-L96】【F:agents/model_based/planner.py†L48-L156】
