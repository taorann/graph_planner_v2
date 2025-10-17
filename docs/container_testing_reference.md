# 容器与测试流程参考

## 运行入口与代理选择
- `scripts/run_rule_agent.py` 提供统一的命令行入口，可选择规则策略或本地 LLM 规划器，内部会创建 `PlannerEnv`、驱动若干步并输出测试结果与补丁 diff。【F:scripts/run_rule_agent.py†L1-L136】
- `run_episode` 会根据 `--agent` 参数实例化 `PlannerAgent`（规则策略）或 `LocalLLMPlannerAgent`（本地模型），确保两种决策器共享相同的环境管线与奖励定义。【F:scripts/run_rule_agent.py†L54-L96】

## 环境封装 `PlannerEnv`
- `PlannerEnv` 负责把 Explore/Memory/Repair/Submit 等动作映射为容器操作：在 `step` 中根据动作类型调用 `_handle_explore`、`_handle_memory`、`_handle_repair` 与 `_handle_submit`，并返回奖励、终止信号及最新观测。【F:env/planner_env.py†L23-L116】【F:env/planner_env.py†L137-L214】
- `reset` 会连接代码图、加载子图、探测容器工作目录并返回统一的观测字典；`close` 在回合结束时持久化子图并释放底层 sandbox。【F:env/planner_env.py†L47-L105】

## SandboxRuntime 与容器后端
- `SandboxRuntime` 支持 `repoenv`、`r2e` 与原生 `docker` 三种后端，并在 `apply_patch`、`lint`、`test` 等接口下屏蔽差异；`backend=repoenv` 时会读取 R2E `ds` 描述并通过 RepoEnv 的 `DockerRuntime` 管理容器生命周期。【F:runtime/sandbox.py†L31-L143】【F:runtime/sandbox.py†L69-L143】
- 在纯 Docker 模式下，运行时会使用 `docker-py` 启动交互式容器、处理宿主挂载并注入 git safe.directory，保证规则代理与本地调试共享同一套执行接口。【F:runtime/sandbox.py†L145-L189】

## Rule pipeline 测试桩
- `tests/test_rule_agent_pipeline.py` 通过 `FakeSandbox` 模拟容器命令、片段读取、补丁写入与测试执行，覆盖规则代理在无 Docker 环境下的端到端决策流程，并记录读取/编辑/命令轨迹。【F:tests/test_rule_agent_pipeline.py†L13-L145】
- 测试开始时会清空 `logs/test_runs.jsonl` 并将生成的遥测 payload 写回磁盘，方便人工检查补丁前后内容与命令序列。【F:tests/test_rule_agent_pipeline.py†L159-L199】

## 日志与遥测
- 遥测配置默认把事件与测试运行写入 `logs/events.jsonl`、`logs/test_runs.jsonl`，可通过环境变量覆盖路径；`telemetry.log_test_result` 会在每次测试后追加 JSONL 记录并附带时间戳。【F:infra/config.py†L31-L96】【F:infra/telemetry.py†L31-L39】
- 运行规则代理或本地 LLM 管线时，`SandboxRuntime.test` / FakeSandbox `test()` 会调用 `log_test_result` 写入测试结果与 `repair_trace`，测试结束后可直接打开 `logs/test_runs.jsonl` 查看补丁细节。【F:runtime/sandbox.py†L191-L216】【F:tests/test_rule_agent_pipeline.py†L120-L145】
