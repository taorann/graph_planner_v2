# 容器与测试流程参考

## 运行入口与代理选择
- `scripts/run_rule_agent.py` 提供统一的命令行入口，可选择规则策略或本地 LLM 规划器，内部会创建 `PlannerEnv`、驱动若干步并输出测试结果与补丁 diff。【F:scripts/run_rule_agent.py†L1-L136】
- `run_episode` 会根据 `--agent` 参数实例化 `PlannerAgent`（规则策略）或 `LocalLLMPlannerAgent`（本地模型），确保两种决策器共享相同的环境管线与奖励定义。【F:scripts/run_rule_agent.py†L54-L96】

## 环境封装 `PlannerEnv`
- `PlannerEnv` 负责把 Explore/Memory/Repair/Submit 等动作映射为容器操作：在 `step` 中根据动作类型调用 `_handle_explore`、`_handle_memory`、`_handle_repair` 与 `_handle_submit`，并返回奖励、终止信号及最新观测。【F:env/planner_env.py†L23-L116】【F:env/planner_env.py†L137-L214】
- `reset` 会连接代码图、加载子图、探测容器工作目录并返回统一的观测字典；`close` 在回合结束时持久化子图并释放底层 sandbox。【F:env/planner_env.py†L47-L105】

## SandboxRuntime 与容器后端
- `SandboxRuntime` 会根据 `SandboxConfig.backend` 在三种容器后端之间切换，并通过统一的 `run`、`apply_patch`、`test` 等接口向上屏蔽差异。【F:runtime/sandbox.py†L37-L68】
- **RepoEnv**：读取 `sandbox.r2e_ds_json` 指向的 R2E 数据集描述，实例化 `RepoEnv` 及其自带的 `DockerRuntime`，并在容器内自动安装 `pytest`、配置 `git safe.directory` 与基础工具，主要用于复现官方评测镜像。【F:runtime/sandbox.py†L69-L129】
- **R2E DockerRuntime**：直接使用 R2E 的 `DockerRuntime` 管理容器，但跳过 `RepoEnv` 的任务编排，适用于训练阶段需要灵活同步宿主挂载或自定义工作目录的场景。【F:runtime/sandbox.py†L131-L162】
- **原生 docker-py**：完全依赖本地 Docker daemon，通过 `docker.from_env()` 拉起交互式容器、挂载宿主目录并执行 `git apply`/`pytest` 等命令，适合无 R2E 数据集时的快速调试。【F:runtime/sandbox.py†L164-L216】
- 三种模式都会在测试后调用 `_finalize_test_result` 写入遥测日志，因此 `logs/test_runs.jsonl` 可统一回放补丁执行轨迹。【F:runtime/sandbox.py†L218-L264】

## Rule pipeline 测试桩
- `tests/test_rule_agent_pipeline.py` 通过 `FakeSandbox` 模拟容器命令、片段读取、补丁写入与测试执行，覆盖规则代理在无 Docker 环境下的端到端决策流程，并记录读取/编辑/命令轨迹。【F:tests/test_rule_agent_pipeline.py†L13-L145】
- 测试开始时会清空 `logs/test_runs.jsonl` 并将生成的遥测 payload 写回磁盘，方便人工检查补丁前后内容与命令序列。【F:tests/test_rule_agent_pipeline.py†L159-L199】

## 日志与遥测
- 遥测配置默认把事件与测试运行写入 `logs/events.jsonl`、`logs/test_runs.jsonl`，可通过环境变量覆盖路径；`telemetry.log_test_result` 会在每次测试后追加 JSONL 记录并附带时间戳。【F:infra/config.py†L31-L96】【F:infra/telemetry.py†L31-L39】
- 运行规则代理或本地 LLM 管线时，`SandboxRuntime.test` / FakeSandbox `test()` 会调用 `log_test_result` 写入测试结果与 `repair_trace`，测试结束后可直接打开 `logs/test_runs.jsonl` 查看补丁细节。【F:runtime/sandbox.py†L191-L216】【F:tests/test_rule_agent_pipeline.py†L120-L145】
