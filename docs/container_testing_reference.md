# 容器与测试流程参考

## 测试命令从何处启动
- `scripts/run_rule_agent.py` 是规则驱动 Agent 的统一入口：解析数据集 JSON、构造 `PlannerEnv` 并循环调用 `env.step` 执行动作，同时在结束时输出测试结果与补丁摘要。【F:scripts/run_rule_agent.py†L1-L125】
- `PlannerEnv.from_dict` 读取 issue/sandbox 配置并实例化 `SandboxRuntime`，后续 `reset` 会建立子图、记录容器工作目录并返回初始 observation。【F:env/planner_env.py†L42-L90】
- `PlannerEnv.step` 根据动作类型路由到 `_handle_explore`、`_handle_memory`、`_handle_repair` 或 `_handle_submit`；修复动作在调用补丁守卫后执行补丁、lint、pytest，再将测试结果封装在 `info` 中。【F:env/planner_env.py†L104-L279】
- `PlannerEnv.close` 会持久化子图并释放 `SandboxRuntime`，确保容器在测试结束后清理干净。【F:env/planner_env.py†L91-L100】

## Sandbox 与容器交互
- `SandboxConfig` 支持 `repoenv`、`r2e`、`docker` 三类后端，默认在提供数据集 JSON 时选择 RepoEnv；`SandboxRuntime.__init__` 依据后端调用 `_init_repoenv_backend` / `_init_r2e_backend` / `_init_docker_backend`。【F:runtime/sandbox.py†L28-L65】
- RepoEnv 路径会读取数据集 JSON，构造 `EnvArgs` → `RepoEnv`，并执行若干初始化命令（安装 pytest、配置 git safe.directory）以保证容器可用。【F:runtime/sandbox.py†L67-L103】
- 统一执行接口通过 `_exec` 将命令转发给底层 runtime；`lint` 走 Ruff/Black 兜底，`test` 先探测官方脚本（`/testbed/run_tests.sh` 等）再退化为 `python -m pytest -q`，并将模式、返回码、stdout 一并返回给上层。【F:runtime/sandbox.py†L170-L249】

## 本地样例容器
- `docker/repoenv_sample/Dockerfile` 基于 `python:3.10-slim` 初始化 git 仓库，将示例项目同步到 `/testbed` 并预装 pytest，便于 Rule Agent 在容器内复现修复流程。【F:docker/repoenv_sample/Dockerfile†L1-L33】
- 示例项目在 `app/calc.py` 中故意把加法实现成减法，对应的 pytest 测试 `tests/test_calc.py` 会失败，给 Agent 提供修复目标。【F:docker/repoenv_sample/sample_repo/app/calc.py†L1-L20】【F:docker/repoenv_sample/sample_repo/tests/test_calc.py†L1-L10】
- 运行 RepoEnv 时使用的最小数据集描述位于 `config/r2e_ds_repoenv_sample.json`，提供镜像名、仓库名与占位的 commit 元数据，满足 RepoEnv 对 `ds` 字段的要求。【F:config/r2e_ds_repoenv_sample.json†L1-L5】

## 官方容器数据集获取情况
- R2E-Gym 官方 README 在页首提供 Hugging Face 组织链接 `https://huggingface.co/R2E-Gym`，可以直接下载环境与模型资源。【F:R2E-Gym/README.md†L17-L25】
- 官方脚本通过 `datasets.load_dataset` 直接拉取 `R2E-Gym/R2E-Gym-Lite` 等数据集，说明 Hugging Face Hub 是推荐的容器镜像/数据集分发渠道。【F:R2E-Gym/README.md†L102-L127】
- 数据预处理代码还引用了 `r2e-edits/r2e-dockers-v1/v2`，并在处理完成后推送回 Hugging Face，进一步佐证官方容器数据集托管在 Hugging Face 上，需要登录后方可访问受限条目。【F:R2E-Gym/src/r2egym/repo_analysis/add_github_issue_to_commit.py†L143-L200】
- RepoEnv/DockerRuntime 的实现默认依赖 `ds["docker_image"]`、`ds["parsed_commit_content"]` 等字段，从 Hugging Face 拉取的数据集中读取镜像名称，再通过 Docker/Kubernetes 启动容器环境。【F:R2E-Gym/src/r2egym/agenthub/runtime/docker.py†L57-L118】

## 测试流程整体思路
1. 通过 `scripts/run_rule_agent.py` 选定数据集条目并实例化 `PlannerEnv`；`PlannerEnv.reset` 启动容器、建立记忆与子图上下文。【F:scripts/run_rule_agent.py†L27-L86】【F:env/planner_env.py†L67-L89】
2. Agent 在每个 step 中利用 `SandboxRuntime` 封装的工具读取代码片段、维护记忆、生成修复计划并调用 `_apply_patch_edits`；成功后执行 `lint` 与 `test` 得到奖励。【F:env/planner_env.py†L127-L279】【F:runtime/sandbox.py†L207-L249】
3. 当 `_handle_submit` 触发或 `run_episode` 达到步数上限时退出，`env.close` 负责保存状态并销毁容器；最终结果在 CLI 中展示测试输出与补丁 diff。【F:env/planner_env.py†L275-L279】【F:scripts/run_rule_agent.py†L89-L125】

## 规则 Agent 管线验证
- `tests/test_rule_agent_pipeline.py` 使用 `FakeSandbox` 取代真实容器，复刻 `pwd`、片段读取、补丁写入与测试执行，从而在无 Docker 环境下覆盖 `PlannerEnv` 的核心交互流程。【F:tests/test_rule_agent_pipeline.py†L1-L125】
- 测试依次触发 `expand → memory → read → repair → submit` 五种动作，断言候选节点、记忆应用、片段内容、补丁执行与测试结果均符合预期，以证明规则 Agent 在本地同样可以打通完整闭环。【F:tests/test_rule_agent_pipeline.py†L69-L119】

## 测试日志写入位置
- `SandboxRuntime.test` 在每次执行测试后会调用 `infra.telemetry.log_test_result`，把后端类型、命令、耗时与 stdout 片段写入 JSONL 记录。【F:runtime/sandbox.py†L231-L249】【F:infra/telemetry.py†L27-L36】
- 日志文件默认位于 `logs/test_runs.jsonl`，路径可以通过配置或设置环境变量 `TEST_RUNS_PATH` 覆盖；CLI 运行结束会打印实际写入位置，便于排查测试历史。【F:infra/config.py†L32-L38】【F:scripts/run_rule_agent.py†L109-L125】

