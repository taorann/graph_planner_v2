# RepoEnv 样例冒烟测试指南

本文档整理了在具备 Docker 权限的主机上，如何使用仓库随附的 RepoEnv 数据集执行一次端到端的规则代理冒烟测试，并解释运行时产出的关键日志文件。若在无 Docker 环境下执行，将会复现当前 CI 中的失败情形。

## 环境准备
- 安装 Python 3.10 及以上版本，并在仓库根目录创建虚拟环境（建议使用 `python -m venv .venv && source .venv/bin/activate`）。
- 安装项目依赖及 R2E-Gym：
  ```bash
  pip install -e .
  pip install -e R2E-Gym
  ```
  R2E-Gym 会携带 `pydantic` 等 PlannerEnv 所需依赖，避免运行期缺包报错。
- 确保主机可以访问 Docker 守护进程，例如通过 `sudo systemctl start docker` 启动本地 daemon，或配置远程 socket。
- 如需接入本地部署的 Planner LLM 或 CGM，可在 `.aci/config.json` 或环境变量中设置 `planner_model`、`cgm` 段的 endpoint、model、API key 等参数，对应字段在 `infra/config.py` 中有默认值说明。

## 数据集与容器镜像
- 数据条目：`datasets/graphplanner_repoenv_sample.jsonl`
- 数据集配置：`config/r2e_ds_repoenv_sample.json`
- 默认镜像：`graph_planner/repoenv-sample:latest`（RepoEnv 会在容器内解析为 `graph-planner/repoenv-sample:latest`）。

在运行前，建议提前拉取镜像以避免首次执行时等待下载：
```bash
docker pull graph-planner/repoenv-sample:latest
```

## 本地快速自检（可选）
若当前环境尚未配置 Docker，可先运行 FakeSandbox 的规则代理单测，验证修复轨迹记录逻辑：
```bash
PYTHONPATH=. pytest tests/test_rule_agent_pipeline.py -q
```
测试会模拟 `read_snippet`、`apply_patch`、`run_pytest` 等操作，并把修复日志写入 `logs/test_runs.jsonl`，可作为后续真实容器运行的参考格式。

## 执行 RepoEnv 冒烟测试
在 Docker daemon 可用的前提下，使用以下命令启动规则代理：
```bash
PYTHONPATH=. python scripts/run_rule_agent.py \
  --backend repoenv \
  --ds-json config/r2e_ds_repoenv_sample.json \
  --max-steps 6 \
  --report smoke_report.json
```

命令流程如下：
1. 解析数据集 JSON，创建 `PlannerEnv` 与 RepoEnv 后端的 `SandboxRuntime`。
2. 规则代理（或通过 `--agent llm` 切换到本地 LLM Planner）在最多 6 步内探索修复策略。
3. 轨迹摘要、奖励、补丁 diff 写入终端，并额外落盘到 `smoke_report.json`。

若 Docker daemon 不可达（常见报错为无法连接 `/var/run/docker.sock`），流程会在初始化阶段失败，可在具备 Docker 的主机上重试以完成冒烟验证。

## 日志与结果解读
- 遥测日志默认写入 `logs/events.jsonl` 与 `logs/test_runs.jsonl`；可通过环境变量 `EVENTS_PATH`、`TEST_RUNS_PATH` 或 `.aci/config.json` 进行重定向。
- 每次测试结束后会调用 `telemetry.log_test_result` 追加一条 JSON 记录，字段中包含：
  - `reads`：代理读取过的文件片段；
  - `edits`：补丁前后差异；
  - `commands`：执行过的 shell/测试命令；
  - `final_files`：关键文件的最终内容快照。
- 真实容器运行与 FakeSandbox 的日志结构一致，可直接在 `logs/test_runs.jsonl` 中对比分析修复步骤。
- `--report` 生成的 `smoke_report.json` 会保存完整的 step-by-step 轨迹、奖励曲线与补丁详情，便于回放或上传到评测系统。

完成以上流程后，即可在本地主机上复现一次 RepoEnv 规则代理的端到端修复测试，并通过日志全面了解补丁产生的全过程。
