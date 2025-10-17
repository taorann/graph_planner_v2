# RepoEnv 样例冒烟测试记录

本节记录如何使用仓库内置的 `repoenv` 数据集快速对规则代理进行冒烟测试，以及本次在无 Docker 守护进程环境下的执行结果，方便后续接入真实容器时复现。

## 使用的数据集
- 数据条目：`datasets/graphplanner_repoenv_sample.jsonl`
- R2E 数据集描述：`config/r2e_ds_repoenv_sample.json`
- 镜像名称：`graph_planner/repoenv-sample:latest`（数据集 JSON 会在容器端解析为 `graph-planner/repoenv-sample:latest`）

## 运行命令
```bash
PYTHONPATH=. python scripts/run_rule_agent.py \
  --backend repoenv \
  --ds-json config/r2e_ds_repoenv_sample.json \
  --max-steps 6 \
  --report smoke_report.json
```

命令会创建 `PlannerEnv`、实例化 `SandboxRuntime` 的 RepoEnv 后端，并执行最多 6 步的规则代理流程，最终在 `smoke_report.json` 中保存轨迹摘要。

## 本次结果
- 由于当前执行环境缺少可访问的 Docker 守护进程，`RepoEnv` 在创建 `DockerRuntime` 时通过 `docker.from_env()` 连接 `unix:///var/run/docker.sock` 失败，抛出 `FileNotFoundError` 并导致流程提前结束。【f47609†L1-L83】
- 在具备 Docker 的主机上，可通过 `sudo systemctl start docker` 或配置远程 Docker socket 后重新运行上述命令，即可拉起样例镜像完成冒烟验证。

生成的 `smoke_report.json` 在异常发生前尚未写入有效内容，可在具备 Docker 的环境中重跑后检视补丁轨迹与测试输出。
