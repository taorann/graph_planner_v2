# 将本地改动同步到远程的操作指南

该流程面向需要更新 Graph Planner 仓库的贡献者，覆盖依赖准备、快速回归测试以及提交规范。

## 1. 环境准备
1. 切换到仓库根目录：
   ```bash
   cd /path/to/graph_planner
   ```
2. 确保已经安装项目所需依赖：
   - Python 3.10+ 与 `pip`；
   - 可选的 Docker 守护进程（运行真实容器时使用）；
   - 若需调用本地 LLM 或 CGM，请在 `.aci/config.json` 或环境变量中配置 `planner_model`、`cgm` 段的 endpoint、model、API Key 等信息。【F:infra/config.py†L49-L176】

## 2. 快速自检
在提交之前推荐执行以下最小检查：

```bash
python -m compileall agents integrations env infra scripts
```
- 该命令覆盖代理、运行时、配置与脚本目录，可在缺少可选依赖（如 rLLM、pydantic）时继续工作。【F:scripts/run_rule_agent.py†L1-L136】【F:integrations/rllm/agent.py†L1-L159】

如需验证规则流程，可以运行 FakeSandbox 驱动的端到端测试：

```bash
PYTHONPATH=. pytest tests/test_rule_agent_pipeline.py -q
```
- 测试会模拟容器交互、生成补丁并将执行轨迹写入 `logs/test_runs.jsonl`，便于检查修改是否破坏现有决策流程。【F:tests/test_rule_agent_pipeline.py†L13-L199】
- 若环境缺少 `pydantic` 等可选依赖，可暂时跳过此步骤，并在提交说明中注明原因。

若需要完整容器验证，可在具备 RepoEnv 数据集的机器上运行：

```bash
python scripts/run_rule_agent.py --backend repoenv --ds-json <path/to/ds.json> --max-steps 8 --agent rule
```
- `--agent llm` 可以切换到本地模型决策；脚本会打印奖励、测试结果以及最终补丁 diff。【F:scripts/run_rule_agent.py†L54-L133】

## 3. 生成遥测报告（可选）
- 任何一次成功的 `SandboxRuntime.test` / FakeSandbox `test()` 调用都会通过 `telemetry.log_test_result` 记录 JSONL，默认写入 `logs/test_runs.jsonl`；必要时请将该文件附在变更说明中，帮助审阅者复现补丁细节。【F:runtime/sandbox.py†L191-L260】【F:infra/telemetry.py†L31-L39】

## 4. Git 提交流程
1. 查看工作区状态：
   ```bash
   git status
   ```
2. 将需要的文件加入暂存区：
   ```bash
   git add <files>
   ```
3. 使用清晰的提交信息：
   ```bash
   git commit -m "feat: <summary>"
   ```
4. 推送或创建 PR：
   - 若直接推送到远程：`git push origin <branch>`；
   - 若走 PR 流程，请将自检命令、可选测试输出及 `logs/test_runs.jsonl` 摘要写入描述，方便审阅。
