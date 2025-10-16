# 将本地代码更新到 GitHub 的操作指南

本文档整理了在当前仓库中将最新的 Agent 与 RepoEnv 集成代码提交到 GitHub 所需的关键步骤，涵盖环境校验、测试执行以及提交 PR 的流程。

## 1. 同步依赖与环境准备
1. 进入仓库根目录：
   ```bash
   cd /path/to/graph_planner
   ```
2. （可选）激活项目使用的虚拟环境，确保已经安装本仓库所需的 Python 依赖、RepoEnv 工具链、以及可选的 rLLM/Verl 组件。
3. 如需使用 CodeFuse CGM 远端补丁服务，确认相关的访问令牌和服务地址已写入环境变量。

## 2. 运行关键测试以确保功能完好
在提交代码之前，请运行核心测试覆盖面：

```bash
pytest
```

该测试套件包含了：
- 规则驱动 Agent 的端到端闭环测试（使用 FakeSandbox 模拟容器）。
- CodeFuse CGM 集成测试（验证远端补丁调用与本地回退逻辑）。
- 其他快速回归测试，确保动作模型、记忆维护与 RepoEnv 适配的基础行为稳定。

如果你调整了 RepoEnv 示例镜像或新增数据集，也请执行：

```bash
bash scripts/build_repoenv_sample.sh
PYTHONPATH=. python scripts/run_rule_agent.py --backend repoenv --ds-json config/r2e_ds_repoenv_sample.json --max-steps 2
```

以确保容器环境仍能被成功拉起并完成最小修复闭环。

## 3. 生成数据集与训练资产（可选）
若更新了 rLLM 训练相关的配置，请重新注册数据集：

```bash
PYTHONPATH=. python scripts/register_graphplanner_dataset.py \
  --ds-json config/r2e_ds_repoenv_sample.json \
  --output datasets/graphplanner_repoenv_sample.jsonl
```

## 4. 提交代码
1. 查看变更：
   ```bash
   git status
   ```
2. 将需要提交的文件加入暂存区：
   ```bash
   git add <files>
   ```
3. 编写清晰的提交信息并提交：
   ```bash
   git commit -m "<描述你的改动>"
   ```

## 5. 推送并直接合并到 `main`
如果你的协作策略允许直接在 `main` 分支上提交（即不经过 Pull Request 流程），推荐按照下面的顺序操作：

1. 确认当前分支就是 `main`，或者执行 `git checkout main` 切换到主分支，并通过 `git pull` 获取最新的远程改动。
2. 将本地验证过的提交推送到远程：
   ```bash
   git push origin main
   ```
3. 观察远程仓库的 CI（如 GitHub Actions）是否顺利完成。如果远程未配置自动测试，可在推送后手动执行关键校验命令并记录结果。
4. 如需保留备份，可在推送前使用 `git tag` 标记一个里程碑版本，以便必要时快速回滚。

通过以上流程，便可以把我们整合了 RepoEnv、CodeFuse CGM 与 rLLM 接入点的最新代码直接合并到远程 `main` 分支，同时仍旧确保基本的质量把控与可追溯性。

## 6. 处理 GitHub 提示“冲突过于复杂”
当 GitHub 在网页端提示 *“These conflicts are too complex to resolve in the web editor”* 时，说明网页冲突解决器无法自动完成合并，需要在本地执行以下步骤：

1. **切换到主分支并拉取最新改动**
   ```bash
   git checkout main
   git pull origin main
   ```
2. **把工作分支与远程 `main` 对齐**（如你在其他分支开发）
   ```bash
   git checkout <your-feature-branch>
   git fetch origin
   git merge origin/main
   ```
   或改用 `git rebase origin/main` 获得线性历史。
3. **手动处理冲突标记**：冲突处会出现 `<<<<<<<` / `=======` / `>>>>>>>`。根据需求手动编辑文件，保留正确的代码后执行：
   ```bash
   git add <resolved-file>
   ```
4. **完成合并或变基并重新测试**：
   - 使用 `merge` 时，运行 `git commit` 生成合并提交。
   - 使用 `rebase` 时，执行 `git rebase --continue`。
   - 随后重新运行 `pytest` 等关键测试，确认冲突解决没有引入回归。
5. **推送更新的分支**：
   ```bash
   git push origin <your-feature-branch>
   ```
   若直接在 `main` 上操作，则 `git push origin main`。如经历过 rebase，需要附加 `--force-with-lease` 以安全地更新远程历史：
   ```bash
   git push --force-with-lease origin <your-feature-branch>
   ```

完成上述步骤后，即可绕过 GitHub 网页端的限制，在本地解决复杂冲突并将结果推送回远程仓库。
