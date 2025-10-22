# Contract-as-Code 修复记录 / Contract-as-Code Notes

为了解决“工具与协议分散”这一痛点，我们把 Planner 与 CGM 的所有提示词、
动作枚举、错误码与校验器统一收敛到 `graph_planner/agents/common/contracts.py`
中，并提供系统化的错误代码。该模块成为唯一事实来源（SSOT），任何新的
动作或补丁约束都必须在此定义、并配套测试。

## 错误码一览

| 类别 | 错误码 | 触发条件 |
| --- | --- | --- |
| Planner | `missing-function-tag` | 未找到 `<function=...>` 或缺少闭合标签 |
| Planner | `invalid-multi-block` | 回复中出现多个 `<function>` 区块 |
| Planner | `duplicate-param` | 同一动作重复声明参数 |
| Planner | `unknown-param` | 未在合同允许列表中的参数 |
| Planner | `missing-required-param` | 如 `repair` 缺少 `subplan` |
| Planner | `invalid-json-param` | 参数内容解析 JSON 失败 |
| CGM | `invalid-patch-schema` | `patch.edits` 为空或字段缺失 |
| CGM | `multi-file-diff` | 同一次补丁跨越多个文件 |
| CGM | `newline-missing` | `new_text` 未以换行结尾 |
| CGM | `range-invalid` | `start/end` 范围非法 |
| CGM | `path-missing` | 缺失 `path` 或为空字符串 |
| CGM | `invalid-unified-diff` | 补丁头部或文件路径与目标不一致 |
| CGM | `hunk-mismatch` | hunk 中旧内容与原文件不匹配 |
| CGM | `encoding-unsupported` | 目标文件解码失败（非 UTF-8） |
| CGM | `dirty-workspace` | 原地落盘失败或检测到部分写入 |
| CGM | `duplicate-patch` | 同一补丁多次应用被拦截 |

## 如何扩展新的动作 / Adding New Actions

1. 在 `contracts.py` 中更新 `PLANNER_CONTRACT.allowed_params` 与
   `required_params`，确保系统提示与校验逻辑一致。
2. 为新动作添加 `validate_planner_action` 分支，并返回对应的
   Pydantic `Action`。
3. 在 `tests/test_contracts_planner.py` 增加成功/失败用例，避免
   协议漂移。
4. 若动作影响 `handle_planner_repair` 或环境逻辑，增补
   `tests/test_text_protocol_e2e.py` 中的端到端测试。

命令 `scripts/validate_contracts.py` 提供了一个轻量化的回归脚本，可在本地或
CI 中快速验证合同实现是否可用。

## Dirty-patch prevention / 原子化补丁保障

为了解决“脏补丁”问题，我们在运行时引入了两层防护：

1. **统一的差异校验**：`validate_cgm_patch` + `validate_unified_diff` 先检查补丁 JSON、统一差分头部、hunk 范围与原文件内容是否完全一致，支持 CRLF → LF 归一化，并在出现 `invalid-unified-diff`、`hunk-mismatch`、`range-invalid` 等错误时立即中止。
2. **原子化落盘**：`runtime.sandbox.PatchApplier.apply_in_temp_then_commit()` 会复制仓库到临时目录，先在副本中写入补丁、执行 lint/测试，再使用 `os.replace` 原子地替换目标文件。一旦测试失败或写入异常，会抛出 `build-failed`、`lint-failed`、`dirty-workspace` 等错误并保留 `temp_path`、`patch_id`、`n_hunks` 等遥测信息，确保主工作区不被污染；重复补丁会触发 `duplicate-patch` 以避免重复应用。

### Error codes added for patch hygiene

| Code | Meaning |
| --- | --- |
| `invalid-unified-diff` | Diff header 或目标路径与预期不一致 |
| `hunk-mismatch` | hunk 中的旧内容与当前文件不匹配 |
| `encoding-unsupported` | 目标文件无法按 UTF-8 解码 |
| `dirty-workspace` | 原子替换阶段出现部分写入/IO 异常 |
| `duplicate-patch` | 同一 `patch_id` 的补丁重复提交 |

### Telemetry contract

所有修复失败都会携带如下字段，方便下游统计：

- `fallback_reason`: `ProtocolError.code`，便于聚合异常来源；
- `patch_id`: 来自 `core.patches.patch_id()` 的去重哈希；
- `n_hunks` / `added_lines` / `removed_lines`: diff 规模统计；
- `temp_path`: 本次试运行的临时目录名称，可用于复现。

> 开发者可以通过 `python scripts/validate_patches.py --file app.py --json candidate.json` 对补丁做离线验证，快速捕获上述错误。
