# Changelog

## Unreleased
- Prune committed run artifacts (`logs/`, `outputs/`, `wandb/`) and `.aci` backups so the repo only tracks source files.
- Refresh README、datasets 指南与脚本文档，突出 `run_eval_graph_planner.sh` 评测入口并记录训练 CLI 正在重构。
- Add `.aci/README.md` 和 `.gitignore` 规则说明缓存目录用途，防止临时文件重新入库。
- Catalog legacy文档于 `docs/legacy_materials.md`，并在相关文件头部加注提示。
