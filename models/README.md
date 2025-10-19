# 本地模型占位符 / Local model placeholders

- `planner_model/`：用于存放 Planner 代理的 Hugging Face checkpoint（如模型权重、tokenizer、config）。
- `cgm_model/`：用于存放 CGM 代理或 Planner 调用的本地补丁模型 checkpoint。

将实际模型文件放入对应目录后，可直接使用仓库中脚本的默认路径。
