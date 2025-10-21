# 本地模型占位符 / Local model placeholders

- `qwen3-14b-instruct/`：用于存放 Planner 代理所需的 Qwen3-14B-Instruct checkpoint（包含权重、tokenizer、config 文件）。
- `codefuse-cgm/`：用于存放 CodeFuse CGM 模型的本地权重与 tokenizer，供 Planner 调用补丁或训练 CGM 代理。

将实际模型文件放入对应目录后，可直接使用仓库中脚本的默认路径。
