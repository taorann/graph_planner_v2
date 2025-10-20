# CodeFuse-CGM 参考对照

本地适配层复用了 `CodeFuse-CGM` 仓库的数据格式与处理约定。下表说明了本包中每个辅助模块与上游脚本之间的对应关系，便于追溯来源。

| 组件 | 本仓库位置 | `CodeFuse-CGM` 对应文件 |
| --- | --- | --- |
| 图节点线性化（列表格式、字段回退、截断策略） | `formatting.py::GraphLinearizer` | `cgm/data/preprocess.py` —— `getJavaSentence`、`getPythonSentence` 与 `graph2embedding` 负责抽取节点名称、注释与代码文本，并在超长时按 `max_len` 截断；子图 JSON 则来自 `retriever/serialize_subgraph.py`，先按 `codegraph` 解析并调用 `serialize_subgraph` 输出节点/边列表。 |
| 候选代码片段序列化（路径/行号范围、片段正文） | `formatting.py::SnippetFormatter` | `cgm/train/train.py` —— `collate_cgm` 读取 `batch['repo']`、`batch['snippets']` 等字段；片段负载结构与检索工具输出的 JSON 保持一致。 |
| 对话提示构造与标签掩码 | `formatting.py::ConversationEncoder` | `cgm/data/encode.py` —— `BaseEncoder`/`CGMEncoder.dataToInput` 负责拼装带角色标签的段落、调用 tokenizer 聊天模板并对用户部分进行掩码。 |
| 样本/JSON 数据集加载（识别 prompt/answer/plan/graph/snippet） | `data.py::CodeFuseCGMDataset` & `CGMExample` | `cgm/train/train.py` —— 数据集 JSON 中包含 `prompt`、`answer`、`plan`、`repo`、`snippets` 以及缺陷元数据，训练循环会将其传入 `collate_cgm`。 |
| 批处理组装（padding、prompt 掩码） | `training.py::CGMBatchCollator` | `cgm/train/train.py` —— `collate_cgm` 会填充到固定长度、掩码查询 token，并返回训练所需张量。 |
| 训练循环（优化器/调度器、梯度累积、可选评估集） | `training.py::CodeFuseCGMTrainer` | `cgm/train/train.py` —— 负责加载 `AutoTokenizer`、`AutoModelForCausalLM`，设定 pad token，结合 `collate_cgm` 构建 DataLoader，并配置余弦学习率与梯度累积。 |
| 本地 Hugging Face 推理封装 | `inference.py::CodeFuseCGMGenerator` | `cgm/modeling/cgm.py` —— CGM 模块会加载 tokenizer/model 对，确保 PAD/EOS 兼容，将模型移动到指定设备，并基于提示编码生成回复。 |

目标是在保持与上游行为一致的同时，提供更清晰且类型友好的 API，以满足 Graph Planner 的集成需求。
