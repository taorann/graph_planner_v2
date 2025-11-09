# .aci 运行时缓存目录

本目录用于存放 Graph Planner 在运行过程中的本地缓存与配置：

- `backups/`：`aci._utils.backup_file` 为被编辑的源文件创建的时间戳备份。
- `subgraphs/`：`memory.subgraph_store` 在持久化工作子图与记忆状态时写出的 JSON 快照。

这些文件与单机调试会话密切相关，通常不需要随仓库共享。为了避免误提交，`backups/` 与 `subgraphs/` 已被添加到 `.gitignore` 中；当需要重置环境时可以直接清空。

若要覆写默认配置，可在此目录创建 `config.json`。除显式维护的配置外，其余内容应视为临时文件。
