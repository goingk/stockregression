# 配置迁移与生效说明

## 单一来源

运行配置现在只从 `config/application.json` 读取。旧 `.local/settings.json`、浏览器 localStorage 草稿和备份文件都不会覆盖主文件。环境变量 `STOCKREGRESSION_HOME` 只改变项目根目录的位置，不覆盖研究、回测、行情或界面字段。

## 自动迁移

以下情况会创建并物化完整主文件：

1. `config/application.json` 不存在；
2. 当前配置目标是项目主文件；
3. 旧 `.local/settings.json` 存在且可解析。

迁移会把旧研究所用的完整默认值和回测默认值归档到 `research.legacy`，然后让新研究使用稳健 J 默认值。兼容且合法的用户值继续继承。旧隐藏文件保留作为人工备份，不再读取。

如果主文件存在，启动不会自动导入旧文件。需要显式执行：

```powershell
.venv\Scripts\python.exe -m stockregression.cli config migrate-legacy
```

主文件已存在时该命令拒绝替换；确认替换需加 `--replace`。写入仍是临时文件加原子替换，并保存上一层备份。

## 浏览器草稿

浏览器只保存当前 revision 下尚未完成的叶子补丁。启动时若文件 revision 已变化，旧草稿不会被自动上传覆盖新文件。正常运行中：

- 未编辑字段接收外部文件变化；
- 不相交修改在下一次保存前合并；
- 同一字段同时出现本地和外部修改时显示冲突并保留本地值；
- 提交研究前先等待保存链完成，再携带同一 `configuration_revision` 创建任务。

## 任务与历史

任务在入队时冻结完整配置文档、有效设置、数据目录、实际数据版本和决策区间。后续配置修改不会改变排队或运行中的任务。结果保留 `configuration_snapshot` 和 `execution` 记录，用于解释当时的费用、滑点、资源与研究参数。

历史研究结果进入任务自己的快照视图，不会改写当前研究默认值。历史版本仍按原 `protocol_version`、`strategy_version` 和原始配置读取；新版默认切换到策略 v4 不会重写旧结果标识。

## 生效方式

- `application.host`、`application.port`、`paths.data_dir`、`paths.package_dir`：重启进程。
- `application.max_workers`：新调度器启动后。
- `backtest.*`、`research.*`、`statistics.*`：下一次任务提交或下一次统计。
- `market_data.*`、`runtime.*`：下一次服务、请求或任务构造；排队任务继续使用自己的快照。
- `ui.*`：界面下一次读取或下一次轮询。
- `schedule.*`：保存后需要按数据中心现有安装／移除流程更新系统计划任务。

API 的 `pending_restart` 明确返回已保存但未生效的路径、地址和端口。定时任务状态与实际注册状态分别展示。

## 回滚

`config/backups/application.previous.json` 是最近一次原子写入前的有效文件，可用于人工审阅和恢复，但不参与正常运行优先级。恢复时应停止服务、通过版本控制或显式替换主文件，再重新启动并运行：

```powershell
.venv\Scripts\python.exe -m stockregression.cli config validate
```

无效文件不会在启动时被默认值静默覆盖。
