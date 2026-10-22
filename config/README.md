# 项目主配置与数据快照

> 研究只支持 `protocol_version=5`、`strategy_version=5`、`ranking_mode=robust_j` 和 `result_schema_version=2`。项目只接受当前分组配置，不再读取扁平文件、`research.legacy` 或旧版本迁移数据。
项目运行配置只有一个持久化来源：`config/application.json`。文件保存后各字段都是显式值，程序不再叠加旧隐藏设置、浏览器缓存或业务环境变量。`STOCKREGRESSION_HOME` 只用于定位项目和本文件，不能覆盖业务参数。`.local/settings.json`、`config/backups/application.previous.json` 和浏览器草稿都不参与运行时优先级。

## 版本、保存与任务边界

- `config_schema_version` 当前为 `5`。`config_revision` 是除自身以外全部规范化内容的 SHA-256。
- 写入顺序为：读取版本、合并叶子字段、完整校验、文件锁、重新检查版本、临时文件、`os.replace` 原子替换。上一层有效文件保存到 `config/backups/application.previous.json`。
- PATCH 必须携带 `base_revision`。版本冲突返回 HTTP 409，缺少版本的旧式 PUT 返回 HTTP 428。未知字段、类型错误和交叉约束错误返回具体字段路径。
- 任务入队时冻结完整配置文档、有效设置、数据路径和配置 revision。之后修改文件只影响新任务；排队和执行中的任务继续使用快照。
- 修改评分参数不会使已有结果的解释失效。修订单笔费用、滑点或初始资金会改变原始回测结果；修改界面轮询间隔不会改变方法标识。
- 配置 API 为 `GET /api/v1/configuration`、`GET /api/v1/configuration/schema`、`POST /api/v1/configuration/validate` 和 `PATCH /api/v1/configuration`。
- 路径和监听地址只在新进程启动后生效，API 的 `pending_restart` 返回待重启字段。研究与界面参数在后续读取、任务入队或界面同步时生效。

主文件缺失时会生成当前默认配置；主文件存在但无效时直接报错，不会用默认值覆盖。旧配置迁移命令和兼容字段已移除。

## 配置清单

类型后的“比例”字段在文件中保存小数，界面显示百分比。参数最小单位、阵列列号、协议标识和交易合法性属于规则，不作为任意可调参数。

### 应用、路径与回测

| 配置键 | 类型 / 单位 | 默认值 | 校验 | 消费模块 | 前端入口 | 生效方式 | 方法标识 |
|---|---|---|---|---|---|---|---|
| `application.host` | 字符串 | `127.0.0.1` | 非空 | CLI、API | 文件 / API | 重启 | 否 |
| `application.port` | 整数 | `8000` | 1–65535 | CLI、待重启检查 | 文件 / API | 重启 | 否 |
| `application.max_workers` | 整数 | `2` | 1–8 | JobManager | 文件 / API | 新调度器 | 否 |
| `paths.data_dir` | 路径 | `data` | 可规范化为项目或绝对路径 | Store、DataService | 文件 / API | 重启 | 否 |
| `paths.package_dir` | 路径或 null | `data/dataset/etf` | 合法目录 | 数据包仓储与同步 | 文件 / API | 重启 | 否 |
| `backtest.grid.grid_mode` | 枚举 | `trailing` | trailing/arithmetic/geometric | 网格引擎 | 回测与研究对象 | 下次提交 | 是 |
| `backtest.grid.buy_step` | 小数 / 比例 | `0.03` | >0，≤1 | 网格买入 | 回测与研究对象 | 下次提交 | 是 |
| `backtest.grid.sell_step` | 小数 / 比例 | `0.03` | >0，≤1 | 网格卖出 | 回测与研究对象 | 下次提交 | 是 |
| `backtest.grid.lower_price` | 价格或 null | `null` | >0，低于上限 | 买入区间 | 回测与研究对象 | 下次提交 | 是 |
| `backtest.grid.upper_price` | 价格或 null | `null` | >0，高于下限 | 卖出区间 | 回测与研究对象 | 下次提交 | 是 |
| `backtest.position.initial_shares` | 整数 / 股或份 | `1000` | 非负、100 整手 | 初始持仓与 P0 | 回测与研究对象 | 下次提交 | 是 |
| `backtest.position.min_shares` | 整数 / 股或份 | `0` | 非负、100 整手、≤初始持股 | 普通回测 | 持仓默认值 | 下次提交 | 是 |
| `backtest.position.max_shares` | 整数或 null | `2000` | 100 整手、≥初始持股 | 普通回测 | 持仓默认值 | 下次提交 | 是 |
| `backtest.position.buy_shares` | 整数 / 股或份 | `100` | ≥100、100 整手；科创板≥200 | 网格买入 | 回测与研究对象 | 下次提交 | 是 |
| `backtest.position.sell_shares` | 整数 / 股或份 | `100` | ≥100、100 整手 | 网格卖出 | 回测与研究对象 | 下次提交 | 是 |
| `backtest.dca.frequency` | 枚举 | `monthly` | daily/weekly/monthly | 定投账户 | 回测与研究对象 | 下次提交 | 是 |
| `backtest.dca.shares` | 整数 / 股或份 | `100` | ≥100、100 整手 | 定投账户 | 回测与研究对象 | 下次提交 | 是 |
| `backtest.dca.initial_shares` | 整数 / 股或份 | `0` | 非负、100 整手 | 定投初始账户 | 回测与研究对象 | 下次提交 | 是 |
| `backtest.fees.commission_rate` | 小数 / 比例 | `0.0005` | 0–0.05 | 成交费用 | 费用默认值 | 重新回测 | 是 |
| `backtest.fees.min_commission` | 金额 / 元 | `0.1` | 非负 | 成交费用 | 费用默认值 | 重新回测 | 是 |

普通回测允许持仓护栏。稳健 J 路径构造任务时强制 `min_shares=0`、`max_shares=null`，以实际观察持仓判断资格和惩罚；`q_buy=buy_shares/P0`、`q_sell=sell_shares/P0` 是结果派生值，不是第二套输入。

### 研究空间、目标、约束与搜索

| 配置键 | 类型 / 单位 | 默认值 | 校验 | 消费模块 | 前端入口 | 生效方式 | 方法标识 |
|---|---|---|---|---|---|---|---|
| `research.defaults.strategy_id` | 枚举 | `geometric_robust_j` | 与版本一致 | 研究调度 | 由策略选择派生 | 下次研究 | 是 |
| `research.defaults.strategy_version` | 枚举 | `5` | 当前只能为 5 | research | 研究策略 | 下次研究 | 是 |
| `research.defaults.protocol_version` | 枚举 | `5` | 当前只能为 5 | research | 研究协议 | 下次研究 | 是 |
| `research.defaults.selection_target` | 枚举 | `fixed` | fixed/adaptive/both | 外层流程 | 参数方案 | 下次研究 | 是 |
| `research.defaults.rebalance_days` | 整数 / 交易日 | `63` | ≥1 | 自适应换参 | 换参周期 | 下次研究 | 是 |
| `research.defaults.seed` | 整数 | `42` | 安全整数 | 稳定随机流 | 随机种子 | 下次研究 | 是 |
| `research.defaults.top_candidates` | 整数 | `20` | 1–100 | 结果展示切片 | 结果候选展示上限 | 下次结果读取 | 否 |
| `research.defaults.max_principal_drawdown` | 小数 / 比例 | `0.20` | >0，≤1 | 本金回撤资格 | 选择与样本约束 | 下次研究 | 是 |
| `research.defaults.min_trades` | 整数 / 笔 | `10` | 非负 | 中心资格 | 选择与样本约束 | 下次研究 | 是 |
| `research.defaults.min_history_days` | 整数 / 交易日 | `250` | ≥20 | 研究请求最低历史 | 选择与样本约束 | 下次研究 | 是 |
| `research.defaults.min_minute_ratio` | 小数 / 比例 | `0.95` | 0–1 | 数据资格 | 选择与样本约束 | 下次研究 | 是 |
| `research.defaults.ranking_mode` | 枚举 | `robust_j` | 当前只能为 robust_j | 稳健 J 排名 | 研究目标 | 下次研究 | 是 |
| `research.defaults.hypercube_edge` | 正奇数 / 研究刻度 | `3` | 1–99 且为奇数 | 邻域与 J | 超正方体边长 | 下次研究 | 是 |
| `research.defaults.granularity` | 四轴对象 | 间距 `0.0001`、股数 `100` | 最小单位正整数倍；四轴完整 | 参数轴与邻域 | 四轴研究细粒度 | 下次研究 | 是 |
| `research.defaults.search_method` | 枚举 | `auto` | auto/hierarchical/exhaustive；穷举须过预检 | CandidateSearcher | 搜索方法 | 下次研究 | 是 |
| `research.defaults.training_budget` | 唯一回测点 | `20000` | 1–1,000,000 | EvaluationSession | 训练预算 | 下次研究 | 是 |
| `research.defaults.validation_budget` | 唯一回测点 | `100000` | 1–2,000,000 | 邻域和峰值核验 | 邻域预算 | 下次研究 | 是 |
| `research.defaults.reporting_budget` | 唯一回测点 | `1000` | 0–2,000,000；展示阶段独立上限 | 展示补算 | 报告预算 | 下次研究 | 是 |
| `research.defaults.total_evaluation_budget` | 唯一回测点或 null | `null` | 1–10,000,000；提交时冻结有效上限 | 全阶段预算管理器 | 总评价预算 | 下次研究 | 是 |
| `research.defaults.max_runtime_seconds` | 秒 | `3600` | 1–86400 | 调度器与预算管理器 | 任务时限 | 下次研究 | 是 |
| `research.defaults.recommendation_count` | 整数 / 组 | `10` | 1–100 | J 排名推荐数量 | 推荐参数个数 | 下次研究 | 是 |
| `research.defaults.max_shares_mode` | 枚举 | `unlimited` | 当前只能为 unlimited | 自然持仓契约 | 固定说明 | 下次研究 | 是 |
| `research.defaults.slippage_mode` | 枚举 | `configured` | configured/zero | 费用与成交 | 滑点处理 | 下次研究 | 是 |
| `research.defaults.neighborhood_fields` | 轴名数组 | 四轴顺序 | v5 必须恰好四轴 | 契约校验 | 固定轴说明 | 下次研究 | 是 |
| `research.defaults.fixed_half_widths` | 四键对象 / 刻度 | 见文件 | 间距按 0.0001、股数按 100 对齐 | 稳定邻域 | 固定半宽 | 下次研究 | 是 |
| `research.defaults.ranges.*.low/high` | 各轴单位 | 见文件 | 有序、按参数最小单位对齐；v5 禁止 step | ParameterSpace | 可调搜索范围 | 下次研究 | 是 |
| `research.defaults.sigma` | 正小数 / 整数刻度 | `1.0` | >0，≤100 | 高斯权重 | J 参数 | 下次研究 | 是 |
| `research.defaults.alpha` | 小数 / 权重 | `0.30` | 0–1 | 中心 R 权重 | J 参数 | 下次研究 | 是 |
| `research.defaults.beta` | 非负小数 | `1.20` | 0–100 | 下行风险权重 | J 参数 | 下次研究 | 是 |
| `research.defaults.tau` | 小数 / 训练累计收益 | `0.01` | −1–10 | 安全垫与下行风险 | J 参数 | 下次研究 | 是 |
| `research.defaults.eta_limit` | P0 倍数 | `2.0` | 非负 | 上界持仓惩罚 | 持仓状态 | 下次研究 | 是 |
| `research.defaults.eta_hard_cap` | P0 倍数或 null | `3.5` | ≥eta_limit；null 关闭 | 中心事后资格 | 持仓状态 | 下次研究 | 是 |
| `research.defaults.gamma_max` | 非负系数 | `0.40` | 0–1000 | 上界二次惩罚 | 持仓状态 | 下次研究 | 是 |
| `research.defaults.eta_floor` | P0 倍数 | `0.0` | 0–1，≤eta_limit | 底仓期触发阈值 | 持仓状态 | 下次研究 | 是 |
| `research.defaults.floor_k_ref` | 卖出整格数 | `10` | >0 | 底仓整格惩罚参考尺度 | 持仓状态 | 下次研究 | 是 |
| `research.defaults.gamma_min` | 非负系数 | `0.0` | 0–1000 | 底仓期上涨整格二次惩罚 | 持仓状态 | 下次研究 | 是 |
| `research.defaults.training_min_days` | 整数 / 交易日 | `250` | 20–5000 | 数据准备与 TrainingWindowPolicy | 训练窗口 | 下次准备与研究 | 是 |
| `research.defaults.training_max_days` | 整数 / 交易日 | `500` | ≥training_min_days | 数据准备与窗口 | 训练窗口 | 下次准备与研究 | 是 |
| `research.defaults.training_window_mode` | 枚举 | `same_price` | same_price/fixed | TrainingWindowPolicy | 训练窗口 | 下次准备与研究 | 是 |
| `research.defaults.peak_tolerance` | 绝对 J 容差 | `1e-12` | >0，≤0.1 | PeakAnalyzer | J 参数 | 下次研究 | 是 |
| `research.defaults.start_min` | 起点数 | `20` | 1–1000，≤start_max | 多起点发现 | 搜索高级参数 | 下次研究 | 是 |
| `research.defaults.start_max` | 起点数 | `64` | 1–1000，≥start_min | 多起点发现 | 搜索高级参数 | 下次研究 | 是 |
| `research.defaults.starts_per_candidate` | 起点数 / 推荐 | `4` | 1–100 | 起点目标 max(start_min,4K) | 搜索高级参数 | 下次研究 | 是 |
| `research.defaults.initial_strides` | 整数刻度数组 | `[1,2]` | 至少一项；最多八项 | 局部搜索 | 搜索高级参数 | 下次研究 | 是 |
| `research.defaults.stride_multiplier` | 整数倍率 | `2` | 2–16 | 停滞后的步长扩展 | 搜索高级参数 | 下次研究 | 是 |
| `research.defaults.max_stride_fraction` | 小数 / 轴跨度 | `0.25` | >0，≤1 | 大步长上限 | 搜索高级参数 | 下次研究 | 是 |
| `research.defaults.validation_reserve_fraction` | 小数 / 预算 | `0.25` | 0–1 | 最终核验预留 | 搜索高级参数 | 下次研究 | 是 |
| `research.defaults.exhaustive_memory_limit_mb` | MiB | `512` | 16–65536 | auto/显式穷举预检 | 搜索高级参数 | 下次研究 | 是 |
| `research.defaults.estimated_bytes_per_point` | 字节 / 点 | `512` | 64–65536 | 工作内存预检 | 搜索高级参数 | 下次研究 | 是 |
| `research.defaults.stress_cost_multiplier` | 正倍率 | `2.0` | >1，≤20 | 费用压力诊断 | 训练与费用压力 | 下次研究 | 是 |
| `research.defaults.stress_region_count` | 峰区数 | `3` | 0–50 | 费用压力诊断 | 训练与费用压力 | 下次研究 | 是 |

新版四轴顺序固定为 `buy_step → sell_step → buy_shares → sell_shares`。中心收益 `R` 是训练区间峰值本金收益率，基础超额为网格收益减同注资持有收益；回撤为逐 bar 最大本金回撤。候选资格要求 `J > 0`。`research.defaults.tau` 只属于 J 下行风险参数。

### 统计、行情、运行时、界面与调度

| 配置键 | 类型 / 单位 | 默认值 | 校验 | 消费模块 | 前端入口 | 生效方式 | 方法标识 |
|---|---|---|---|---|---|---|---|
| `market_data.cache_mb` | MiB | `256` | 0–16384 | ByteLRU | 文件 | 保存后配置缓存 | 否 |
| `market_data.recent_minute_source` | 枚举 | `sina` | sina/tdx | 近期分钟主来源；同花顺为自动回退 | 文件 / 数据操作 | 新同步 | 否 |
| `market_data.etf_daily_source` | 枚举 | `baostock` | baostock/eastmoney | ETF 日线来源 | 文件 / 数据操作 | 新同步 | 否 |
| `market_data.network_timeout_seconds` | 秒 | `15` | 1–300 | HTTP 请求 | 文件 | 下次请求 | 否 |
| `market_data.request_retries` | 次数 | `1` | 0–5 | 来源重试 | 文件 | 下次请求 | 否 |
| `market_data.failure_cooldown_hours` | 小时 | `24` | 0–720 | 来源健康 | 文件 / 数据诊断 | 下次请求 | 否 |
| `market_data.retry_after_seconds` | 秒 | `60` | 1–86400 | 准备重试提示 | 文件 | 下次准备 | 否 |
| `market_data.source_lock_timeout_seconds` | 秒 | `30` | 1–600 | 来源与 Store 锁 | 文件 | 下次任务构造 | 否 |
| `market_data.sync_lock_timeout_seconds` | 秒 | `60` | 1–3600 | 单标的同步锁 | 文件 | 下次同步 | 否 |
| `market_data.prepare_lock_timeout_seconds` | 秒 | `300` | 1–86400 | 数据准备锁 | 文件 | 下次准备 | 否 |
| `market_data.package_lock_timeout_seconds` | 秒 | `600` | 1–86400 | 数据包安装锁 | 文件 | 下次安装 | 否 |
| `market_data.sqlite_timeout_seconds` | 秒 | `60` | 1–600 | SQLite | 文件 | 下次连接 | 否 |
| `market_data.daily_batch_size` | 标的数 | `128` | 1–2048 | 日线同步批次 | 文件 | 下次同步 | 否 |
| `market_data.service_batch_size` | 记录数 | `128` | 1–2048 | 行情服务批次 | 文件 | 下次请求 | 否 |
| `market_data.scheduler_poll_seconds` | 秒 | `0.5` | >0，≤60 | 调度与异步等待 | 文件 | 新调度器 | 否 |
| `runtime.cpu_budget_mode` | 枚举 | `auto` | custom 时必须填预算 | 资源租约 | 文件 / 性能 API | 新任务 | 否 |
| `runtime.cpu_budget` | 线程数或 null | `null` | 1–128 | CPU 租约 | 文件 / 性能 API | 新任务 | 否 |
| `runtime.download_workers` | 线程数 | `4` | 1–4 | 下载任务 | 文件 | 新任务 | 否 |
| `runtime.validation_workers` | 线程数 | `4` | 1–4 | 数据验证 | 文件 | 新任务 | 否 |
| `runtime.blas_threads` | 线程数 | `8` | 1–16 | NumPy/BLAS | 文件 | 新任务 | 否 |
| `runtime.workspace_mb` | MiB | `8192` | 256–65536 | 工作内存租约 | 文件 | 新任务 | 否 |
| `runtime.evaluation_chunk_size` | 参数点数 | `512` | 16–8192 | BatchEvaluator | 文件 | 下次研究 | 否 |
| `runtime.job_cancel_poll_ms` | 毫秒 | `300` | 100–10000 | 取消检查 | 文件 | 新任务 | 否 |
| `runtime.progress_update_ms` | 毫秒 | `500` | 100–60000 | 进度写入 | 文件 | 新任务 | 否 |
| `ui.autosave_debounce_ms` | 毫秒 | `500` | 100–10000 | 共享配置仓库 | 自动保存 | 下次编辑 | 否 |
| `ui.polling_interval_ms` | 毫秒 | `2000` | 500–60000 | 配置轮询 | 交互行为 | 下次轮询 | 否 |
| `ui.preview_debounce_ms` | 毫秒 | `300` | 50–10000 | 回测预览 | 交互行为 | 下次编辑 | 否 |
| `ui.page_size` | 行数 | `50` | 10–500 | 通用分页 | 交互行为 | 下次读取 | 否 |
| `ui.chart_points` | 点数 | `2000` | 100–5000，匹配行情接口 | K 线 | 交互行为 | 下次读取 | 否 |
| `ui.default_ma` | 正整数数组 | `[5,20,60]` | 每项 1–1000 | 行情均线 | 交互行为 | 下次读取 | 否 |
| `ui.task_polling_interval_ms` | 毫秒 | `2500` | 500–60000 | 任务中心 | 交互行为 | 下次轮询 | 否 |
| `ui.data_refresh_interval_ms` | 毫秒 | `5000` | 1000–300000 | 数据中心与行情面板 | 交互行为 | 下次轮询 | 否 |
| `ui.data_health_page_size` | 行数 | `100` | 10–500 | 数据体检分页 | 交互行为 | 下次读取 | 否 |
| `ui.inventory_page_size` | 行数 | `50` | 10–500 | 数据库存分页 | 交互行为 | 下次读取 | 否 |
| `ui.instrument_lookup_limit` | 条数 | `40` | 5–200 | 证券搜索 | 交互行为 | 下次搜索 | 否 |
| `ui.instrument_lookup_debounce_ms` | 毫秒 | `250` | 50–5000 | 搜索防抖 | 交互行为 | 下次输入 | 否 |
| `schedule.enabled` | 布尔 | `false` | 无 | 定时更新 | 数据中心 | 保存；注册状态另显示 | 否 |
| `schedule.daily_time` | HH:MM | `18:30` | 安装时校验 | 每日补数 | 数据中心 | 下次安装 | 否 |
| `schedule.weekly_day` | 三天缩写 | `SAT` | 安装时校验 | 周末核对 | 数据中心 | 下次安装 | 否 |
| `schedule.weekly_time` | HH:MM | `10:00` | 安装时校验 | 周末核对 | 数据中心 | 下次安装 | 否 |
| `schedule.subprocess_timeout_seconds` | 秒 | `45` | 5–900 | 计划任务子进程 | 文件 | 下次安装 | 否 |
| `schedule.lock_timeout_seconds` | 秒 | `30` | 1–300 | 计划任务锁 | 文件 | 下次安装 | 否 |

## 拒绝与恢复

- 主文件缺失时生成完整默认主文件；主文件存在但无效时启动直接报错，不恢复默认值覆盖原文件。
- 草稿只保留浏览器中未完成的编辑。页面重新聚焦或定时轮询会比较 revision；未编辑字段接收外部修改，同字段并发修改保留双方并阻止自动保存。
- 加载历史结果进入该任务快照视图，不把历史参数写回当前项目配置。
- 已入队和运行中的任务继续使用原快照。数据目录或端口修改后显示“已保存、待生效”。

## 数据快照

本目录保存从本机应用数据库导出的自选快照，便于通过 Git 备份及跨机器转移。导出日期为 2026-10-20（Asia/Shanghai）。

- `watchlist.json`：自选列表，包含证券代码、显示名称及加入时间，按原加入顺序保存；当前快照为 12 条。

应用启动或读取自选时会导入 `watchlist.json` 中未处理的新增条目，按代码去重，不删除本地新增项。应用内修改不会自动回写快照；需要跨机器同步时应重新导出。行情数据、数据库、任务历史、结果报告和日志保存在应用数据目录中。
