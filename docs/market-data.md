# 周期行情数据管理

## 同步与校验并发

线上同步默认下载并发为 4，实际值还受“计算资源”设置和共享 CPU 预算限制。新浪分钟使用带超时和连接复用的专用适配器，每只证券只发送一次未复权分钟请求。BaoStock 日线批处理最多使用两个独立进程，每个进程维护自己的会话和登录；网页任务、CLI 与计划任务通过同一数据目录下的槽位文件限制来源并发。

校验先检查行情版本、文件状态、规则版本和逐日校验账本。完整且无异常变化的区间直接复用，不再读取行情正文。需要重验时按自然月划分，最多 4 个进程独立计算，父进程汇总进度；历史修订仍只重验受影响日期。任务工作集预算限制并行月份和中间帧数量。

行情由 `stockregression.market_data` 管理。系统维护一个当前数据包，作为唯一的正式行情来源。行情浏览、回测、预览和参数研究都读取该数据包；线上增量、历史补齐、手动导入和本地聚合也都写回该数据包。

## 基础底座、增量覆盖与合规截止

正式更新的基础周期只有 `1day` 和 `1min`。已有 Parquet 文件保持原布局，作为历史底座直接读取；后续新增和修订写入当前数据包 `.stockregression/increments/` 下的不可变分片。分钟分片按月、日线及周期分片按年生成，查询时按时间合并底座和覆盖文件。旧版本继续引用原文件，因此增量更新不会重写历史底座。

导入支持两种模式：`merge` 合并新增和重叠修订，`snapshot` 以新文件为准替换当前可见底座并清除现有覆盖。全量替换只撤销内容变化的交易日校验结果，未变化的日线摘要继续复用。重复导入先检查文件内容指纹；文件大小和修改时间只用于发现变化。

`5min/15min/30min/60min` 只能由 `1min` 聚合，`1week/1month` 只能由 `1day` 聚合。旧粗周期文件只用于追溯，新查询会在需要时生成并持久化派生缓存；基础数据修订会使相关范围失效。基础数据缺失时返回补齐指引，不从旧粗周期文件兜底。

每个标的分别保存日线和 1 分钟的连续合规截止日期。校验结果按交易日保存，绑定行情摘要、规则版本、交易日历、日线依赖和公司行动依赖。纯追加只校验新增日期，历史修订只撤销并重验受影响日期；已合规区间再次校验不读取 K 线。原始外部文件可通过 `/api/v1/data/archive`、`data archive` 或页面按钮复制到 `.stockregression/archives/`。归档前不要直接覆盖原始文件。

所有导入、同步、读取、聚合和回测共用同一套价格校验。OHLC 必须是有限正数，成交量和成交额不能为负，最高价、最低价与开收盘价必须满足顺序关系。异常记录在读取时隔离并写入质量报告，形成不可自动填充的缺口；浏览请求仍可取得该区间内其余有效 K 线，回测、研究和参数聚合遇到未解决异常会返回 `DATA_NOT_READY`。零成交量不是停牌，也不会被推断为停牌。

SQLite 中的 `catalog.sqlite` 保存可重建索引、任务、配置和结果索引。它不是行情正文。迁移后即使索引损坏，也可以通过数据包登记恢复目录。

## 数据包

活动 ETF 数据包固定为应用数据目录下的 `data/dataset/etf/`；股票数据包预留为 `data/dataset/stock/`，两者使用相同的 Parquet 布局。当前活动数据包由 `config/application.json` 的 `paths.package_dir` 指定，相对路径按项目根目录解析；应用不会回退到旧 `package/`、`manifests/` 或 `parquet/` 快照。`data/` 下的其他子目录保存索引、任务、结果和日志，行情正文只放在 `data/dataset/<asset>/`。

ETF 数据包目录为 `data/dataset/etf/`，股票数据包目录为 `data/dataset/stock/`。两者分别使用以下布局：

```text
data/dataset/etf/
  etf_basic_data.parquet
  etf_daily.parquet
  etf_1min/<代码>.parquet
  etf_5min/<代码>.parquet
  etf_15min/<代码>.parquet
  etf_30min/<代码>.parquet
  etf_60min/<代码>.parquet
  etf_1week/<代码>.parquet
  etf_1month/<代码>.parquet

  auxiliary/
    calendars/SH.parquet
    calendars/SZ.parquet
    actions/<代码>.parquet

  .stockregression/
    manifest.json
    quality/
    history/
    staging/
    transactions/
    locks/

data/dataset/stock/
  stock_basic_data.parquet
  stock_daily.parquet
  stock_1min/<代码>.parquet
  stock_5min/<代码>.parquet
  stock_15min/<代码>.parquet
  stock_30min/<代码>.parquet
  stock_60min/<代码>.parquet
  stock_1week/<代码>.parquet
  stock_1month/<代码>.parquet

  auxiliary/
    calendars/SH.parquet
    calendars/SZ.parquet
    actions/<代码>.parquet

  .stockregression/
    manifest.json
    quality/
    history/
    staging/
    transactions/
    locks/
```

ETF 文件保留原有路径。A 股使用同类布局。日线是每个市场类型一个集中文件，分钟、周线和月线按标的拆文件。

数据包清单保存稳定身份 `package_id` 和单调递增的 `package_version`。文件引用保存相对路径，因此整个目录可以搬移。

## 单一写入链路

所有写入均经过以下步骤：

1. 识别证券、周期、时间标签和单位。
2. 规范化字段并与包内当前记录比较。
3. 生成暂存文件和变更报告。
4. 备份会被替换的物理文件。
5. 原子替换正式文件。
6. 发布包内清单和数据库索引。
7. 计算本次范围的质量状态。

重复导入相同内容不会产生新行情版本。低优先级派生记录不能覆盖更高优先级的经验证记录。同等优先级的数据修订允许更新，并记录新增、修正、未变化和冲突数量。

日线集中文件按批提交：同一批最多 128 个证券完成下载后只重写一次集中文件。下载仍可并行。使用 DuckDB/Arrow 合并，保留其他证券和未知源列。

## 时间、单位与来源

| 源字段 | 正式字段 | 规则 |
|---|---|---|
| `ts_code` | `symbol` | 与文件名核对 |
| `trade_time` | `timestamp` | 分钟、周线、月线时间 |
| `trade_date` | `timestamp` | 日线日期 |
| `pre_close` | `preclose` | 保留来源值 |
| `vol` | `volume` | 日线源单位为手；分钟源按实际单位校准 |
| `amount` | `turnover` | 日线源单位为千元；分钟源按实际单位校准 |
| `adj_factor` | `adj_factor` | 保留，不自动复权 |

每个行情文件的清单记录时间标签、成交量倍率、成交额倍率、来源核验状态、解析器版本和内容版本。

开始标签转为结束标签、09:30 开盘集合竞价合并、成交量换算都只应用一次。重复读取、重新校验或后续增量不能再次平移或重复合并。

无法确认时间标签或单位的内容保留在包内暂存区，不直接混入正式文件。缺少日历、公司行动或 ST 历史状态时，行情可以浏览，但严格回测会返回具体缺口。

## 文件版本与恢复

增量更新前，旧物理文件复制到 `.stockregression/history/files/`。同一份旧文件只保存一次。集中日线被替换后：

- 所有相关证券的旧快照继续指向旧物理文件。
- 所有相关证券的当前引用切换到新物理文件。
- 只有自身内容变化的证券更新内容时间；其他证券只更新物理文件版本引用。
- 读取和写入同一物理文件时共享文件锁。

提交日志位于 `.stockregression/transactions/`。启动时：

- 清单尚未发布的未完成事务，从包内历史文件恢复旧目标文件。
- 已发布但索引尚未更新的记录标记为需要重建索引。
- 已完成且索引一致的记录清理日志。

外部程序改动正式文件后，大小、修改时间或内容摘要不一致会使质量状态失效；重新登记并校验后才能再次显示为就绪。

## 登记与索引

“登记并连接”是幂等操作：

- 只更新指定的证券、周期和文件。
- 不删除同周期其他证券、其他周期或历史版本。
- 文件未变化时保留现有版本和校验结果。
- 文件变化时发布新版本并要求重新核验。
- 缺少文件时标记为缺失，历史快照仍保留。
- 支持 `etf_*` 和 `stock_*` 两种布局。

登记只读取 Parquet 元数据，不复制全部分钟 K 线。索引存在时打开证券目录只访问 SQLite；行情查询只打开目标文件并用 DuckDB 过滤所需区间。

首次接入时执行一次全量登记即可写入每个物理文件的大小、修改时间和版本；该操作只读元数据，耗时与文件数量有关，不重写 K 线。旧格式行情可以使用 `data cleanup-legacy` 查看清单，确认后加 `--apply` 删除；清理保留结果、任务、配置、自选和当前数据包版本。

## 本地导入

支持单文件、文件夹和拖拽导入。导入结果合并到当前数据包。

```powershell
uv run --frozen stockregression data import-file --frequency 1day --path .\imports\1day\600000.parquet --calendar-path .\calendar.csv
uv run --frozen stockregression data import-file --frequency 15min --path .\imports\15min\600000.csv --calendar-path .\calendar.csv --label-convention end
```

CSV 自动识别 UTF-8、UTF-8 BOM 和 GB18030；Parquet 的多层时间索引会提升为列。没有日历或公司行动文件时，数据先发布为可浏览状态，质量状态保持待核验。

## 线上同步

```powershell
# 只更新数据包内已有日线证券
uv run --frozen stockregression data sync --frequency 1day --mode incremental --scope market --resume

# 显式初始化全市场日线
uv run --frozen stockregression data sync --frequency 1day --mode history --scope market --start 2015-01-01 --initialize

# 只更新指定证券和周期
uv run --frozen stockregression data sync --frequency 1min --mode repair --symbol 600000.SH --start 2026-09-28 --end 2026-10-16
```

日常增量默认不扩大证券范围。新证券必须通过指定代码或显式 `--initialize` 初始化。

日线回看最近三个交易日，1 分钟回看最近两个已结束交易日。来源空响应或失败不会推进已核验覆盖终点。严格校验通过后才推进对应周期的合规截止日期。

5/15/30/60 分钟和周期数据不再联网修补；基础数据更新使相关派生范围失效，查询时从 1 分钟或日线重新生成。交易所日历、停牌和已有集合竞价占位分别处理。

### 数据源策略与健康状态

| 用途 | 默认或必需来源 | 可选来源 |
|---|---|---|
| 股票日线 | BaoStock | — |
| 1 分钟 | 新浪（AKShare）近期 | 通达信近期；更早历史只来自本地包或导入 |
| 5/15/30/60 分钟、周线、月线 | 本地基础数据聚合 | — |
| ETF 日线 | BaoStock | 东方财富（AKShare） |
| 1 分钟历史 | 当前数据包 | — |
| 交易日历、股票公司行动 | BaoStock | — |
| ETF 分红、拆分 | AKShare | — |

```powershell
uv run --frozen stockregression data sources
uv run --frozen stockregression data probe --source tdx --capability recent_minute --market all
uv run --frozen stockregression data probe --source eastmoney --capability etf_daily --market etf
```

默认来源直接工作，可选来源必须先通过股票和 ETF 的对应探测。健康状态按来源、接口能力和市场类型保存在应用目录的 SQLite 状态表中，多个证券、后台进程和新任务共享同一个结果。失败状态默认冷却 24 小时；冷却期间使用默认来源，手动探测成功会清除冷却。连接或超时错误最多重试一次，协议错误不重试，可选来源失败时当前请求最多回退默认来源一次。检测任务不写入行情。

来源页面不会因打开而联网。ETF 日线的失败与 ETF 分红、拆分接口彼此独立；一次东方财富日线故障不会停用 AKShare 的基金资料。

### 1 分钟历史边界

1 分钟历史初始化只接受本地数据包登记或导入，API 与 CLI 会拒绝在线历史请求并返回导入指引。增量与缺口修补对每只证券最多获取一次近期窗口，再按本地缺口合并。新浪返回范围之外的缺口保留为待导入状态，不推进该段已核验覆盖。

## 质量状态

状态拆分为：

| 维度 | 值 |
|---|---|
| 文件 | `readable`、`missing`、`changed` |
| 质量 | `pending`、`passed`、`failed` |
| 覆盖 | `unknown`、`partial`、`complete` |
| 派生 | `valid`、`stale` |

列表中的数据集“可用”要求当前文件的完整范围已有通过的质量报告。回测准备检查按本次请求范围判断；区间外缺口不会自动阻塞本次请求，除非本次任务依赖它。

分钟覆盖日按有效交易日日期去重统计，不能使用记录行数。容量按物理文件去重统计。统计与分页由后端完成，不受页面单页数量影响。

## 回测准备任务

回测或研究返回 `DATA_NOT_READY` 和具体缺口。数据管理页先预览：

- 必备证券资料和真实交易日历。
- 日线、前置估值收盘和公司行动。
- 目标分钟周期。
- 校验和发布步骤。

用户确认后提交一个父任务，按依赖执行。只缺日历或公司行动时，只补相应辅助资料；已有完整行情不会因辅助资料缺口被无意义重写。

父任务保留每步状态。部分成功后，已发布数据立即有效；重试只处理失败或剩余缺口。完成后重新判断原请求，并提供返回回测或研究草稿的入口。

数据包的辅助交易日历可能只包含最近一段增量。历史校验、覆盖判断和缺口修补会结合包内日线的实际交易日期，并保留历史失败记录；不会仅按这份短日历缩小已校验范围。仅缺质量核验时直接在本地先校验日线、再校验目标周期，无需重新下载或导入。准备任务只校验请求区间及必要的日线前置历史，显示已校验／待校验天数；同月行情批量读取，已通过日期可复用。

## 本地聚合

查询、回测和研究缺少目标粗周期时，会从对应基础周期自动生成所需范围并持久化。显式聚合命令仍可用于预先建立缓存；不再从旧粗周期文件继续导入或联网同步。

```powershell
uv run --frozen stockregression data aggregate --symbol 600000.SH --from 1min --to 15min
uv run --frozen stockregression data aggregate --symbol 600000.SH --from 1day --to 1week
```

聚合只读取本地数据包，不联网。1 分钟只生成 5/15/30/60 分钟，日线只生成周线和月线；日线保持独立基础来源。源版本变化后，派生数据标记为待重新生成。

## 缓存

查询缓存按文件版本、查询区间、字段和解析规则生成键。文件替换后旧缓存自然失效。默认预算为 256 MiB，由 `market_cache_mb` 控制。

## API

| 接口 | 用途 |
|---|---|
| GET /api/v1/data/frequencies | 周期与来源能力 |
| GET /api/v1/data/sources | 来源选择、最近探测、冷却与失败原因；只读缓存 |
| POST /api/v1/data/sources/probe | 提交来源探测任务，不写入行情 |
| GET /api/v1/data/package | 数据包身份、版本、路径、容量和恢复状态 |
| GET /api/v1/data/inventory | 按证券分组的服务器分页目录与完整统计 |
| GET /api/v1/data/datasets | 兼容的按周期数据集列表 |
| GET /api/v1/data/coverage/{symbol}?timeframe=1day | 覆盖、版本与质量 |
| GET /api/v1/data/bars/{symbol}?frequency=1day | 数据包行情 |
| POST /api/v1/data/import-package/scan | 预览 ETF、股票或混合数据包 |
| POST /api/v1/data/import-package | 幂等登记数据包并建立索引 |
| POST /api/v1/data/import-batch/scan | 预览普通文件夹 |
| POST /api/v1/data/import-batch | 批量导入并写回数据包 |
| POST /api/v1/data/sync | 指定周期同步；`initialize` 显式扩大新证券范围 |
| POST /api/v1/data/validate | 本地重新校验 |
| POST /api/v1/data/archive | 保护原始 Parquet 历史快照 |
| GET /api/v1/data/compliance/{symbol} | 日线和 1 分钟合规截止、待检验与失败日期 |
| POST /api/v1/data/aggregate | 显式聚合 |
| POST /api/v1/data/readiness | 回测或研究就绪检查 |
| POST /api/v1/data/preparation/plan | 生成准备步骤 |
| POST /api/v1/data/preparation | 提交准备父任务 |
| GET /api/v1/jobs/{job_id}/items | 分页读取逐项任务报告 |

