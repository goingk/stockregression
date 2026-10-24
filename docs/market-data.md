# 周期行情数据管理

## 同步与校验并发

线上同步默认下载并发为 4，实际值还受“计算资源”设置和共享 CPU 预算限制。新浪分钟使用带超时和连接复用的专用适配器，每只证券只发送一次未复权分钟请求。BaoStock 日线批处理最多使用两个独立进程，每个进程维护自己的会话和登录；网页任务、CLI 与计划任务通过同一数据目录下的槽位文件限制来源并发。

校验先检查行情版本、文件状态、规则版本和逐日校验账本。完整且无异常变化的区间直接复用，不再读取行情正文。需要重验时按自然月划分，最多 4 个进程独立计算，父进程汇总进度；历史修订仍只重验受影响日期。任务工作集预算限制并行月份和中间帧数量。

行情由 `stockregression.market_data` 管理。系统维护一个当前数据包，作为唯一的正式行情来源。行情浏览、回测、预览和参数研究都读取该数据包；线上增量、历史补齐、手动导入和本地聚合也都写回该数据包。

## 统一分片、版本与合规截止

正式基础行情只有 `1day` 和 `1min`，全部使用按时间分区的不可变 Parquet 文件。日线按年，分钟线按月；写入只重建受影响的完整时间分区，其他分片保持不变。包清单只引用当前数据集版本，独立版本清单保留历史文件引用。SQLite 以单一 `parquet_shards` 表保存所有版本的分片引用，是可重建索引；发布依据始终是数据包清单。增量仅表示更新方式，更新后生成受影响周期的新完整分片。旧索引表在首次启动时事务迁移，保留所有历史版本引用；行情文件、包身份和代际不变。

导入支持两种模式：`merge` 保留未涉及日期并合并新增或修订；`snapshot` 以完整输入替换当前视图，撤销输入中缺失的日期。两个模式都保留已经发布的旧版本。完全相同的重复导入不会增加行情版本。普通 CSV 和 Parquet 导入、线上同步、历史修订及聚合统一写入规范分片。

`5min/15min/30min/60min` 只能由 `1min` 聚合，`1week/1month` 只能由 `1day` 聚合。旧粗周期文件只用于追溯，新查询会在需要时生成并持久化派生缓存；基础数据修订会使相关范围失效。基础数据缺失时返回补齐指引，不从旧粗周期文件兜底。

每个标的分别保存日线和 1 分钟的连续合规截止日期。校验结果按交易日保存，绑定行情摘要、规则版本、交易日历、日线依赖和公司行动依赖。纯追加只校验新增日期，历史修订只撤销并重验受影响日期；已合规区间再次校验不读取 K 线。完整年/月分片发布后不可变，历史清单直接引用旧分片；不再另存“本体”或“增量覆盖层”，也不需要复制一套归档行情。

所有导入、同步、读取、聚合和回测共用同一套价格校验。OHLC 必须是有限正数，成交量和成交额不能为负，最高价、最低价与开收盘价必须满足顺序关系。异常记录在读取时隔离并写入质量报告，形成不可自动填充的缺口；浏览请求仍可取得该区间内其余有效 K 线，回测、研究和参数聚合遇到未解决异常会返回 `DATA_NOT_READY`。零成交量不是停牌，也不会被推断为停牌。

SQLite 中的 `catalog.sqlite` 保存行情可重建索引、任务、配置、自选和结果索引。重建时只替换行情表，业务记录保持不变。分片清单缺失时会明确报错，不通过扫描目录猜测当前版本。

## 数据包

活动 ETF 数据包固定为应用数据目录下的 `data/dataset/etf/`；股票包可使用相同布局。当前活动包由 `config/application.json` 的 `paths.package_dir` 指定，相对路径按项目根目录解析。正式读取只支持 schema 2 清单；旧整包布局只由一次性迁移命令读取，不保留运行时兼容分支。`data/` 下另存索引、迁移报告、导入、结果和日志。

ETF 数据包目录为 `data/dataset/etf/`，股票数据包目录为 `data/dataset/stock/`。两者分别使用以下布局：

```text
data/dataset/etf/
  etf_basic_data.parquet
  partitions/
    1day/symbol=<代码>/adjustment=none/year=YYYY/<sha256>.parquet
    1min/symbol=<代码>/adjustment=none/month=YYYY-MM/<sha256>.parquet
    <派生周期>/...                       # 按需生成并缓存
  auxiliary/
    calendars/SH.parquet
    calendars/SZ.parquet
    actions/<代码>.parquet

  .stockregression/
    manifest.json
    datasets/<周期>/<代码>/<复权口径>/v<版本>.json
    versions/                              # 迁移后任务快照
    quality/
    history/
    quarantine/unsupported-symbols/
    staging/
    transactions/
    locks/

data/dataset/stock/
  stock_basic_data.parquet
  partitions/                              # 与 ETF 相同的年/月分片规则
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

每个行情分片只归属一个证券、周期、复权口径和时间桶；文件名由内容摘要确定。相同包搬迁保留 `package_id` 和 `generation`，迁移建立新的 generation，使旧行情快照明确不可用于重跑。

数据包清单保存稳定身份 `package_id` 和单调递增的 `package_version`。文件引用保存相对路径，因此整个目录可以搬移。

## 单一写入链路

所有写入均经过以下步骤：

1. 识别证券、周期、时间标签和单位。
2. 规范化字段并与包内当前记录比较。
3. 锁定证券、周期和复权口径，生成受影响时间桶的完整暂存分片。
4. 校验分片并按内容摘要原子安装。
5. 写不可变数据集版本清单，并原子更新包级当前引用。
6. 事务更新 SQLite 索引并撤销受影响的质量结论、使派生数据失效。
7. 完成事务日志；索引晚于已发布清单时，启动恢复会按清单修复。

重复导入相同内容不会产生新行情版本。低优先级派生记录不能覆盖更高优先级的经验证记录。同等优先级的数据修订允许更新，并记录新增、修正、未变化和冲突数量。

日线下载仍可并行，但每个证券分别提交自己的年分片，不再争写共享日线大文件。分钟及分钟派生分片也分别按月提交。

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

旧分片不可变。新版本只引用未变分片和已替换时间桶的新分片，已提交任务继续读取自己的数据集版本。

提交日志位于 `.stockregression/transactions/`。清单发布前失败不会改变当前引用；清单发布后 SQLite 更新失败，启动时会根据已发布清单补齐索引。孤立的内容分片不会被自动认作当前行情。`data verify-package --full` 会校验所有活跃分片的 SHA-256、Parquet 行数和时间范围。

## 登记与索引

“从清单重建索引”是幂等操作：

- 只更新指定的证券、周期和文件。
- 不删除同周期其他证券、其他周期或历史版本。
- 文件未变化时保留现有版本和校验结果。
- 仅从 schema 2 包清单读取已发布的分片引用。
- 保留每个数据集的历史版本引用和未变化的质量记录。
- 重建行情表时保留任务、结果、自选和配置业务表。

目录列表使用 SQLite 索引；行情查询按证券、周期、版本及时间范围选择分片，再由一次 DuckDB 读取合并所需字段和时间过滤。

迁移和索引维护命令：

```powershell
uv run stockregression data migrate-package --source D:\ETF --target .\data\dataset\etf --dry-run
uv run stockregression data migrate-package --source D:\ETF --target .\data\dataset\etf --apply
uv run stockregression data verify-package --full
uv run stockregression data rebuild-index
```

如果迁移在目录切换阶段失败，且 `migration.json` 显示 `status=failed`、`rolled_back=true` 并保留 `stage_path`，可用同一个迁移编号恢复已校验的暂存包：

```powershell
uv run stockregression data migrate-package --source D:\ETF --target .\data\dataset\etf --apply --migration-id <编号>
```

恢复会重新核对冻结基线并验证暂存清单；基线变化或暂存文件缺失时会停止，不会重用不完整数据。

清理迁移源或旧目标前，必须先用 `data verify-package --full --source-unavailable` 验证源目录不可用时仍能完成服务冒烟检查，再通过 `data migrate-package --delete-source --delete-replaced --migration-id <编号>` 清理。

## 本地导入

支持单文件、文件夹和拖拽导入。导入结果合并到当前数据包。

```powershell
uv run --frozen stockregression data import-file --frequency 1day --path .\imports\1day\600000.parquet --calendar-path .\calendar.csv
uv run --frozen stockregression data import-file 600000.SH --frequency 1min --path .\imports\1min\600000.csv --calendar-path .\calendar.csv --label-convention end
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

全历史初始化与日常增量的工作量不同：全历史任务会建立并检查历史账本，按最多 512 个交易日一批校验后原子发布；增量下载只回看已发布行情尾部的近期重叠区间。质量账本缺失、落后或中断时，仍从本地分片完成历史补校；质量水位不会扩大网络下载范围。任务进度报告当前阶段、日数、成功／失败／复用／跳过／未处理数量，以及实际下载、BaoStock 会话和写入并发。进度是已完成工作的观测值，不提供总耗时承诺。

任务中心可从失败、取消或部分完成的 `market_sync` 任务继续。继续会使用原任务冻结的范围和截止日，生成新的任务记录；原任务保持终态。取消任务即使已有结果编号，结果也只是截至中断时的部分报告，包含已处理批次与计数，不等于完整同步。

日线下载保留 3 个自然日重叠窗口；1 分钟从已发布尾部回看 2 根基础 bar，并以冻结截止日为上限。来源空响应或失败不会推进已核验覆盖终点。严格校验通过后才推进对应周期的合规截止日期。

5/15/30/60 分钟和周期数据不再联网修补；基础数据更新使相关派生范围失效，查询时从 1 分钟或日线重新生成。交易所日历、停牌和已有集合竞价占位分别处理。

### 数据源策略与健康状态

| 用途 | 默认或必需来源 | 可选来源 |
|---|---|---|
| 股票日线 | BaoStock | — |
| 1 分钟 | 新浪（AKShare）近期 | 通达信近期；两者都失败时自动回退同花顺；更早历史只来自本地包或导入 |
| 5/15/30/60 分钟、周线、月线 | 本地基础数据聚合 | — |
| ETF 日线 | BaoStock | 东方财富（AKShare），按备用开关和真实故障冷却决定是否尝试 |
| 1 分钟历史 | 当前数据包 | — |
| 交易日历、股票公司行动 | BaoStock | — |
| ETF 分红、拆分 | AKShare | — |

```powershell
uv run --frozen stockregression data sources
uv run --frozen stockregression data probe --source tdx --capability recent_minute --market all
uv run --frozen stockregression data probe --source eastmoney --capability etf_daily --market etf
```

默认与可选来源均可直接使用，手动检测仅供诊断，不作为启用门槛。健康状态按来源、接口能力和市场类型保存在应用目录的 SQLite 状态表中，多个证券、后台进程和新任务共享同一个结果。网络或接口级失败状态默认冷却 24 小时，单标的空响应不触发来源冷却；冷却期间使用默认来源，手动探测成功会清除冷却。连接或超时错误最多重试一次，协议错误不重试，可选来源失败时当前请求最多回退默认来源一次。ETF 日线备用源由 `market_data.etf_daily_fallback_enabled` 控制（默认 `true`），不要求 24 小时资格或预先手动检测，实际返回的行情仍需满足计算必需的字段和价格要求。同花顺回退仅用于近期分钟，不提供 ETF 日线。检测任务不写入行情。

新浪和通达信均失败，或返回的同日分钟明显不完整（首末分钟缺失、单日缺超过 5 根或连续缺超过 3 根）时，当前请求继续尝试同花顺近期 1 分钟接口。深市收盘集合竞价的 14:58、14:59 零量占位不算不完整。同花顺覆盖约 40 个交易日的滚动窗口，返回结束标签的 241 根／日（含 09:30 开盘集合竞价），OHLC 与成交量按来源原值使用。该接口不提供逐分钟成交额，系统用成交量与 OHLC4 估算分钟成交额，再按当日官方成交总额缩放；缩放不修改 OHLC 和成交量。来源状态会记录 `ths` 的成功或失败，页面可将同花顺作为自动回退来源单独检测。

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
| GET /api/v1/data/package | 包身份、generation、schema、分片容量和索引状态 |
| GET /api/v1/data/inventory | 按证券分组的服务器分页目录与完整统计 |
| GET /api/v1/data/datasets | 按周期数据集列表 |
| GET /api/v1/data/coverage/{symbol}?timeframe=1day | 覆盖、版本与质量 |
| GET /api/v1/data/bars/{symbol}?frequency=1day | 数据包行情 |
| POST /api/v1/data/import-package/scan | 预览 ETF、股票或混合数据包 |
| POST /api/v1/data/import-package | 从 schema 2 清单幂等重建行情索引 |
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


### 同步任务恢复与诊断

`POST /api/v1/jobs/{job_id}/resume` 支持已失败、取消或部分完成的行情同步，返回新的 `job_id` 和 `resumed_from`。恢复使用原任务冻结的证券、包身份和截止日；已完成检查点仍需核对行情版本、质量修订、依赖与规则。行情已发布而质量未完成时先进行本地补校。

任务响应的可选 `progress_detail` 提供阶段、当前证券、已检查／待检查天数、本次结果计数、检查点复用数量和实际资源分配。取消任务保留 `cancelled` 状态及关联的部分报告，区分行情已发布但质量未完成、已下载未提交、失败和未处理项。空响应归类为 `empty_unresolved`，不得推进水位；同步操作完成但质量有缺口时仍为部分完成，严格回测按原规则拒绝不合规区间。

离线性能比较可运行 `.venv/Scripts/python.exe scripts/benchmark_daily_validation.py`。脚本在临时目录创建固定 1,250 个交易日样本，比较首次建账、512 日批次发布次数和新增 1 日时的分片读取，不访问网络或现有数据包。
