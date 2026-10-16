# 本地行情文件格式

本页说明 `stockregression import-data` 和“行情”辅助区域的“本地文件导入”接受的 CSV / Parquet 格式。支持沪深 A 股和场内 ETF，数量单位分别为股和份。样例块只列出表头，不含价格或成交量样本。请导入真实、未复权、时间和单位可核验的数据。

## 导入命令

CLI 的股票代码与日线文件是两个必需的位置参数：

```powershell
uv run stockregression import-data 600000.SH .\daily.csv `
  --minute-path .\minute.csv `
  --actions-path .\actions.csv `
  --calendar-path .\calendar.csv `
  --label-convention end
```

`symbol` 和 `daily_path` 不能写为 `--symbol` 或 `--daily-path`。文件扩展名为 `.parquet` 时按 Parquet 读取；其他扩展名按 CSV 读取。CLI、网页和 API 都在运行程序所在的本机读取该路径，不会上传到远端处理。

若 Parquet 把 K 线时钟放在索引而不是列里（例如 Tushare 导出的 `trade_date` / `trade_time` 两级索引），导入器会把日期时间索引层级提升为列后再按上表别名识别，不需要事先 `reset_index`。索引中的其他层级会作为普通列保留。

示例假定分钟时间戳是结束标签，因此写了 `--label-convention end`；若供应方给的是开始标签，应改为 `start`。不确定时省略该选项，默认值 `auto` 会尝试校准，未核验的交易日按日线处理。请根据数据源说明选择标签，不要仅凭文件中时间看起来接近收盘就判断。

日线文件在首次导入时必需；API / 网页导入在该股票已有数据时可以省略日线路径并沿用当前版本日线，但要确保数据确实完整。CLI 的 `daily_path` 是必需的位置参数，所以通过 CLI 时每次都要提供一个日线文件路径。导入会发布一个新的不可变数据版本。

## 日线

必需行情列为日期、开盘、最高、最低、收盘和成交量。准确导入还要求原始文件明确提供 `preclose`、`tradestatus`；股票额外要求 `is_st` 或 `isST` 历史字段，ETF 不要求 ST 字段。这些准确性字段不由导入器推算；缺少必需字段会拒绝日线导入。

ETF 导入可通过网页类型选择，或 CLI `--etf-category stock|bond|crossborder|commodity|money` 覆盖证券清单识别结果，分别对应股票、债券、跨境、黄金/商品和货币 ETF。股票可忽略该选项。ETF OHLC 按 0.001 元精度、成交量按份提供；如果来源的成交量单位为手，须先正确转换，不能把手数直接当份数。

```powershell
uv run --frozen stockregression import-data 518880.SH .\etf-daily.csv --calendar-path .\calendar.csv --actions-path .\etf-actions.csv --etf-category commodity
```

ETF 分红资料未核验的区间不能严格回测。可提供经核实的公司行动文件（确认无行动时使用带规范表头的空文件），切勿为了绕过核验而伪造空资料。

| 含义 | 规范列名 | 可识别的别名 / 说明 |
| --- | --- | --- |
| 交易日期 | `date` | 也可用 `timestamp`、`datetime`、`day`、`dt`、`time`、`trade_date` 或中文 `日期`；按日期归一化。 |
| 开盘价 | `open` | 或 `开盘` |
| 最高价 | `high` | 或 `最高` |
| 最低价 | `low` | 或 `最低` |
| 收盘价 | `close` | 或 `收盘` |
| 成交量 | `volume` | 或 `vol`、`成交量`；单位须与分钟文件一致并按股数提供。 |
| 前收盘 | `preclose` | 准确导入必需。应提供真实未复权交易前收盘，不能用前一条记录的收盘值取代除权参考价。 |
| 交易状态 | `tradestatus` | 准确导入必需；`1` 为正常交易，`0` 为确认停牌。状态未知不要伪填为 `1`。 |
| ST 历史状态 | `is_st` 或 `isST` | 准确导入必需；建议用 `0`/`1`，分别表示否/是。 |
| 股票上市日期 | `listing_date` | 可选；如数据源能提供，可在每行重复填写该股票的实际上市日期（`YYYY-MM-DD`），用于识别上市初期交易规则。 |
| 成交额 | `turnover` | 可选，也可用 `amount` 或 `成交额`。 |

CSV 表头格式（不含历史行情行）：

```csv
date,open,high,low,close,volume,preclose,tradestatus,is_st,listing_date,turnover
```

日期应覆盖与交易日历相交的全部交易日，包括停牌日；停牌时仍需有日线价格和 `tradestatus=0`，以便后端明确识别停牌，而不是把缺日当成未交易或缺失。状态列和 OHLCV 数据应是同一股票、同一未复权口径。

## 分钟线

后端只把通过校准的 **1 分钟**输入作为分钟原始行情，再按回测选定频率重采样为 5/15/30/60 分钟。不是直接导入 5 分钟或小时线的入口。

| 含义 | 规范列名 | 可识别的别名 / 说明 |
| --- | --- | --- |
| 分钟时间 | `timestamp` | 也可用 `datetime`、`day`、`dt`、`time`、`trade_time` 或中文 `时间`；日线式 `date`、`trade_date` 也可映射，但分钟记录仍必须带具体时分。 |
| 开盘价 | `open` | 或 `开盘` |
| 最高价 | `high` | 或 `最高` |
| 最低价 | `low` | 或 `最低` |
| 收盘价 | `close` | 或 `收盘` |
| 成交量 | `volume` | 或 `vol`、`成交量`。单位应与日线成交量一致；后端会测试 1 倍与 100 倍候选并记录校准结果。 |
| 成交额 | `turnover` | 可选，也可用 `amount` 或 `成交额`。 |
| 来源 | `source` | 可选，仅作来源标识。 |

分钟 CSV 表头格式（不含历史行情行）：

```csv
timestamp,open,high,low,close,volume,turnover
```

时间可以带时区；如带时区，导入器先转换到 `Asia/Shanghai`，再去掉时区标识用于本地交易时段校验。请保证源数据实际对应上海市场时钟。分钟记录的开盘、最高、最低、收盘必须为正；出现 0 或负值的文件会被整帧拒绝，因为非正价格是来源缺陷，不是真实成交。默认 `label_convention=auto`，系统会尝试校准时间标签；无法确认的交易日会标记为未校准并使用日线。数据供应方已明确标签定义时，可选择 `end`（K 线时间是分钟结束时间）或 `start`（分钟开始时间，导入器会向后平移 1 分钟）。选择前请核对数据源说明；选错标签可能导致分钟行情无法通过核验，届时覆盖报告会标记未校准并回退日线。

## 交易日历

日历文件必需包含一列 `date`，每个真实沪深交易日一行。表头格式：

```csv
date
```

未提供 `--calendar-path` 时导入器会尝试从 BaoStock 获取真实交易日历；如果联网或日历源失败，任务会失败并提示必须提供交易日历文件。不要用 `pandas.bdate_range`、周一至周五序列或未经核实的日期列表替代交易日历，否则节假日、停牌日及上市区间判断会不准确。

## 公司行动（分红、送转等）

有现金分红或送转的真实未复权价格序列应提供覆盖研究区间的公司行动资料。公司行动文件可选，但若除权前收盘变化没有对应的公司行动，回测会因无法解释跳变而拒绝运行。提供文件时至少要能归一化为下列字段：

| 规范列名 | 含义 | 要求 |
| --- | --- | --- |
| `record_date` | 股权登记日 | 必需。 |
| `ex_date` | 除权除息日 | 必需。 |
| `pay_date` | 现金红利派息日 | `cash_per_share > 0` 时必需。 |
| `stock_date` | 送转股可交易日 | `bonus_ratio > 0` 时必需。 |
| `cash_per_share` | 每股现金红利 | 可为 0；供应方 BaoStock 字段 `dividCashPsBeforeTax` 表示税前金额。 |
| `bonus_ratio` | 每股新增股份比例 | 可为 0；供应方的送股与转增比例会合并为该字段。 |
| `implemented` | 是否确认已实施 | 必需事件需为真。CSV 建议用数值 `1`/`0`，避免将字符串 `"False"` 误读为真值。 |
| `unsupported_reason` | 未建模事项说明 | 可选；非空时后端会阻止回测。 |

文件列名支持部分 BaoStock 分红字段别名：`dividRegistDate`、`dividOperateDate`、`dividPayDate`、`dividStockMarketDate`、`dividCashPsBeforeTax`，并可从 `dividStocksPs` 和 `dividReserveToStockPs` 合并送转比例。推荐自行转换为规范列名，确保字段完整、日期语义明确。

公司行动 CSV 表头格式（不含历史公司行动行）：

```csv
record_date,ex_date,pay_date,stock_date,cash_per_share,bonus_ratio,implemented,unsupported_reason
```

现金红利是税前值；本版不计算个人所得税。配股、合并、拆股等未建模的特殊公司行动会阻止相关回测。不能确认是否实施或缺少必要日期时，应先完善公司行动数据，不要填造日期以绕过校验。

## 分钟质量核验与回测降级

后端以沪市/深市连续竞价的 240 个分钟时段校验每个交易日：上午 `09:31–11:30`、下午 `13:01–15:00`，均按结束标签记时。校准档案会记录 `convention`、`volume_multiplier`、`verified_days`、交叉源匹配分数和原因。时间标签/单位未核验时，页面仍保留档案信息，但该交易日不会用未验证分钟价格路径，回测会改用日线。

对于分钟档案通过校验的交易日，最多允许整日缺 5 根且每个交易时段中连续缺失不超过 3 根；上午和下午首分钟必须存在。后端还会验证 OHLC 合法性、分钟成交量与日线总量、完整日分钟价格聚合。少量合格缺口只以前值补价格、成交量置零，并标记 `is_synthetic`；这种合成分钟不能发起交易。超过阈值、首分钟缺失、价格/成交量不一致或来源未校准时，整日回退日线。覆盖报告中的 `minute_ratio`、逐日模式、缺失分钟数和 `warnings` 是解释回测的依据。

日线降级只提供每日 OHLC，不能重建盘中先后顺序或网格触发点。因此分钟覆盖率不足时，不能把这类日线降级回测当成分钟策略完整执行结果。
