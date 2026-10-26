# 智能定投规则与数据边界

本项目提供普通固定股数定投、均线智能定投和涨跌幅智能定投。回测与参数研究共用 `ConfigDcaFields` 和同一股数账户执行链路。

## 参考范围与公开说明

本功能参考“下跌时增加、上涨时减少投入”的智能定投理念。2026-10-26 尝试查阅[支付宝服务大厅](https://help.alipay.com/lab/index.htm)与[蚂蚁财富官网](https://www.antfortune.com/)；搜索中继返回 502，直接 HTTP 访问未取得可核验的智能定投算法正文。因此下文是本项目明确约定、可解释和可测试的规则，**不声称精确复刻支付宝或蚂蚁未公开的算法、分档或参数**。

本功能用于历史回测，不连接真实账户执行申购。

## 配置与兼容

配置保存在现有 `BacktestConfig.dca` 内，保存的回测配置和工作台默认值使用同一校验模型。

| 字段 | 默认值 | 含义及范围 |
| --- | --- | --- |
| `mode` | `fixed_shares` | `fixed_shares`、`ma`、`price_change` |
| `frequency` | `monthly` | `daily`、`weekly`、`monthly` |
| `shares` | 100 | 普通定投每期股数，100 的正整数倍 |
| `initial_shares` | 0 | 定投账户开始前已有持仓，100 的非负整数倍 |
| `base_amount` | 1000 | 智能定投每期基准金额，元；大于 0、不超过 10 亿元 |
| `reference` | `target` | `target` 当前交易标的、`security` 另一证券/ETF、`index` 指数 |
| `reference_symbol` | `null` | 外部参考完整代码；指数与证券使用独立命名空间 |
| `reference_version` | `null` | 外部参考冻结版本，提交时解析；执行时必须存在 |
| `ma_window` | 250 | 均线收盘样本数，2–1000 |
| `change_window` | 1 | 涨跌幅跨越的交易日数，1–1000，需要 N+1 条收盘价 |
| `sensitivity` | 5 | 调整力度，大于 0、不超过 100 |
| `min_multiplier` | 0.5 | 最低投入倍数，0–1 |
| `max_multiplier` | 2 | 最高投入倍数，1–20 |

旧配置没有新增字段时自动采用 `fixed_shares`，保留原有定投行为。智能定投适用于 `schema_version=2/3`；当前 UI/API 提交使用版本 3。科创板普通定投要求每期至少 200 股；智能模式不检查隐藏的 `shares=100` 默认值，由实际委托数量遵守 200 股门槛。

所有数值必须有限；模型拒绝未知字段和无效模式。指数当前提供上证指数 `000001.SH`、沪深300 `000300.SH`、中证500 `000905.SH`、上证50 `000016.SH`、深证成指 `399001.SZ`、创业板指 `399006.SZ`。例如指数 `000001.SH` 不会按股票代码补成 `000001.SZ`。

## 信号与公式

对执行日 D，只取日期严格小于 D 的参考日线。记最近一次可用参考收盘日期为 T，收盘价为 C(T)。正常情况下 T 必须覆盖交易标的已知的前一交易日；若参考缺失、停牌或落后，不能往前找一个陈旧价格继续执行。

证券跨除权日比较时，将窗口内历史价格按**截至 T 已发生**的公司行动换算至 T 的价格口径；指数直接使用其收盘点位。

均线模式：

```text
MA = 最近 ma_window 条此前收盘价的算术平均（包含 T）
signal = C(T) / MA - 1
```

涨跌幅模式：

```text
N = change_window
signal = C(T) / C(T-N) - 1
```

这里的 N 是相邻日线交易样本间隔，默认 1 表示此前最近两个收盘价之间的变化。窗口内已知交易日缺口会使信号不可用。

两种模式共用投入规则：

```text
multiplier = min(max_multiplier, max(min_multiplier, 1 - sensitivity × signal))
planned_amount = round(base_amount × multiplier, 2)
```

例如调整力度 5、上下限 0.5–2：信号为 -10% 时投入 1.5 倍，+10% 时投入 0.5 倍，-30% 时最多投入 2 倍。

## 成交、外部本金与公司行动

每期在现有调度规则允许的第一个可执行开盘尝试下单。当天只使用同一份此前收盘信号，不读取当天收盘或日内后续走势。

智能模式的 `planned_amount` 是买入证券金额上限。引擎先按原有规则计算含滑点、价格最小单位和涨跌停边界的买入价，再将金额换成股数，按 100 股单位向下取整。科创板不足 200 股不能买入。手续费另计，**实际买入金额与费用只在成功成交后计入累计外部本金**；未成交、信号不可用、零倍数或不足交易单位不会注入本金。剩余金额不跨期累积。

未成功执行的期内，可在后续可交易日用该日之前的收盘资料重新尝试。每期最多成功一笔；整期未成交记为跳过。停牌、不可用交易日、涨跌停及原有拒单规则仍适用。

定投初始持仓沿用开始前估值，不额外收取开仓费用。持仓、分红应收、送转和拆分继续由原生账户结算。普通固定股数会依公司行动转换股数；智能模式的金额基准不会被拆分比例放大，实际下单股数由当时价格重新计算。收益仍使用既有现金流本金口径，不将追加本金算成盈利。

## 数据准备与版本冻结

- 当前标的：使用目标回测已经冻结的行情快照，读取其中开始日前的日线预热和对应版本的公司行动。
- 其他证券/ETF：先在数据管理中同步日线和公司行动；任务提交时将选定快照版本写入 `dca.reference_version`，执行与重载仅读取该版本。
- 指数：表单中点击“准备指数参考数据”，调用 `POST /api/v1/dca/reference/prepare`，请求体为当前完整 `BacktestConfig`。接口通过现有串行、限速的 BaoStock 连接读取未复权指数收盘，并缓存内容版本；返回 `symbol/kind/source/version/first_date/last_date/row_count`。前端写回 `reference_version`。首次获取的预热日历长度是请求边界，实际计算仍要求足量交易日收盘。

指数缓存同时保留内容版本与当前指针；正式提交和预览在入队前锁定版本。已经锁定的配置、候选重载和任务执行不再读取最新指针。旧版本丢失或不匹配会报错，不能回退至最新版本或另一个参考。切换参考类型或代码时清空旧版本。证券快照中没有公司行动版本时，也不会读取该证券后续新增的最新公司行动。

参数研究冻结时同时更新 `base.dca` 与 `research_context.dca`。这是必需的：worker 反序列化 `OptimizationConfig` 时会以研究上下文重建 `base`，仅更新 `base` 会丢失参考版本。测试同时覆盖传统研究配置、独立 v5 请求的真实提交，以及序列化后重新校验的路径。

提交缺少可用外部参考时返回 HTTP 409，错误码 `DCA_REFERENCE_NOT_READY`。准备指数网络或提供方失败会显示接口错误，不静默切换数据来源。参数保存本身可先于数据准备。

## 可解释输出与拒单

`strategies.dca.dca_rule` 为 `contrarian_linear_v1`。详细结果的 `signals[]` 记录每个评估交易日的信号证据；成功成交和对应拒单附带 `dca_signal`：

```text
mode, reference, reference_symbol, reference_version,
execution_date, signal_date, window, base_amount,
close, baseline, signal, multiplier, planned_amount, status
```

不可用信号使用 `status=skipped`，增加 `reason/message`；尚未取得可用信号时 `signal_date` 可以为 null，`close/baseline/signal/multiplier/planned_amount` 不出现。`window` 表示所需收盘样本数，因此涨跌幅模式为 `change_window+1`。

| 原因 | 含义 |
| --- | --- |
| `dca_reference_stale` | 没有此前参考收盘，或缺少执行日前一交易日参考 |
| `dca_insufficient_history` | 此前历史样本数不足 |
| `dca_reference_invalid` | 非法价格或窗口包含停牌 |
| `dca_reference_gap` | 窗口内缺少已知交易日收盘 |
| `dca_reference_action` | 未支持/不可核验公司行动，或除权跳变没有资料解释 |
| `dca_amount_below_lot` | 当期金额或零投入倍数不足交易单位 |
| `dca_signal_unavailable` | 内核信号不可用的兜底原因，通常由上述详细原因替换 |
| `dca_period_skipped` | 整期没有成功买入，沿用原有汇总记录 |

指标 `signal_unavailable_days` 是整个评估区间内信号不可用的交易日数，包含未到期日期；`skipped_periods` 是实际没有成功成交的定投期数，二者含义不同。摘要计算仍保留计数和规则标识，但不返回完整信号数组。

## 验证

`tests/test_smart_dca.py` 覆盖普通定投兼容、两模式与倍数限制、目标/外部证券/指数、执行日及未来价格和公司行动扰动、历史不足与陈旧/缺口、拆分除权、滑点和费用、STAR 实际手数、独立账户、参数验证、配置与默认值持久化、准备 API、任务提交冻结、参考更新后重载旧版本、旧版本缺失报错，以及 Numba/解释执行一致性。

`frontend/tests/smartDca.test.mjs` 覆盖旧配置补默认、两模式与外部参考配置保留、无效参数和摘要；纯 TS 工具采用 Node 可解析的 `.ts` 导入。

指数提供方测试使用确定性的模拟日线，不依赖运行时公网；生产数据准备能否成功取决于提供方连接和该指数所选历史的可用性。
