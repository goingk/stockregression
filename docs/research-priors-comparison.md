# 研究先验对照实验

本记录对应 2026-10-25 在 ADX/NATR golden 修复落盘后，用
`scripts/compare_research_priors.py` 完成的离线对照。`docs/research-priors-comparison.json`
为同次运行的原始报告，`benchmark_run=true`，生成时间为
`2026-10-25T03:34:03.369575+00:00`。六个“标的 × 策略”分支都完成了既定搜索并正常退出，
但每个分支均未找到 `J>0` 的合格局部峰，因此顶层和建议分支状态为 `partial`。这不是运行超时或
数据下载失败，也不表示 `daily_state_v1` 改善或恶化了收益。

## 数据、区间与日历审计

本次使用本地现有数据，不下载行情，不写入应用任务或持久化研究 result。脚本按代码顺序从最近
可用共同截止日向前选择正式研究预检 ready、活动快照可按原版本重载、且完整训练前置就绪的 ETF，
得到：

| ETF | 锁定快照 | 训练窗口 |
| --- | --- | --- |
| `159611.SZ` | `mkt-7a988f71944d6ea3cf32c2c8` | 2024-05-27 至 2025-09-17，323 个交易日，同价匹配 |
| `159995.SZ` | `mkt-e21502ab6499ee768e58fb41` | 2023-08-28 至 2025-09-17，500 个交易日，无同价候选时回退 |
| `512880.SH` | `mkt-ab3af88482d8f8c4e6804eb2` | 2023-08-28 至 2025-09-17，500 个交易日，无同价候选时回退 |

`159516.SZ` 的 250 日统计预检可用，但完整 500 日前置训练未通过质量、范围和参考价检查，故被
排除。`159792.SZ`、`515450.SH`、`517520.SH` 虽有正式 ready 行，活动快照却不能按版本重载，
也未进入样本。脚本没有用其他快照补位、没有下载数据、没有改用户配置，也没有把 1 分钟研究降级
成日线或缩短评估期。

共同日历从 `2025-03-29` 回看至 `2026-09-30`，其中去重归一后有 368 个共同交易日；这是 550 日
回看窗口中实际可用的交易日数，**不是评估样本量**。正式评估只取最后 250 个共同交易日：
`2025-09-18` 至 `2026-09-30`。该区间跨约 270 个自然工作日，经中国交易日历的非交易日过滤后
为 250 个交易日，计数口径与计划一致。

审计只读源文件，在内存中按 Asia/Shanghai 解释时间戳、去除时区、归一化到日期、排序去重并过滤
周末，不修改同步源。三只标的的审计结果相同：

| 日历层 | 原始时间戳 | 无效 | 含时分 | 周末 | 归一化重复日 | 窗口内唯一日 | 组合交易日 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auxiliary calendar payload | 3 | 0 | 0 | 0 | 0 | 0 | — |
| 单标的交易日历 | 368 | 0 | 0 | 0 | 0 | 368 | 368 |
| 三标的共同交易日历 | 368 | 0 | 0 | 0 | 0 | 368 | 368 |

辅助 payload 只有 `2026-10-21`、`2026-10-22`、`2026-10-23` 三个增量尾部日期，均位于本次
回看窗口之后，所以窗口内计数为 0；报告的 `outside_range_timestamps=3` 明确保留这一点。评估
段的 `evaluation_weekend_days=0`、`evaluation_duplicate_dates=0`、
`evaluation_intraday_timestamps=0`，且评估日数与回看共同日数分开记录。

## 实验配置

两只策略臂使用同一标的、快照、共同评估日期、搜索范围、费用、窗口策略和随机种子。
`selection_target=fixed`，种子 42，频率 1 分钟；`prior_policy` 依次为 `off` 和
`daily_state_v1`。两臂都显式不绑定 `source_etf`，用来隔离 `daily_state_v1` 的状态候选。
每只 ETF 的实际源版本与目标版本相同；配置中 `lower_price`、`upper_price` 均为空。

预算固定为训练 100、邻域 200、展示 20、总评价 1000，每次运行限时 180 秒。三只标的的完整
3×3×3 邻域均满足 81 点。三区域压力评价上界为 1875，高于冻结总预算 1000，报告保留三条预算
警告；实际运行只使用 240 个候选评价，剩余总预算 760，未触发预算停止或时间停止。

`fixed` 与 `current` 的决策日都是 `2025-09-17`，六次运行均记录
`same_anchor_as_fixed=true`。`current` 在此报告中复用评估开始前的固定选择并独立展示，不是
最新行情推荐。自适应分支未在本轮运行；实现中它会在每次生效日前使用当时可见历史，而本轮
`fixed/current` 只使用评估开始日之前的数据。

## 对照结果

| ETF | 策略 | 耗时 (s) | 候选 / 账本评价 | J / 合格峰 | planned | quota_used | actually_evaluated | annotation_only | skipped |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `159611.SZ` | `off` | 5.09 | 240 / 480 | 空 / 0 | — | — | — | — | — |
| `159611.SZ` | `daily_state_v1` | 2.00 | 240 / 480 | 空 / 0 | 2 | 1 | 2 | 1 | 0 |
| `159995.SZ` | `off` | 2.55 | 240 / 480 | 空 / 0 | — | — | — | — | — |
| `159995.SZ` | `daily_state_v1` | 2.24 | 240 / 480 | 空 / 0 | 3 | 3 | 3 | 0 | 1 |
| `512880.SH` | `off` | 2.50 | 240 / 480 | 空 / 0 | — | — | — | — | — |
| `512880.SH` | `daily_state_v1` | 2.20 | 240 / 480 | 空 / 0 | 4 | 3 | 4 | 1 | 0 |

`planned_count` 是进入先验队列的坐标数；`quota_used` 是真正注入初始采样、因而不是原有初始坐标
的那部分；`actually_evaluated_count` 是计划坐标最终存在于实际评价行中的数量，也包含已由原粗
网格评价、仅获得来源标注的坐标。因此旧报告中写作“使用 2/3/4”的值实际上只对应本轮
`planned_count`，不能解释为 `quota_used` 或独立新增的候选评价数。

具体状态：

- `159611.SZ` 从两项状态指标计划 2 个坐标。NATR/HV 对称坐标与原有初始坐标重合，状态为
  `annotation_only`、原因为 `duplicate_initial_coordinate`；它最终已被评价，但没有额外占用先验
  注入名额。另一个振幅 Q75 对称坐标实际注入并评价。
- `159995.SZ` 看到 4 个状态候选。3 个在配额内实际评价；振幅 Q75 的向上不对称候选因
  `sell_step` 越界被跳过，原因为 `out_of_range:sell_step`，没有钳制到边界。
- `512880.SH` 从 4 个状态候选计划 4 个坐标。NATR/HV 向上不对称坐标与原有初始坐标重合，记为
  `annotation_only`；其余 3 个由先验注入并实际评价。

三只在 `daily_state_v1` 下的最终状态统计分别为
`actually_evaluated=1, annotation_only=1`、`actually_evaluated=3, skipped=1`、
`actually_evaluated=3, annotation_only=1`，与 `quota_used=1/3/3` 及
`actually_evaluated_count=2/3/4` 的分层含义一致。所有 `research_prior.phase` 均为 `final`，
配额均为 8，未超过训练预算的 20%，也未超过 8。

## 样本外与失败解释

所有固定分支都没有合格局部峰或推荐参数，因此没有可执行的样本外策略，样本外超额收益、峰值
本金回撤和配对持有回撤均为空；`daily_state_minus_off` 的三项差值也为空。不能把这组结果解释成
“先验收益为零”，更不能据此宣称正收益、显著性或者某一臂更优。

六次运行各自只评价训练 100 点与邻域 140 点，没有展示补算，原因是候选搜索未产生 `J>0` 的合格
峰；报告中的 `partial` 是结果不完整的领域状态，不代表命令失败。`failures` 数组为空，所有分支
`runtime_limit_hit=false`。`daily_state_v1` 的 `source_seed_count=0` 是本实验有意不绑定
`source_etf` 的结果；`source_check` 因而记录 `source_as_of_missing`，没有自动切换到最新结果、
其他 ETF 或静默转为无来源以外的模式。

本次指标代码由 `market_state.py` 的 SHA-256
`a3c250810c1aae86bde048fb6b7518d239f9a9b4e5770565af26128aa6d2f08a` 标识，报告同时记录
ADX golden 测试 `test_wilder_adx_and_natr_match_fixed_talib_064_golden` 及 C TA-Lib 0.6.4
参考。运行源码提交为 `fcb740463eee96d10d6b34f8397bf4e36e23adc3`，报告 JSON 的 SHA-256 为
`577ad8003462ce0bd4944ea61a5483eb67c7de710a7d6c54d06c8b555230f131`。

## 验证

本次完成后运行了状态指标与研究来源相关的限定测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_market_state.py tests/test_research_prior.py tests/test_research_origin.py
```

结果为 `37 passed`。另外逐项核对了报告不变量：评估恰为 250 个唯一交易日，回看共同日为 368；
每只 ETF 的源版本、目标版本和两次运行的执行版本一致，两臂频率为 1 分钟，固定参数上下价格界
均为空，范围、策略和种子相同；六次运行的 `current.decision_date` 均为 2025-09-17 且
`same_anchor_as_fixed=true`；每个 `daily_state_v1` 的 `phase=final`，候选状态计数与
`quota_used`、`actually_evaluated_count`、`annotation_only_count` 一致。

## 复现命令

先做不运行候选评价的选择与日历审计：

```powershell
.\.venv\Scripts\python.exe scripts/compare_research_priors.py --selection-only
```

正式三 ETF 对照命令：

```powershell
.\.venv\Scripts\python.exe scripts/compare_research_priors.py `
  --training-budget 100 `
  --validation-budget 200 `
  --reporting-budget 20 `
  --total-evaluation-budget 1000 `
  --max-runtime-seconds 180 `
  --seed 42 `
  --output docs/research-priors-comparison.json
```

原始 JSON 中的 `selection.calendar_audit_by_symbol` 保留每个标的的辅助 payload、单标的日历和
共同日历审计；`symbols.*.configs`、`runs.*.prior.candidate_checks` 和
`runs.*.candidate_failure_reasons` 保留完整配置、逐候选状态和失败理由。
