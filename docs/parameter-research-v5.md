# 参数研究协议 v5

新版参数研究使用 `protocol_version=5`、`strategy_version=5` 和 `result_schema_version=2`。普通股数回测仍使用输入契约 `schema_version=3`。旧研究请求、旧结果和旧排队任务不再进入当前计算链路；当前数据目录只保留可读取的 v5 结果。

## 单一训练窗口

每次决策只使用一段连续训练历史，内部验证窗数量固定为 0。`training_window_mode` 有两种选择：

- `same_price`（默认）：在评估日前 250 至 `training_max_days` 个交易日之间，从远到近寻找严格满足 `daily_low < anchor_close < daily_high` 的交易日。
- `fixed`：使用 `training_max_days` 个完整交易日；历史不足时明确报告数据问题，不缩短后继续。

`same_price` 模式先把历史选窗价格转换到决策日价格基准。对交易日 `d`，仅应用满足 `d < ex_date <= anchor` 且已实施的公司行动；同日行动先合并，连续行动按日期累乘。除权因子优先使用与行动推算值核验一致的除权参考价；参考价缺失或与行动不一致时，由现金分红和送转比例推算。缺少必要前收盘价、未确认实施、无效因子或未支持行动均返回独立的 `data_issue`，不会退回未复权比较，也不会把数据问题误报成“没有同价日”或“历史不足”。

成交账本始终使用真实未复权成交价。训练窗口诊断记录锚日、价格基准、行动因子、转换指纹、匹配日、回退状态和回看范围。固定窗口与同价窗口共用相同数据准备、预检和补算策略。

## 收益、风险与配对基准

对每个账户，在本次实际评价区间内定义：

```text
I_t = 初始外部本金 + 累计外部注资
P_t = 资产总值 - I_t
K   = max_t I_t
R   = P_T / K
D_t = (max(0, max_{s<=t} P_s) - P_t) / K
```

`peak_principal_return` 是 `R`，`max_principal_drawdown` 是逐 bar 路径上的 `max(D_t)`，`principal_drawdown` 是逐点值。`K=0` 时收益和回撤为空，并记录 `unevaluable_reason`。资产总值包含现金、持仓和公司行动应收权益；内部卖出回款与分红不增加外部本金。评价区间外或后续评估期的注资不进入当前区间分母。

J 的主基准是 `matched_funding_hold`。它和网格账户从相同初始持股、现金和权益开始；网格实际成交并发生外部补款时，对照账户在同一事件时点收到相同金额。拒单和未成交信号不补款。追加资金与此前整手余款组成待投资资金池，在最早可成交开盘按含费用的最大合法整手买入；不能成交时留作现金并在后续 bar 重试。现金分红留在账户内，不自动进入待投资资金池；送转和应收权益沿用相同账本规则。

基础超额为：

```text
E = peak_principal_return(grid) - peak_principal_return(matched_funding_hold)
```

网格、J 中心收益、邻域均值和下行风险都使用同一 `E`。不使用复利相对收益，也不使用时间加权日收益：

```text
mu          = sum_i(w_i * E_i)
downside    = sqrt(sum_i(w_i * min(0, E_i - tau)^2))
w_i         = exp(-distance_i^2 / (2*sigma^2))
J           = alpha*E_center + (1-alpha)*mu - beta*downside - Cpos
```

邻域权重在裁剪后的合法超正方体内重新归一化。`Cpos` 由最大持仓超限惩罚和底仓分段惩罚组成。候选资格要求 `J > 0`，使用 `max_principal_drawdown`，同时要求最少成交、买卖两向、持仓与峰值证据完整。费用压力仍为诊断，不构成时间上的验证窗。

## 样本外与统计

固定参数方案在评估开始前选择一次。自适应方案按 `rebalance_days` 重新选择参数，但换参不重置账户、现金、持仓或资金流；两条流程都汇报整个实际评估区间的本金收益和配对超额。

`fixed` 与 `current` 都以评估开始日前最后一个收盘为决策锚，并使用该日及更早的历史。`current` 复用评估开始前生成的固定分支选择，仅作为同一因果点的独立展示，不是评估结束后的最新行情推荐，也不读取样本外区间来重选参数。自适应分支则在每次生效日前使用当时最后一个已收盘交易日及更早历史，绝不使用该次生效日或评估结束日之后的数据。各分支的先验状态也按该分支实际决策日计算。

本版本不提供峰值本金超额的显著性检验，协议状态固定为 `not_applicable`，也不输出 p 值、bootstrap 区间或显著性标签。邻域稳定、约束合格和真实样本外表现是不同判断。

## ETF 来源与先验闭环

参数研究请求新增三个相互独立的显式字段：

```json
{
  "source_etf": {"result_id": "...", "candidate_ids": [1, 2]},
  "prior_policy": "daily_state_v1",
  "requested_data_version": "mkt-..."
}
```

`source_etf` 引用一条仍可读取的 ETF 研究结果。`candidate_ids` 省略时，服务只取该 ETF 在来源结果中合格的前三个样本内候选；显式给出时，必须全部属于该结果和同一标的。来源候选只通过 `buy_step`、`sell_step`、`buy_shares`、`sell_shares` 四个相对参数进入目标研究的初始覆盖集。原 ETF 的绝对价格上下界、参考价和行情版本不会被复制到目标研究；目标配置的 `lower_price` 和 `upper_price` 继续清空。目标研究仍使用目标 ETF 自己的行情、费用和快照，来源候选不直接成为推荐参数。

服务端从冻结的 ETF 结果解析来源，并在预检、提交和执行时记录 `research_origin`，包括来源结果、候选编号、来源截止日、来源数据版本、ETF 方法版本、GSS、原始指标与四轴参数。`GET /api/v1/etf-research/{result_id}/research-links?symbol=...` 提供关联研究的分页摘要。执行工作进程只消费已冻结的来源，不重新读取当前 ETF 结果或当前行情。

来源版本和目标版本分开保存。`source_data_version` 标识 ETF 结果当时冻结的行情；`requested_data_version` 标识目标参数研究要求的数据快照，执行时的 `execution_data_version` 单独记录并必须与请求版本一致。目标版本不存在时返回明确错误，不能改用来源 ETF 的版本，也不能静默回落到当前目标版本。

时间因果以目标决策日为硬门槛。冻结来源的 `source_as_of` 必须不晚于该决策日；未来来源只会跳过其静态候选，并保留 `future_source` 诊断，市场状态先验仍可使用当时可见的数据。来源缺失、不可读取、标的不匹配或候选不属于来源时，请求返回 `RESEARCH_SOURCE_INVALID`。系统不会自动切换到其他历史结果、最新结果或“无来源”模式；用户需要解除来源或重新选择候选后提交。

## `daily_state_v1` 先验

`prior_policy` 默认是 `off`。`off` 保持原有的起点、邻域和评分顺序，不生成先验诊断；`daily_state_v1` 只改变有限个初始候选，J 的定义、候选评价、约束阈值、样本外账本和结论口径全部不变。完整枚举模式下，先验只给已存在的格子附加来源标签，不改变枚举结果。

状态读取各分支实际决策日之前最多 250 个连续有效日线交易日，并额外使用紧邻的前置收盘。`fixed` 与 `current` 共用评估开始前的决策日；自适应分支使用该次生效日前的最后收盘。日线缺口会截断可用历史，不能跨缺口拼接；日线与公司行动先转换到决策日价格基准。计算失败或样本不足时保留原因，不以零值、日线替代值或未来数据补齐。

| 状态字段 | 公式 |
| --- | --- |
| `natr14` | `ATR14 / close_t`。真实波幅 `TR_t=max(H_t-L_t, |H_t-C_{t-1}|, |L_t-C_{t-1}|)`；前 14 个 ATR 取算术平均，之后按 Wilder 递推 |
| `hv10`、`hv60` | `std(ln(C_t/C_{t-1}), ddof=1)`，分别使用最近 10 和 60 个连续日对数收益 |
| `hv_ratio` | `hv10 / hv60`；`hv60=0` 时标记零分母，不伪造比值 |
| `+DI14`、`-DI14`、`ADX14` | 对 TR 和方向运动做 Wilder 平滑；`DI=100*DM/smoothed_TR`，`DX=100*|+DI--DI|/(+DI+-DI)`；ADX 先取前 14 个 DX 均值，再 Wilder 递推 |
| `amplitude_q75` | 最近 20 日 `(H_t-L_t)/C_{t-1}` 的线性插值 75 分位，零振幅保留 |

先验候选由两条对称基准及 ADX 方向不对称变体组成：

```text
hv_multiplier = 0.65 + 0.5 * max(0, hv_ratio - 1)   # hv_ratio 缺失时取 0.65
base_natr     = NATR14 * hv_multiplier
base_q75      = amplitude_q75
alpha         = 1 + min(1, (ADX14 - 22) / 28 * 0.85)   # 仅 ADX14 > 25 且方向明确时使用
```

每条有效基准先产生对称候选 `(base, base)`。满足 `ADX14 > 25` 时，`up` 方向产生 `(0.85*base, alpha*base)`，`down` 方向产生 `(alpha*base, 0.85*base)`；其余情况不产生不对称候选。步长参数按研究刻度和合法范围转换，股数参数取合法范围中点。超出目标范围、非法坐标或刻度不匹配的候选会跳过并记录具体原因，例如 `out_of_range:buy_step`、`invalid_source_granularity:buy_step`、`illegal_coordinate`。

先验配额为：

```text
quota = min(8, floor(training_budget * 20%))
```

冻结来源候选先于状态候选进入队列；重复坐标合并来源，超过配额的候选标记 `quota_exhausted`。控制流中的先验坐标排在原有粗网格之后、Sobol 样本之前，因此不会悄悄替换旧候选序列。`research_prior` 记录配额、实际使用数、状态指标、来源检查、每个候选的转换结果和跳过原因。Hurst、IV、BBW 以及跨轮次冷却不在本轮实现。

## 计算预算与状态

任务级预算覆盖训练、邻域评分、峰值核验、展示补算、费用压力和样本外回放：

- `training_budget` 和 `validation_budget` 继续以候选参数点计。
- `reporting_budget` 默认 1000，可设为 0。
- `total_evaluation_budget` 在预检时按决策数和各阶段上界形成有限默认值，提交后冻结。
- `max_runtime_seconds` 默认 3600，内存继续受 `workspace_mb` 约束。
- 一次候选点同时评价网格与同注资持有两个账本，因此同时记录候选评价数和账本评价数。

展示目标在阶段开始时冻结为当时的收益领先点和待展示候选。新发现的更高收益邻点不会自动扩展下一轮展示目标。批次启动前预占评价额度，已启动的失败评价计入消耗；合法缓存命中不重复计费。预算或时间截断时保留已完成结果，记录停止原因，不补零，也不把有限搜索完成误写为全空间穷举完成。

任务状态、评分完整性、局部峰值核验和全局覆盖分别记录。`data_issue`、`insufficient_history`、`no_qualified`、`budget_exhausted`、`partial`、`cancelled` 和统计不适用不会共用同一个含混状态。

## 结果、补算与计算来源

结果摘要只保存协议、诊断、推荐和候选档案引用，普通读取不递归解压候选。候选保存为独立不可变 SQLite 档案；分页、排序、区域和切片查询在存储层执行，相同规范化候选集由不同角色复用引用。大型 JSON 和 CSV 使用流式读取。

结果 GET 只读取归档和补算状态，不加载行情、不调用评估器、不创建任务。缺失邻域通过 `research_completion` 后台任务补算：

```text
POST /api/v1/results/{result_id}/completion
```

请求指定决策分支、决策序号、目标和独立补算预算。同一结果、决策、目标和版本的并发提交合并成同一活动任务。每批保存新增候选、消耗、冻结目标和剩余工作；取消或异常后的恢复不重置已消耗预算。补算生成独立修订，不覆写原始研究，也不追溯改变历史推荐参数。

任务提交和执行时记录引擎版本、Git 提交、未提交修改指纹、参与计算的源码指纹、依赖锁哈希、Python 与数值依赖版本，以及独立的数据版本和配置指纹。补算要求源码、依赖、配置、数据版本、费用、窗口、参数空间和收益口径一致；缺失指纹、运行期间改码或依赖变化均拒绝混算。

## 本机同价窗口与固定窗口对比

使用本地锁定快照 `159611.SZ` / `mkt-55b86ebb66f6e417d7ce3623`，评估区间 2024-02-02 至 2026-10-21，种子 42。两次运行都使用训练 100、邻域 100、展示 20、总评价 1000 和 180 秒时限。

| 模式 | 训练窗口 | 锚日 | 匹配 / 回退 | 候选评价 | 已完成结果 |
| --- | --- | --- | --- | ---: | --- |
| 同价 | 2022-06-07 至 2024-02-01，408 日 | 2024-02-01 | 同价匹配 | 320 | 有限搜索完成，未找到合格参数 |
| 固定 | 2022-03-31 至 2024-02-01，450 日 | 2024-02-01 | 固定窗口 | 320 | 有限搜索完成，未找到合格参数 |

两种模式均未越过冻结的总评价上限。本样本没有合格推荐，因此不对哪种窗口收益更高作结论；原始对比见 [`v5-training-window-comparison.json`](v5-training-window-comparison.json)。

离线复现命令：

```powershell
.\.venv\Scripts\python.exe scripts/compare_training_windows.py `
  --symbol 159611.SZ `
  --version mkt-55b86ebb66f6e417d7ce3623 `
  --start 2024-02-02 --end 2026-10-21 `
  --training-budget 100 --validation-budget 100 --reporting-budget 20 `
  --total-evaluation-budget 1000 --max-runtime-seconds 180 --seed 42 `
  --output docs/v5-training-window-comparison.json
```

## 先验开关离线对照

`scripts/compare_research_priors.py` 是不写入应用任务或研究结果的离线对照。它从当前目录最近截止日向前检查，找出至少三只同时满足正式研究预检、活动快照可按版本重载和完整训练前置就绪的 ETF，再按代码取前三只；不会把 `limited`、不可重新加载的快照或无法覆盖训练期的数据混入。脚本要求完整的 250 个共同交易日评估区间，锁定每只 ETF 的同一数据版本，先运行 `off`，再运行 `daily_state_v1`；种子 42、搜索范围、费用、配置和 `selection_target=fixed` 全部相同。结果中的 `current` 与 `fixed` 共用评估开始前的同一决策锚，是同一选择的独立展示，不是最新行情推荐；本轮不要求自适应分支。

训练仍使用现有窗口策略和 250–500 日前置历史。若 1 分钟数据、日线、公司行动或质量账本不足以覆盖训练和评估区间，脚本记录具体缺口并停止该标的，不切换日线、不缩短历史，也不自动补数。默认预算为训练 100、邻域 200、展示 20、总评价 1000，每个先验分支单独限时 180 秒；提交前记录预检中的最小邻域、三区域压力评价上界和冻结总预算。

报告记录数据版本、完整配置及指纹、J 分数与合格候选数、候选评价数、先验配额和使用数、来源候选检查及失败原因、固定参数样本外配对超额、本金回撤、耗时、预算停止原因和 `partial` 状态。结果只写入 [`research-priors-comparison.json`](research-priors-comparison.json)，不预设先验会带来正收益。

2026-10-25 在 ADX/NATR golden 修复后运行了正式对照。最近可组成三只完整可运行样本的共同截止日为 2026-09-30；550 日回看窗口从 2025-03-29 开始时，去重归一后有 368 个可用共同交易日，但正式评估只取**最后 250 个共同交易日**，即 2025-09-18 至 2026-09-30。回看 368 日不是评估样本量。辅助 calendar payload 的 3 个增量尾部日期位于回看窗口之后；单标的日历和共同日历均无重复日、时分时间戳或周末日。所选标的为 `159611.SZ`、`159995.SZ`、`512880.SH`；`159516.SZ` 在 250 日统计预检可用，但 500 日前置训练质量不足，因而被排除，其他不可按版本重载的快照也记录了原因。

| 标的 | 策略 | 候选评价 | 训练 / 邻域 | planned / quota_used / actually_evaluated / annotation_only / skipped | J / 合格峰 | 样本外超额 / 回撤 | 状态 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 159611.SZ | `off` | 240 | 100 / 140 | — | 无 / 0 | 无 / 无 | `partial` |
| 159611.SZ | `daily_state_v1` | 240 | 100 / 140 | 2 / 1 / 2 / 1 / 0 | 无 / 0 | 无 / 无 | `partial` |
| 159995.SZ | `off` | 240 | 100 / 140 | — | 无 / 0 | 无 / 无 | `partial` |
| 159995.SZ | `daily_state_v1` | 240 | 100 / 140 | 3 / 3 / 3 / 0 / 1 | 无 / 0 | 无 / 无 | `partial` |
| 512880.SH | `off` | 240 | 100 / 140 | — | 无 / 0 | 无 / 无 | `partial` |
| 512880.SH | `daily_state_v1` | 240 | 100 / 140 | 4 / 3 / 4 / 1 / 0 | 无 / 0 | 无 / 无 | `partial` |

旧版把这组试运行中的 2/3/4 写成“使用”；它们实际是 `planned_count`。`quota_used` 才是真正注入初始采样并评价的坐标数，分别为 1/3/3；`actually_evaluated_count` 2/3/4 还包含已由原粗网格评价、仅附加来源标注的重复坐标，`annotation_only` 分别为 1/0/1。`159995.SZ` 的另一个振幅不对称候选因 `sell_step` 越界被跳过。

六个分支均未发现 `J>0` 的已完成局部峰，也没有可回放的合格推荐，因此样本外超额和最大本金回撤保持为空；不能据此判断先验优劣或收益方向。每次运行使用 240 个候选评价，均未达到 1000 总预算或 180 秒时限。`fixed` 与 `current` 的决策日均为 2025-09-17，六次运行都确认二者同锚，`current` 不是最新行情建议。所有先验 `phase=final`，`planned_count` 未超过配额 8；本轮不绑定 `source_etf`，所以 `source_seed_count=0`。预检的最小完整邻域为 81；三区域压力评价上界 1875 作为独立预算警告写入报告，但实际有效总预算仍受 1000 上限约束。完整结果和日历审计见 [`research-priors-comparison.md`](research-priors-comparison.md) 与 [`research-priors-comparison.json`](research-priors-comparison.json)。

完整复现命令：

```powershell
.\.venv\Scripts\python.exe scripts/compare_research_priors.py `
  --training-budget 100 --validation-budget 200 --reporting-budget 20 `
  --total-evaluation-budget 1000 --max-runtime-seconds 180 --seed 42 `
  --output docs/research-priors-comparison.json
```
