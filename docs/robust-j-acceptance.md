# 稳健 J v4 真实日线验收

> 历史资料：本记录对应 `protocol_version=4`，其旧候选存储和结果页面不可由当前引擎接续。现行收益、风险、预算和补算契约见[参数研究 v5](parameter-research-v5.md)。

验收日期：2026-10-20（Asia/Shanghai）  
标的：`159611.SZ`  
数据版本：`mkt-2fe613b32b3cb6f36535ca00`  
评估区间：2026-04-17 至 2026-10-16  
协议：`schema_version=3`、`protocol_version=4`、`strategy_version=4`

## 固定样本

本次评估 33 个合法中心。训练预算 100、邻域预算 2000、Top-K 3。搜索器选择 `exhaustive`；合法空间不超过训练预算且满足工作内存上限时自动完整枚举。

训练窗口取 2024-09-30 至 2026-10-16，共 493 个交易日。窗口参数为 250–500 日，起点原因 `same_price_farthest`，回退状态 `False`。

## 结果

| 项目 | 结果 |
|---|---|
| 研究状态 | `completed` |
| 固定方案状态 | `no_qualified` |
| 完整评价中心 | 33 |
| 完整 J 中心 | 33 |
| 合格中心 | 0 |
| 正式推荐峰区 | 0 |
| 评分完整 | true |
| 局部峰值已核验 | false |
| 全局空间已覆盖 | true |
| 训练唯一点 | 33 |
| 邻域新增唯一点 | 0 |
| 训练时段准备 | 1.49 秒 |
| 研究计算 | 1.15 秒 |

本次没有中心同时满足交易、风险、`J>0`、持仓资格和完整平台核验，因此正式推荐为空，方案保持暂停交易语义。三个状态分别输出：评分完整、局部峰值核验和全局空间覆盖互不替代；没有把诊断中心当作正式执行参数。

原始结果：[robust-j-acceptance.json](robust-j-acceptance.json)。

## 覆盖的边界

- 四轴当前坐标顺序为 `buy_step → sell_step → buy_shares → sell_shares`。
- 新版强制自然持仓，结果不把训练期观察最大持仓转成可执行护栏。
- 所有合法中心的 J 均完整计算；没有用中心 `R` 过滤候选。
- 平台没有合格入口，所以没有生成推荐；这与预算不足或评分失败是不同状态。
- 核验脚本为 [acceptance_robust_j.py](../scripts/acceptance_robust_j.py)。

盈利不是验收条件。本次证明新版真实数据路径、窗口、四维枚举、J 计算和结果状态已接通；若要评价搜索算法质量，应使用 [robust-search-benchmark.md](robust-search-benchmark.md) 中同一目标的参考峰区与召回结果。
