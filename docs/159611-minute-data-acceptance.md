# 159611 分钟行情只读验收

验收时间：2026-10-17T16:52:41.513266+08:00（本机时区）

活动 snapshot `4aa1d7dbe8a0d0d2ae3f9b19`，非演示数据。全量指纹前后相同：**True**；下载/同步调用：**0**。

原始分钟行 **276,946**，交易日 **1,154**，重复时间戳 **0**；来源分布为 import **1,147×240** = 275,280 行，Sina **7×238** = 1,666 行。这表示导入日各 240 条，Sina 后续日各 238 条；不能描述为每个 raw 交易日都有 240 条。

真实槽位空缺分布：`14:58` 7 日, `14:59` 7 日. 原始槽位数与合成占位数分开记录；结构性收盘集合竞价候选槽不计入普通分钟缺口。

原始记录含全部 240 个网格时间戳的最后交易日为 **2026-09-30**；排除 14:58/14:59 结构性收盘竞价槽后完整的最后交易日为 **2026-10-16**。

重算覆盖：分钟日 **1154** / 总交易日 **1154**，分钟比例 **1.0**，收盘竞价占位 **14**，实际有效分钟缺口 **0**。

已保存研究 `1d88cb912cb84d5f84e52719b6b381a5` 样本外区间 2025-10-16～2026-10-16，锁定 snapshot 与当前相同。1m fixed/adaptive 参数回放结果一致：**True / True**。未执行参数搜索。

| 检查 | 结果 |
|---|---:|
| real_non_demo_snapshot | 通过 |
| current_version_ready | 通过 |
| locked_ensure_bundle_version | 通过 |
| locked_ensure_matches_load | 通过 |
| coverage_has_minute_data | 通过 |
| coverage_matches_bundle_ratio | 通过 |
| effective_missing_minutes_zero | 通过 |
| auction_placeholders_14 | 通过 |
| marketbars_locked_to_active_version | 通过 |
| marketbars_quality_matches_coverage | 通过 |
| saved_study_uses_active_snapshot | 通过 |
| public_result_uses_locked_snapshot | 通过 |
| public_result_quality_matches_coverage | 通过 |
| public_result_has_import_source_span | 通过 |
| public_root_nested_status_consistent | 通过 |
| fixed_replay_equal | 通过 |
| adaptive_replay_equal | 通过 |
| snapshot_unchanged | 通过 |
| readonly_facade_suppressed_all_cache_writes | 通过 |
| no_sync_calls | 通过 |
| fixed_replay_metrics | 通过 |
| fixed_replay_equity | 通过 |
| fixed_replay_trades | 通过 |
| fixed_replay_rejections | 通过 |
| fixed_replay_period_returns | 通过 |
| adaptive_replay_metrics | 通过 |
| adaptive_replay_equity | 通过 |
| adaptive_replay_trades | 通过 |
| adaptive_replay_rejections | 通过 |
| adaptive_replay_period_returns | 通过 |

本记录由 `scripts/validate_minute_data.py` 从现有本地 snapshot 和保存研究生成；脚本只读取 SQLite/Parquet/JSON，缓存写入被抑制，只更新本验收记录。
