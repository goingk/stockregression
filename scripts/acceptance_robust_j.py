"""Run a small real-data acceptance for the four-axis robust-J protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockregression.data import DataService
from stockregression.models import BacktestConfig, OptimizationConfig
from stockregression.research_preparation import prepare_research_bundle
from stockregression.scientific import optimize_scientific
from stockregression.settings import load_settings
from stockregression.storage import Store


def render_markdown(payload: dict) -> str:
    diagnostics = payload["diagnostics"]
    window = diagnostics.get("training_window") or {}
    timing = payload["timing_seconds"]
    lines = [
        "# 稳健 J v4 真实日线验收",
        "",
        f"验收日期：{payload['generated_at'][:10]}（Asia/Shanghai）  ",
        f"标的：`{payload['symbol']}`  ",
        f"数据版本：`{payload['data_version']}`  ",
        f"评估区间：{payload['evaluation']['start']} 至 {payload['evaluation']['end']}  ",
        f"协议：`schema_version={payload['strategy']['schema_version']}`、"
        f"`protocol_version={payload['strategy']['protocol_version']}`、"
        f"`strategy_version={payload['strategy']['strategy_version']}`",
        "",
        "## 固定样本",
        "",
        f"本次评估 {diagnostics.get('legal_combination_count')} 个合法中心。训练预算 "
        f"{payload['budgets']['training']}、邻域预算 {payload['budgets']['validation']}、"
        f"Top-K {payload['strategy']['recommendation_count']}。"
        f"搜索器选择 `{diagnostics.get('search_method')}`；合法空间不超过训练预算且满足工作内存上限时自动完整枚举。",
        "",
        f"训练窗口取 {window.get('start')} 至 {window.get('end')}，共 {window.get('days')} 个交易日。"
        f"窗口参数为 {window.get('min_offset_trading_days')}–{window.get('max_offset_trading_days')} 日，"
        f"起点原因 `{window.get('start_reason')}`，回退状态 `{window.get('fallback')}`。",
        "",
        "## 结果",
        "",
        "| 项目 | 结果 |",
        "|---|---|",
        f"| 研究状态 | `{payload['status']}` |",
        f"| 固定方案状态 | `{payload['fixed_status']}` |",
        f"| 完整评价中心 | {diagnostics.get('training_unique_count')} |",
        f"| 完整 J 中心 | {diagnostics.get('scored_center_count')} |",
        f"| 合格中心 | {diagnostics.get('eligible_center_count')} |",
        f"| 正式推荐峰区 | {payload['recommendation_count']} |",
        f"| 评分完整 | {str(diagnostics.get('score_complete')).lower()} |",
        f"| 局部峰值已核验 | {str(diagnostics.get('local_peak_verified')).lower()} |",
        f"| 全局空间已覆盖 | {str(diagnostics.get('global_exhaustive')).lower()} |",
        f"| 训练唯一点 | {diagnostics.get('training_unique_count')} |",
        f"| 邻域新增唯一点 | {diagnostics.get('neighborhood_unique_count')} |",
        f"| 训练时段准备 | {timing['data_preparation']:.2f} 秒 |",
        f"| 研究计算 | {timing['research']:.2f} 秒 |",
        "",
    ]
    if payload["recommendation_count"]:
        summary = "本次存在正式推荐峰区；原始收益峰仍只作诊断，不会替代 J 峰值。"
    else:
        summary = (
            "本次没有中心同时满足交易、风险、`J>0`、持仓资格和完整平台核验，因此正式推荐为空，"
            "方案保持暂停交易语义。三个状态分别输出：评分完整、局部峰值核验和全局空间覆盖互不替代；"
            "没有把诊断中心当作正式执行参数。"
        )
    lines += [
        summary,
        "",
        "原始结果：[robust-j-acceptance.json](robust-j-acceptance.json)。",
        "",
        "## 覆盖的边界",
        "",
        "- 四轴当前坐标顺序为 `buy_step → sell_step → buy_shares → sell_shares`。",
        "- 新版强制自然持仓，结果不把训练期观察最大持仓转成可执行护栏。",
        "- 所有合法中心的 J 均完整计算；没有用中心 `R` 过滤候选。"
        if diagnostics.get("global_exhaustive") else
        "- 稀疏搜索只认证实际完整评分的中心；未覆盖空间会明确报告为搜索覆盖不足。",
        "- 平台没有合格入口，所以没有生成推荐；这与预算不足或评分失败是不同状态。",
        "- 核验脚本为 [acceptance_robust_j.py](../scripts/acceptance_robust_j.py)。",
        "",
        "盈利不是验收条件。本次证明新版真实数据路径、窗口、四维枚举、J 计算和结果状态已接通；"
        "若要评价搜索算法质量，应使用 [robust-search-benchmark.md](robust-search-benchmark.md) 中"
        "同一目标的参考峰区与召回结果。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="159611.SZ")
    parser.add_argument("--start", default="2026-04-17")
    parser.add_argument("--end", default="2026-10-16")
    parser.add_argument("--frequency", default="1d", choices=("1d", "1m"))
    parser.add_argument("--training-budget", type=int, default=100)
    parser.add_argument("--validation-budget", type=int, default=2000)
    parser.add_argument("--output", default=str(ROOT / "docs" / "robust-j-acceptance.json"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "robust-j-acceptance.md"))
    args = parser.parse_args()
    settings = load_settings()
    store = Store(settings.data_dir)
    manifest = store.snapshot(args.symbol)
    if not manifest or manifest.get("metadata", {}).get("is_demo"):
        raise SystemExit("需要已有真实数据快照")
    base = BacktestConfig(
        schema_version=3, symbol=args.symbol, start_date=args.start, end_date=args.end,
        frequency=args.frequency, initial_shares=1000, min_shares=0, max_shares=None,
        buy_shares=1000, sell_shares=1000, grid_mode="geometric",
        buy_step=.015, sell_step=.015, fees=settings.fee_defaults.model_dump(),
    )
    config = OptimizationConfig.model_validate({
        "base": base.model_dump(mode="python"), "protocol_version": 4, "selection_target": "fixed",
        "bootstrap_reps": 200, "min_trades": 1, "max_drawdown": .5, "min_minute_ratio": 0,
        "ranges": {
            "buy_step": {"low": .015, "high": .016},
            "sell_step": {"low": .015, "high": .015},
            "buy_shares": {"low": 900, "high": 1100},
            "sell_shares": {"low": 1000, "high": 1000},
        },
        "strategy": {
            "strategy_version": 4, "search_method": "auto", "ranking_mode": "fixed_mean",
            "granularity": {"buy_step": .0001, "sell_step": .0001, "buy_shares": 100, "sell_shares": 100},
            "hypercube_edge": 3, "sigma": 1, "alpha": .3, "beta": 1.2, "tau": .01,
            "eta_limit": 2, "eta_hard_cap": 3.5, "gamma_max": .4,
            "training_budget": args.training_budget, "validation_budget": args.validation_budget,
            "recommendation_count": 3, "training_min_days": 250, "training_max_days": 500,
            "stress_region_count": 3,
        },
    })
    started = time.perf_counter()
    bundle = prepare_research_bundle(
        DataService(store), base, manifest["version"], lookback_days=config.strategy.training_max_days,
    )
    prepared = time.perf_counter() - started
    research_started = time.perf_counter()
    result = optimize_scientific(config, bundle)
    seconds = time.perf_counter() - research_started
    fixed = result.get("fixed") or {}
    current = fixed.get("current") or {}
    diagnostics = current.get("diagnostics") or fixed.get("diagnostics") or {}
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "symbol": args.symbol, "frequency": args.frequency, "data_version": bundle.version,
        "evaluation": {"start": args.start, "end": args.end},
        "strategy": {
            "schema_version": config.base.schema_version,
            "protocol_version": config.protocol_version,
            "strategy_version": config.strategy.strategy_version,
            "strategy_id": config.strategy.strategy_id,
            "recommendation_count": config.strategy.recommendation_count,
            "training_min_days": config.strategy.training_min_days,
            "training_max_days": config.strategy.training_max_days,
        },
        "budgets": {"training": args.training_budget, "validation": args.validation_budget},
        "status": result.get("status"),
        "fixed_status": fixed.get("status"),
        "recommendation_status": fixed.get("recommendation_status"),
        "recommendation_count": len(fixed.get("recommendation_regions") or []),
        "diagnostics": {
            key: diagnostics.get(key) for key in (
                "geometry", "strategy_id", "strategy_version", "search_method",
                "legal_combination_count", "training_unique_count", "neighborhood_unique_count",
                "scored_center_count", "eligible_center_count", "score_complete",
                "local_peak_verified", "global_exhaustive", "coverage_status",
                "training_window", "platforms", "computation",
            )
        },
        "regions": [{
            "region_id": item.get("region_id"), "center_coordinate": item.get("center_coordinate"),
            "center_params": item.get("center_params"), "q_buy": item.get("q_buy"), "q_sell": item.get("q_sell"),
            "j_score": item.get("j_score"), "center_excess_return": item.get("center_excess_return"),
            "neighborhood_mean": item.get("mean_excess_return"), "downside_risk": item.get("downside_risk_tau"),
            "position_penalty": item.get("pos_penalty"), "actual_min_pos_ratio": item.get("actual_min_pos_ratio"),
            "actual_max_pos_ratio": item.get("actual_max_pos_ratio"), "stockout_count": item.get("stockout_count"),
            "tplus1_block_count": item.get("tplus1_block_count"),
            "score_complete": item.get("score_complete"), "local_peak_verified": item.get("local_peak_verified"),
            "platform_type": item.get("platform_type"), "platform_size": item.get("platform_size"),
            "global_exhaustive": item.get("global_exhaustive"), "stress": item.get("stress"),
        } for item in fixed.get("recommendation_regions") or []],
        "timing_seconds": {"data_preparation": prepared, "research": seconds},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
