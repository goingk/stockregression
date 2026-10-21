"""Reproducible search-layer benchmark for the robust-J candidate finder."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockregression.models import BacktestConfig, OptimizationConfig, ResearchStrategySpec
from stockregression.numerics.robust import PeakAnalyzer, eligible, robust_score, tensor_robust_scores
from stockregression.research.robust import (
    EvaluationSession,
    _initial_coordinates,
    _local_search,
    _score_sort,
    _start_coordinates,
    _verify_peak,
)
from stockregression.research_v4.space import ParameterSpace

K = 3
TOL = 1e-12


def make_config(training_budget: int, validation_budget: int) -> OptimizationConfig:
    base = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date="2025-01-02",
        frequency="1d", initial_shares=1000, min_shares=0, max_shares=None,
        buy_shares=100, sell_shares=100, grid_mode="geometric",
        buy_step=.03, sell_step=.03,
    )
    strategy = ResearchStrategySpec.model_validate({
        "strategy_version": 5, "search_method": "hierarchical", "ranking_mode": "robust_j",
        "granularity": {"buy_step": .001, "sell_step": .001, "buy_shares": 100, "sell_shares": 100},
        "hypercube_edge": 3, "sigma": 1.0, "alpha": .30, "beta": 1.20, "tau": .01,
        "eta_limit": 2.0, "eta_hard_cap": 3.5, "gamma_max": .40,
        "eta_floor": 0.0, "gamma_min": 0.0, "floor_k_ref": 10.0, "peak_tolerance": TOL,
        "training_budget": training_budget, "validation_budget": validation_budget,
        "recommendation_count": K, "start_min": 20, "start_max": 64,
        "starts_per_candidate": 4, "initial_strides": [1, 2],
        "stride_multiplier": 2, "max_stride_fraction": .25,
        "validation_reserve_fraction": .25, "stress_region_count": 0,
    })
    return OptimizationConfig(
        base=base, protocol_version=5, strategy=strategy, selection_target="fixed",
        ranges={
            "buy_step": {"low": .001, "high": .007},
            "sell_step": {"low": .001, "high": .007},
            "buy_shares": {"low": 100, "high": 300},
            "sell_shares": {"low": 100, "high": 300},
        },
    )


def gaussian(coord: tuple[int, ...], center: tuple[int, ...], width: float) -> float:
    distance = sum((left - right) ** 2 for left, right in zip(coord, center))
    return math.exp(-distance / (2 * width * width))


def surface(scenario: str, coordinate: tuple[int, ...]) -> tuple[float, float, int]:
    x = tuple((value - 1) / 6 for value in coordinate[:2]) + tuple((value - 1) / 2 for value in coordinate[2:])
    value = .002
    max_position = 1.0
    if scenario == "narrow_peak":
        value += .060 * gaussian(coordinate, (4, 4, 2, 2), .55)
    elif scenario == "wide_peak":
        value += .050 * gaussian(coordinate, (3, 4, 2, 2), 1.45)
    elif scenario == "boundary_peak":
        value += .055 * gaussian(coordinate, (1, 7, 1, 3), .75)
    elif scenario == "multi_peak":
        value += .060 * gaussian(coordinate, (2, 2, 1, 3), .65)
        value += .050 * gaussian(coordinate, (6, 6, 3, 1), .85)
        value += .025 * gaussian(coordinate, (1, 7, 3, 3), .60)
    elif scenario == "constraint_split":
        value += .060 * gaussian(coordinate, (2, 6, 1, 3), .70)
        if x[0] < .42 and x[1] > .50:
            max_position = 4.0
    elif scenario == "flat":
        value = .020
    else:
        raise ValueError("unknown scenario")
    return float(value), max_position, 20


def row(space: ParameterSpace, coordinate: tuple[int, ...], scenario: str) -> dict[str, Any]:
    excess, max_position, trades = surface(scenario, coordinate)
    return {
        "params": space.params(coordinate), "key": coordinate, "coordinate": list(coordinate),
        "evaluation_status": "completed", "training_excess_return": excess,
        "training_drawdown": 0.0 if coordinate[0] == coordinate[1] else .05,
        "trade_count": trades, "buy_count": trades // 2, "sell_count": trades // 2,
        "actual_max_pos_ratio": max_position,
        "actual_min_pos_ratio": 0.0 if coordinate[0] % 2 else 1.0,
        "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
    }


@dataclass
class SyntheticEvaluator:
    space: ParameterSpace
    scenario: str
    chunk_size: int = 64

    def evaluate_training(self, coordinates: list[tuple[int, ...]]):
        return [row(self.space, coordinate, self.scenario) for coordinate in coordinates]


def exhaustive_reference(config: OptimizationConfig, space: ParameterSpace, scenario: str):
    started = time.perf_counter()
    rows = {coordinate: row(space, coordinate, scenario) for coordinate in space.iterate()}
    scores = tensor_robust_scores(space, rows, config.strategy)
    candidates = [
        coordinate for coordinate, score in scores.items()
        if eligible(rows.get(coordinate), score, max_drawdown=config.max_drawdown, min_trades=config.min_trades, strategy=config.strategy)
    ]
    candidates.sort(key=lambda coordinate: _score_sort(scores[coordinate], coordinate))
    analyzer = PeakAnalyzer(space, TOL)
    peaks = {}
    known_plateau = set()
    for coordinate in candidates:
        if coordinate in known_plateau:
            continue
        analysis = analyzer.inspect(
            coordinate,
            lambda point: scores.get(point),
            lambda point: eligible(rows.get(point), scores.get(point), max_drawdown=config.max_drawdown, min_trades=config.min_trades, strategy=config.strategy),
        )
        if analysis.complete and analysis.representative is not None:
            peaks.setdefault(analysis.representative, coordinate)
            known_plateau.update(analysis.plateau)
    ordered = sorted(peaks, key=lambda point: _score_sort(scores[point], point))
    return {
        "peaks": ordered,
        "best_j": float(scores[ordered[0]].j_score) if ordered else None,
        "scored_points": len(rows),
        "seconds": time.perf_counter() - started,
    }


def sparse_search(config: OptimizationConfig, space: ParameterSpace, scenario: str):
    started = time.perf_counter()
    evaluator = SyntheticEvaluator(space, scenario)
    session = EvaluationSession(evaluator, space, config, None)
    initial = _initial_coordinates(space, config, "2025-01-02", {})
    session.ensure(initial, "training", int(config.strategy.training_budget))
    target_starts = min(
        int(config.strategy.start_max),
        max(int(config.strategy.start_min), int(config.strategy.starts_per_candidate) * K),
        len(session.rows),
    )
    starts = _start_coordinates(session, space, target_starts)
    search_cap = max(0, int(math.floor(int(config.strategy.validation_budget) * (1 - float(config.strategy.validation_reserve_fraction)))))
    _local_search(session, space, starts, search_cap)
    for coordinate in session.rows:
        if coordinate not in session.scores:
            session.scores[coordinate] = robust_score(coordinate, session.rows, space, config.strategy)
    analyzer = PeakAnalyzer(space, TOL)
    candidates = [
        coordinate for coordinate, score in session.scores.items()
        if eligible(session.rows.get(coordinate), score, max_drawdown=config.max_drawdown, min_trades=config.min_trades, strategy=config.strategy)
    ]
    candidates.sort(key=lambda coordinate: _score_sort(session.scores[coordinate], coordinate))
    peaks = {}
    incomplete = 0
    for coordinate in candidates:
        if len(peaks) >= K or session.spent["validation"] >= int(config.strategy.validation_budget):
            break
        analysis = _verify_peak(session, analyzer, coordinate, int(config.strategy.validation_budget))
        if analysis.complete and analysis.representative is not None:
            peaks.setdefault(analysis.representative, coordinate)
        elif not analysis.complete:
            incomplete += 1
    ordered = sorted(peaks, key=lambda point: _score_sort(session.scores[point], point))
    return {
        "peaks": ordered,
        "best_j": float(session.scores[ordered[0]].j_score) if ordered else None,
        "training_points": session.spent["training"],
        "neighborhood_points": session.spent["validation"],
        "unique_backtests": session.attempts,
        "cache_hits": max(0, session.requested - session.attempts),
        "start_count": len(starts),
        "incomplete_verifications": incomplete,
        "seconds": time.perf_counter() - started,
    }


def benchmark(training_budget: int, validation_budget: int) -> dict[str, Any]:
    config = make_config(training_budget, validation_budget)
    space = ParameterSpace(config)
    scenarios = []
    for scenario in ("narrow_peak", "wide_peak", "boundary_peak", "multi_peak", "constraint_split", "flat"):
        reference = exhaustive_reference(config, space, scenario)
        sparse = sparse_search(config, space, scenario)
        wanted = set(reference["peaks"][:K])
        found = set(sparse["peaks"])
        matched = found & wanted
        recall = len(matched) / len(wanted) if wanted else 1.0
        best = reference["best_j"]
        sparse_best = sparse["best_j"]
        regret = max(0.0, best - sparse_best) if best is not None and sparse_best is not None else None
        scenarios.append({
            "name": scenario,
            "reference_peak_count": len(reference["peaks"]),
            "reference_top_k": [list(point) for point in reference["peaks"][:K]],
            "sparse_top_k": [list(point) for point in sparse["peaks"]],
            "top_k_recall": recall,
            "best_j_regret": regret,
            "reference_scored_points": reference["scored_points"],
            "sparse": sparse,
            "reference_seconds": reference["seconds"],
        })
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "space": {
            "axes": ["buy_step", "sell_step", "buy_shares", "sell_shares"],
            "shape": [7, 7, 3, 3], "legal_points": space.size(),
            "hypercube_edge": 3, "sigma": 1.0, "top_k": K,
        },
        "budgets": {
            "hierarchical_training": training_budget,
            "hierarchical_neighborhood": validation_budget,
            "exhaustive_reference": space.size(),
            "cost_model": "one unique synthetic point equals one raw backtest",
        },
        "scenarios": scenarios,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 稳健 J 搜索层固定样本基准",
        "",
        "本文件由 `scripts/benchmark_robust_search.py` 生成。完整枚举使用全部合法中心作为参考；分层搜索使用生产版 Sobol 覆盖、多来源起点、轮转局部搜索和峰值核验。两者共用同一个四维空间、J、平台定义和单位回测成本。",
        "",
        "这里使用合成多峰收益场验证搜索层，不替代真实行情的回测验收。合成点不包含真实成交延迟或滑点；其作用是让最优峰区和搜索召回可以精确计算。",
        "",
        f"- 空间：{payload['space']['shape']}，合法中心 {payload['space']['legal_points']}",
        f"- 四轴：{', '.join(payload['space']['axes'])}",
        f"- 分层训练 / 邻域预算：{payload['budgets']['hierarchical_training']} / {payload['budgets']['hierarchical_neighborhood']}",
        "- 完整枚举参考成本：" + str(payload['budgets']['exhaustive_reference']) + " 个唯一点",
        "",
        "| 场景 | 参考峰数 | Top-K 召回 | 最优 J 遗憾 | 分层唯一点 | 缓存命中 | 起点 | 核验未闭合 | 分层耗时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["scenarios"]:
        sparse = item["sparse"]
        regret = item["best_j_regret"]
        lines.append(
            "| {name} | {reference_peak_count} | {recall:.0%} | {regret} | {points} | {hits} | {starts} | {incomplete} | {seconds:.3f}s |".format(
                name=item["name"], reference_peak_count=item["reference_peak_count"],
                recall=item["top_k_recall"],
                regret="—" if regret is None else f"{regret:.8f}",
                points=sparse["unique_backtests"], hits=sparse["cache_hits"],
                starts=sparse["start_count"], incomplete=sparse["incomplete_verifications"],
                seconds=sparse["seconds"],
            )
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- 稀疏搜索可能遗漏全局峰；本基准只验证它输出的局部峰与完整计算定义一致。",
        "- 参考峰按完整一格联合邻域和合格中心可行域计算，平台去重后再取 Top-K。",
        "- 分层运行的邻域预算包含局部搜索与最终核验；缓存命中不重复计入唯一回测点。",
        "- 小生境差分进化和多种群粒子群尚未实现，因此本表不用于宣称它们与当前方法的绝对排名。",
        "",
        "原始数据保存在 `docs/robust-search-benchmark.json`。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-budget", type=int, default=72)
    parser.add_argument("--validation-budget", type=int, default=360)
    parser.add_argument("--json", default=str(ROOT / "docs" / "robust-search-benchmark.json"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "robust-search-benchmark.md"))
    args = parser.parse_args()
    payload = benchmark(args.training_budget, args.validation_budget)
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown).write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
