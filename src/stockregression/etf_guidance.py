"""Explainable search hints derived only from an ETF's frozen statistics.

These are starting points for a subsequent study, never optimized parameters
or estimates of achievable excess return. No replay performance is consumed.
"""
from __future__ import annotations

import math
from datetime import date

from .etf_models import EtfResearchRequest


GUIDANCE_VERSION = "etf_parameter_guidance_v1"
UNIT = .0001


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _ceil(value, unit=UNIT):
    return round(math.ceil(value / unit - 1e-10) * unit, 8)


def parameter_guidance(row: dict, request: EtfResearchRequest) -> dict:
    """Build a versioned, JSON-safe recommendation without reading live data."""
    as_of = row.get("window_end")
    output = {
        "method_version": GUIDANCE_VERSION, "as_of": as_of,
        "data_version": row.get("version"), "status": "unavailable",
        "reason": None, "ranges": {}, "granularity": {}, "tau": None,
        "evidence": {}, "notes": [],
    }

    def unavailable(reason):
        output["reason"] = reason
        return output

    try:
        date.fromisoformat(as_of)
    except (TypeError, ValueError):
        return unavailable("缺少有效的历史截止日，无法给出参数建议。")

    state = row.get("market_state") or {}
    diagnostics = row.get("diagnostics") or {}
    for stamp in (state.get("as_of"), diagnostics.get("window_end"),
                  (state.get("samples") or {}).get("selected_end"),
                  (row.get("grid_basis") or {}).get("current_price_date")):
        if stamp:
            try:
                if date.fromisoformat(stamp) > date.fromisoformat(as_of):
                    return unavailable("统计样本晚于研究截止日，不能用于参数建议。")
            except (TypeError, ValueError):
                return unavailable("统计样本日期无效，不能用于参数建议。")
    basis = row.get("grid_basis") or {}
    metrics = state.get("metrics") or {}
    details = state.get("metric_details") or {}
    atr30 = _number(basis.get("atr30_pct")) if row.get("status") == "completed" else None
    natr = _number(metrics.get("natr14"))
    amplitude = _number(metrics.get("amplitude_q75"))
    hv = details.get("hv_ratio") or {}
    sigma = _number(hv.get("hv60"))
    samples = int(_number(hv.get("sample_count")) or 0)
    if not state.get("as_of") or sigma is None or sigma <= 0 or samples < 60:
        return unavailable("需要截止日前至少 60 个连续有效日收益样本，才能估计波动尺度和安全垫。")
    if not natr or not amplitude:
        return unavailable("日线真实波幅或振幅样本不足，暂不生成搜索范围。")
    price = _number((row.get("metrics") or {}).get("mean_raw_price"))
    if not price:
        price = _number(basis.get("current_raw_price_unadjusted"))
    tick = _number(basis.get("price_tick"))
    if not price or not tick:
        return unavailable("缺少可信的未复权价格或最小报价单位，无法核算费用与间距下限。")

    # Use the minimum ETF order (100 units), conservatively allowing for the
    # minimum commission even when the user's subsequent search varies size.
    fees = request.execution.fees
    fee_floor = 2 * (max(fees.commission_rate, fees.min_commission / (100 * price))
                     + fees.slippage_bps / 10_000)
    tick_ratio = tick / price
    volatility_floor = 1.2 * atr30 if atr30 and atr30 > 0 else .5 * natr
    lower = max(volatility_floor, 5 * tick_ratio, fee_floor, UNIT)
    upper = min(.10, max(3 * lower, 2 * natr, 2 * amplitude))
    if lower >= upper:
        return unavailable("费用、报价刻度或波动要求的最小间距已达到 10% 探索上限；请单独检查数据和费用。")
    # Around 25 points per spacing axis, with no precision finer than a tick.
    step = _ceil(max(tick_ratio, (upper - lower) / 24, UNIT))
    low = _ceil(lower, step)
    high = round(math.floor(upper / step + 1e-10) * step, 8)
    if high - low < 2 * step - 1e-10:
        return unavailable("按可成交价格刻度对齐后不足三个间距点，暂不能建议稳健搜索范围。")

    # A deliberately modest hurdle, expressed on a 250-session horizon. The
    # 10% coefficient is a policy choice, NOT a confidence level or forecast.
    horizon = 250
    volatility_buffer = .10 * sigma * math.sqrt(horizon)
    tau = _ceil(max(fee_floor, volatility_buffer), .001)
    if tau > .10:
        return unavailable("历史波动或交易费用异常偏高，自动安全垫超过 10%，请人工核查后设定。")
    output.update(
        status="limited" if not atr30 or row.get("data_status") == "limited" else "ready",
        ranges={name: {"low": low, "high": high} for name in ("buy_step", "sell_step")},
        granularity={"buy_step": step, "sell_step": step, "buy_shares": 100, "sell_shares": 100},
        tau={
            "value": tau, "sensitivity": sorted(set([_ceil(tau * .5, .001), tau, _ceil(tau * 1.5, .001)])),
            "reference_days": horizon,
            "reason": "取 100 份往返费用率与 10% × 60 日日收益标准差 × √250 的较大值，向上对齐 0.1 个百分点。",
        },
        evidence={
            "daily_return_samples": samples, "daily_sample_start": (state.get("samples") or {}).get("selected_start"),
            "daily_sample_end": (state.get("samples") or {}).get("selected_end"),
            "daily_volatility": sigma, "natr14": natr, "amplitude_q75": amplitude,
            "atr30_pct": atr30, "raw_price": price, "price_tick": tick,
            "tick_ratio": tick_ratio, "round_trip_cost": fee_floor, "cost_order_shares": 100,
            "spacing_points": int(round((high - low) / step)) + 1,
        },
        notes=[
            "间距下限取波幅尺度、5 个报价刻度及往返费用率的最大值；上限取下限的 3 倍、2 倍 NATR14、2 倍日振幅 Q75 的最大值（最多 10%）。",
            "间距细粒度按约 25 个采样点及至少一个报价刻度设置；数量细粒度为 ETF 最小交易单位 100 份，数量范围保留研究草稿设置。",
            "τ 是整段评价区间的峰值本金差值超额收益门槛，不是单日涨跌幅或收益预测；250 日与 10% 系数是启发式假设，训练跨度改变时需重新比较敏感性值。",
            "只用于截止日之后的参数探索；尚未验证最优性，后续须通过样本外研究。风险偏好、评分权重、持仓上限和运行预算仍由研究设置决定。",
        ],
    )
    if output["status"] == "limited":
        output["notes"].append("分钟样本受限时使用日线波幅辅助建议，不能替代分钟成交验证。")
    return output
