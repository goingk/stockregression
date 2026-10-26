"""Resolve ETF source references into immutable, server-owned research inputs."""
from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from typing import Any


class ResearchOriginError(ValueError):
    def __init__(self, message: str, *, status: int = 422):
        super().__init__(message)
        self.status = status


def origin_fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, default=str).encode("utf-8")).hexdigest()


def resolve_research_origin(store, config) -> dict | None:
    """Never trust a caller's resolved origin or reread it in a worker."""
    reference = config.source_etf
    if reference is None:
        return None
    try:
        original = store.result(reference.result_id)
    except (OSError, ValueError) as exc:
        raise ResearchOriginError("来源 ETF 研究结果不可读取，请解除来源后继续独立研究", status=404) from exc
    if not original or original.get("kind") != "etf_research":
        raise ResearchOriginError("来源 ETF 研究结果不存在，请解除来源后继续独立研究", status=404)
    symbol = config.base.symbol
    row = next((item for item in original.get("rows", []) if item.get("symbol") == symbol), None)
    if row is None:
        raise ResearchOriginError("研究标的不在来源 ETF 研究中", status=409)
    try:
        as_of = date.fromisoformat(str(row.get("window_end") or original.get("as_of"))).isoformat()
    except ValueError as exc:
        raise ResearchOriginError("来源 ETF 研究缺少有效截止日") from exc
    guidance = row.get("parameter_guidance") or {}
    if reference.apply_parameter_guidance:
        if (guidance.get("status") not in {"ready", "limited"}
                or guidance.get("as_of") != as_of
                or guidance.get("data_version") != row.get("version")):
            raise ResearchOriginError("来源 ETF 缺少同一数据版本的有效历史参数建议，请重新运行 ETF 研究")
        if config.base.start_date <= date.fromisoformat(as_of):
            raise ResearchOriginError("应用 ETF 历史参数建议时，研究开始日必须晚于建议截止日，避免未来信息泄漏")
    available = (row.get("validation") or {}).get("candidates") or []
    by_id = {item.get("id"): item for item in available if isinstance(item, dict)}
    if reference.candidate_ids is None:
        selected = [item for item in available if item.get("qualified")][:3]
    else:
        if any(key not in by_id for key in reference.candidate_ids):
            raise ResearchOriginError("指定候选不属于来源 ETF 研究")
        selected = [by_id[key] for key in dict.fromkeys(reference.candidate_ids)]
    execution = (original.get("request") or {}).get("execution") or {}
    seeds = []
    for item in selected:
        replay = item.get("replay_config") or {}
        params = {key: item.get(key, replay.get(key, execution.get(key)))
                  for key in ("buy_step", "sell_step", "buy_shares", "sell_shares")}
        try:
            if any(isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0
                   for value in params.values()):
                raise ValueError("invalid parameter")
            if any(float(params[key]) % 100 for key in ("buy_shares", "sell_shares")):
                raise ValueError("invalid quantity")
        except (ValueError, TypeError) as exc:
            raise ResearchOriginError("来源候选缺少有效的四轴参数") from exc
        seeds.append({"candidate_id": item["id"], "params": params})
    origin = {
        "schema_version": 1, "etf_result_id": reference.result_id, "symbol": symbol,
        "method_version": original.get("method_version"), "source_as_of": as_of,
        "source_data_version": row.get("version") or (
            ((original.get("universe") or {}).get("versions") or {}).get(symbol) or {}).get("version"),
        "source_window_start": row.get("window_start"),
        "gss_score": row.get("gss_score"), "source_metrics": row.get("metrics") or {},
        "market_state": row.get("market_state"), "seeds": seeds,
    }
    if row.get("parameter_guidance"):
        origin["parameter_guidance"] = row["parameter_guidance"]
    if reference.apply_parameter_guidance:
        origin["applied_parameter_guidance"] = True
    origin["source_hash"] = origin_fingerprint(origin)
    return origin


def prior_contract_hash(config) -> str:
    """Bind rules and the resolved source without changing sampling seeds."""
    contract = {"prior_policy": config.prior_policy,
                "research_origin": config.research_origin}
    links = getattr(config.base, "parameter_links", None)
    if links and (links.spacing or links.shares):
        contract["parameter_links"] = {
            "spacing": bool(links.spacing), "shares": bool(links.shares),
        }
    return origin_fingerprint(contract)


def linked_result_summary(result: dict | None) -> dict:
    if not result:
        return {}
    fixed, adaptive = result.get("fixed") or {}, result.get("adaptive") or {}
    current = fixed.get("current") or {}
    return {
        "fixed": {"status": fixed.get("status"), "score": fixed.get("score"),
                  "eligible": fixed.get("eligible"), "oos_metrics": fixed.get("oos_metrics")},
        "adaptive": {"status": adaptive.get("status"), "oos_metrics": adaptive.get("oos_metrics")},
        "current": {"status": current.get("status"), "as_of": current.get("as_of"),
                    "score": current.get("score"), "eligible": current.get("eligible"),
                    "params": current.get("recommended_params")},
    }
