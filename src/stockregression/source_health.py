from __future__ import annotations

import threading
import time
import re
from contextlib import contextmanager
from typing import Callable, Iterable

import pandas as pd
from filelock import FileLock

from .settings import Settings, load_settings
from .storage import Store, now_iso


SOURCE_POLICY_VERSION = "market-sources-v1"
SOURCE_COOLDOWN_HOURS = 24
SOURCE_STATE_VERSION = 1
SOURCE_QUALIFICATION_VERSION = "etf-daily-type-probe-v1"
SOURCE_QUALIFICATION_MAX_AGE_HOURS = 24

SOURCE_CAPABILITIES = {
    "recent_minute": {
        "label": "近期 1 分钟",
        "default": "sina",
        "optional": "tdx",
        "fallback": "ths",
        "markets": ("stock", "etf"),
        "description": "新浪提供近期分钟行情；通达信为检测通过后可选的来源；两者都失败时自动转同花顺。",
    },
    "etf_daily": {
        "label": "ETF 日线",
        "default": "baostock",
        "optional": "eastmoney",
        "markets": ("etf",),
        "description": "BaoStock 为默认日线来源；东方财富为检测通过后可选的来源。",
    },
}

SOURCE_LABELS = {
    "sina": "新浪（AKShare）",
    "tdx": "通达信",
    "ths": "同花顺",
    "baostock": "BaoStock",
    "eastmoney": "东方财富（AKShare）",
    "akshare": "AKShare",
}

_PROBE_LOCK = threading.Lock()


class SourceFailure(RuntimeError):
    def __init__(
        self, source: str, capability: str, category: str, message: str,
        source_attempts: list[dict] | None = None,
        etf_daily_route: dict | None = None, source_events: list[dict] | None = None,
    ):
        super().__init__(message)
        self.source = source
        self.capability = capability
        self.category = category
        self.source_attempts = list(source_attempts or [])
        self.etf_daily_route = dict(etf_daily_route or {})
        self.source_events = list(source_events or [])

    def __reduce__(self):
        # ProcessPoolExecutor must reconstruct all routing fields, not just args[0].
        return type(self), (
            self.source, self.capability, self.category, str(self), self.source_attempts,
            self.etf_daily_route, self.source_events,
        )


def classify_etf_product(metadata: dict | None) -> str:
    """Classify only from explicit listing/product labels, never symbol-derived terms."""
    metadata = metadata if isinstance(metadata, dict) else {}
    labels = [str(metadata.get(key, "")) for key in (
        "product_type", "security_type", "instrument_type", "etf_type", "listing_type",
        "listing_label", "name",
    )]
    text = " ".join(labels).casefold()
    has_lof = bool(re.search(r"lof|上市开放式基金", text, re.IGNORECASE))
    has_reit = bool(re.search(r"reits?|不动产投资信托", text, re.IGNORECASE))
    if has_lof and has_reit:
        return "unknown"
    if has_lof:
        return "lof"
    if has_reit:
        return "reit"
    explicit_type = " ".join(labels).casefold()
    if re.search(r"etf", text, re.IGNORECASE) or explicit_type.strip() == "交易型开放式指数基金":
        return "etf"
    return "unknown"


def market_for_symbol(symbol: str) -> str:
    return "etf" if str(symbol).split(".")[0].startswith(("1", "5")) else "stock"


def classify_source_error(exc: BaseException) -> str:
    if isinstance(exc, InterruptedError):
        return "cancelled"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "network"
    if isinstance(exc, SourceFailure):
        return exc.category
    explicit_category = getattr(exc, "category", None)
    if explicit_category in {"empty_unresolved", "empty", "unsupported", "data_quality", "provider", "network", "timeout", "protocol"}:
        return explicit_category
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and (500 <= status < 600 or status in {408, 429}):
        return "network"
    message = str(exc).lower()
    original = getattr(exc, "original_exception", None)
    if original is not None:
        message += " " + str(original).lower()
    if "unpack requires" in message or "protocol" in message or "解析" in message:
        return "protocol"
    if any(word in message for word in ("timeout", "timed out", "超时")):
        return "timeout"
    if any(word in message for word in ("connection", "proxy", "network", "连接失败", "remote end closed")):
        return "network"
    if any(word in message for word in ("未返回", "没有返回", "empty", "无数据")):
        return "empty"
    if any(word in message for word in ("不存在", "无此代码", "未上市", "不支持该标的", "unsupported symbol")):
        return "unsupported"
    if any(word in message for word in ("非正", "字段", "质量", "无效", "invalid")):
        return "data_quality"
    return "provider"


def is_transient(category: str) -> bool:
    return category in {"network", "timeout"}


def _health_key(source: str, capability: str, market: str) -> str:
    return f"source-health:{SOURCE_POLICY_VERSION}:{capability}:{source}:{market}"


def _qualification_key(source: str, product_type: str) -> str:
    return f"source-qualification:{SOURCE_QUALIFICATION_VERSION}:etf_daily:{source}:{product_type}"


def source_qualification(store: Store, source: str, product_type: str) -> dict:
    value = store.state(_qualification_key(source, product_type))
    if not isinstance(value, dict):
        return {"status": "unknown", "source": source, "product_type": product_type,
                "policy_version": SOURCE_QUALIFICATION_VERSION}
    result = dict(value)
    if result.get("policy_version") != SOURCE_QUALIFICATION_VERSION:
        result["status"] = "unknown"
        result["qualification_error"] = "policy_version_mismatch"
        return result
    try:
        expires = pd.Timestamp(result.get("expires_at"))
        checked = pd.Timestamp(result.get("checked_at"))
        now = pd.Timestamp.now(tz="UTC")
        fresh = checked <= now < expires and expires <= checked + pd.Timedelta(hours=24)
        evidence_valid = (
            int(result.get("rows") or 0) > 0 and bool(result.get("sample_symbol"))
            and pd.notna(pd.Timestamp(result.get("coverage_start")))
            and pd.notna(pd.Timestamp(result.get("coverage_end")))
            and pd.Timestamp(result["coverage_start"]) <= pd.Timestamp(result["coverage_end"])
        )
    except (TypeError, ValueError):
        fresh = False
        evidence_valid = False
    if result.get("product_type") != product_type or result.get("source") != source:
        result["status"] = "unknown"
    elif result.get("status") == "passed" and not evidence_valid:
        result["status"] = "unknown"
        result["qualification_error"] = "missing_probe_evidence"
    elif not fresh and result.get("status") == "passed":
        result["status"] = "expired"
    return result


def _record_source_qualification(
    store: Store, source: str, product_type: str, sample_symbol: str,
    *, status: str, rows: int = 0, coverage_start=None, coverage_end=None, error=None,
) -> dict:
    checked = pd.Timestamp.now(tz="UTC")
    value = {
        "status": status, "source": source, "product_type": product_type,
        "sample_symbol": sample_symbol, "rows": int(rows),
        "coverage_start": str(coverage_start) if coverage_start is not None else None,
        "coverage_end": str(coverage_end) if coverage_end is not None else None,
        "checked_at": checked.isoformat(),
        "expires_at": (checked + pd.Timedelta(hours=SOURCE_QUALIFICATION_MAX_AGE_HOURS)).isoformat(),
        "policy_version": SOURCE_QUALIFICATION_VERSION,
        "error_category": classify_source_error(error) if error is not None else None,
        "error": str(error) if error is not None else None,
    }
    store.state(_qualification_key(source, product_type), value)
    return value


def _market_list(capability: str, market: str | None) -> tuple[str, ...]:
    allowed = SOURCE_CAPABILITIES[capability]["markets"]
    if market in (None, "all"):
        return tuple(allowed)
    if market not in allowed:
        raise ValueError(f"{capability} 不支持市场类型 {market}")
    return (str(market),)


def source_state(store: Store, source: str, capability: str, market: str) -> dict:
    value = store.state(_health_key(source, capability, market))
    if not isinstance(value, dict):
        value = {}
    return {
        "source": source,
        "capability": capability,
        "market": market,
        "status": value.get("status", "unknown"),
        "rows": value.get("rows"),
        "coverage_start": value.get("coverage_start"),
        "coverage_end": value.get("coverage_end"),
        "duration_seconds": value.get("duration_seconds"),
        "last_checked_at": value.get("last_checked_at"),
        "last_success_at": value.get("last_success_at"),
        "last_failure_at": value.get("last_failure_at"),
        "cooldown_until": value.get("cooldown_until"),
        "error_category": value.get("error_category"),
        "error": value.get("error"),
        "policy_version": SOURCE_POLICY_VERSION,
    }


def _record(
    store: Store,
    source: str,
    capability: str,
    market: str,
    *,
    error: BaseException | None = None,
    rows: int | None = None,
    coverage_start=None,
    coverage_end=None,
    duration_seconds: float | None = None,
    market_settings=None,
) -> dict:
    market_policy = market_settings or store.market_settings
    lock_path = store.root / ".source-health.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=market_policy.source_lock_timeout_seconds):
        previous = store.state(_health_key(source, capability, market)) or {}
        checked = now_iso()
        if error is None:
            value = {
                **previous, "status": "passed", "source": source,
                "capability": capability, "market": market,
                "rows": rows, "coverage_start": str(coverage_start) if coverage_start is not None else None,
                "coverage_end": str(coverage_end) if coverage_end is not None else None,
                "duration_seconds": duration_seconds, "last_checked_at": checked,
                "last_success_at": checked, "cooldown_until": None,
                "error_category": None, "error": None,
                "state_version": SOURCE_STATE_VERSION,
            }
        else:
            category = classify_source_error(error)
            cooldown = None
            if category != "cancelled":
                cooldown = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=market_policy.failure_cooldown_hours)).isoformat()
            value = {
                **previous, "status": "cancelled" if category == "cancelled" else "failed",
                "source": source, "capability": capability, "market": market,
                "rows": rows, "coverage_start": None, "coverage_end": None,
                "duration_seconds": duration_seconds, "last_checked_at": checked,
                "last_failure_at": checked if category != "cancelled" else previous.get("last_failure_at"),
                "cooldown_until": cooldown, "error_category": category,
                "error": str(error), "state_version": SOURCE_STATE_VERSION,
            }
        store.state(_health_key(source, capability, market), value)
        return source_state(store, source, capability, market)


def record_success(store: Store, source: str, capability: str, market: str, **values) -> dict:
    return _record(store, source, capability, market, **values)


def record_failure(store: Store, source: str, capability: str, market: str, error: BaseException, **values) -> dict:
    return _record(store, source, capability, market, error=error, **values)


def _cooldown_active(state: dict) -> bool:
    value = state.get("cooldown_until")
    if not value or state.get("status") != "failed":
        return False
    try:
        return pd.Timestamp(value) > pd.Timestamp.now(tz="UTC")
    except (TypeError, ValueError):
        return False


def _recovery_due(state: dict) -> bool:
    return state.get("status") == "failed" and not _cooldown_active(state)


def source_is_ready(store: Store, source: str, capability: str, markets: Iterable[str] | None = None) -> bool:
    selected = tuple(markets or _market_list(capability, None))
    return all(source_state(store, source, capability, market)["status"] == "passed" for market in selected)


def decode_coverage(frame: pd.DataFrame):
    if frame is None or frame.empty:
        return None, None
    column = "timestamp" if "timestamp" in frame.columns else "date"
    return frame[column].min(), frame[column].max()


class SourceRouter:
    """Resolve configured sources and keep health shared across jobs and processes."""

    def __init__(self, store: Store, *, settings: Settings | None = None):
        self.store = store
        self.settings = settings or store.settings
        self.market_settings = self.settings.market_data

    def route(self, capability: str, market: str, product_type: str | None = None) -> dict:
        spec = SOURCE_CAPABILITIES[capability]
        default = spec["default"]
        optional = spec["optional"]
        field = "recent_minute_source" if capability == "recent_minute" else "etf_daily_source"
        configured = getattr(self.settings, field)
        if configured == default:
            result = {"configured": default, "effective": default, "fallback": False, "reason": None}
            if capability == "etf_daily":
                result.update({
                    "product_type": product_type or "unknown",
                    "qualification": source_qualification(self.store, optional, product_type or "unknown"),
                })
            return result
        state = source_state(self.store, optional, capability, market)
        type_qualification = None
        if capability == "etf_daily":
            type_qualification = source_qualification(
                self.store, optional, product_type or "unknown",
            )
        qualified = type_qualification is None or type_qualification["status"] == "passed"
        effective = optional if qualified and (state["status"] == "passed" or _recovery_due(state)) else default
        reason = None
        if effective == default and optional == configured:
            if not qualified:
                reason = f"Eastmoney 未通过 {product_type or 'unknown'} 类型的 24 小时资格探测"
            else:
                reason = "可选来源尚未通过检测" if state["status"] == "unknown" else state.get("error") or "可选来源暂不可用"
        return {
            "configured": configured, "effective": effective,
            "fallback": effective != configured, "reason": reason,
            **({"product_type": product_type or "unknown", "qualification": type_qualification}
               if capability == "etf_daily" else {}),
        }

    def call(
        self,
        capability: str,
        market: str,
        calls: dict[str, Callable[[], object]],
        *,
        record: bool = True,
        fallbacks: Iterable[str] = (),
        accept: Callable[[object, str], None] | None = None,
        product_type: str | None = None,
    ):
        choice = self.route(capability, market, product_type=product_type)
        primary = choice["effective"]

        optional = SOURCE_CAPABILITIES[capability]["optional"]
        recovery_lock = None
        if primary == optional:
            lock = FileLock(str(self.store.root / f".source-probe-{capability}-{primary}-{market}.lock"), timeout=0)
            try:
                lock.acquire()
                recovery_lock = lock
                state = source_state(self.store, primary, capability, market)
                if state["status"] != "passed" and not _recovery_due(state):
                    primary = SOURCE_CAPABILITIES[capability]["default"]
                    choice = {**choice, "effective": primary, "fallback": True,
                              "reason": state.get("error") or "可选来源未通过检测"}
            except Exception:
                primary = SOURCE_CAPABILITIES[capability]["default"]
                choice = {**choice, "effective": primary, "fallback": True,
                          "reason": "另一个任务正在检测或使用可选来源"}

        def invoke(source: str):
            started = time.monotonic()
            category = "provider"
            retries = int(self.market_settings.request_retries)
            try:
                value = calls[source]()
                if isinstance(value, pd.DataFrame) and value.empty:
                    raise SourceFailure(source, capability, "empty_unresolved", f"{source} 返回空数据")
                if accept is not None:
                    accept(value, source)
                if record:
                    rows = len(value) if isinstance(value, (pd.DataFrame, list, tuple)) else None
                    start, end = decode_coverage(value) if isinstance(value, pd.DataFrame) else (None, None)
                    record_success(
                        self.store, source, capability, market, rows=rows,
                        coverage_start=start, coverage_end=end,
                        duration_seconds=round(time.monotonic() - started, 3),
                        market_settings=self.market_settings,
                    )
                return value, source, {
                    "source": source, "status": "passed",
                    "rows": len(value) if isinstance(value, (pd.DataFrame, list, tuple)) else None,
                    "coverage_start": str(decode_coverage(value)[0]) if isinstance(value, pd.DataFrame) and decode_coverage(value)[0] is not None else None,
                    "coverage_end": str(decode_coverage(value)[1]) if isinstance(value, pd.DataFrame) and decode_coverage(value)[1] is not None else None,
                }
            except InterruptedError:
                raise
            except Exception as exc:
                category = classify_source_error(exc)
                if record and category not in {"empty", "empty_unresolved", "unsupported", "data_quality"}:
                    record_failure(
                        self.store, source, capability, market, exc,
                        duration_seconds=round(time.monotonic() - started, 3),
                        market_settings=self.market_settings,
                    )
                for _attempt in range(retries if is_transient(category) else 0):
                    try:
                        value = calls[source]()
                        if isinstance(value, pd.DataFrame) and value.empty:
                            raise SourceFailure(source, capability, "empty_unresolved", f"{source} 返回空数据")
                        if accept is not None:
                            accept(value, source)
                        if record:
                            rows = len(value) if isinstance(value, (pd.DataFrame, list, tuple)) else None
                            start, end = decode_coverage(value) if isinstance(value, pd.DataFrame) else (None, None)
                            record_success(
                                self.store, source, capability, market, rows=rows,
                                coverage_start=start, coverage_end=end,
                                duration_seconds=round(time.monotonic() - started, 3),
                                market_settings=self.market_settings,
                            )
                        start, end = decode_coverage(value) if isinstance(value, pd.DataFrame) else (None, None)
                        return value, source, {
                            "source": source, "status": "passed",
                            "rows": len(value) if isinstance(value, (pd.DataFrame, list, tuple)) else None,
                            "coverage_start": str(start) if start is not None else None,
                            "coverage_end": str(end) if end is not None else None,
                        }
                    except InterruptedError:
                        raise
                    except Exception as retry_error:
                        if record and classify_source_error(retry_error) not in {"empty", "empty_unresolved", "unsupported", "data_quality"}:
                            record_failure(
                                self.store, source, capability, market, retry_error,
                                duration_seconds=round(time.monotonic() - started, 3),
                                market_settings=self.market_settings,
                            )
                        exc = retry_error
                        category = classify_source_error(exc)
                        if not is_transient(category):
                            break
                attempt = {
                    "source": source, "status": "failed", "error_category": category,
                    "error": str(exc),
                }
                raise SourceFailure(source, capability, category, str(exc), [attempt]) from exc

        if primary not in calls:
            raise ValueError(f"来源 {primary} 不支持 {capability}")
        chain = []
        for source in (primary, SOURCE_CAPABILITIES[capability]["default"], *fallbacks):
            if source in calls and source not in chain:
                chain.append(source)
        failures: list[str] = []
        attempts: list[dict] = []
        last_category = "provider"
        try:
            for index, source in enumerate(chain):
                try:
                    value, actual, successful_attempt = invoke(source)
                    attempts.append(successful_attempt)
                    if index == 0:
                        return value, actual, {**choice, "source_attempts": attempts}
                    return value, actual, {
                        **choice, "effective": actual, "fallback": True,
                        "reason": "；".join(failures) or choice.get("reason"),
                        "source_attempts": attempts,
                    }
                except SourceFailure as exc:
                    failures.append(str(exc))
                    attempts.extend(exc.source_attempts or [{
                        "source": source, "status": "failed",
                        "error_category": exc.category, "error": str(exc),
                    }])
                    last_category = exc.category
            if failures:
                raise SourceFailure(primary, capability, last_category, "；".join(failures), attempts)
            raise SourceFailure(primary, capability, "provider", f"来源 {primary} 不可用")
        finally:
            if recovery_lock is not None:
                recovery_lock.release()


@contextmanager
def probe_guard(store: Store, source: str, capability: str, market: str):
    """Serialize a manual probe so concurrent requests cannot stampede a source."""
    lock = store.root / f".source-probe-{capability}-{source}-{market}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with _PROBE_LOCK:
        with FileLock(str(lock), timeout=store.market_settings.source_lock_timeout_seconds):
            yield


def configured_sources(store: Store) -> dict:
    settings = load_settings()
    router = SourceRouter(store, settings=settings)
    capabilities = []
    for name, spec in SOURCE_CAPABILITIES.items():
        field = "recent_minute_source" if name == "recent_minute" else "etf_daily_source"
        configured = getattr(settings, field)
        sources = []
        fallback = spec.get("fallback")
        options = (spec["default"], spec["optional"], *([fallback] if fallback else []))
        for source in options:
            states = [source_state(store, source, name, market) for market in spec["markets"]]
            sources.append({
                "source": source,
                "label": SOURCE_LABELS.get(source, source),
                "role": "default" if source == spec["default"] else "fallback" if source == fallback else "optional",
                "selected": configured == source,
                "selectable": source != fallback and (
                    source == spec["default"] or all(state["status"] == "passed" for state in states)
                ),
                "states": states,
            })
        capabilities.append({
            "capability": name,
            "label": spec["label"],
            "description": spec["description"],
            "configured": configured,
            "default": spec["default"],
            "optional": spec["optional"],
            "routes": [{"market": market, **router.route(name, market)} for market in spec["markets"]],
            "sources": sources,
        })
    return {
        "policy_version": SOURCE_POLICY_VERSION,
        "cooldown_hours": settings.market_data.failure_cooldown_hours,
        "request_retries": settings.market_data.request_retries,
        "capabilities": capabilities,
    }


def validate_source_settings(store: Store, values: dict) -> None:
    if values.get("recent_minute_source") == "tdx" and not source_is_ready(store, "tdx", "recent_minute"):
        raise ValueError("通达信尚未通过股票和 ETF 的近期分钟检测，暂不能设为默认来源")
    if values.get("etf_daily_source") == "eastmoney":
        qualification = source_qualification(store, "eastmoney", "etf")
        if qualification["status"] != "passed":
            raise ValueError("东方财富尚未通过 24 小时内的 ETF 类型探测，暂不能设为默认来源")


def _component_call(
    store: Store,
    source: str,
    capability: str,
    market: str,
    callback: Callable[[], object],
):
    started = time.monotonic()
    retries = int(store.market_settings.request_retries)
    for attempt in range(retries + 1):
        try:
            value = callback()
        except InterruptedError:
            raise
        except Exception as exc:
            category = classify_source_error(exc)
            if category not in {"empty", "empty_unresolved", "unsupported", "data_quality", "cancelled"}:
                record_failure(
                    store, source, capability, market, exc,
                    duration_seconds=round(time.monotonic() - started, 3),
                )
            if not is_transient(category) or attempt >= retries:
                raise
            continue
        record_success(
            store, source, capability, market,
            duration_seconds=round(time.monotonic() - started, 3),
        )
        return value


def fetch_auxiliary_routed(
    store: Store,
    symbol: str,
    start,
    end,
    cancelled: Callable[[], bool] = lambda: False,
):
    """Fetch daily, calendar and actions through independently health-checked adapters."""
    from .providers import (
        fetch_etf_daily,
        fetch_market_actions,
        fetch_market_calendar,
        fetch_stock_daily,
    )

    market = market_for_symbol(symbol)
    frozen_settings = store.settings
    router = SourceRouter(store, settings=frozen_settings)
    network_timeout = router.market_settings.network_timeout_seconds
    metadata: dict = {}
    events: list[dict] = []
    if market == "etf":
        instrument = store.instrument(symbol) or {}
        product_type = classify_etf_product(instrument)
        market_settings = router.market_settings
        configured_source = market_settings.etf_daily_source
        fallback_qualification = source_qualification(store, "eastmoney", product_type)
        fallback_allowed = (
            configured_source == "baostock"
            and market_settings.etf_daily_fallback_enabled
            and product_type in {"etf", "lof", "reit"}
            and fallback_qualification["status"] == "passed"
        )
        fallback_reasons = []
        if not market_settings.etf_daily_fallback_enabled:
            fallback_reasons.append("ETF 日线备用来源已关闭")
        elif product_type == "unknown":
            fallback_reasons.append("证券产品类型未能从元数据确认")
        elif fallback_qualification["status"] != "passed":
            fallback_reasons.append("Eastmoney 未通过该产品类型的 24 小时探测")
        fallbacks = ("eastmoney",) if fallback_allowed else ()
        try:
            daily, daily_source, route = router.call(
                "etf_daily", market,
            {
                "baostock": lambda: fetch_etf_daily(
                    symbol, start, end, "baostock", cancelled,
                    network_timeout=network_timeout,
                ),
                "eastmoney": lambda: fetch_etf_daily(
                    symbol, start, end, "eastmoney", cancelled,
                    network_timeout=network_timeout,
                ),
            },
                fallbacks=fallbacks,
                product_type=product_type,
            )
        except SourceFailure as exc:
            fallback_attempted = any(
                attempt.get("source") == "eastmoney"
                for attempt in exc.source_attempts
            )
            failure_route = {
                "configured": configured_source,
                "effective": None,
                "fallback": False,
                "product_type": product_type,
                "qualification": fallback_qualification,
                "fallback_attempted": fallback_attempted,
                "fallback_status": (
                    "failed" if fallback_allowed and fallback_attempted
                    else "not_eligible" if not fallback_allowed else "not_attempted"
                ),
                "fallback_reason": "；".join(fallback_reasons) or str(exc),
                "source_attempts": exc.source_attempts,
            }
            failure_events = ([{
                "kind": "source_fallback", "source": configured_source,
                "fallback": "eastmoney", "product_type": product_type,
                "message": f"{configured_source} ETF 日线失败，Eastmoney 回退也未完成：{exc}",
            }] if fallback_attempted else [])
            if exc.category in {"empty", "empty_unresolved"}:
                from .providers import EmptyUnresolvedError
                raise EmptyUnresolvedError(
                    f"日线来源未返回可确认数据：{exc}",
                    source_attempts=exc.source_attempts,
                    product_type=product_type,
                    fallback_status=failure_route["fallback_status"],
                    fallback_reason=failure_route["fallback_reason"],
                    configured_source=configured_source,
                    fallback_attempted=fallback_attempted,
                    etf_daily_route=failure_route,
                    source_events=failure_events,
                ) from exc
            exc.etf_daily_route = failure_route
            exc.source_events = failure_events
            raise
        route.update({
            "product_type": product_type,
            "fallback_attempted": any(item.get("source") == "eastmoney" for item in route.get("source_attempts", [])),
            "fallback_status": (
                "succeeded" if daily_source == "eastmoney" and configured_source == "baostock"
                else "failed" if fallback_allowed and any(item.get("source") == "eastmoney" for item in route.get("source_attempts", []))
                else "not_eligible" if not fallback_allowed else "not_attempted"
            ),
            "fallback_reason": route.get("reason") or ("；".join(fallback_reasons) or None),
            "qualification": fallback_qualification,
        })
        metadata.update({
            "daily_source": daily_source,
            "product_type": product_type,
            "source_attempts": route.get("source_attempts", []),
            "etf_daily_route": route,
        })
        if route.get("fallback"):
            events.append({
                "kind": "source_fallback", "source": configured_source, "fallback": daily_source,
                "product_type": product_type,
                "message": f"{configured_source} ETF 日线不可用，本次已使用 {daily_source}。"
                + (" " + str(route.get("reason") or "") if route.get("reason") else ""),
            })
    else:
        daily, stock_metadata = _component_call(
            store, "baostock", "stock_daily", "stock",
            lambda: fetch_stock_daily(symbol, start, end, cancelled),
        )
        metadata.update(stock_metadata)
        metadata["daily_source"] = "baostock"
    calendar = _component_call(
        store, "baostock", "calendar", market,
        lambda: fetch_market_calendar(start, end, cancelled),
    )
    if calendar is None or calendar.empty:
        raise ValueError("BaoStock 交易日历未返回所选区间数据")
    actions, action_metadata = _component_call(
        store, "akshare" if market == "etf" else "baostock",
        "etf_actions" if market == "etf" else "stock_actions", market,
        lambda: fetch_market_actions(
            symbol, start, end, cancelled, network_timeout=network_timeout,
        ),
    )
    metadata.update(action_metadata)
    metadata.update({
        "calendar_source": "baostock",
        "actions_source": "akshare" if market == "etf" else "baostock",
        "source_policy_version": SOURCE_POLICY_VERSION,
        "source_events": events,
    })
    metadata["warnings"] = [*(metadata.get("warnings") or []), *[event["message"] for event in events]]
    return daily, actions, calendar, metadata


def probe_source(
    store: Store,
    source: str,
    capability: str,
    market: str = "all",
    cancelled: Callable[[], bool] = lambda: False,
    product_type: str = "etf",
    sample_symbol: str | None = None,
) -> dict:
    """Probe a source without writing market data or changing the selected source."""
    from .providers import fetch_etf_daily, fetch_sina, fetch_tdx, fetch_ths, latest_closed_date

    if capability not in SOURCE_CAPABILITIES:
        raise ValueError("未知数据源能力：" + capability)
    spec = SOURCE_CAPABILITIES[capability]
    allowed = {spec["default"], spec["optional"]}
    if spec.get("fallback"):
        allowed.add(spec["fallback"])
    if source not in allowed:
        raise ValueError(f"{source} 不适用于 {capability}")
    if capability == "etf_daily" and source not in {"baostock", "eastmoney"}:
        raise ValueError("ETF 日线探测来源应为 baostock 或 eastmoney")
    if capability != "etf_daily" and (product_type != "etf" or sample_symbol is not None):
        raise ValueError("产品类型和样本证券只适用于 ETF 日线探测")
    if capability == "etf_daily" and product_type not in {"etf", "lof", "reit"}:
        raise ValueError("ETF 日线探测类型应为 etf/lof/reit")
    if capability == "etf_daily" and product_type in {"lof", "reit"} and sample_symbol is None:
        raise ValueError(f"{product_type.upper()} 类型探测必须提供本地元数据匹配的 sample_symbol")
    if capability == "etf_daily" and market not in {"all", "etf"}:
        raise ValueError("ETF 日线探测仅支持 ETF 市场")
    markets = _market_list(capability, market)
    network_timeout = store.market_settings.network_timeout_seconds
    end = pd.Timestamp(latest_closed_date()).normalize()
    start = end - pd.Timedelta(days=35)
    results = []
    for item in markets:
        if cancelled():
            raise InterruptedError("任务已取消")
        symbol = (sample_symbol or "159611.SZ") if capability == "etf_daily" else ("159611.SZ" if item == "etf" else "600000.SH")
        if capability == "etf_daily" and sample_symbol is not None:
            stored = store.instrument(sample_symbol)
            if classify_etf_product(stored) != product_type:
                raise ValueError(f"样本证券 {sample_symbol} 的本地元数据不能确认类型 {product_type}")
        started = time.monotonic()
        try:
            with probe_guard(store, source, capability, item):
                if capability == "recent_minute":
                    if source == "tdx":
                        frame = fetch_tdx(
                            symbol, earliest=start, cancelled=cancelled,
                            network_timeout=network_timeout,
                        )
                    elif source == "ths":
                        frame = fetch_ths(
                            symbol, network_timeout=network_timeout,
                        )
                    else:
                        frame = fetch_sina(
                            symbol, network_timeout=network_timeout,
                        )
                else:
                    frame = fetch_etf_daily(
                        symbol, start, end, source, cancelled,
                        network_timeout=network_timeout,
                    )
            if frame is None or frame.empty:
                from .providers import EmptyUnresolvedError
                raise EmptyUnresolvedError(f"{source} 探测未返回数据", product_type=product_type)
            coverage_start, coverage_end = decode_coverage(frame)
            if capability == "etf_daily":
                coverage_end = pd.Timestamp(coverage_end).normalize()
                latest = pd.Timestamp(end).normalize()
                if coverage_end > latest or coverage_end < latest - pd.Timedelta(days=7):
                    raise ValueError(f"{source} 样本数据未覆盖最近已收盘日附近：{coverage_end.date()}")
            state = record_success(
                store, source, capability, item, rows=len(frame),
                coverage_start=coverage_start, coverage_end=coverage_end,
                duration_seconds=round(time.monotonic() - started, 3),
            )
            result = {"market": item, "symbol": symbol, "status": "passed", "state": state}
            if capability == "etf_daily":
                result["qualification"] = _record_source_qualification(
                    store, source, product_type, symbol, status="passed", rows=len(frame),
                    coverage_start=coverage_start, coverage_end=coverage_end,
                )
            results.append(result)
        except InterruptedError:
            raise
        except Exception as exc:
            category = classify_source_error(exc)
            if category not in {"empty", "empty_unresolved", "unsupported", "data_quality", "cancelled"}:
                state = record_failure(
                    store, source, capability, item, exc,
                    duration_seconds=round(time.monotonic() - started, 3),
                )
            else:
                state = source_state(store, source, capability, item)
            result = {
                "market": item, "symbol": symbol, "status": "failed",
                "error_category": category, "error": str(exc), "state": state,
            }
            if capability == "etf_daily":
                result["qualification"] = _record_source_qualification(
                    store, source, product_type, symbol, status="failed", error=exc,
                )
            results.append(result)
    passed = [item for item in results if item["status"] == "passed"]
    status = "completed" if len(passed) == len(results) else "partial" if passed else "failed"
    return {
        "status": status, "source": source, "capability": capability,
        "results": results,
        "message": f"{SOURCE_LABELS.get(source, source)} 探测完成：{len(passed)}/{len(results)} 通过",
    }
