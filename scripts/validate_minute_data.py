"""Read-only local acceptance for the real 159611.SZ minute snapshot.

This script does not contact providers and does not write to the market-data
Store. It reads an existing locked snapshot, recomputes local coverage, checks
the saved scientific result's source projection, and replays its saved 1m
fixed/adaptive parameters through the real share engine. Only the compact
acceptance records in docs/ are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from stockregression.api import _result_data_metadata, public_result
from stockregression.data import DataService
from stockregression.models import BacktestConfig, OptimizationConfig
from stockregression.research_preparation import preparation_config
from stockregression.settings import PROJECT_ROOT, load_settings
from stockregression.share_engine import run_share_backtest
from stockregression.storage import Store, dumps, json_safe


_NO_STATE_VALUE = object()
_QUALITY_KEYS = (
    "minute_days", "daily_days", "suspended_days", "total_days",
    "pending_calibration_days", "minute_ratio", "synthetic_bars",
    "auction_placeholders",
)


class ReadOnlyStore(Store):
    """Store read facade using SQLite mode=ro and swallowing cache writes."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.db_path = self.root / "catalog.sqlite"
        if not self.db_path.is_file():
            raise FileNotFoundError(f"本地 catalog 不存在：{self.db_path}")
        self.lock = None
        self.suppressed_state_writes: list[str] = []

    @contextmanager
    def db(self):
        uri = self.db_path.as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def state(self, key: str, value: Any = _NO_STATE_VALUE):
        if value is not _NO_STATE_VALUE:
            self.suppressed_state_writes.append(key)
            return value
        # Ignore every persisted state value, including older coverage keys and
        # newer versioned quality caches. This forces local recomputation while
        # keeping both reads and writes of the cache out of the acceptance path.
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_fingerprint(store: ReadOnlyStore, manifest: dict) -> dict[str, Any]:
    manifest_path = store.root / "manifests" / f"{manifest['version']}.json"
    files = sorted({relative for paths in manifest["files"].values() for relative in paths})
    file_hashes = {
        relative: _sha256(store.root / relative)
        for relative in files
    }
    return {
        "version": manifest["version"],
        "manifest_sha256": _sha256(manifest_path),
        "files": file_hashes,
        "aggregate_sha256": hashlib.sha256(dumps(file_hashes).encode("utf-8")).hexdigest(),
    }


def _minute_rows(store: ReadOnlyStore, manifest: dict) -> pd.DataFrame:
    import pyarrow.parquet as pq

    frames = []
    wanted = ("timestamp", "source", "source_verified", "is_synthetic", "volume", "close")
    for relative in manifest["files"].get("minute", []):
        path = store.root / relative
        available = set(pq.read_schema(path).names)
        columns = [name for name in wanted if name in available]
        if "timestamp" in columns:
            frames.append(pq.read_table(path, columns=columns).to_pandas())
    if not frames:
        return pd.DataFrame(columns=wanted)
    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result = result[result.timestamp.notna()].sort_values("timestamp").reset_index(drop=True)
    return result


def _expected_times(day: pd.Timestamp) -> set[str]:
    am = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min")
    pm = pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=120, freq="min")
    return set(am.strftime("%H:%M")) | set(pm.strftime("%H:%M"))


def _raw_minute_summary(minute: pd.DataFrame) -> dict[str, Any]:
    if minute.empty:
        return {"rows": 0, "days": 0, "sources": {}, "duplicate_timestamps": 0, "raw_hole_slots": {}, "post_2026_09_30_days": []}
    minute = minute.copy()
    minute["day"] = minute.timestamp.dt.normalize()
    minute["clock"] = minute.timestamp.dt.strftime("%H:%M")
    source_counts = minute.source.fillna("unknown").astype(str).value_counts().sort_index().to_dict() if "source" in minute else {}
    duplicates = int(minute.timestamp.duplicated(keep=False).sum())
    holes: dict[str, int] = {}
    daily = []
    complete_slot_days = []
    complete_except_auction_days = []
    source_daily_coverage = {}
    if "source" in minute:
        for source, source_group in minute.groupby("source", dropna=False, sort=True):
            source_day_counts = source_group.groupby(source_group.timestamp.dt.normalize()).size().value_counts().sort_index()
            source_daily_coverage[str(source or "unknown")] = {
                "days": int(source_group.timestamp.dt.normalize().nunique()),
                "rows_per_day_distribution": {str(int(rows)): int(days) for rows, days in source_day_counts.items()},
            }
    for day, group in minute.groupby("day", sort=True):
        missing = sorted(_expected_times(day) - set(group.clock))
        if not missing:
            complete_slot_days.append(day.date().isoformat())
        elif set(missing) <= {"14:58", "14:59"}:
            complete_except_auction_days.append(day.date().isoformat())
        for slot in missing:
            holes[slot] = holes.get(slot, 0) + 1
        if day.date() > date(2026, 9, 30):
            auction = [slot for slot in missing if slot in {"14:58", "14:59"}]
            daily.append({
                "date": day.date().isoformat(),
                "raw_minute_bars": int(len(group)),
                "missing_slots": missing,
                "structural_auction_candidate_slots": auction,
                "other_missing_slots": [slot for slot in missing if slot not in auction],
            })
    return {
        "rows": int(len(minute)),
        "days": int(minute.day.nunique()),
        "first_timestamp": minute.timestamp.min().isoformat(),
        "last_timestamp": minute.timestamp.max().isoformat(),
        "sources": {str(key): int(value) for key, value in source_counts.items()},
        "source_daily_coverage": source_daily_coverage,
        "duplicate_timestamps": duplicates,
        "raw_hole_slots": dict(sorted(holes.items())),
        "days_with_all_240_expected_timestamps": len(complete_slot_days),
        "last_day_with_all_240_expected_timestamps": complete_slot_days[-1] if complete_slot_days else None,
        "last_day_complete_except_structural_auction_candidates": (
            max([*complete_slot_days, *complete_except_auction_days])
            if complete_slot_days or complete_except_auction_days else None
        ),
        "post_2026_09_30_days": daily,
    }


def _quality_summary(quality: dict | None) -> dict[str, Any]:
    quality = quality or {}
    return {key: quality.get(key) for key in _QUALITY_KEYS}


def _diagnostic_summary(value: dict | None) -> dict[str, Any]:
    value = value or {}
    keys = (
        "schema_version", "status", "checked_start", "checked_end",
        "minute_days", "daily_days", "total_days", "suspended_days",
        "pending_calibration_days", "minute_ratio", "synthetic_bars",
        "auction_placeholders", "repaired_bars", "repaired_days",
        "missing_raw_minutes", "grid_missing_minutes", "raw_complete",
        "minute_through", "daily_through", "missing_minute_dates",
        "source_spans",
    )
    return {key: value.get(key) for key in keys}


def _study_intervals(result: dict) -> dict[str, Any]:
    evidence = result.get("evidence") or {}
    check = evidence.get("final_126_day_check") or {}
    return {
        "fixed_as_of": (result.get("fixed") or {}).get("as_of"),
        "schedule": [
            {key: row.get(key) for key in ("decision_date", "effective_date")}
            for row in (result.get("adaptive") or {}).get("schedule", [])
        ],
        "final_126_day_check": {key: check.get(key) for key in ("start", "end", "days", "status")},
    }


def _select_saved_study(store: ReadOnlyStore, version: str, requested_id: str | None) -> dict:
    rows = store.results()
    candidates = [row for row in rows if row.get("kind") == "optimization"]
    if requested_id:
        candidates = [row for row in candidates if row.get("id") == requested_id]
    for row in candidates:
        result = store.result(row["id"])
        config = result.get("config", {}) if result else {}
        base = config.get("base", {}) if isinstance(config, dict) else {}
        symbol = (result or {}).get("symbol") or base.get("symbol")
        if (result and result.get("protocol_version") == 2
                and symbol == "159611.SZ"
                and result.get("data_version") == version
                and result.get("fixed", {}).get("test_result")
                and result.get("adaptive", {}).get("test_result")):
            return result
    raise RuntimeError("找不到使用当前 snapshot 且含 fixed/adaptive 真实回放的已保存 v2 研究；请传入 --result-id")


def _replay_branch(config: BacktestConfig, bundle, schedule=None) -> dict:
    return run_share_backtest(
        config,
        bundle,
        parameter_schedule=schedule,
        daily_details=True,
    )


def _same_replay(expected: dict, actual: dict) -> tuple[bool, dict[str, Any]]:
    expected_grid = expected.get("strategies", {}).get("grid", {})
    actual_grid = actual.get("strategies", {}).get("grid", {})
    pieces = ("metrics", "equity", "trades", "rejections", "period_returns")
    checks = {key: dumps(expected_grid.get(key, [] if key != "metrics" else {})) == dumps(actual_grid.get(key, [] if key != "metrics" else {})) for key in pieces}
    return all(checks.values()), checks


def _markdown(report: dict) -> str:
    snapshot = report["snapshot"]
    raw = report["raw_minutes"]
    coverage = report["current_snapshot_recomputation"]["coverage"]
    checks = report["checks"]
    source_coverage = raw.get("source_daily_coverage", {})
    import_coverage = source_coverage.get("import", {})
    sina_coverage = source_coverage.get("sina", {})
    import_rows = import_coverage.get("rows_per_day_distribution", {})
    sina_rows = sina_coverage.get("rows_per_day_distribution", {})
    source_pattern = (
        f"import **{import_coverage.get('days', 0):,}×{next(iter(import_rows), 'n/a')}** = {raw['sources'].get('import', 0):,} 行，"
        f"Sina **{sina_coverage.get('days', 0):,}×{next(iter(sina_rows), 'n/a')}** = {raw['sources'].get('sina', 0):,} 行"
    )
    lines = [
        "# 159611 分钟行情只读验收",
        "",
        f"验收时间：{report['checked_at']}（本机时区）",
        "",
        f"活动 snapshot `{snapshot['version']}`，非演示数据。全量指纹前后相同：**{checks['snapshot_unchanged']}**；下载/同步调用：**{report['sync_calls']}**。",
        "",
        f"原始分钟行 **{raw['rows']:,}**，交易日 **{raw['days']:,}**，重复时间戳 **{raw['duplicate_timestamps']}**；来源分布为 {source_pattern}。这表示导入日各 240 条，Sina 后续日各 238 条；不能描述为每个 raw 交易日都有 240 条。",
        "",
        "真实槽位空缺分布：" + (", ".join(f"`{slot}` {count} 日" for slot, count in raw["raw_hole_slots"].items()) or "无") + ". 原始槽位数与合成占位数分开记录；结构性收盘集合竞价候选槽不计入普通分钟缺口。",
        "",
        f"原始记录含全部 240 个网格时间戳的最后交易日为 **{raw.get('last_day_with_all_240_expected_timestamps')}**；排除 14:58/14:59 结构性收盘竞价槽后完整的最后交易日为 **{raw.get('last_day_complete_except_structural_auction_candidates')}**。",
        "",
        f"重算覆盖：分钟日 **{coverage['quality'].get('minute_days')}** / 总交易日 **{coverage['quality'].get('total_days')}**，分钟比例 **{coverage['quality'].get('minute_ratio')}**，收盘竞价占位 **{coverage['quality'].get('auction_placeholders')}**，实际有效分钟缺口 **{coverage['diagnostics'].get('missing_raw_minutes')}**。",
        "",
        f"已保存研究 `{report['saved_study']['result_id']}` 样本外区间 {report['saved_study']['evaluation_start']}～{report['saved_study']['evaluation_end']}，锁定 snapshot 与当前相同。1m fixed/adaptive 参数回放结果一致：**{checks['fixed_replay_equal']} / {checks['adaptive_replay_equal']}**。未执行参数搜索。",
        "",
        "| 检查 | 结果 |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {'通过' if value else '失败'} |" for name, value in checks.items())
    lines.extend(["", "本记录由 `scripts/validate_minute_data.py` 从现有本地 snapshot 和保存研究生成；脚本只读取 SQLite/Parquet/JSON，缓存写入被抑制，只更新本验收记录。", ""])
    return "\n".join(lines)


def run(data_dir: Path, result_id: str | None) -> dict[str, Any]:
    store = ReadOnlyStore(data_dir)
    service = DataService(store)
    sync_calls: list[str] = []

    def forbidden_sync(*args, **kwargs):
        sync_calls.append("sync")
        raise AssertionError("validation must never synchronize market data")

    service.sync = forbidden_sync
    manifest = store.snapshot("159611.SZ")
    if not manifest:
        raise RuntimeError("本地 Store 没有 159611.SZ 活动 snapshot")
    version = manifest["version"]
    if manifest.get("metadata", {}).get("is_demo"):
        raise RuntimeError("活动 snapshot 是演示数据，拒绝作为真实验收")
    before = _snapshot_fingerprint(store, manifest)
    raw_minute = _minute_rows(store, manifest)
    raw_summary = _raw_minute_summary(raw_minute)

    study = _select_saved_study(store, version, result_id)
    optimization_config = OptimizationConfig.model_validate(study["config"])
    prepared_config = preparation_config(store, optimization_config.base)
    prepared_config.end_date = optimization_config.base.end_date

    current_ready_version = service.current_version_if_ready(prepared_config)
    bundle = service.load_bundle(prepared_config, version=version)
    ensured = service.ensure_bundle(prepared_config, version=version)
    coverage = service.coverage("159611.SZ")
    marketbars = service.market_bars(
        "159611.SZ", frequency="1m",
        start_date=str(optimization_config.base.start_date),
        end_date=str(optimization_config.base.end_date), limit=5000,
    )
    result_metadata = _result_data_metadata(store, study)
    public = public_result(study, data_metadata=result_metadata)

    fixed_result = study["fixed"]["test_result"]
    adaptive_result = study["adaptive"]["test_result"]
    fixed_config = BacktestConfig.model_validate(fixed_result["config"])
    adaptive_config = BacktestConfig.model_validate(adaptive_result["config"])
    fixed_replay = _replay_branch(fixed_config, bundle)
    adaptive_replay = _replay_branch(adaptive_config, bundle, study["adaptive"].get("schedule") or [])
    fixed_equal, fixed_parts = _same_replay(fixed_result, fixed_replay)
    adaptive_equal, adaptive_parts = _same_replay(adaptive_result, adaptive_replay)
    coverage_diagnostics = coverage.get("diagnostics", {})
    market_diagnostics = marketbars.get("diagnostics", {})
    public_diagnostics = public.get("diagnostics", {})
    nested_diagnostics = [
        ((public.get(branch) or {}).get("test_result") or {}).get("diagnostics", {})
        for branch in ("fixed", "adaptive")
    ]
    public_statuses = [public_diagnostics.get("status"), *(row.get("status") for row in nested_diagnostics)]

    after_manifest = store.snapshot("159611.SZ")
    after = _snapshot_fingerprint(store, after_manifest)
    checks = {
        "real_non_demo_snapshot": not bool(manifest.get("metadata", {}).get("is_demo")),
        "current_version_ready": current_ready_version == version,
        "locked_ensure_bundle_version": ensured.version == version,
        "locked_ensure_matches_load": dumps(_quality_summary(ensured.quality)) == dumps(_quality_summary(bundle.quality)),
        "coverage_has_minute_data": int(coverage.get("minute_days", 0)) > 0 and float(coverage.get("minute_ratio", 0)) >= 0.95,
        "coverage_matches_bundle_ratio": float(coverage.get("minute_ratio", 0)) == float(bundle.quality.get("minute_ratio", 0)),
        "effective_missing_minutes_zero": int(coverage_diagnostics.get("missing_raw_minutes", -1)) == 0,
        "auction_placeholders_14": int(coverage_diagnostics.get("auction_placeholders", -1)) == 14,
        "marketbars_locked_to_active_version": marketbars.get("version") == version,
        "marketbars_quality_matches_coverage": all(
            market_diagnostics.get(key) == coverage_diagnostics.get(key)
            for key in ("status", "minute_ratio", "auction_placeholders", "missing_raw_minutes", "raw_complete")
        ),
        "saved_study_uses_active_snapshot": study.get("data_version") == version,
        "public_result_uses_locked_snapshot": public.get("data_version") == version,
        "public_result_quality_matches_coverage": all(
            public_diagnostics.get(key) == coverage_diagnostics.get(key)
            for key in ("status", "minute_ratio", "auction_placeholders", "missing_raw_minutes", "raw_complete")
        ),
        "public_result_has_import_source_span": any(span.get("source_kind") == "import" for span in public.get("diagnostics", {}).get("source_spans", [])),
        "public_root_nested_status_consistent": bool(public_statuses) and len(set(public_statuses)) == 1,
        "fixed_replay_equal": fixed_equal,
        "adaptive_replay_equal": adaptive_equal,
        "snapshot_unchanged": before == after,
        "readonly_facade_suppressed_all_cache_writes": isinstance(store, ReadOnlyStore),
        "no_sync_calls": not sync_calls,
    }
    checks.update({f"fixed_replay_{key}": value for key, value in fixed_parts.items()})
    checks.update({f"adaptive_replay_{key}": value for key, value in adaptive_parts.items()})

    from datetime import datetime
    from zoneinfo import ZoneInfo

    report = {
        "schema_version": 1,
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "symbol": "159611.SZ",
        "snapshot": {
            "version": version,
            "created_at": manifest.get("created_at"),
            "is_demo": bool(manifest.get("metadata", {}).get("is_demo")),
            "manifest_sha256": before["manifest_sha256"],
            "data_files_sha256": before["aggregate_sha256"],
            "raw_metadata_quality_summary": manifest.get("metadata", {}).get("quality_summary"),
            "raw_metadata_warnings": manifest.get("metadata", {}).get("warnings", []),
            "profiles": manifest.get("profiles", {}),
        },
        "raw_minutes": raw_summary,
        "current_snapshot_recomputation": {
            "coverage": {
                "quality": _quality_summary(coverage),
                "diagnostics": _diagnostic_summary(coverage.get("diagnostics")),
                "source_events_count": len(coverage.get("source_events", [])),
            },
            "prepared_bundle": {
                "version": bundle.version,
                "quality": _quality_summary(bundle.quality),
                "diagnostics": _diagnostic_summary((bundle.metadata or {}).get("diagnostics")),
            },
            "market_bars": {
                "version": marketbars.get("version"),
                "frequency": marketbars.get("frequency"),
                "returned_bars": len(marketbars.get("bars", [])),
                "first_timestamp": marketbars.get("bars", [{}])[0].get("timestamp") if marketbars.get("bars") else None,
                "last_timestamp": marketbars.get("bars", [{}])[-1].get("timestamp") if marketbars.get("bars") else None,
                "diagnostics": _diagnostic_summary(marketbars.get("diagnostics")),
                "warnings": marketbars.get("warnings", []),
            },
            "current_version_if_ready": current_ready_version,
            "suppressed_cache_writes": sorted(set(store.suppressed_state_writes)),
            "suppressed_cache_write_count": len(store.suppressed_state_writes),
        },
        "saved_study": {
            "result_id": study.get("id"),
            "data_version": study.get("data_version"),
            "status": study.get("status"),
            "evaluation_start": (study.get("config", {}).get("base") or {}).get("start_date"),
            "evaluation_end": (study.get("config", {}).get("base") or {}).get("end_date"),
            "quality": _quality_summary(study.get("quality")),
            "public_diagnostics": _diagnostic_summary(public.get("diagnostics")),
            "public_warnings": public.get("warnings", []),
            "source_events_count": len(public.get("source_events", [])),
            "intervals": _study_intervals(study),
            "raw_result_file_bytes": (store.root / "results" / f"{study['id']}.json").stat().st_size if study.get("id") else None,
            "public_result_keys": sorted(public.keys()),
        },
        "model_replay": {
            "engine": "stockregression.share_engine.run_share_backtest",
            "searches_run": 0,
            "fixed_equal": fixed_equal,
            "fixed_components_equal": fixed_parts,
            "adaptive_equal": adaptive_equal,
            "adaptive_components_equal": adaptive_parts,
        },
        "after": {"version": after["version"], "manifest_sha256": after["manifest_sha256"], "data_files_sha256": after["aggregate_sha256"]},
        "sync_calls": len(sync_calls),
        "checks": checks,
        "passed": all(checks.values()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="现有本地 Store 目录；默认读 .local/settings.json")
    parser.add_argument("--result-id", help="指定一个已保存、同 snapshot 的科学研究结果")
    parser.add_argument("--json-out", type=Path, default=PROJECT_ROOT / "docs" / "159611-minute-data-acceptance.json")
    parser.add_argument("--markdown-out", type=Path, default=PROJECT_ROOT / "docs" / "159611-minute-data-acceptance.md")
    args = parser.parse_args()
    data_dir = args.data_dir or Path(load_settings().data_dir)
    report = run(data_dir, args.result_id)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(args.json_out.resolve()), "passed": report["passed"], "checks": report["checks"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
