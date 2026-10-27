from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .data import DataService
from .models import BacktestConfig, OptimizationConfig
from .market_data import MarketDataService
from .market_data.aggregation import AggregationService
from .market_data.package import import_etf_package, index_etf_package_direct
from .market_data.maintenance import rebuild_state, update_rebuild_state
from .market_data.preparation import MarketPreparationService
from .market_data.sync import MarketDataSyncService
from .market_data.models import MarketSyncCancelled
from .performance import PerformanceTrace, activate_performance, reset_performance
from .research_preparation import prepare_research_bundle
from .resources import ResourceLease, clear_stale_leases
from .settings import Settings, load_settings
from .source_health import probe_source
from .storage import Store


class ClaimLostError(RuntimeError):
    pass


TIMEOUT_GRACE_SECONDS = 30
"""Extra seconds a compute job may use to stop cleanly after its limit."""


def _runtime_limit_seconds(job: dict) -> int:
    """Wall-clock limit of a running compute job (0 means unlimited)."""
    payload = job.get("payload") or {}
    if not isinstance(payload, dict):
        return 0
    if job.get("kind") in {"etf_research", "etf_validation"}:
        config = payload.get("config") or {}
        return int(config.get("max_runtime_seconds", 3600))
    if job.get("kind") == "research_completion":
        return int(payload.get("max_runtime_seconds", 0) or 0)
    if job.get("kind") == "research_validation":
        return int((payload.get("config") or {}).get("max_runtime_seconds", 86400))
    config = payload.get("config")
    strategy = config.get("strategy") if isinstance(config, dict) else None
    if not isinstance(strategy, dict):
        return 0
    return int(strategy.get("max_runtime_seconds", 0) or 0)


def _elapsed_seconds(job: dict, now: datetime | None = None) -> float:
    """Seconds since the job was claimed."""
    try:
        claimed = datetime.fromisoformat(str(job.get("claimed_at")))
    except (TypeError, ValueError):
        return 0.0
    if claimed.tzinfo is None:
        claimed = claimed.replace(tzinfo=timezone.utc)
    return ((now or datetime.now(timezone.utc)) - claimed).total_seconds()


def timeout_stage(elapsed: float, limit: int, grace: int = TIMEOUT_GRACE_SECONDS) -> str:
    """Escalation stage for a job that ran past its runtime limit.

    ``stop`` asks the worker to cancel cleanly so it can checkpoint and
    publish the evidence it already produced.  ``terminate`` kills the
    process tree and is only reached once the extra grace window is used up.
    """
    if limit <= 0:
        return "none"
    if elapsed > limit + max(0, int(grace)):
        return "terminate"
    if elapsed > limit:
        return "stop"
    return "none"


def _request_stop(store: Store, job_id: str, job: dict, limit: int) -> bool:
    """Ask a running compute job to stop and keep its partial evidence.

    The worker polls ``cancel_requested``; it finishes the current batch,
    saves what it already evaluated and releases the claim instead of being
    killed in the middle of a computation.
    """
    if job.get("cancel_requested"):
        return False
    return bool(store.update_job(
        job_id, require_claim=True, expected_claim_token=job.get("claim_token"),
        cancel_requested=1,
        message="任务超过时间上限，正在安全停止并保存已完成结果",
        error=f"任务运行时间超过 {limit} 秒，已请求安全停止，已完成结果保留",
    ))


def _with_completion_revision(payload: dict, result: dict) -> dict:
    """Return the job payload with the produced completion revision attached.

    The revision identifies the immutable artefact the completion produced,
    so the job can later resolve exactly that revision instead of whatever
    revision happens to be the newest one.
    """
    revision = result.get("completion_revision") if isinstance(result, dict) else None
    if revision is None:
        return payload
    return {**payload, "completion_revision": int(revision)}


def _cleanup_uploaded_input(store: Store, job_id: str):
    job = store.job(job_id) or {}
    payload = job.get("payload") or {}
    directories = payload.get("upload_dirs") or ([payload["upload_dir"]] if payload.get("upload_dir") else [])
    import_root = (store.root / "imports").resolve()
    # Resolve the package actually attached to this catalog. Global settings
    # may point to another data root; cleanup must not initialize a repository.
    with store.db() as db:
        has_meta = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='parquet_meta'"
        ).fetchone()
        package = db.execute(
            "SELECT value FROM parquet_meta WHERE key='package_dir'"
        ).fetchone() if has_meta else None
    package_root = Path(package[0]) if package else None
    if package_root is not None and not package_root.is_absolute():
        package_root = store.root / package_root
    staging_uploads = (
        (package_root / ".stockregression" / "staging" / "uploads").resolve()
        if package_root is not None else None
    )
    for relative in directories:
        path = Path(relative)
        target = path.resolve() if path.is_absolute() else (store.root / path).resolve()
        owned = (
            target != import_root and target.is_relative_to(import_root)
        ) or (
            staging_uploads is not None
            and target != staging_uploads and target.is_relative_to(staging_uploads)
        )
        if owned and target.is_dir():
            shutil.rmtree(target, ignore_errors=True)


def _research_protocol_hash(config: OptimizationConfig) -> str:
    payload = config.model_dump(mode="json")
    payload.pop("configuration_revision", None)
    from .research.space import UNITS
    payload["protocol_thresholds"] = {
        "protocol_version": 5,
        "result_schema_version": 2,
        "strategy": dict(payload.get("strategy", {})),
        "grid_mode": "geometric",
        "units": UNITS,
        "share_unit": 100,
        "negative_tolerance": 1e-12,
        "funding_mode": "top_up_on_fill",
        "return_basis": "peak_principal",
        "risk_basis": "principal_profit_drawdown",
        "ranking_baseline": "matched_funding_hold",
        "training_window": config.strategy.training_window_mode,
        "statistics": "not_applicable",
        "pressure_is_diagnostic_only": True,
        "validation_account_mode": "matched_funding_hold",
    }
    base = dict(payload.get("base", {}))
    for key in ("symbol", "start_date", "end_date"):
        base.pop(key, None)
    for key in ("grid_step", "buy_step", "sell_step", "buy_shares", "sell_shares", "min_shares", "max_shares"):
        base.pop(key, None)
    payload["base"] = base
    if isinstance(payload.get("research_context"), dict):
        context = dict(payload["research_context"])
        for key in ("symbol", "start_date", "end_date"):
            context.pop(key, None)
        payload["research_context"] = context
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _evaluation_intervals(start: str, end: str | None) -> list[dict]:
    """The v5 study has one real out-of-sample evaluation interval."""
    return [{"start": start, "end": end or start, "kind": "input_evaluation"}]


def _research_failed(result: dict) -> bool:
    if result.get("status") in {"failed", "error"}:
        return True
    branches = [result.get(name, {}) for name in ("fixed", "adaptive") if isinstance(result.get(name), dict)]
    statuses = [branch.get("status") for branch in branches if branch.get("status") is not None]
    return bool(statuses) and all(status in {"failed", "error"} for status in statuses)


def _aggregate_failure_categories(runs: list[dict], fallback: dict | None = None) -> dict[str, int]:
    """Aggregate per-run failure categories without counting run failures twice."""
    aggregated: dict[str, int] = {}
    for run in runs:
        categories = run.get("failure_categories")
        if isinstance(categories, dict) and categories:
            for category, count in categories.items():
                if isinstance(count, (int, float)) and not isinstance(count, bool):
                    key = str(category or "unknown")
                    aggregated[key] = aggregated.get(key, 0) + int(count)
            continue
        for failure in run.get("failures") or []:
            if not isinstance(failure, dict):
                continue
            key = str(failure.get("category") or failure.get("error_category") or "unknown")
            aggregated[key] = aggregated.get(key, 0) + 1
    if not aggregated and isinstance(fallback, dict):
        return {
            str(category or "unknown"): int(count)
            for category, count in fallback.items()
            if isinstance(count, (int, float)) and not isinstance(count, bool)
        }
    return aggregated


def _market_sync_cancel_report(completed_runs: list[dict], current: dict | None,
                               timeframes: list[str]) -> dict:
    """Merge finished timeframes with the partial report for the cancelled one."""
    report = current if isinstance(current, dict) else {}
    current_runs = report.get("runs")
    if isinstance(current_runs, list):
        runs = [*completed_runs, *[item for item in current_runs if isinstance(item, dict)]]
    elif report:
        runs = [*completed_runs, report]
    else:
        runs = list(completed_runs)
    failures = []
    failure_keys = set()
    for failure in [
        *(report.get("failures") or []),
        *[
            {**failure, **({"timeframe": run.get("timeframe")} if run.get("timeframe") else {})}
            for run in runs for failure in (run.get("failures") or [])
        ],
    ]:
        if not isinstance(failure, dict):
            continue
        key = json.dumps(failure, sort_keys=True, ensure_ascii=False, default=str)
        if key not in failure_keys:
            failure_keys.add(key)
            failures.append(failure)
    quality_partial = any(
        run.get("status") == "partial"
        or any(item.get("quality_status") in {"partial", "failed"}
               for item in [*(run.get("symbols") or []), *(run.get("skipped") or [])]
               if isinstance(item, dict))
        for run in runs
    ) or report.get("quality_status") in {"partial", "failed"}
    run_timeframes = [str(run["timeframe"]) for run in runs if run.get("timeframe")]
    merged = {
        **report,
        "status": "cancelled",
        "timeframes": list(dict.fromkeys([*timeframes, *run_timeframes])),
        "runs": runs,
        "failures": failures,
        "failure_categories": _aggregate_failure_categories(runs, report.get("failure_categories")),
        "message": "同步已取消，已保留各周期已完成与当前周期部分结果",
    }
    if quality_partial:
        merged["quality_status"] = "partial"
    else:
        merged.pop("quality_status", None)
    return merged


def execute_job(root: str, job_id: str, observer=None):
    store = Store(root)
    job = store.job(job_id)
    if not job:
        return
    claim_token = str(job.get("claim_token") or "")
    if job.get("status") != "running" or not claim_token:
        # A worker may only run a job it has atomically claimed.  A queued or
        # unclaimed job must be left untouched so the scheduler can still
        # claim it: an unguarded write would move it to "running" outside the
        # queue's ordering, cancellation and capacity rules.
        return
    frozen = (job.get("payload") or {}).get("configuration_snapshot")
    frozen_settings = (frozen or {}).get("settings") if isinstance(frozen, dict) else None
    try:
        configured = Settings.model_validate(frozen_settings) if isinstance(frozen_settings, dict) else load_settings()
    except ValueError:
        configured = load_settings()
    store = Store(root, settings=configured)
    trace = PerformanceTrace(job_id=job_id)
    lease = ResourceLease(store, job_id, configured.performance)
    cpu_units = lease.acquire()
    if cpu_units <= 0:
        # The configured CPU budget is fully committed to the other running
        # compute jobs.  Activating the performance context would floor the
        # share at one unit and oversubscribe the machine, so the job is
        # failed explicitly and both the claim and the lease are released.
        message = "计算配额已被其它任务占满，未能获得 CPU 份额；请稍后重试或调整计算预算"
        store.update_job(
            job_id, require_claim=True, expected_claim_token=claim_token,
            status="failed", message=message, error=message,
            claim_token=None, claimed_at=None,
        )
        lease.release()
        return
    performance_tokens = activate_performance(
        cpu_budget=cpu_units,
        workspace_mb=configured.performance.workspace_mb,
        trace=trace,
        rebalance=lease.rebalance,
        blas_threads=configured.performance.blas_threads,
    )
    trace.update(
        cpu_budget=cpu_units,
        workspace_budget_mb=configured.performance.workspace_mb,
        performance_settings=configured.performance.model_dump(mode="json"),
    )
    claim_lost = False

    def write_job(**values) -> None:
        nonlocal claim_lost
        if values.get("status") in {"completed", "partial", "failed", "cancelled"}:
            # The terminal write still validates the token above, then releases
            # ownership so a restarted process cannot continue this task.
            values.setdefault("claim_token", None)
            values.setdefault("claimed_at", None)
        updated = store.update_job(
            job_id, require_claim=True, expected_claim_token=claim_token, **values,
        )
        if not updated:
            claim_lost = True
            raise ClaimLostError("任务领取令牌已失效")

    try:
        write_job(
            status="running",
            # A job without a share is never started, so the message always
            # states the units the worker really got.
            message=f"正在执行 · {cpu_units} 个计算线程",
            pid=os.getpid(),
        )
    except ClaimLostError:
        # The claim was lost before the job could start.  Release the lease
        # and the performance context instead of leaking them.
        lease.release()
        reset_performance(performance_tokens)
        return
    last_progress = 0.0
    pending_progress_detail: dict | None = None
    last_progress_detail_stage: str | None = None
    progress_detail_lock = threading.Lock()
    last_cancel_check = 0.0
    cancel_state = False
    cancel_poll_seconds = max(.05, int(configured.performance.job_cancel_poll_ms) / 1000)
    progress_update_seconds = max(.05, int(configured.performance.progress_update_ms) / 1000)

    def cancelled():
        nonlocal last_cancel_check, cancel_state
        tick = time.monotonic()
        if tick - last_cancel_check > cancel_poll_seconds:
            current = store.job(job_id) or {}
            if current.get("claim_token") != claim_token:
                cancel_state = True
            else:
                cancel_state = bool(current.get("cancel_requested"))
            last_cancel_check = tick
        return cancel_state

    def progress(fraction, message):
        nonlocal last_progress, pending_progress_detail
        tick = time.monotonic()
        if tick - last_progress > progress_update_seconds or fraction <= 0 or fraction >= 1:
            rendered = str(message)
            if job.get("kind") in {"backtest", "optimization", "batch", "batch_backtest", "etf_research", "etf_validation", "research_validation"} and 0 < fraction < 1:
                rendered += f" · {max(1, lease.units)} 线程"
            normalized_fraction = max(0, min(1, float(fraction)))
            with progress_detail_lock:
                detail = pending_progress_detail
                pending_progress_detail = None
                write_job(
                    progress=normalized_fraction, message=rendered,
                    **({"progress_detail": detail} if detail is not None else {}),
                )
            if observer:
                observer(fraction, rendered)
            last_progress = tick

    def flush_progress_detail() -> None:
        nonlocal pending_progress_detail
        with progress_detail_lock:
            detail = pending_progress_detail
            pending_progress_detail = None
            if detail is not None:
                write_job(progress_detail=detail)
    try:
        service = DataService(store, settings=configured)
        payload, kind = job["payload"], job["kind"]
        computation_started = None
        submission_provenance = None
        if kind in {"optimization", "etf_research", "etf_validation", "research_validation"}:
            from .reproducibility import computation_provenance, reproducibility_mismatches
            from .settings import PROJECT_ROOT
            submission_provenance = payload.get("submission_provenance")
            computation_started = computation_provenance(
                PROJECT_ROOT, configuration=payload.get("config"),
            )
            mismatches = reproducibility_mismatches(submission_provenance, computation_started)
            if mismatches:
                raise RuntimeError("计算来源在任务提交后发生变化，请重新提交：" + "；".join(mismatches))
        if cancelled():
            raise InterruptedError("任务已取消")
        if kind == "backtest":
            from .engine import run_backtest
            config = BacktestConfig.model_validate(payload["config"])
            progress(0.01, "检查本地行情并准备回测数据")
            with trace.measure("data_preparation"):
                bundle = service.ensure_bundle(config, payload.get("version"), lambda fraction, message: progress(0.02 + 0.30 * fraction, message), cancelled)
            payload["version"] = bundle.version
            write_job(payload=payload)
            progress(0.33, "行情准备完成，开始回测")
            with trace.measure("backtest"):
                result = run_backtest(config, bundle, lambda fraction, message: progress(0.33 + 0.67 * fraction, message), cancelled)
            symbol, title = config.symbol, "现金补足网格 / 定投 / 买入持有" if config.schema_version == 3 else "股数网格 / 定投 / 买入持有" if config.schema_version == 2 else "网格 / 定投 / 初始全仓"
        elif kind == "research_validation":
            from .validation_models import ResearchValidationRequest
            from .validation_planner import load_validation_bundles
            from .validation_runner import run_validation
            from .validation_store import ValidationStore
            request = ResearchValidationRequest.model_validate(payload["config"])
            audit_store = ValidationStore(store)
            run_id = payload["run_id"]
            if not payload.get("resume_from_job"):
                audit_store.register_plan(run_id, payload["config"], payload["plan"], submission_provenance)
            progress(.01, "读取完整验证的冻结数据与检查点")
            bundles = load_validation_bundles(store, request, payload["plan"], settings=configured, cancelled=cancelled)

            def save_inner(inner, inner_symbol, arm_group):
                identity = uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{inner_symbol}:{arm_group}").hex
                inner = {**inner, "validation_run_id": run_id, "computation_provenance": {
                    "submission": submission_provenance, "started": computation_started,
                    "reproducible": None, "completion_allowed": False,
                    "reason": "内部单元的计算来源以所属完整验证最终审计为准",
                }}
                return store.save_result("optimization", inner, inner_symbol, "完整验证内部研究 · " + arm_group, result_id=identity)

            result = run_validation(
                request, payload["plan"], bundles,
                progress=lambda fraction, message: progress(.05 + .94 * fraction, message),
                cancelled=cancelled, runtime=configured.performance,
                checkpoint=audit_store.load_checkpoint(run_id),
                save_checkpoint=lambda value: audit_store.save_checkpoint(run_id, value),
                audit_event=lambda unit, event, value=None: audit_store.append_event(run_id, unit, event, value),
                save_inner_result=save_inner,
                save_unit_ledger=lambda unit, value: audit_store.save_ledger(run_id, unit, value),
                load_unit_ledger=audit_store.load_ledger,
            )
            result.update(kind="research_validation", run_id=run_id,
                          validation_protocol_version=1, result_schema_version=1,
                          config=request.model_dump(mode="json"), plan=payload["plan"],
                          checkpoint_snapshot=audit_store.freeze_checkpoint(run_id),
                          audit=audit_store.audit_summary(run_id))
            symbol, title = ",".join(request.symbols), "完整研究验证 · 自适应主方案"
        elif kind in {"etf_research", "etf_validation"}:
            from .etf_models import EtfResearchRequest
            from .etf_research import run_etf_research, run_local_validation
            config = EtfResearchRequest.model_validate(payload["config"])
            function = run_etf_research if kind == "etf_research" else run_local_validation
            result = function(
                store, config, payload["plan"],
                market=MarketDataService(store, settings=configured),
                progress=progress, cancelled=cancelled,
            )
            if payload.get("source_result_id"):
                result["source_result_id"] = payload["source_result_id"]
            symbol, title = "", "ETF 网格研究" if kind == "etf_research" else "ETF 本地批量校验"
        elif kind == "research_completion":
            from .research_completion import complete_research_result
            result_id = str(payload["result_id"])
            original = store.result(result_id)
            if original is None:
                raise ValueError("原始结果不存在")
            result = complete_research_result(
                store, service, result_id,
                decision=str(payload.get("decision", "current")),
                decision_index=int(payload.get("decision_index", 0)),
                reporting_budget=int(payload.get("reporting_budget", 1000)),
                target=str(payload.get("target", "return_leader")),
                # A fresh job owns a new checkpoint; a resumed job continues
                # the checkpoint of the job it resumes.
                checkpoint_id=str(payload.get("checkpoint_id") or job_id),
                progress=lambda fraction, message: progress(0.02 + 0.97 * fraction, message),
                cancelled=cancelled,
            )
            result["protocol"] = original.get("protocol")
            result["data_version"] = original.get("data_version")
            symbol = original.get("symbol") or ""
            title = f"研究结果补算 · {payload.get('decision', 'current')} · 修订 {result['completion_revision']}"
            # Record the produced revision next to the job so the exact
            # artefact this run created can be resolved later, even after
            # newer revisions were published for the same result.
            payload = _with_completion_revision(payload, result)
            write_job(payload=payload)
        elif kind == "optimization":
            from .optimization import optimize
            config = OptimizationConfig.model_validate(payload["config"])
            is_v5_research = config.protocol_version == 5
            progress(0.01, "检查本地行情并准备研究数据" if not is_v5_research else "准备评估区间及前置训练时段")
            with trace.measure("data_preparation"):
                if is_v5_research:
                    bundle = prepare_research_bundle(
                        service,
                        config.base,
                        payload.get("version"),
                        lambda fraction, message: progress(0.02 + 0.30 * fraction, message),
                        cancelled,
                        lookback_days=int(config.strategy.training_max_days),
                    )
                else:
                    bundle = service.ensure_bundle(config.base, payload.get("version"), lambda fraction, message: progress(0.02 + 0.30 * fraction, message), cancelled)
            payload["version"] = bundle.version
            write_job(payload=payload)
            progress(0.33, "行情准备完成，开始参数研究" if is_v5_research else "行情准备完成，开始参数搜索")
            with trace.measure("research"):
                result = optimize(
                    config, bundle,
                    lambda fraction, message: progress(0.33 + 0.67 * fraction, message),
                    cancelled, configured.performance, payload.get("budget_contract"),
                )
            if is_v5_research:
                result["symbol"] = config.base.symbol
                result["data_version"] = bundle.version
            result["diagnostics"] = bundle.metadata.get("diagnostics", {})
            result["source_events"] = bundle.metadata.get("source_events", [])
            symbol = config.base.symbol
            if is_v5_research:
                audit = store.register_research(
                    registry_id=job_id,
                    protocol_hash=_research_protocol_hash(config),
                    symbol=symbol,
                    input_start=str(config.base.start_date),
                    input_end=str(config.base.end_date) if config.base.end_date else str(bundle.calendar.max())[:10] if len(bundle.calendar) else None,
                    data_version=bundle.version,
                    evaluation_intervals=_evaluation_intervals(
                        str(config.base.start_date),
                        str(config.base.end_date) if config.base.end_date else str(bundle.calendar.max())[:10] if len(bundle.calendar) else None,
                    ),
                )
                result["evidence"] = {**(result.get("evidence") if isinstance(result.get("evidence"), dict) else {}), "local_audit": audit}
                result["protocol_hash"] = audit["protocol_hash"]
                title = "稳健 J v5 峰值研究" + (" · 固定与自适应" if config.selection_target == "both" else f" · {config.selection_target}")
            else:
                title = "历史搜索 / 样本外验证"
        elif kind == "market_import":
            market = MarketDataService(store, settings=configured)
            result = market.import_file(
                payload["symbol"], payload["timeframe"], payload["path"],
                label_convention=payload.get("label_convention", "auto"),
                adjustment=payload.get("adjustment", "none"), etf_category=payload.get("etf_category"),
                calendar_path=payload.get("calendar_path"), actions_path=payload.get("actions_path"),
                skip_invalid_rows=payload.get("skip_invalid_rows", False),
                import_mode=payload.get("import_mode", "merge"),
                progress=progress, cancelled=cancelled,
            )
            symbol, title = payload["symbol"], f"导入 {payload['timeframe']} 行情"
        elif kind == "market_import_batch":
            market = MarketDataService(store, settings=configured)
            result = market.import_folder(
                payload["folder"], payload["timeframe"], recursive=payload.get("recursive", False),
                label_convention=payload.get("label_convention", "auto"),
                adjustment=payload.get("adjustment", "none"), etf_category=payload.get("etf_category"),
                resume=payload.get("resume", True), calendar_path=payload.get("calendar_path"),
                workers=payload.get("workers", configured.performance.download_workers), skip_invalid_rows=payload.get("skip_invalid_rows", False),
                import_mode=payload.get("import_mode", "merge"),
                progress=progress, cancelled=cancelled,
            )
            symbol, title = "", f"批量导入 {payload['timeframe']}"
        elif kind == "market_import_package":
            market = MarketDataService(store, settings=configured)
            update_rebuild_state(store, stage="importing")
            result = import_etf_package(
                market, payload["folder"], timeframes=payload.get("timeframes"),
                symbols=payload.get("symbols"),
                label_convention=payload.get("label_convention", "auto"),
                resume=payload.get("resume", True), workers=payload.get("workers", configured.performance.download_workers),
                skip_invalid_rows=payload.get("skip_invalid_rows", False),
                progress=progress, cancelled=cancelled,
            )
            update_rebuild_state(store, stage="finished", result=result)
            symbol, title = "", "ETF 数据包导入"
        elif kind == "market_index_package":
            update_rebuild_state(store, stage="indexing")
            result = index_etf_package_direct(
                MarketDataService(store, settings=configured), payload["folder"],
                timeframes=payload.get("timeframes"), symbols=payload.get("symbols"),
                progress=progress, cancelled=cancelled,
            )
            update_rebuild_state(store, stage="finished", result=result)
            symbol, title = "", "登记 ETF 原始 Parquet 数据包"
        elif kind == "market_sync":
            timeframes = payload.get("timeframes") or [payload["timeframe"]]
            symbols = payload.get("symbols", [])
            scope = payload.get("scope")
            if "1min" in timeframes and payload.get("mode") == "history":
                raise ValueError("1 分钟历史行情只从本地数据包读取；请先登记数据包或导入文件，近期缺口使用缺口修补。")
            if "1min" in timeframes and scope == "market" and payload.get("initialize"):
                raise ValueError("1 分钟行情不支持全市场在线初始化；请登记本地数据包或导入文件。")
            combined = []
            failures = []
            sync_cancelled = False
            ordered = sorted(set(timeframes), key=lambda value: 0 if value == "1day" else 1)
            cutoff_date = payload.get("cutoff_date")
            if not cutoff_date:
                from .providers import latest_closed_date
                cutoff_date = str(latest_closed_date().date())
            write_job(message="正在核对数据包清单与同步检查点", progress_detail={
                "stage": "package_recovery", "cutoff_date": str(cutoff_date),
                "counts": {"total": len(symbols), "processed": 0, "succeeded": 0,
                           "failed": 0, "reused": 0, "skipped": 0,
                           "unprocessed": len(symbols)},
                "resources": {"download_workers": 0, "baostock_sessions": 0, "write_workers": 1},
            })
            sync_service = MarketDataSyncService(MarketDataService(store, settings=configured))
            package_id = str(sync_service.repository.package_id)
            if payload.get("package_id") and str(payload["package_id"]) != package_id:
                raise ValueError("任务绑定的行情数据包已切换，不能继续原同步检查点")
            package_generation = str(sync_service.repository.generation)
            if payload.get("package_generation") and str(payload["package_generation"]) != package_generation:
                raise ValueError("任务绑定的行情数据包代际已变化，不能继续原同步检查点")

            def period_progress(index, fraction, message):
                base = index / len(ordered)
                span = 1 / len(ordered)
                progress(base + span * max(0.0, min(1.0, float(fraction))), message)

            frozen_by_timeframe = payload.get("sync_symbols_by_timeframe")
            if not isinstance(frozen_by_timeframe, dict):
                frozen_by_timeframe = {}
            if not frozen_by_timeframe:
                if scope == "market" and payload.get("initialize"):
                    service.refresh_instruments(
                        lambda fraction, message: period_progress(0, .1 * fraction, message),
                        cancelled,
                    )
                    selected_symbols = [row["symbol"] for row in store.instruments(limit=None)]
                    frozen_by_timeframe = {timeframe: list(selected_symbols) for timeframe in ordered}
                elif scope == "market":
                    frozen_by_timeframe = {
                        timeframe: list(sync_service.repository.symbols_for_timeframe(timeframe))
                        for timeframe in ordered
                    }
                elif scope == "watchlist":
                    selected_symbols = [row["symbol"] for row in store.watches()]
                    frozen_by_timeframe = {timeframe: list(selected_symbols) for timeframe in ordered}
                else:
                    frozen_by_timeframe = {timeframe: list(symbols) for timeframe in ordered}
            else:
                frozen_by_timeframe = {
                    timeframe: list(frozen_by_timeframe.get(timeframe, symbols))
                    for timeframe in ordered
                }
            symbols = list(dict.fromkeys(
                symbol for timeframe in ordered for symbol in frozen_by_timeframe[timeframe]
            ))
            payload.update({
                "symbols": symbols,
                "sync_symbols_by_timeframe": frozen_by_timeframe,
                "package_id": package_id,
                "package_generation": package_generation,
                "cutoff_date": str(cutoff_date),
            })
            write_job(payload=payload)

            def sync_detail(detail):
                nonlocal pending_progress_detail, last_progress_detail_stage
                if not isinstance(detail, dict):
                    return
                counts = detail.get("counts") if isinstance(detail.get("counts"), dict) else {}
                resources = detail.get("resources") if isinstance(detail.get("resources"), dict) else {}
                snapshot = {
                    "stage": str(detail.get("stage") or "sync"),
                    "symbol": detail.get("symbol"),
                    "checked_days": detail.get("checked_days"),
                    "total_days": detail.get("total_days"),
                    "counts": {
                        name: int(counts.get(name, 0) or 0)
                        for name in ("total", "processed", "succeeded", "failed", "reused", "skipped", "unprocessed")
                    },
                    "resources": {
                        "download_workers": int(resources.get("download_workers", configured.performance.download_workers)),
                        "baostock_sessions": int(resources.get("baostock_sessions", configured.performance.download_workers)),
                        "write_workers": int(resources.get("write_workers", configured.performance.download_workers)),
                    },
                    "cutoff_date": str(detail.get("cutoff_date") or cutoff_date),
                }
                with progress_detail_lock:
                    stage_changed = snapshot["stage"] != last_progress_detail_stage
                    last_progress_detail_stage = snapshot["stage"]
                    pending_progress_detail = snapshot
                    if stage_changed:
                        pending_progress_detail = None
                        write_job(progress_detail=snapshot)
            # Every timeframe reports its own 0..1 fraction.  Giving each one
            # a slice of a single scale keeps the bar from restarting at 0%
            # every time the next period begins, so the job reads as one
            # continuous run instead of several short ones.
            for index, timeframe in enumerate(ordered):
                base = index / len(ordered)
                span = 1 / len(ordered)

                def period_progress(fraction, message, base=base, span=span):
                    progress(base + span * max(0.0, min(1.0, float(fraction))), message)

                selected = list(frozen_by_timeframe[timeframe])
                sync_progress = period_progress
                if scope == "market" and payload.get("initialize") and index == 0:
                    sync_progress = lambda fraction, message: period_progress(
                        0.1 + 0.9 * max(0.0, min(1.0, float(fraction))), message,
                    )
                try:
                    one = MarketDataSyncService(
                        sync_service.service, job_id=job_id, job_claim_token=claim_token,
                        detail=sync_detail, cutoff_date=cutoff_date,
                    ).sync(
                        selected, timeframe, payload.get("mode", "incremental"),
                        start=payload.get("start_date"), end=payload.get("end_date"),
                        resume=payload.get("resume", False), retry_failed=payload.get("retry_failed", False),
                        workers=payload.get("workers", configured.performance.download_workers),
                        progress=sync_progress, cancelled=cancelled,
                    )
                except MarketSyncCancelled as exc:
                    result = _market_sync_cancel_report(combined, exc.report, ordered)
                    sync_cancelled = True
                    break
                combined.append(one)
                failures.extend({**item, "timeframe": timeframe} for item in one.get("failures", []))
            if not sync_cancelled:
                succeeded = sum(len(run.get("symbols", [])) for run in combined)
                skipped = sum(len(run.get("skipped", [])) for run in combined)
                quality_partial = any(
                    run.get("status") == "partial"
                    or any(item.get("quality_status") in {"partial", "failed"}
                           for item in [*(run.get("symbols") or []), *(run.get("skipped") or [])]
                           if isinstance(item, dict))
                    for run in combined
                )
                sync_status = "partial" if quality_partial or (failures and (succeeded or skipped)) else "failed" if failures else "completed"
                result = {
                    "status": sync_status,
                    "timeframes": sorted(set(timeframes)),
                    "runs": combined, "failures": failures,
                    "failure_categories": _aggregate_failure_categories(combined),
                    "message": f"处理 {len(combined)} 个基础周期：成功 {succeeded} 项，跳过 {skipped} 项，失败 {len(failures)} 项",
                }
            symbol, title = ",".join(symbols), f"基础行情同步 · {', '.join(sorted(set(timeframes)))}"
        elif kind == "market_source_probe":
            probe_options = {
                key: payload[key] for key in ("product_type", "sample_symbol")
                if payload.get(key) is not None
            }
            result = probe_source(
                store, payload["source"], payload["capability"],
                payload.get("market", "all"), cancelled, **probe_options,
            )
            symbol, title = "", f"数据源探测 · {payload['source']}"
        elif kind == "market_validate":
            result = MarketDataService(store, settings=configured).revalidate(
                payload["symbol"], payload["timeframe"],
                label_convention=payload.get("label_convention"),
                mode=payload.get("mode", "incremental"),
                start=payload.get("start_date"), end=payload.get("end_date"),
            )
            symbol, title = payload["symbol"], f"校验 {payload['timeframe']} 行情"
        elif kind == "market_archive":
            market = MarketDataService(store, settings=configured)
            values = []
            for symbol in payload.get("symbols", []):
                values.append(market.repository.archive_dataset(
                    symbol, payload["timeframe"], progress=progress, cancelled=cancelled,
                ))
            result = {
                "status": "completed", "timeframe": payload["timeframe"],
                "symbols": values,
            }
            symbol, title = ",".join(payload.get("symbols", [])), "保护历史行情快照"
        elif kind == "market_prepare":
            market = MarketDataService(store, settings=configured)
            config = BacktestConfig.model_validate(payload["config"])
            result = MarketPreparationService(market).execute(
                config, progress=progress, cancelled=cancelled,
            )
            symbol, title = config.symbol, f"准备 {config.frequency} 行情"
        elif kind == "market_aggregate":
            market = MarketDataService(store, settings=configured)
            aggregated = AggregationService(market)
            values = []
            aggregate_failures = []
            for symbol in payload["symbols"]:
                if cancelled():
                    raise InterruptedError("任务已取消")
                try:
                    values.append(aggregated.aggregate(
                        symbol, payload["source_timeframe"], payload["target_timeframe"],
                        start=payload.get("start_date"), end=payload.get("end_date"),
                        progress=progress, cancelled=cancelled,
                    ))
                except InterruptedError:
                    raise
                except Exception as exc:
                    aggregate_failures.append({"symbol": symbol, "reason": str(exc)})
            failures = [*aggregate_failures, *[item for item in values if item.get("status") == "failed"]]
            result = {"status": "completed" if not failures else "partial", "results": values,
                      "targets": len(payload["symbols"]), "completed": len(values) - len(failures),
                      "failures": failures, "message": f"聚合完成 {len(values) - len(failures)} 个标的"}
            symbol, title = ",".join(payload["symbols"]), f"本地聚合 {payload['source_timeframe']} → {payload['target_timeframe']}"
        elif kind == "instruments":
            result = service.refresh_instruments(progress, cancelled)
            symbol, title = "", "证券清单更新"
        else:
            raise ValueError("未知任务类型：" + kind)
        if cancelled() and kind not in {"optimization", "etf_research", "etf_validation", "research_validation"} \
                and not (kind == "market_sync" and result.get("status") == "cancelled"):
            raise InterruptedError("任务已取消")
        was_cancelled = cancelled()
        failed_sync = kind in {"market_sync", "market_source_probe", "market_import", "market_import_batch", "market_import_package", "market_index_package", "market_aggregate", "market_prepare", "market_archive"} and result.get("status") == "failed"
        failed_research = (kind == "optimization" and payload.get("config", {}).get("protocol_version") == 5 and _research_failed(result)) or (kind == "research_validation" and result.get("status") == "failed")
        result_cancelled = kind in {"optimization", "etf_research", "etf_validation", "research_validation"} and result.get("status") == "cancelled"
        cancelled_job = was_cancelled or result_cancelled or (
            kind == "market_sync" and result.get("status") == "cancelled"
        )
        if kind == "optimization" and payload.get("config", {}).get("protocol_version") == 5:
            if failed_research:
                title += " · 失败"
            elif cancelled_job:
                title += " · 已取消"
            elif result.get("status") == "partial":
                title += " · 部分完成"
        if isinstance(result, dict):
            result["performance"] = trace.snapshot()
            result["configuration_snapshot"] = frozen if isinstance(frozen, dict) else None
            if kind in {"optimization", "etf_research", "etf_validation", "research_validation"}:
                from .reproducibility import computation_provenance, reproducibility_mismatches
                from .settings import PROJECT_ROOT
                computation_finished = computation_provenance(
                    PROJECT_ROOT, configuration=payload.get("config"),
                )
                finish_mismatches = reproducibility_mismatches(computation_started, computation_finished)
                result["computation_provenance"] = {
                    "submission": submission_provenance,
                    "started": computation_started,
                    "finished": computation_finished,
                    "reproducible": not finish_mismatches,
                    "mismatches": finish_mismatches,
                    "completion_allowed": bool(
                        not finish_mismatches
                        and computation_finished.get("engine_version")
                        and computation_finished.get("git_commit")
                        and computation_finished.get("source_fingerprint")
                        and computation_finished.get("dependency_lock_hash")
                        and computation_finished.get("configuration_fingerprint")
                    ),
                }
                if finish_mismatches:
                    result["status"] = "failed"
                    if kind == "research_validation":
                        result["completion_status"] = "incomplete"
                        result["qualification_status"] = "unevaluable"
                    result["reproducibility_error"] = (
                        "任务运行期间计算源码、配置或依赖发生变化，结果不可复现，请重新提交"
                    )
                    failed_research = True
                    title += " · 计算来源变化"
            result["execution"] = {
                "configuration_revision": (frozen or {}).get("revision") if isinstance(frozen, dict) else None,
                "data_version": payload.get("version"),
                "resource_settings": configured.performance.model_dump(mode="json"),
                "cpu_units": int(cpu_units),
            }
        if kind == "market_sync":
            flush_progress_detail()
        if kind == "research_completion":
            # The revision is stored beside the original immutable result.  A
            # separate result artifact would look like a second research study
            # and would not contain the original protocol/configuration.
            result_id = str(result.get("result_id") or "")
        else:
            deterministic_id = (
                f"market-sync-cancel-{job_id}"
                if kind == "market_sync" and result.get("status") == "cancelled" else None
            )
            result_id = store.save_result(kind, result, symbol, title, result_id=deterministic_id)
        sync_error = None
        if failed_sync:
            failures = result.get("failures", [])
            sync_error = "; ".join(f"{item.get('symbol', '标的')}: {item.get('reason', '未知错误')}" for item in failures) or result.get("message") or "所有同步目标失败"
        if cancelled_job:
            message = "同步已取消，已保存部分报告" if kind == "market_sync" else "已取消，已完成的搜索候选保留"
        elif failed_sync:
            message = f"同步失败：{sync_error}"
        elif failed_research:
            message = "参数研究失败：" + str(result.get("reason") or result.get("message") or "查看结果中的失败原因")
        elif kind == "optimization" and payload.get("config", {}).get("protocol_version") == 5 and result.get("status") == "partial":
            message = "参数研究部分完成；已保留已完成的候选与结果"
        else:
            message = result.get("message", "完成")
        partial = result.get("status") == "partial" or (kind == "research_validation" and result.get("completion_status") == "incomplete")
        if kind == "research_completion" and partial:
            message = "补算部分完成，已保存可追踪修订与检查点"
        terminal_error = sync_error or (result.get("reason") if failed_research else None)
        if terminal_error is None and cancelled_job:
            # Keep the stop reason the dispatcher recorded, so a run that was
            # stopped because it ran out of time does not look like a plain
            # user cancellation.
            terminal_error = (store.job(job_id) or {}).get("error")
        terminal_values = {
            "status": "cancelled" if cancelled_job else "failed" if failed_sync or failed_research else "partial" if partial else "completed",
            "message": message, "error": terminal_error, "result_id": result_id,
        }
        if not cancelled_job:
            terminal_values["progress"] = 1
        write_job(**terminal_values)
    except ClaimLostError:
        return
    except InterruptedError as exc:
        original_result = None
        if job.get("kind") == "research_completion":
            original_result = (job.get("payload") or {}).get("result_id")
        completion_revision = getattr(exc, "completion_revision", None)
        if original_result and completion_revision is not None:
            payload = _with_completion_revision(
                job.get("payload") or {}, {"completion_revision": completion_revision},
            )
            write_job(status="cancelled", message=str(exc), result_id=original_result, payload=payload)
            return
        write_job(status="cancelled", message=str(exc), **({"result_id": original_result} if original_result else {}))
    except Exception as exc:
        folder = store.root / "logs"
        folder.mkdir(exist_ok=True)
        (folder / f"{job_id}.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_job(status="failed", message="任务失败，查看原因后可重试", error=str(exc))
    finally:
        if not claim_lost:
            _cleanup_uploaded_input(store, job_id)
        lease.release()
        reset_performance(performance_tokens)


class JobManager:
    def __init__(self, root: str, workers=2, poll_seconds: float | None = None):
        self.root, self.workers = root, workers
        self.poll_seconds = float(poll_seconds or load_settings().market_data.scheduler_poll_seconds)
        self.stop_event = threading.Event()
        self.claim_token = uuid.uuid4().hex
        self.active: dict[str, multiprocessing.Process] = {}
        self.last_recovery = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True, name="grid-job-dispatcher")

    def _recover_orphans(self, store: Store, startup_grace: float = 30.0) -> None:
        cursor = None
        now = datetime.now(timezone.utc)
        while True:
            page = store.jobs_by_status("running", after=cursor, limit=200)
            if not page:
                break
            for job in page:
                if job.get("pid") and _pid_alive(job["pid"]):
                    continue
                updated: datetime | None = None
                try:
                    updated = datetime.fromisoformat(str(job.get("updated_at")))
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    pass
                starting = job.get("message") == "正在启动计算进程"
                if starting and updated is not None and (now - updated).total_seconds() < startup_grace:
                    continue
                marked = store.update_job(
                    job["id"], require_claim=bool(job.get("claim_token")),
                    expected_claim_token=job.get("claim_token"), status="failed",
                    claim_token=None, claimed_at=None,
                    message="启动超时" if starting else "执行中断",
                    error="应用或计算进程中断；原数据版本保留，请重新运行",
                )
                current = store.job(job["id"]) or {}
                if marked or current.get("status") != "running":
                    # The dead job must not keep holding its CPU lease.
                    _release_job_lease(store, job["id"])
                    _cleanup_uploaded_input(store, job["id"])
            cursor = (page[-1]["created_at"], page[-1]["id"])

    def start(self):
        # Previously interrupted application jobs are reviewable and retryable.
        store = Store(self.root)
        self._recover_orphans(store)
        clear_stale_leases(store)
        self.thread.start()

    def _loop(self):
        context = multiprocessing.get_context("spawn")
        store = Store(self.root)
        while not self.stop_event.wait(self.poll_seconds):
            if time.monotonic() - self.last_recovery >= 5.0:
                self._recover_orphans(store)
                self.last_recovery = time.monotonic()
            for job_id, process in list(self.active.items()):
                if not process.is_alive():
                    process.join()
                    self.active.pop(job_id)
                    job = store.job(job_id)
                    if job and job["status"] == "running":
                        store.update_job(
                            job_id, require_claim=bool(job.get("claim_token")),
                            expected_claim_token=job.get("claim_token"), status="failed",
                            claim_token=None, claimed_at=None,
                            error="计算进程异常退出", message="进程中断",
                        )
                    _release_job_lease(store, job_id)
                    _cleanup_uploaded_input(store, job_id)
                    continue
                job = store.job(job_id) or {}
                limit = _runtime_limit_seconds(job)
                if job.get("kind") in {"optimization", "research_completion", "etf_research", "etf_validation", "research_validation"} and limit > 0:
                    elapsed = _elapsed_seconds(job)
                    stage = timeout_stage(elapsed, limit)
                    if stage == "stop":
                        # Ask the worker to stop first: it checkpoints and
                        # publishes the evidence it already produced.
                        _request_stop(store, job_id, job, limit)
                    elif stage == "terminate":
                        current = store.job(job_id) or {}
                        if current.get("status") != "running":
                            # The worker already published its final evidence.
                            continue
                        _terminate_process_tree(process.pid)
                        process.join(timeout=1)
                        self.active.pop(job_id, None)
                        store.update_job(
                            job_id, require_claim=True,
                            expected_claim_token=current.get("claim_token"),
                            status="failed", message="任务超过时间上限，计算进程已终止",
                            error="达到时间预算后 30 秒仍未退出，已终止进程",
                            claim_token=None, claimed_at=None,
                        )
                        _release_job_lease(store, job_id)
                        _cleanup_uploaded_input(store, job_id)
            if len(self.active) >= self.workers:
                continue
            cancelled_ids: list[str] = []
            claimed_ids = store.claim_jobs(
                self.claim_token, self.workers, cancelled_ids=cancelled_ids,
            )
            for cancelled_id in cancelled_ids:
                _cleanup_uploaded_input(store, cancelled_id)
            for job_id in claimed_ids:
                # Batch workers spawn their own BaoStock/validation children.
                # A daemonic parent is forbidden from creating those children.
                process = context.Process(target=execute_job, args=(self.root, job_id), daemon=False)
                try:
                    process.start()
                    if not store.update_job(
                        job_id, require_claim=True, expected_claim_token=self.claim_token,
                        pid=process.pid,
                    ):
                        _terminate_process_tree(process.pid)
                        process.join(timeout=1)
                        _cleanup_uploaded_input(store, job_id)
                        continue
                    self.active[job_id] = process
                except Exception as exc:
                    store.update_job(
                        job_id, require_claim=True, expected_claim_token=self.claim_token,
                        status="failed", error="无法启动计算进程：" + str(exc),
                        message="进程启动失败", claim_token=None, claimed_at=None,
                    )
                    _cleanup_uploaded_input(store, job_id)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        store = Store(self.root)
        for job_id, process in self.active.items():
            store.update_job(job_id, cancel_requested=1)
            process.join(timeout=3)
            if process.is_alive():
                _terminate_process_tree(process.pid)
                process.join(timeout=1)
                job = store.job(job_id) or {}
                store.update_job(
                    job_id, require_claim=bool(job.get("claim_token")),
                    expected_claim_token=job.get("claim_token"),
                    status="cancelled", message="应用已退出；计算任务已停止",
                    claim_token=None, claimed_at=None,
                )
            _cleanup_uploaded_input(store, job_id)
            _release_job_lease(store, job_id)


def _terminate_process_tree(pid: int | None) -> None:
    """Cancel a job together with validation/session child processes."""
    if not pid:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        return
    try:
        os.kill(int(pid), 15)
    except OSError:
        pass


def _release_job_lease(store: Store, job_id: str) -> None:
    with store.db() as db:
        db.execute("DELETE FROM resource_leases WHERE job_id=?", (job_id,))


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
