from __future__ import annotations

import pandas as pd
import pytest

from stockregression.market_data import MarketDataService
from stockregression.market_data.coverage_index import CoverageArchive
from stockregression.market_data.repository import validation_rules_version
from stockregression.quality import minute_slots
from stockregression.settings import load_settings
from stockregression.storage import Store


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("missing_opening", [False, True])
def test_minute_revalidation_publishes_current_rules_and_keeps_real_gaps(
    tmp_path, workers, missing_opening,
):
    settings = load_settings()
    settings.performance.cpu_budget_mode = "custom"
    settings.performance.cpu_budget = 2
    settings.performance.validation_workers = workers
    service = MarketDataService(Store(tmp_path / "data"), settings=settings)
    symbol = "512880.SH"
    days = pd.DatetimeIndex(["2026-01-05", "2026-02-02"])
    daily = pd.DataFrame({
        "timestamp": days, "open": 10.0, "high": 10.0, "low": 10.0,
        "close": 10.0, "preclose": 10.0, "volume": 24_000.0,
        "turnover": 240_000.0, "tradestatus": 1, "is_st": False,
        "source": "fixture", "source_verified": True,
        "is_synthetic": False, "is_auction_placeholder": False,
    })
    service.repository.commit_bars(symbol, "1day", daily, source="fixture")
    minute_parts = []
    for day in days:
        slots = minute_slots(day)
        omitted = {237, 238}
        if missing_opening and day == days[-1]:
            omitted.update(range(50))
        minute_parts.append(pd.DataFrame({
            "timestamp": [stamp for index, stamp in enumerate(slots) if index not in omitted],
            "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
            "volume": 100.0, "turnover": 1000.0, "source": "fixture",
            "source_verified": True, "is_synthetic": False,
            "is_auction_placeholder": False,
        }))
    service.repository.commit_bars(
        symbol, "1min", pd.concat(minute_parts, ignore_index=True), source="fixture",
    )
    service.repository.save_auxiliary(
        symbol, "calendar", [{"date": day} for day in days],
        version="fixture-calendar", verified=True,
    )
    service.repository.save_auxiliary(symbol, "actions", [], version="fixture-actions", verified=True)

    result = service.revalidate(symbol, "1min", start=days[0], end=days[-1])

    assert result["report"]["checked_days"] == 2
    assert result["report"]["cached"] is False
    assert result["report"]["parallel_workers"] == (2 if workers == 2 else 0)
    state = service.repository.validation_state(symbol, "1min", start=days[0], end=days[-1])
    assert state["pending"] == []
    assert state["failed"] == (["2026-02-02"] if missing_opening else [])
    assert result["status"] == ("partial" if missing_opening else "complete")
    rows = CoverageArchive(service.repository).rows(symbol, "1min", start=days[0], end=days[-1])
    assert len(rows) == 2
    for row in rows:
        assert row["rules_version"] == validation_rules_version("1min")
        assert not row.get("dirty_reason")
        assert row["auction_placeholders"] == 2
        assert row["synthetic_rows"] == 0
        assert row["report"]["expected_rows"] == 240
    assert rows[-1]["report"]["unfillable"] == (50 if missing_opening else 0)

    repeated = service.revalidate(symbol, "1min", start=days[0], end=days[-1])
    assert repeated["report"]["checked_days"] == 0
    assert repeated["report"]["cached"] is True
    assert repeated["status"] == result["status"]
