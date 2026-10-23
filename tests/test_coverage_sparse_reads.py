from __future__ import annotations

import pandas as pd

from stockregression.market_data import MarketDataService
from stockregression.storage import Store


def _write_daily(path, dates, close=10.2):
    pd.DataFrame({
        "date": dates,
        "open": 10.0,
        "high": 10.5,
        "low": 9.5,
        "close": close,
        "volume": 24_000.0,
        "turnover": 240_000.0,
        "preclose": 10.0,
        "tradestatus": 1,
        "is_st": 0,
    }).to_csv(path, index=False)
    return path


def test_sparse_merge_reads_only_affected_storage_periods(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    repository = service.repository
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": ["2019-06-03", "2022-06-03", "2026-06-03"]}).to_csv(
        calendar, index=False,
    )
    service.import_file(
        "600000.SH", "1day",
        _write_daily(tmp_path / "initial.csv", [
            "2019-06-03", "2022-06-03", "2026-06-03",
        ]),
        calendar_path=calendar,
    )

    with repository._connect() as db:
        shard_paths = {
            row["period_key"]: row["path"]
            for row in repository._current_shards(
                db, "600000.SH", "1day", "none", 1,
            )
        }
    read_calls = []
    original_read_paths = repository._read_paths

    def capture(paths, **kwargs):
        read_calls.append(tuple(paths))
        return original_read_paths(paths, **kwargs)

    monkeypatch.setattr(repository, "_read_paths", capture)
    service.import_file(
        "600000.SH", "1day",
        _write_daily(tmp_path / "sparse.csv", ["2019-06-03", "2026-06-03"]),
        calendar_path=calendar,
    )

    flattened = [path for call in read_calls for path in call]
    assert len(flattened) == 2
    assert set(flattened) == {shard_paths["year=2019"], shard_paths["year=2026"]}
    assert shard_paths["year=2022"] not in flattened
    assert repository.active_rows("600000.SH", "1day").timestamp.dt.year.tolist() == [2019, 2022, 2026]
