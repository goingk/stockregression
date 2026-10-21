"""v5 研究协议 S4：候选档案（SQLite）的分页、惰性读取与导出契约。

这些用例只覆盖 v5 新契约（protocol_version=5 / result_schema_version=2）：
候选行落盘为 root/result_archives/<sha256>.sqlite，结果 JSON 里只保留
{"__candidate_archive__": ...} 引用；分页读取时只反序列化当前页；
摘要读取不打开候选档案；导出时把档案流式展开成候选行。

不涉及 protocol v4 的 bootstrap/gzip 归档路径。
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stockregression import storage
from stockregression.api import create_app
from stockregression.storage import Store


ARCHIVE_REFERENCE_KEY = "__candidate_archive__"
ARCHIVE_FOLDER_NAME = "result_archives"


def _candidate_row(index: int) -> dict:
    """构造一条 v5 候选行；index 同时决定 candidate_id 与四轴参数。"""
    return {
        "candidate_id": index + 1,
        "params": {
            "buy_step": round(1 + index * 0.1, 6),
            "sell_step": round(2 + index * 0.1, 6),
            "buy_shares": round(100 + index, 6),
            "sell_shares": round(200 + index, 6),
        },
        "j_score": round(10.0 - index * 0.01, 8),
        "training_excess_return": round(0.01 * index, 10),
        "max_principal_drawdown": round(-0.02 * index, 10),
        "neighborhood_mean": round(0.5 - index * 0.001, 10),
        "downside_risk_tau": round(0.3 + index * 0.001, 10),
        "pos_penalty": round(index * 0.0001, 10),
        "score_complete": True,
        "region_ids": [1 + (index % 3)],
    }


def _v5_result(current_rows, fixed_rows=()) -> dict:
    """构造一个 protocol_version=5 / result_schema_version=2 的研究结果。"""
    return {
        "status": "completed",
        "protocol_version": 5,
        "result_schema_version": 2,
        "symbol": "600000.SH",
        "data_version": "test-version",
        "config": {
            "base": {
                "symbol": "600000.SH",
                "start_date": "2026-01-05",
                "end_date": "2026-02-27",
            },
            "top_candidates": 20,
        },
        "fixed": {"status": "completed", "test_result": None},
        "adaptive": {"status": "not_requested", "schedule": [], "test_result": None},
        "evidence": {
            "candidate_archives": {
                "fixed": list(fixed_rows),
                "current": list(current_rows),
                "adaptive": [],
            },
            "trial_archives": {"fixed": [], "current": [], "adaptive": []},
            "search_candidate_archives": {
                "fixed": list(fixed_rows),
                "current": list(current_rows),
                "adaptive": [],
            },
            "decisions": [],
            "reasons": [],
        },
        "computation": {
            "decision_count": 1,
            "candidate_evaluations": len(current_rows),
            "ledger_evaluations": 0,
        },
    }


def _completion_revision(store, result_id, *, decision="current", decision_index=0,
                         group="candidate_archives", overlay_rows=()):
    """写一条补算修订：合并档案覆盖某个决策分支。"""
    stored = store.result(result_id)
    base = stored["evidence"][group][decision]
    overlay = storage._write_candidate_archive(store.root, list(overlay_rows))
    merged = store.merge_candidate_archives(base, overlay)
    return store.save_completion_revision(result_id, {
        "decision": decision, "decision_index": decision_index, "status": "completed",
        "candidate_archive": overlay, "merged_candidate_archive": merged,
    })


def _store_with_archives(tmp_path: Path):
    """落盘一个 v5 结果：current 分支 100 行，fixed 分支 30 行。"""
    store = Store(tmp_path)
    rows = [_candidate_row(index) for index in range(100)]
    fixed_rows = [_candidate_row(1000 + index) for index in range(30)]
    result_id = store.save_result(
        "research", _v5_result(rows, fixed_rows=fixed_rows), "600000.SH", "v5 候选档案",
    )
    return store, result_id, rows, fixed_rows


@contextmanager
def _record_json_loads(monkeypatch):
    """记录 json.loads 的调用文本，用来判断实际反序列化的规模。"""
    real_loads = json.loads
    parsed: list[str] = []

    def counting_loads(payload, *args, **kwargs):
        if isinstance(payload, (bytes, bytearray)):
            parsed.append(bytes(payload).decode("utf-8", "replace"))
        elif isinstance(payload, str):
            parsed.append(payload)
        return real_loads(payload, *args, **kwargs)

    monkeypatch.setattr(storage.json, "loads", counting_loads)
    try:
        yield parsed
    finally:
        monkeypatch.setattr(storage.json, "loads", real_loads)


def _candidate_row_payloads(parsed: list[str]) -> list[str]:
    """候选行 JSON 的特征：以 '{' 开头且含 '"params"'。"""
    return [text for text in parsed if text.startswith("{") and '"params"' in text]


@contextmanager
def _record_sqlite_connects(monkeypatch):
    """记录 sqlite3.connect 被打开的路径。"""
    real_connect = sqlite3.connect
    opened: list[str] = []

    def recording_connect(target, *args, **kwargs):
        opened.append(str(target))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(storage.sqlite3, "connect", recording_connect)
    try:
        yield opened
    finally:
        monkeypatch.setattr(storage.sqlite3, "connect", real_connect)


def _archive_sqlite_opens(opened: list[str], root: Path) -> list[str]:
    """筛出位于 root/result_archives 下的 .sqlite 打开记录。"""
    folder = (root / ARCHIVE_FOLDER_NAME).resolve()
    hits: list[str] = []
    for raw in opened:
        try:
            path = Path(raw).resolve()
        except (TypeError, OSError, ValueError):
            continue
        if path.parent == folder and path.suffix == ".sqlite":
            hits.append(str(path))
    return hits


def test_candidate_archive_paginates_and_only_deserializes_the_requested_page(tmp_path, monkeypatch):
    """分页只反序列化当前页的候选行，total 始终是整份档案的行数。"""
    store, result_id, rows, fixed_rows = _store_with_archives(tmp_path)
    stored = store.result(result_id)

    branch = stored["evidence"]["candidate_archives"]["current"]
    assert branch[ARCHIVE_REFERENCE_KEY] == f"{ARCHIVE_FOLDER_NAME}/{branch['archive_hash']}.sqlite"
    assert branch["row_count"] == 100
    assert stored["evidence"]["candidate_archives"]["fixed"]["row_count"] == 30

    with _record_json_loads(monkeypatch) as parsed:
        first = store.candidate_archive(
            stored, "candidate_archives", "current", limit=10, offset=0, sort_by="ordinal",
        )
    assert first["total"] == 100
    assert first["offset"] == 0
    assert first["limit"] == 10
    assert len(first["rows"]) == 10
    assert [row["candidate_id"] for row in first["rows"]] == list(range(1, 11))
    # 关键契约：只反序列化本页的 10 行，而不是整份档案的 100 行。
    assert len(_candidate_row_payloads(parsed)) == 10

    with _record_json_loads(monkeypatch) as parsed:
        second = store.candidate_archive(
            stored, "candidate_archives", "current", limit=10, offset=10, sort_by="ordinal",
        )
    assert second["total"] == 100
    assert second["offset"] == 10
    assert len(second["rows"]) == 10
    assert [row["candidate_id"] for row in second["rows"]] == list(range(11, 21))
    assert len(_candidate_row_payloads(parsed)) == 10

    first_ids = {row["candidate_id"] for row in first["rows"]}
    second_ids = {row["candidate_id"] for row in second["rows"]}
    assert first_ids.isdisjoint(second_ids)

    # 对照组：一次性读完整份档案会反序列化全部 100 行，说明上面的计数确实能区分规模。
    with _record_json_loads(monkeypatch) as parsed:
        everything = store.candidate_archive(
            stored, "candidate_archives", "current", limit=100, offset=0, sort_by="ordinal",
        )
    assert len(everything["rows"]) == 100
    assert len(_candidate_row_payloads(parsed)) == 100


def test_result_summary_never_opens_candidate_archive_sqlite(tmp_path, monkeypatch):
    """摘要读取（store.result / store.results / GET 结果详情）不打开候选档案 SQLite。"""
    store, result_id, rows, fixed_rows = _store_with_archives(tmp_path)
    assert (store.root / ARCHIVE_FOLDER_NAME).is_dir()

    with _record_sqlite_connects(monkeypatch) as opened:
        summary = store.result(result_id)
        listing = store.results()
    assert summary is not None
    assert summary["protocol_version"] == 5
    assert summary["result_schema_version"] == 2
    assert listing and listing[0]["id"] == result_id
    assert _archive_sqlite_opens(opened, store.root) == []
    # 摘要里只有引用对象，没有候选行本身。
    assert summary["evidence"]["candidate_archives"]["current"][ARCHIVE_REFERENCE_KEY]

    app = create_app(str(tmp_path), start_jobs=False)
    with _record_sqlite_connects(monkeypatch) as opened:
        with TestClient(app) as client:
            response = client.get(f"/api/v1/results/{result_id}")
    assert response.status_code == 200
    assert _archive_sqlite_opens(opened, store.root) == []
    assert '"params"' not in response.text


def test_identical_candidate_sets_reuse_a_single_archive_sqlite(tmp_path):
    """候选行完全相同的两个结果共用同一个档案 SQLite 与同一个 archive_hash。"""
    store = Store(tmp_path)
    rows = [_candidate_row(index) for index in range(20)]
    first_id = store.save_result("research", _v5_result(rows), "600000.SH", "同一候选集 A")
    second_id = store.save_result(
        "research", _v5_result(list(reversed(rows))), "600000.SH", "同一候选集 B",
    )

    archives = sorted((store.root / ARCHIVE_FOLDER_NAME).glob("*.sqlite"))
    assert len(archives) == 1

    first = store.result(first_id)["evidence"]["candidate_archives"]["current"]
    second = store.result(second_id)["evidence"]["candidate_archives"]["current"]
    assert first[ARCHIVE_REFERENCE_KEY] == second[ARCHIVE_REFERENCE_KEY]
    assert first["archive_hash"] == second["archive_hash"]
    assert first["row_count"] == second["row_count"] == 20
    assert archives[0].name == f"{first['archive_hash']}.sqlite"


def test_adaptive_decisions_keep_per_decision_archive_references(tmp_path):
    """自适应分支按决策保存，不能把整个决策列表误写成一份候选档案。"""
    store = Store(tmp_path)
    first = [_candidate_row(index) for index in range(3)]
    second = [_candidate_row(index + 10) for index in range(4)]
    payload = _v5_result([])
    payload["evidence"]["candidate_archives"]["adaptive"] = [
        {
            "effective_date": "2026-01-05", "decision_date": "2026-01-02",
            "candidate_archive": first, "search_candidate_archive": first,
        },
        {
            "effective_date": "2026-02-02", "decision_date": "2026-01-30",
            "candidate_archive": second, "search_candidate_archive": second,
        },
    ]

    result_id = store.save_result("research", payload, "600000.SH", "自适应档案")
    stored = store.result(result_id)
    decisions = stored["evidence"]["candidate_archives"]["adaptive"]

    assert isinstance(decisions, list) and len(decisions) == 2
    assert decisions[0]["candidate_archive"][ARCHIVE_REFERENCE_KEY]
    assert decisions[1]["candidate_archive"][ARCHIVE_REFERENCE_KEY]
    assert decisions[0]["candidate_archive"]["row_count"] == 3
    assert decisions[1]["candidate_archive"]["row_count"] == 4
    assert len(store.candidate_archive(
        stored, "candidate_archives", "adaptive", 0, limit=10,
    )["rows"]) == 3
    second_page = store.candidate_archive(
        stored, "candidate_archives", "adaptive", 1, limit=10,
    )
    assert [row["candidate_id"] for row in second_page["rows"]] == [11, 12, 13, 14]


def test_changed_candidate_row_creates_a_second_archive_sqlite(tmp_path):
    """候选集只要有一行不同，archive_hash 就不同，并落成第二个档案文件。"""
    store = Store(tmp_path)
    rows = [_candidate_row(index) for index in range(20)]
    changed = [_candidate_row(index) for index in range(20)]
    changed[0]["j_score"] = changed[0]["j_score"] + 0.5

    first_id = store.save_result("research", _v5_result(rows), "600000.SH", "候选集 A")
    second_id = store.save_result("research", _v5_result(changed), "600000.SH", "候选集 B")

    archives = sorted((store.root / ARCHIVE_FOLDER_NAME).glob("*.sqlite"))
    assert len(archives) == 2

    first = store.result(first_id)["evidence"]["candidate_archives"]["current"]
    second = store.result(second_id)["evidence"]["candidate_archives"]["current"]
    assert first["archive_hash"] != second["archive_hash"]
    assert first[ARCHIVE_REFERENCE_KEY] != second[ARCHIVE_REFERENCE_KEY]
    assert {archive.name for archive in archives} == {
        f"{first['archive_hash']}.sqlite",
        f"{second['archive_hash']}.sqlite",
    }


def test_iter_candidate_archive_walks_the_archive_in_fixed_size_pages(tmp_path, monkeypatch):
    """iter_candidate_archive 以 1000 行一页遍历整份档案并保持顺序。"""
    store = Store(tmp_path)
    rows = [_candidate_row(index) for index in range(1200)]
    result_id = store.save_result("research", _v5_result(rows), "600000.SH", "v5 分页迭代")
    stored = store.result(result_id)

    calls: list[tuple] = []
    real_candidate_archive = store.candidate_archive

    def recording_candidate_archive(result, group, decision, decision_index=0, **query):
        calls.append((query.get("offset"), query.get("limit")))
        return real_candidate_archive(result, group, decision, decision_index, **query)

    monkeypatch.setattr(store, "candidate_archive", recording_candidate_archive)
    collected = list(store.iter_candidate_archive(stored, "candidate_archives", "current"))

    assert len(collected) == 1200
    assert [row["candidate_id"] for row in collected] == list(range(1, 1201))
    assert calls == [(0, 1000), (1000, 1000)]


def test_candidate_axis_values_returns_sorted_distinct_four_axis_parameters(tmp_path):
    """candidate_axis_values 只返回四轴参数的去重升序列表。"""
    store = Store(tmp_path)
    rows = [_candidate_row(index) for index in range(20)]
    result_id = store.save_result("research", _v5_result(rows), "600000.SH", "v5 轴值")
    stored = store.result(result_id)

    axis = store.candidate_axis_values(
        stored, "candidate_archives", "current", names=["buy_step", "sell_step", "j_score"],
    )
    assert set(axis) == {"buy_step", "sell_step"}
    assert axis["buy_step"] == pytest.approx([round(1 + index * 0.1, 6) for index in range(20)])
    assert axis["sell_step"] == pytest.approx([round(2 + index * 0.1, 6) for index in range(20)])
    assert axis["buy_step"] == sorted(set(axis["buy_step"]))


def test_export_json_streams_candidate_rows_instead_of_the_reference(tmp_path):
    """kind=json 的导出把候选档案流式展开成候选行，而不是保留引用对象。"""
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    rows = [_candidate_row(index) for index in range(100)]
    fixed_rows = [_candidate_row(1000 + index) for index in range(30)]
    result_id = store.save_result(
        "research", _v5_result(rows, fixed_rows=fixed_rows), "600000.SH", "v5 JSON 导出",
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v1/results/{result_id}/export?kind=json")

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    payload = json.loads(response.text)

    archives = payload["evidence"]["candidate_archives"]
    assert isinstance(archives["current"], list)
    assert len(archives["current"]) == 100
    assert archives["current"][0]["candidate_id"] == 1
    assert archives["current"][0]["params"]["buy_step"] == pytest.approx(1.0)
    assert isinstance(archives["fixed"], list)
    assert len(archives["fixed"]) == 30
    assert archives["fixed"][0]["candidate_id"] == 1001
    # 引用对象不能残留在导出结果里。
    assert ARCHIVE_REFERENCE_KEY not in json.dumps(archives, ensure_ascii=False)


def test_export_candidates_csv_writes_every_candidate_row(tmp_path):
    """kind=candidates 的导出逐行写出固定与当前两个选择的全部候选行。"""
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    rows = [_candidate_row(index) for index in range(100)]
    fixed_rows = [_candidate_row(1000 + index) for index in range(30)]
    result_id = store.save_result(
        "research", _v5_result(rows, fixed_rows=fixed_rows), "600000.SH", "v5 候选导出",
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v1/results/{result_id}/export?kind=candidates")

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]

    # 导出响应带 UTF-8 BOM 前缀，去掉后再交给 csv 解析。
    reader = csv.DictReader(io.StringIO(response.text.lstrip("\ufeff"), newline=""))
    records = list(reader)
    assert reader.fieldnames is not None
    assert "candidate_id" in reader.fieldnames
    assert len(records) == len(rows) + len(fixed_rows)

    by_selection: dict[str, list[int]] = {}
    for record in records:
        by_selection.setdefault(record["selection"], []).append(int(record["candidate_id"]))
    assert set(by_selection) == {"fixed", "current"}
    assert by_selection["current"] == list(range(1, 101))
    assert by_selection["fixed"] == list(range(1001, 1031))


def test_completion_merge_keeps_region_membership_and_candidate_order(tmp_path):
    """补算合并不得丢失推荐域成员，也不得打乱候选顺序。"""
    store = Store(tmp_path)
    base_rows = [
        {**_candidate_row(0), "region_ids": [1]},
        {**_candidate_row(1), "region_ids": [1]},
    ]
    # 覆盖层重新评价了第一个候选，但没有区域信息，也没有候选号。
    overlay_rows = [{
        key: value for key, value in {**_candidate_row(0), "j_score": 99.0}.items()
        if key not in {"candidate_id", "region_ids"}
    }]
    result_id = store.save_result("research", _v5_result(base_rows), "600000.SH", "合并档案")
    stored = store.result(result_id)

    base_members = store.candidate_archive(
        stored, "candidate_archives", "current", region_id=1,
    )["rows"]
    assert len(base_members) == 2

    _completion_revision(store, result_id, overlay_rows=overlay_rows)
    revised = store.result(result_id)
    rows = store.candidate_archive(revised, "candidate_archives", "current", revision=1)["rows"]
    assert [row["candidate_id"] for row in rows] == [1, 2], "合并后候选顺序必须保持"
    assert rows[0]["j_score"] == pytest.approx(99.0), "覆盖层必须替换同一坐标的评分"

    members = store.candidate_archive(
        revised, "candidate_archives", "current", revision=1, region_id=1,
    )["rows"]
    assert len(members) == len(base_members), "合并后区域成员数量不得变化"
    assert [row["candidate_id"] for row in members] == [1, 2]
    assert members[0]["region_ids"] == [1], "覆盖层必须继承基础行的区域成员"


def test_candidate_reference_refuses_a_revision_from_another_decision(tmp_path):
    """修订只覆盖自己的决策分支，别的分支必须明确失败。"""
    store = Store(tmp_path)
    rows = [_candidate_row(index) for index in range(5)]
    fixed_rows = [_candidate_row(1000 + index) for index in range(3)]
    result_id = store.save_result(
        "research", _v5_result(rows, fixed_rows=fixed_rows), "600000.SH", "修订作用域",
    )
    _completion_revision(store, result_id, decision="current", overlay_rows=[rows[0]])
    stored = store.result(result_id)

    reference = store.candidate_reference(
        stored, "candidate_archives", "current", 0, revision=1,
    )
    assert reference and reference["archive_hash"]

    with pytest.raises(storage.CompletionRevisionError) as failure:
        store.candidate_reference(stored, "candidate_archives", "fixed", 0, revision=1)
    assert failure.value.code == "COMPLETION_REVISION_OUT_OF_SCOPE"
    assert failure.value.scope == {"decision": "current", "decision_index": 0}

    with pytest.raises(storage.CompletionRevisionError) as missing:
        store.candidate_reference(stored, "candidate_archives", "current", 0, revision=7)
    assert missing.value.code == "COMPLETION_REVISION_NOT_FOUND"


def test_export_json_resolves_the_covered_branch_to_the_merged_archive(tmp_path):
    """带修订的 JSON 导出必须把被覆盖的分支解析成合并档案。"""
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    rows = [_candidate_row(index) for index in range(10)]
    fixed_rows = [_candidate_row(1000 + index) for index in range(4)]
    result_id = store.save_result(
        "research", _v5_result(rows, fixed_rows=fixed_rows), "600000.SH", "修订导出",
    )
    overlay_rows = [{
        key: value for key, value in {**_candidate_row(0), "j_score": 42.0}.items()
        if key != "region_ids"
    }]
    _completion_revision(store, result_id, decision="current", overlay_rows=overlay_rows)

    with TestClient(app) as client:
        response = client.get(f"/api/v1/results/{result_id}/export?kind=json&revision=1")

    assert response.status_code == 200
    payload = json.loads(response.text)
    current = payload["evidence"]["candidate_archives"]["current"]
    assert len(current) == 10
    assert current[0]["j_score"] == pytest.approx(42.0), "被覆盖的分支必须是合并后的档案"
    fixed = payload["evidence"]["candidate_archives"]["fixed"]
    assert fixed[0]["candidate_id"] == 1001, "未覆盖的分支必须保持原档案"
    assert payload["export_revision"]["decision"] == "current"
    assert payload["export_revision"]["revision"] == 1


def test_export_candidates_csv_marks_the_revised_branch(tmp_path):
    """候选 CSV 只能对被修订的分支应用修订，并逐行标明来源。"""
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    rows = [_candidate_row(index) for index in range(10)]
    fixed_rows = [_candidate_row(1000 + index) for index in range(4)]
    result_id = store.save_result(
        "research", _v5_result(rows, fixed_rows=fixed_rows), "600000.SH", "修订候选导出",
    )
    overlay_rows = [{
        key: value for key, value in {**_candidate_row(0), "j_score": 42.0}.items()
        if key != "region_ids"
    }]
    _completion_revision(store, result_id, decision="current", overlay_rows=overlay_rows)

    with TestClient(app) as client:
        response = client.get(f"/api/v1/results/{result_id}/export?kind=candidates&revision=1")

    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text.lstrip("\ufeff"), newline=""))
    records = list(reader)
    assert reader.fieldnames is not None
    assert "archive_source" in reader.fieldnames and "revision" in reader.fieldnames

    by_selection: dict[str, list[dict]] = {}
    for record in records:
        by_selection.setdefault(record["selection"], []).append(record)
    assert set(by_selection) == {"fixed", "current"}
    assert {record["archive_source"] for record in by_selection["current"]} == {"revised"}
    assert {record["revision"] for record in by_selection["current"]} == {"1"}
    assert {record["archive_source"] for record in by_selection["fixed"]} == {"base"}
    assert {record["revision"] for record in by_selection["fixed"]} == {""}
    assert by_selection["current"][0]["candidate_id"] == "1"


def test_equity_export_streams_without_building_a_data_frame(tmp_path, monkeypatch):
    """账本导出必须是生成器驱动的流式响应，不能再构造 pandas 表格。"""
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    result = _v5_result([])
    result["fixed"]["test_result"] = {
        "strategies": {"grid": {"equity": [
            {"timestamp": "2026-01-05", "value": 1.0},
            {"timestamp": "2026-01-06", "value": 1.5},
        ]}},
    }
    result_id = store.save_result("research", result, "600000.SH", "账本导出")

    def forbidden(*args, **kwargs):
        raise AssertionError("账本导出不得构造 pandas 表格")

    monkeypatch.setattr("stockregression.api.pd.DataFrame", forbidden)
    with TestClient(app) as client:
        response = client.get(f"/api/v1/results/{result_id}/export?kind=equity")

    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text.lstrip("\ufeff"), newline=""))
    records = list(reader)
    assert reader.fieldnames is not None
    assert set(reader.fieldnames) == {"selection", "strategy", "timestamp", "value"}
    assert len(records) == 2
    assert records[0]["selection"] == "fixed"
    assert records[0]["strategy"] == "grid"
