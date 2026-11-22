"""盘后观察事件：本地追加、三态历史、严格对日的十交易日事后复核。"""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Barrier

import pandas as pd
import pytest

from ashare import watch_journal as journal


def snap(day, matches=(), *, unknown=(), pins=(), **extra):
    return {"date": day, "score_date": day, "score_age": 0, "gate": True,
            "matches": [{"code": c} for c in matches], "unknown": list(unknown),
            "pinned": [{"code": c} for c in pins], **extra}


def at(value):
    return datetime.fromisoformat(value)


def test_read_does_not_create_new_db_or_directory(tmp_path):
    db = tmp_path / "read only" / "nothing.sqlite3"
    assert journal.load_history(db) == []
    assert not db.parent.exists()


def test_roundtrip_all_rows_and_reopen_idempotence(tmp_path):
    db = tmp_path / "space & 中文" / "观察#1.db"
    raw = snap("2024-02-08", ("g", "c", "d", "e", "f"), pins=("p", "d"), note={"文本": "观察"})
    clock = at("2024-02-08T07:00:00+00:00")  # 上海恰好 15:00
    assert journal.save_snapshot(db, raw, now=clock) == "inserted"
    loaded, = journal.load_history(db)
    assert {k: loaded[k] for k in raw} == raw and loaded["kind"] == "record"
    assert loaded["recorded_at"] == "2024-02-08T15:00:00+08:00"
    # 读回元数据不可变成输入字段；键顺序和第二次运行时间都不创造新事件。
    replay = dict(reversed(list(loaded.items())))
    assert journal.save_snapshot(db, replay, now=at("2024-02-08T20:00:00+08:00")) == "unchanged"
    assert journal.load_history(db) == [loaded]


def test_baseline_is_once_honest_and_strictly_append_only(tmp_path):
    db = tmp_path / "daily.db"
    baseline = snap("2024-02-08", ("A",), kind="record", recorded_at="spoof")
    latest = snap("2024-02-19", ("A", "B"))
    clock = at("2024-02-19T19:22:31+08:00")
    assert journal.save_snapshot(db, baseline, now=clock, baseline=True) == "inserted"
    assert journal.save_snapshot(db, baseline, now=clock, baseline=True) == "unchanged"
    first, = journal.load_history(db)
    assert (first["kind"], first["recorded_at"]) == ("baseline", clock.isoformat())
    with pytest.raises(ValueError):
        journal.save_snapshot(db, latest, now=clock, baseline=True)
    assert journal.save_snapshot(db, latest, now=clock) == "inserted"
    for bad, is_base in [(dict(latest, gate=False), False), (baseline, True)]:
        with pytest.raises(ValueError):
            journal.save_snapshot(db, bad, now=clock, baseline=is_base)
    assert [s["date"] for s in journal.load_history(db)] == [baseline["date"], latest["date"]]


@pytest.mark.parametrize("day,clock,baseline", [
    ("2024-02-08", "2024-02-08T06:59:59+00:00", False),  # 上海收市前一秒
    ("2024-02-08", "2024-02-08T14:59:59+08:00", True),
    ("2024-02-08", "2024-02-09T15:01:00+08:00", False),  # 不能补录
    ("2024-02-19", "2024-02-08T20:00:00+08:00", True),  # 不能登记未来
    ("2024-02-10", "2024-02-10T20:00:00+08:00", False),  # 周六
])
def test_record_clock_is_bound_to_shanghai_session(tmp_path, day, clock, baseline):
    db = tmp_path / "absent.db"
    with pytest.raises(ValueError):
        journal.save_snapshot(db, snap(day), now=at(clock), baseline=baseline)
    assert not db.exists()


@pytest.mark.parametrize("override", [
    {"date": "20240208"}, {"score_date": "2024-02-09"}, {"score_age": -1}, {"gate": 1},
    {"pinned": [{"code": ""}]}, {"matches": [{"code": "A"}, {"code": "A"}]},
    {"matches": [{"code": "A"}], "unknown": ["A"]}, {"unknown": [None]},
    {"not_json": float("nan")}, {"not_json": {"x"}},
])
def test_invalid_snapshot_is_not_persisted(tmp_path, override):
    db = tmp_path / "invalid.db"
    with pytest.raises(ValueError):
        journal.save_snapshot(db, dict(snap("2024-02-08"), **override), now=at("2024-02-08T18:00:00+08:00"))
    assert not db.exists()


def test_two_writers_with_conflicting_observations_cannot_both_commit(tmp_path):
    db = tmp_path / "race.db"
    ready = Barrier(2)

    def append(code):
        ready.wait(timeout=5)
        try:
            return journal.save_snapshot(db, snap("2024-02-08", (code,)), now=at("2024-02-08T18:00:00+08:00"))
        except ValueError:
            return "conflict"

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(append, ["A", "B"])) == ["conflict", "inserted"]
    row, = journal.load_history(db)
    assert row["matches"] in [[{"code": "A"}], [{"code": "B"}]]


def test_changes_initial_recovery_loss_and_reentry_use_last_known_state():
    a = snap("2024-02-01", ("A", "C"), unknown=("U",), pins=("P",))
    b = snap("2024-02-02", ("B",), unknown=("A", "U"), pins=("P",))
    c = snap("2024-02-05", ("A", "U", "C"), unknown=("B",))
    assert journal.changes([], a) == {"initial": ["A", "C"], "new": [], "kept": [], "lost": [], "unknown": ["U"]}
    assert journal.changes([a], b) == {"initial": [], "new": ["B"], "kept": [], "lost": ["C"], "unknown": ["A", "U"]}
    expected = {"initial": ["U"], "new": ["C"], "kept": ["A"], "lost": [], "unknown": ["B"]}
    assert journal.changes([a, b], c) == expected
    # 查看历史某天时，不拿当天自己的已存行或后续记录来决定“新进”。
    assert journal.changes([a, b, dict(c, kind="record"), snap("2024-02-06")], c) == expected


def test_review_triggers_are_first_or_reentry_plus_first_pin_never_daily():
    days = pd.bdate_range("2024-01-02", periods=8)
    s = [d.date().isoformat() for d in days]
    rows = [dict(snap(s[0], ("A", "B"), pins=("A", "P")), kind="baseline"),
            dict(snap(s[1], ("B",), unknown=("A",), pins=("A", "P")), kind="record"),
            dict(snap(s[2], ("A",), pins=("P",)), kind="record"),
            dict(snap(s[3], ("A", "B"), pins=("P",)), kind="record")]
    results = journal.review_events(rows, days, lambda *_: 10., "idx", horizon=2)
    assert [(r["signal_date"], r["code"]) for r in results] == [(s[0], "A"), (s[0], "B"), (s[0], "P"), (s[3], "B")]
    assert results[0]["sources"] == ["match", "pinned"] and results[2]["sources"] == ["pinned"]
    assert not results[0]["prospective"] and results[0]["kind"] == "baseline"
    assert results[-1]["prospective"] and results[-1]["kind"] == "record"


def test_t_plus_one_to_t_plus_eleven_counts_only_market_sessions():
    # 春节后第一日是 2/19；包括两个周末，不能按自然日或每天换仓。
    dates = pd.DatetimeIndex(["2024-02-08", *pd.bdate_range("2024-02-19", periods=11)])
    signal = snap("2024-02-08", ("A",), score_date="2024-02-05", score_age=3)
    quotes = {("A", dates[1]): 10., ("A", dates[11]): 11.5,
              ("sh000300", dates[1]): 100., ("sh000300", dates[11]): 105.}
    out, = journal.review_events([signal], dates, lambda c, d: quotes[c, d], "sh000300")
    assert (out["signal_date"], out["entry_date"], out["exit_date"], out["status"]) == (
        "2024-02-08", "2024-02-19", "2024-03-04", "complete")
    assert [out[k] for k in ("stock_return", "benchmark_return", "excess_return")] == pytest.approx([.15, .05, .10])


def test_pending_event_reads_no_quotes_even_if_entry_has_arrived():
    dates = pd.bdate_range("2024-01-02", periods=11)  # t..t+10：还差一个实际交易日
    result, = journal.review_events([snap("2024-01-02", ("A",))], dates,
                                    lambda *_: pytest.fail("pending read a quote"), "idx")
    assert result["status"] == "pending" and result["entry_date"] == "2024-01-03"
    assert result["exit_date"] is None and result["excess_return"] is None


@pytest.mark.parametrize("gapped,failed_leg", [(float("nan"), "A"), (0., "idx"), (None, "A"), (float("inf"), "idx")])
def test_mature_but_missing_exact_open_does_not_shift_or_fill(gapped, failed_leg):
    dates = pd.bdate_range("2024-01-02", periods=5)
    calls = []

    def price(c, d):
        calls.append((c, d))
        return gapped if c == failed_leg and d == dates[1] else 100.

    out, = journal.review_events([snap("2024-01-02", ("A",))], dates, price, "idx", horizon=2)
    assert out["status"] == "missing" and out["excess_return"] is None
    assert (out["stock_return"], out["benchmark_return"]) == ((None, 0.) if failed_leg == "A" else (0., None))
    assert set(d for _, d in calls) == {dates[1], dates[3]}


@pytest.mark.parametrize("calendar", [pd.DatetimeIndex([]), pd.DatetimeIndex(["2200-01-02"])])
def test_no_realized_sessions_exposes_no_event_or_future_quote(calendar):
    events = [snap("2024-02-08", ("past",)), snap("2200-01-02", ("future",))]
    assert journal.review_events(events, calendar, lambda *_: pytest.fail("unavailable quote was read"), "idx") == []


def test_confirmed_loss_after_multiple_unknown_reports_is_real_exit():
    rows = [snap("2024-02-01", ("A",)), snap("2024-02-02", unknown=("A",)),
            snap("2024-02-05", unknown=("A",)), snap("2024-02-06")]
    assert journal.changes(rows[:-1], rows[-1])["lost"] == ["A"]
    assert journal.changes(rows, snap("2024-02-07", ("A",)))["new"] == ["A"]


def test_nonjournal_database_and_corrupt_file_are_not_modified(tmp_path):
    snapshot = snap("2024-02-08", ("A",))
    when = at("2024-02-08T16:00:00+08:00")
    db = tmp_path / "foreign.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE important(name TEXT)")
        conn.commit()
    finally:
        conn.close()
    corrupt = tmp_path / "not-a-database.db"
    corrupt.write_bytes(b"not a SQLite database")
    for path in (db, corrupt):
        original = path.read_bytes()
        with pytest.raises(ValueError):
            journal.load_history(path)
        with pytest.raises(ValueError):
            journal.save_snapshot(path, snapshot, now=when)
        assert original == path.read_bytes()


def test_mature_event_can_report_one_observable_leg_even_when_file_missing():
    days = pd.date_range("2024-01-02", periods=3)

    def price(code, ts):
        if code == "A":
            raise FileNotFoundError(code)
        return 100. if ts == days[1] else 103.

    row, = journal.review_events([snap("2024-01-02", ("A",))], days, price, "idx", horizon=1)
    assert row["status"] == "missing" and row["excess_return"] is None
    assert row["stock_return"] is None and row["benchmark_return"] == pytest.approx(.03)


def test_signal_date_absent_from_calendar_never_moves_to_next_day():
    days = pd.DatetimeIndex(["2024-02-07", "2024-02-19", "2024-02-20"])
    event, = journal.review_events([snap("2024-02-08", ("A",))], days,
                                   lambda *_: pytest.fail("missing anchor read a quote"), "idx", horizon=1)
    assert event["status"] == "missing" and event["entry_date"] is None


@pytest.mark.parametrize("dates,horizon", [(["2024-02-07"], 0), (["2024-02-08", "2024-02-07"], 10),
                                          (["2024-02-07", "2024-02-07"], 10)])
def test_horizon_and_calendar_must_be_unambiguous(dates, horizon):
    with pytest.raises(ValueError):
        journal.review_events([], pd.DatetimeIndex(dates), lambda *_: 1., "idx", horizon=horizon)
