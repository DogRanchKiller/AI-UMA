"""收盘观察名单的本地追加日志。观察事件不是盘中信号，也不是成交记录。"""
from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
from itertools import chain
from pathlib import Path

import pandas as pd

_SHANGHAI = timezone(timedelta(hours=8))
_CLOSE = time(15)
_TABLE = """CREATE TABLE IF NOT EXISTS watch_snapshots (
    day TEXT PRIMARY KEY NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('record', 'baseline')),
    recorded_at TEXT NOT NULL, payload TEXT NOT NULL
)"""


def _day(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"日期须为 YYYY-MM-DD：{value!r}") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"日期须为 YYYY-MM-DD：{value!r}")
    return parsed


def _codes(snapshot: dict, key: str) -> set[str]:
    values = snapshot.get(key)
    if not isinstance(values, list):
        raise ValueError(f"{key} 须为列表")
    result = set()
    for item in values:
        code = item if key == "unknown" else item.get("code") if isinstance(item, dict) else None
        if not isinstance(code, str) or not code or code.strip() != code or code in result:
            raise ValueError(f"{key} 含空白、重复或无效代码：{code!r}")
        result.add(code)
    return result


def _members(snapshot: dict) -> tuple[set[str], set[str], set[str]]:
    hits, unknown, pinned = (_codes(snapshot, k) for k in ("matches", "unknown", "pinned"))
    if hits & unknown:
        raise ValueError("同一观察不能同时为达标和未知")
    return hits, unknown, pinned


def _checked_payload(snapshot: dict) -> tuple[date, str]:
    if not isinstance(snapshot, dict):
        raise ValueError("观察快照须为 JSON 对象")
    day, score_day = _day(snapshot.get("date")), _day(snapshot.get("score_date"))
    age = snapshot.get("score_age")
    if score_day > day or type(age) is not int or age < 0 or type(snapshot.get("gate")) is not bool:
        raise ValueError("评分日期、交易日龄或闸门状态无效")
    if "latest_data_date" in snapshot and _day(snapshot["latest_data_date"]) != day:
        raise ValueError("登记日期不是最新行情日")
    _members(snapshot)
    # 存储元数据由数据库生成；读回快照不能通过重写 kind 或 recorded_at 伪装前瞻记录。
    data = {k: v for k, v in snapshot.items() if k not in ("kind", "recorded_at")}
    try:
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        raise ValueError("观察快照必须可编码为 JSON，不能含 NaN/Infinity") from exc
    if json.loads(payload) != data:
        raise ValueError("观察快照不能含非字符串对象键等非 JSON 数据")
    return day, payload


def _has_journal(conn: sqlite3.Connection) -> bool:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT GLOB 'sqlite_*'")}
    if tables - {"watch_snapshots"}:
        raise ValueError("指定的数据库不是观察日志；请选择专用文件")
    return "watch_snapshots" in tables


def load_history(db: Path) -> list[dict]:
    """按日升序只读已登记快照；即使路径不存在，也不创建文件或目录。"""
    db = Path(db)
    if not db.exists():
        return []
    try:
        with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            if not _has_journal(conn):
                return []
            return [dict(json.loads(payload), kind=kind, recorded_at=stamp) for payload, kind, stamp in
                    conn.execute("SELECT payload,kind,recorded_at FROM watch_snapshots ORDER BY day")]
    except sqlite3.Error as exc:
        raise ValueError(f"无法读取观察日志：{exc}") from exc


def save_snapshot(db: Path, snapshot: dict, *, baseline: bool = False, now: datetime | None = None) -> str:
    """首次/重新运行返回 inserted 或 unchanged；不同内容、不同性质或时间不合法则拒绝。

    日志不访问行情；调用方用实际基准行情校验有交易、且 baseline 取最新已收盘缓存日。
    有时区的 now 转换为上海时间；无时区的注入值按上海墙上时间解释，不依赖主机时区。
    """
    day, payload = _checked_payload(snapshot)
    clock = now if now is not None else datetime.now(_SHANGHAI)
    if not isinstance(clock, datetime) or type(baseline) is not bool:
        raise ValueError("无效的记录时间或基线标记")
    clock = clock.astimezone(_SHANGHAI) if clock.tzinfo is not None else clock.replace(tzinfo=_SHANGHAI)
    if day.weekday() >= 5 or day > clock.date() or (day == clock.date() and clock.time() < _CLOSE):
        raise ValueError("没有可登记的已收盘交易日；不得登记周末、未来或盘中数据")
    if not baseline and day != clock.date():
        raise ValueError("正式观察仅可在上海当天收市后登记，不能补填历史")
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    kind = "baseline" if baseline else "record"
    try:
        with closing(sqlite3.connect(db)) as conn, conn:
            # 先取得写锁，再查最后一行；两个进程不能同时将空库判断为可初始化的基线。
            conn.execute("BEGIN IMMEDIATE")
            if not _has_journal(conn):
                conn.execute(_TABLE)
            last = conn.execute("SELECT day,kind,payload FROM watch_snapshots ORDER BY day DESC LIMIT 1").fetchone()
            key = day.isoformat()
            if last:
                if key < last[0]:
                    raise ValueError("不得向已有观察日志插入更早的日期")
                if key == last[0]:
                    if (kind, payload) == last[1:]:
                        return "unchanged"
                    raise ValueError("同日已有不同内容或性质的登记，不可覆盖")
                if baseline:
                    raise ValueError("基线只可在空库登记一次")
            conn.execute("INSERT INTO watch_snapshots(day,kind,recorded_at,payload) VALUES(?,?,?,?)",
                         (key, kind, clock.isoformat(), payload))
    except sqlite3.Error as exc:
        raise ValueError(f"未能登记观察日志：{exc}") from exc
    return "inserted"


def _transitions(history: Iterable[dict]):
    """逐笔推进确认状态；unknown 延续既有结论，但自身不触发入选或离场。"""
    active, unresolved = set(), set()
    previous_day = None
    for snap in history:
        day = _day(snap.get("date"))
        if previous_day is not None and day <= previous_day:
            raise ValueError("观察历史必须按唯一日期严格递增")
        hits, unknown, pinned = _members(snap)
        # 首次即未知且从未有确证的标的，后来首次可判断为真，只能标作「首见」。
        initial = hits if previous_day is None else hits & unresolved
        delta = {"initial": initial, "new": hits - active - initial, "kept": hits & active,
                 "lost": active - hits - unknown, "unknown": unknown}
        unresolved = unknown if previous_day is None else unresolved & unknown
        active = hits | (active & unknown)
        previous_day = day
        yield snap, delta, hits, pinned


def changes(history: list[dict], snapshot: dict) -> dict[str, list[str]]:
    """对比更早的实际登记而非昨天；同日重复预览及后来的快照不参与对比。"""
    day = snapshot["date"]
    earlier = (s for s in history if s["date"] < day)
    delta = next(d for row, d, _, _ in _transitions(chain(earlier, [snapshot])) if row is snapshot)
    return {key: sorted(codes) for key, codes in delta.items()}


def _market_days(calendar: pd.DatetimeIndex) -> tuple[list[pd.Timestamp], dict[str, int]]:
    if (not isinstance(calendar, pd.DatetimeIndex) or calendar.hasnans or not calendar.is_monotonic_increasing
            or not calendar.is_unique or not calendar.equals(calendar.normalize())):
        raise ValueError("复核需要按日升序、唯一且有效的真实交易日历")
    now = datetime.now(_SHANGHAI)
    today = now.date()
    closed_through = today if now.time() >= _CLOSE else today - timedelta(days=1)
    observed = []
    positions = {}
    for ts in calendar:
        trading_day = (ts.tz_convert(_SHANGHAI) if ts.tzinfo else ts).date()
        if trading_day > closed_through:
            break  # 即使误给了未来排期，也不得以尚无日线的交易日进行复核。
        positions[trading_day.isoformat()] = len(observed)
        observed.append(ts)
    return observed, positions


def review_events(history: list[dict], calendar: pd.DatetimeIndex, open_at: Callable[[str, pd.Timestamp], float],
                  benchmark_code: str, horizon: int = 10) -> list[dict]:
    """在已观察到的交易日历上复核：t+1 → t+1+horizon 开盘价收益，不扣费、不模拟成交。

    calendar 为基准日线截至查询日的日历（不是未来排期）；open_at 须提供该日真实开盘，
    停牌/零量/无行情返回缺失。不前填、不另寻可成交日。事件的 date 是观察日，不是评分日。
    sources 是触发来源；matched/pinned 表示同日实际所属名单。baseline 永不当作前瞻记录。
    """
    if type(horizon) is not int or horizon < 1 or not isinstance(benchmark_code, str) or not benchmark_code.strip():
        raise ValueError("复核期限必须为正整数，且须指定基准代码")
    days, pos = _market_days(calendar)
    cache = {}

    def price(code, day):
        key = (code, day)
        if key not in cache:
            try:
                value = open_at(code, day)
            except (KeyError, FileNotFoundError):
                value = None
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                value = math.nan
            cache[key] = value if math.isfinite(value) and value > 0 else None
        return cache[key]

    def price_return(code, entry, exit_):
        start, end = price(code, entry), price(code, exit_)
        if start is None or end is None:
            return None
        result = end / start - 1
        return result if math.isfinite(result) else None

    result, seen_pins = [], set()
    if not pos:
        return []
    asof = next(reversed(pos))
    snapshots = (s for s in history if s["date"] <= asof)
    for snap, delta, hits, pinned in _transitions(snapshots):
        selected, added_pins = delta["initial"] | delta["new"], pinned - seen_pins
        seen_pins.update(pinned)
        signal = snap["date"]
        i = pos.get(signal)
        entry = days[i+1] if i is not None and i+1 < len(days) else None
        end = days[i+1+horizon] if i is not None and i+1+horizon < len(days) else None
        for code in sorted(selected | added_pins):
            kind = snap.get("kind", "unregistered")
            sources = (["match"] if code in selected else []) + (["pinned"] if code in added_pins else [])
            row = {"signal_date": signal, "code": code, "sources": sources, "matched": code in hits, "pinned": code in pinned,
                   "kind": kind, "recorded_at": snap.get("recorded_at"), "prospective": kind == "record",
                   "observation_type": "after_close_watchlist", "score_date": snap.get("score_date"),
                   "status": "pending" if i is not None else "missing",
                   "entry_date": entry.date().isoformat() if entry is not None else None,
                   "exit_date": end.date().isoformat() if end is not None else None,
                   "stock_return": None, "benchmark_return": None, "excess_return": None}
            if end is not None:
                stock, bench = price_return(code, entry, end), price_return(benchmark_code, entry, end)
                row.update(stock_return=stock, benchmark_return=bench, status="missing")
                if stock is not None and bench is not None:
                    row.update(status="complete", excess_return=stock-bench)
            result.append(row)
    return result
