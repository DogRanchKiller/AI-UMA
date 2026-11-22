"""行情数据获取与本地缓存。

数据源：
  * BaoStock 不复权日线、真实成交额和上市以来累计后复权因子
  * 腾讯批量快照接口（市值、换手、PE、PB 等截面字段）
  * BaoStock 提供交易日历、停牌状态和 A 股主列表（含证券名称，用于 ST 判定）

设计原则：
  1. 单只失败不阻断整体；BaoStock 进程共享会话必须串行访问。
  2. 落盘缓存为逐股 CSV，二次运行不重复下载。
  3. 返回的 DataFrame 统一列名，日期为 datetime，按日期升序。
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import requests

from .config import CACHE_DIR, RAW_DIR, Config

log = logging.getLogger(__name__)

KLINE_COLS = ["date", "open", "close", "high", "low", "volume", "amount", "adj_factor"]
_PRICE_COLS = ["open", "close", "high", "low"]
SHANGHAI = timezone(timedelta(hours=8))
REFERENCE_INDICES = ("sh000300", "sh000905", "sh000852", "sh000001")
_STOCK_PREFIX = ("sh.600", "sh.601", "sh.603", "sh.605", "sh.688", "sz.000", "sz.001", "sz.002", "sz.003", "sz.300", "sz.301")
_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://gu.qq.com/",
}

_BS_LOCK = Lock()
_RAW_VERSION = "baostock_raw_v1"
_BS_BAR_FIELDS = ("date", "code", "open", "high", "low", "close", "volume", "amount", "adjustflag")


@contextmanager
def _bs_session():
    # SDK 的 socket 和 user_id 都在模块全局；锁覆盖登录、所有分页、登出。
    import baostock as bs

    with _BS_LOCK:
        if bs.login().error_code != "0":
            raise RuntimeError("BaoStock 登录失败")
        try:
            yield bs
        finally:
            bs.logout()


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_UA)
    return s


def _get(sess: requests.Session, url: str, retries: int = 3, **kw) -> requests.Response:
    last = None
    for i in range(retries):
        try:
            r = sess.get(url, timeout=kw.pop("timeout", 20), **kw)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001 - 网络层错误统一重试
            last = exc
        time.sleep(0.4 * (i + 1))
    raise RuntimeError(f"请求失败 {url}: {last}")


def _num(x) -> float:
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- #
# 批次交易日与有证据的停牌（不能用工作日或最新本地行情猜交易日）
# --------------------------------------------------------------------------- #
def _bs_rows(rs, required):
    if rs.error_code != "0" or not set(required).issubset(rs.fields):
        raise RuntimeError(f"BaoStock 查询失败或缺少字段 {required}：{rs.error_code}")
    columns = [rs.fields.index(name) for name in required]
    rows = []
    while rs.next():  # 不用 get_data()：部分版本在分页时调用已移除的 pandas.append。
        row = rs.get_row_data()
        rows.append(tuple(row[i] for i in columns))
    # SDK 0.9.40 遇到空网络响应时 next() 也返回 False，却不更新 error_code；
    # 真正读完整页后必收到短页或空页，不能把仍停在满页当成正常结束。
    page = getattr(rs, "data", None)
    if rs.error_code != "0" or (page and len(page) == int(rs.per_page_count)):
        raise RuntimeError(f"BaoStock 分页中断：{rs.error_code}")
    return rows


def _roster_on(bs, day: date) -> tuple[frozenset[str], dict[str, str]]:
    rows = _bs_rows(bs.query_all_stock(day=day.isoformat()), ("code", "tradeStatus", "code_name"))
    if not rows or len({r[0] for r in rows}) != len(rows):
        raise ValueError("BaoStock 状态清单为空或重复")
    stocks = {code: name for code, _, name in rows if code.startswith(_STOCK_PREFIX)}
    if not stocks or any(not re.fullmatch(r"(?:sh|sz)\.[0-9]{6}", c) or not isinstance(n, str) or not n.strip()
                         for c, n in stocks.items()):
        raise ValueError("目标日股票名单为空或含无效名称/代码")
    return frozenset(to_tencent(code) for code, state, _ in rows if state == "0"), stocks


def _historical_halts(day: date) -> frozenset[str]:
    with _bs_session() as bs:
        return _roster_on(bs, day)[0]


@dataclass
class MarketCalendar:
    """本批已收盘的实际交易日。历史确证停牌仅在遇到缺行时查询，每日全批最多查一次。"""
    sessions: pd.DatetimeIndex
    halted: dict[date, frozenset[str]] = field(default_factory=dict)
    stocks: dict[str, str] = field(default_factory=dict)  # 本批目标日的 BaoStock 原代码和名称
    query_halts: Callable[[date], frozenset[str]] | None = field(default=None, repr=False)
    covered_from: date | None = None  # 原始日历起点可能早于第一个开市日。
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        idx = self.sessions
        if (not isinstance(idx, pd.DatetimeIndex) or idx.empty or idx.hasnans or not idx.is_unique
                or not idx.is_monotonic_increasing or idx.tz is not None or not idx.equals(idx.normalize())):
            raise ValueError("交易日历必须为升序且不重复的本地交易日期")

    @property
    def target(self) -> date:
        return self.sessions[-1].date()

    def is_halted(self, code: str, day: date) -> bool:
        # BaoStock 的会话是进程共享的；同一历史日的状态不能在行情线程间重复登录查询。
        with self._lock:
            if day not in self.halted:
                try:
                    self.halted[day] = self.query_halts(day) if self.query_halts else frozenset()
                except Exception as exc:
                    log.warning("无法核实 %s 停牌状态，无行仍按未知：%s", day, exc)
                    self.halted[day] = frozenset()
            return code in self.halted[day]


def resolve_market_calendar(cfg: Config, *, now: datetime | None = None) -> MarketCalendar:
    """上海15:00（含）后可选当天，否则只考虑前一自然日及以前；缺日历即失败。"""
    clock = now if now is not None else datetime.now(SHANGHAI)
    clock = clock.astimezone(SHANGHAI) if clock.tzinfo is not None else clock.replace(tzinfo=SHANGHAI)
    end = clock.date() if clock.time() >= wall_time(15) else clock.date() - timedelta(days=1)
    if cfg.end_date:
        end = min(end, date.fromisoformat(cfg.end_date))
    start = min(date.fromisoformat(cfg.start_date), end - timedelta(days=31))
    with _bs_session() as bs:
        rows = _bs_rows(bs.query_trade_dates(start_date=start.isoformat(), end_date=end.isoformat()),
                        ("calendar_date", "is_trading_day"))
        expected = pd.date_range(start, end)
        if (len(rows) != len(expected) or any(day != stamp.date().isoformat() or flag not in ("0", "1")
                                               for (day, flag), stamp in zip(rows, expected))):
            raise ValueError("交易日历不连续、过期或标记无效；不能推定最近收盘日")
        sessions = expected[[flag == "1" for _, flag in rows]]
        cal = MarketCalendar(sessions, query_halts=_historical_halts, covered_from=start)
        if cal.target < date.fromisoformat(cfg.start_date):
            raise ValueError("请求起点晚于最近已收盘交易日")
        try:
            cal.halted[cal.target], cal.stocks = _roster_on(bs, cal.target)
        except Exception as exc:
            log.warning("当日停牌状态不可用，缺行不能认定停牌：%s", exc)
            cal.halted[cal.target] = frozenset()
        return cal


# --------------------------------------------------------------------------- #
# A 股主列表
# --------------------------------------------------------------------------- #
def _baostock_master() -> pd.DataFrame:
    rows = []
    with _bs_session() as bs:
        for back in range(20):
            day = (datetime.now() - timedelta(days=back)).strftime("%Y-%m-%d")
            rows = _bs_rows(bs.query_all_stock(day=day), ("code", "code_name"))
            if rows:
                break
    df = pd.DataFrame(rows, columns=["code", "code_name"])
    if df.empty:
        raise RuntimeError("baostock 返回空主列表")
    return df.rename(columns={"code": "code", "code_name": "name"})


def master_list(cfg: Config, force: bool = False) -> pd.DataFrame:
    """返回全 A 股/指数主列表：code, name, kind。"""
    path = CACHE_DIR / "master_list.csv"
    if path.exists() and not force:
        return pd.read_csv(path, dtype={"code": str})

    df = _baostock_master()
    df["code"] = df["code"].astype(str)
    # 仅保留 A 股个股。baostock 的 query_all_stock 同时返回 ETF/LOF/指数/债券，
    # 这里用交易所内股票代码段白名单过滤，而不是用后缀排除法。
    df = df[df["code"].str.startswith(_STOCK_PREFIX)].copy()
    df["kind"] = "stock"
    out = df[["code", "name", "kind"]].drop_duplicates("code").reset_index(drop=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("主列表已缓存：%d 只", len(out))
    return out


def to_tencent(code: str) -> str:
    """baostock 风格代码 sh.600000 -> 腾讯风格 sh600000。"""
    return code.replace(".", "").strip().lower()


# --------------------------------------------------------------------------- #
# 行业分类
# --------------------------------------------------------------------------- #
def industry_map(cfg: Config, force: bool = False) -> dict[str, str]:
    """返回 {腾讯风格代码: 行业名}。

    数据源为 baostock 的证监会行业分类。注意这是**当前快照**而非历史分类，
    个股历史上改行业的极少，但这一近似会带来轻微的前视偏差（见 README 局限）。
    """
    path = CACHE_DIR / "industry.csv"
    if path.exists() and not force:
        df = pd.read_csv(path, dtype={"code": str})
    else:
        fields = ["code", "industry"]
        with _bs_session() as bs:
            rows = _bs_rows(bs.query_stock_industry(), fields)
        if not rows:
            log.warning("行业分类获取失败，返回空映射")
            return {}
        df = pd.DataFrame(rows, columns=fields)
        df = df.rename(columns={"code": "code_raw"})
        df["code"] = df["code_raw"].map(to_tencent)
        df["industry"] = df["industry"].astype(str).str.strip()
        df = df[["code", "industry"]]
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log.info("行业分类已缓存：%d 只", len(df))

    blank = df["industry"].isin(["", "None", "nan"])
    df = df[~blank]
    return dict(zip(df["code"], df["industry"]))


# --------------------------------------------------------------------------- #
# 单只 K 线（不复权，成交额元、成交量手）
# --------------------------------------------------------------------------- #
def fetch_kline(code: str, cfg: Config, *, start: date | None = None,
                end: date | None = None) -> pd.DataFrame:
    """不复权 OHLC 搭配截至报价日已生效的累计后复权乘数；不回填未来事件。"""
    tcode = to_tencent(code)
    if not re.fullmatch(r"(?:sh|sz)[0-9]{6}", tcode):
        raise ValueError(f"无效证券代码：{code!r}")
    bs_code = f"{tcode[:2]}.{tcode[2:]}"
    start = start or date.fromisoformat(cfg.start_date)
    end = end or resolve_market_calendar(cfg).target
    is_index = tcode in {*REFERENCE_INDICES, to_tencent(cfg.benchmark)}
    with _bs_session() as bs:
        rs = bs.query_history_k_data_plus(bs_code, ",".join(_BS_BAR_FIELDS), start_date=start.isoformat(),
                                          end_date=end.isoformat(), frequency="d", adjustflag="3")
        rows = _bs_rows(rs, _BS_BAR_FIELDS)
        events = []
        if rows and not is_index:
            # 从沪深开市之前起查，增量/短窗口也必须继承此前累计因子，而非重新锚为1。
            events = _bs_rows(bs.query_adjust_factor(bs_code, start_date="1990-01-01", end_date=end.isoformat()),
                              ("code", "dividOperateDate", "backAdjustFactor"))
    df = pd.DataFrame(rows, columns=_BS_BAR_FIELDS)
    # 0.9.40 实盘 ResultData.adjustflag 头为空；股票和指数逐行才返回所请求的 '3'。
    if not df.pop("adjustflag").eq("3").all():
        raise ValueError(f"{tcode} 未返回不复权日线")
    if not df.code.eq(bs_code).all() or any(c != bs_code for c, _, _ in events):
        raise ValueError(f"{tcode} 行情或复权因子证券代码不匹配")
    df["code"] = tcode
    df["volume"] = pd.to_numeric(df.volume, errors="raise") / 100.0
    df["date"] = pd.to_datetime(df.date, errors="raise")
    df["adj_factor"] = 1.0
    if events:
        factor = pd.Series([float(v) for _, _, v in events],
                           index=pd.to_datetime([d for _, d, _ in events], errors="raise")).sort_index()
        if (factor.index.has_duplicates or factor.index.hasnans or not np.isfinite(factor).all()
                or (factor <= 0).any() or not factor.index.equals(factor.index.normalize())):
            raise ValueError(f"{tcode} 复权日期或累计因子无效")
        df["adj_factor"] = factor.reindex(df.date, method="ffill").fillna(1.0).to_numpy()
    frame = _dated_bars(df, tcode, start=start, end=end)
    _validate_bars(frame, is_index)
    return frame.reset_index()


# --------------------------------------------------------------------------- #
# 批量快照（截面字段）
# --------------------------------------------------------------------------- #
def fetch_snapshot(codes: Sequence[str], batch: int = 200) -> pd.DataFrame:
    """批量取实时快照：总市值、流通市值、换手率、PE、PB、量比等。"""
    sess = _session()
    tcs = [to_tencent(c) for c in codes]
    recs = []
    for i in range(0, len(tcs), batch):
        chunk = tcs[i : i + batch]
        url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
        try:
            r = _get(sess, url)
            r.encoding = "gbk"
        except RuntimeError as exc:
            log.warning("快照批次 %d 失败: %s", i // batch, exc)
            continue
        for line in r.text.split("\n"):
            line = line.strip()
            if not line.startswith("v_") or "~" not in line:
                continue
            head, _, payload = line.partition("=")
            f = payload.strip().strip('";').split("~")
            if len(f) < 50:
                continue
            recs.append(
                {
                    "code": head[2:].strip(),
                    "name": f[1],
                    "price": _num(f[3]),
                    "prev_close": _num(f[4]),
                    "open": _num(f[5]),
                    "volume": _num(f[6]),
                    "turnover": _num(f[38]),        # 换手率 %
                    "pe_ttm": _num(f[39]),
                    "amplitude": _num(f[43]),
                    "float_mv": _num(f[44]),        # 流通市值（亿）
                    "total_mv": _num(f[45]),        # 总市值（亿）
                    "pb": _num(f[46]),
                    "limit_up": _num(f[47]),
                    "limit_down": _num(f[48]),
                }
            )
        time.sleep(0.05)
    df = pd.DataFrame(recs)
    if not df.empty:
        span = (df["limit_up"] - df["limit_down"]) / df["prev_close"].replace(0, pd.NA)
        df["limit_pct"] = span.round(3)  # 0.20=20cm, 0.10=10cm, 0.05=5cm(ST)
    return df


# --------------------------------------------------------------------------- #
# 批量下载 + 缓存
# --------------------------------------------------------------------------- #
def _cache_path(code: str) -> Path:
    code = to_tencent(code)
    if not re.fullmatch(r"(?:sh|sz)\d{6}", code, flags=re.ASCII):
        raise ValueError(f"无效证券代码：{code!r}")
    return RAW_DIR / _RAW_VERSION / f"{code}.csv"


def load_cached(code: str) -> pd.DataFrame | None:
    p = _cache_path(code)
    if not p.exists():
        if (RAW_DIR / p.name).exists():
            raise ValueError(f"{p.name} 仅有旧腾讯缓存；请重新下载 BaoStock 不复权行情，旧文件保持原样")
        return None
    frame = _dated_bars(pd.read_csv(p), to_tencent(code))
    if frame.empty:
        raise ValueError(f"{p.name} 行情缓存为空，请重新下载")
    _validate_bars(frame, to_tencent(code) in REFERENCE_INDICES)
    return frame.reset_index()


_BAR_COLUMNS = [*KLINE_COLS, "code"]
_FRESH = {"cached", "incremental", "full", "rebased"}


class _Incomplete(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


def _dated_bars(frame, code, *, start=None, end=None):
    """先限定返回窗口再核验价量，且永不静默去重/舍弃损坏的历史行。"""
    if not set(_BAR_COLUMNS).issubset(frame.columns):
        raise ValueError(f"{code} 日线缺少 不复权 OHLCV、amount、adj_factor 或标识字段；旧缓存必须重新下载")
    frame = frame[_BAR_COLUMNS].copy()
    dates = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"), errors="raise"))
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError(f"{code} 日期为空、带时区或含非日线时间")
    frame.index = dates.rename("date")
    if start is not None:
        frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
    if frame.index.has_duplicates or not frame.code.eq(code).all():
        raise ValueError(f"{code} 返回重复日期或证券代码不符")
    fields = [*_PRICE_COLS, "volume", "amount", "adj_factor"]
    frame[fields] = frame[fields].apply(pd.to_numeric, errors="raise")
    return frame.sort_index()


def _validate_bars(frame, is_index):
    prices = frame[_PRICE_COLS].to_numpy(dtype=float)
    if (not np.isfinite(frame[[*_PRICE_COLS, "volume", "amount", "adj_factor"]]).all().all()
            or (prices <= 0).any() or (frame.volume < 0).any() or (frame.amount < 0).any()
            or ((frame.amount == 0) != (frame.volume == 0)).any()
            or (frame.adj_factor <= 0).any() or (is_index and not frame.adj_factor.eq(1).all())
            or (frame.high < frame[["open", "close"]].max(axis=1)).any()
            or (frame.low > frame[["open", "close"]].min(axis=1)).any()):
        raise ValueError("日线数值无效、实际交易锚不为正或 OHLC 区间矛盾")


def _target_status(frame, code, cal, is_index):
    target = pd.Timestamp(cal.target)
    halted = not is_index and cal.is_halted(code, cal.target)
    has_row = target in frame.index
    if has_row and (not np.isfinite(frame.at[target, 'volume']) or frame.at[target, 'volume'] < 0):
        raise ValueError("目标日成交量无效")
    if not has_row or frame.at[target, 'volume'] == 0:
        if halted:
            raise _Incomplete("suspended", "BaoStock 明确该证券目标日停牌，不生成价格行")
        if not has_row:
            raise _Incomplete("missing_target", "来源未返回目标日，停牌未证实；未写行情")
        raise _Incomplete("no_trade_unconfirmed", "目标日零成交量，不能证实有效报价或停牌")
    if halted:
        raise _Incomplete("invalid", "供应方有成交，但 BaoStock 确认当日停牌；来源冲突")


def _check_session_dates(frame, cal):
    # 只在本批官方日历的覆盖区间内判断，不推定缺行原因。
    within = frame.index[frame.index >= pd.Timestamp(cal.covered_from or cal.sessions[0])]
    if not within.isin(cal.sessions).all():
        raise ValueError("日线含权威交易日历以外的日期")


def _check_sessions(frame, code, cal, is_index, start):
    expected = cal.sessions[cal.sessions >= pd.Timestamp(start)]
    missing = expected.difference(frame.index[frame.volume > 0])
    for ts in missing:
        if is_index or not cal.is_halted(code, ts.date()):
            raise _Incomplete("missing_sessions", f"{ts.date()} 缺有效日线且未证实停牌，保留旧缓存")


def save_fetch_table(frame: pd.DataFrame, path: Path, *, write_index=False):
    """在同目录生成并刷盘，只有完整写入才一次性替换目标。"""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.fetch-', suffix='.tmp', delete=False) as temp:
        tmp = Path(temp.name)
    try:
        with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
            frame.to_csv(stream, index=write_index)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _download_one(code: str, cfg: Config, force: bool, cal: MarketCalendar) -> dict:
    code = to_tencent(code)
    target = cal.target
    is_index = code in {*REFERENCE_INDICES, to_tencent(cfg.benchmark)}
    original_rows, last = 0, None

    def result(status, detail=""):
        return {"code": code, "target_date": str(target), "last_date": str(last) if last else None,
                "rows": original_rows, "status": status, "fresh": status in _FRESH, "detail": detail}

    def request(start):
        df = fetch_kline(code, cfg, start=start, end=target)
        if df.empty:
            df = pd.DataFrame(columns=_BAR_COLUMNS)
        out = _dated_bars(df, code, start=start, end=target)
        _check_session_dates(out, cal)
        _target_status(out, code, cal, is_index)
        _validate_bars(out, is_index)
        return out

    try:
        path = _cache_path(code)
        old = None
        start = date.fromisoformat(cfg.start_date)
        repair = False
        if path.exists():
            old = _dated_bars(pd.read_csv(path), code)
            if old.empty:
                raise ValueError("已有缓存无可校验的历史日线")
            original_rows, last = len(old), old.index[-1].date()
            if last > target:
                raise _Incomplete("cache_ahead", "缓存含晚于本批已收盘日的行；不采信、不截断")
            _check_session_dates(old, cal)
            start = min(start, old.index[0].date())  # --force 也不得丢掉更早历史。
            try:
                _validate_bars(old, is_index)
            except ValueError:
                if not force:
                    raise
                repair = True  # 仅在完整重取且旧日期全部覆盖后允许修复。
            if not repair and not force and last == target:
                _target_status(old, code, cal, is_index)
                if is_index:
                    _check_sessions(old, code, cal, True, start)
                return result("cached", "已有目标日完整有效的日线")

        if old is not None and not force:
            # 锚在同源最近三条可观察日线上。历史价格或累计因子修订时须全量重取。
            anchors = old.index[-3:]
            incoming = request(anchors[0].date())
            shared = incoming.reindex(anchors)
            if not anchors.isin(incoming.index).all():
                raise _Incomplete("incomplete_history", "供应方短页遗漏历史锚点，不能验证历史连续性")
            if ((old.loc[anchors, 'volume'] > 0) & (shared.volume == 0)).any():
                raise ValueError("历史锚点从有成交变为零量；拒绝增量覆盖，请核对停牌或全量行情")
            anchors_cols = [*_PRICE_COLS, "adj_factor"]
            prices_changed = not np.allclose(old.loc[anchors, anchors_cols], shared[anchors_cols], rtol=0, atol=1e-10)
            if not prices_changed:
                out = pd.concat([old.loc[old.index < anchors[0]], incoming])
                _check_sessions(out, code, cal, is_index, start if is_index else last + timedelta(days=1))
                status = "incremental"
            else:
                out, status = request(start), "rebased"
        else:
            out, status = request(start), "full"
        if status != "incremental":
            if old is not None and not old.index.isin(out.index).all():
                raise _Incomplete("incomplete_history", "全量结果缺少已有历史日期，拒绝替换旧缓存")
            # 个股上市前不要求报价；旧缓存前的未知历史不当作追补成功的依据。
            first_needed = start if is_index else out.index[0].date()
            _check_sessions(out, code, cal, is_index, first_needed)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_fetch_table(out, path, write_index=True)
        original_rows, last = len(out), target
        return result(status)
    except _Incomplete as exc:
        return result(exc.status, str(exc))
    except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        return result("invalid", str(exc))
    except Exception as exc:  # 单证券失败须可见，但不应中止其余证券的验证。
        return result("error", f"{type(exc).__name__}: {exc}")


def download_all(codes: Iterable[str], cfg: Config, force: bool = False, workers: int = 12,
                 *, calendar: MarketCalendar | None = None) -> pd.DataFrame:
    """全批只解析一次收盘目标；rows 是保留的实际行数，不是行情已新的标记。"""
    cal = calendar if calendar is not None else resolve_market_calendar(cfg)
    codes = list(dict.fromkeys(to_tencent(c) for c in codes))
    if workers < 1 or not codes:
        raise ValueError("工作线程与证券清单均须非空")
    recs = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_download_one, c, cfg, force, cal) for c in codes]
        for n, fut in enumerate(as_completed(futures), 1):
            recs.append(fut.result())
            if n % 200 == 0 or n == len(codes):
                log.info("下载进度 %d/%d  用时 %.0fs", n, len(codes), time.time() - t0)
    return pd.DataFrame(recs).sort_values('code').reset_index(drop=True)


def load_panel(
    codes: Sequence[str], cfg: Config, min_rows: int = 60
) -> dict[str, pd.DataFrame]:
    """从缓存读取面板数据。"""
    panel: dict[str, pd.DataFrame] = {}
    for c in codes:
        df = load_cached(c)
        if df is None or len(df) < min_rows:
            continue
        panel[to_tencent(c)] = df
    return panel
