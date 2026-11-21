"""行情数据获取与本地缓存。

数据源：
  * 腾讯财经 K 线接口（前复权，速度快，支持分页取全历史）
  * 腾讯批量快照接口（市值、换手、PE、PB 等截面字段）
  * baostock 仅用于一次性拉取 A 股主列表（含证券名称，用于 ST 判定）

设计原则：
  1. 所有网络访问都有重试与超时，单只失败不阻断整体。
  2. 落盘缓存为逐股 CSV，二次运行不重复下载。
  3. 返回的 DataFrame 统一列名，日期为 datetime，按日期升序。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

from .config import CACHE_DIR, RAW_DIR, Config

log = logging.getLogger(__name__)

KLINE_COLS = ["date", "open", "close", "high", "low", "volume"]
_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://gu.qq.com/",
}

# 腾讯 K 线单次最多返回 640 根
_PAGE_LIMIT = 640

# 腾讯的反爬是按域名而不是按 IP 生效的：实测 web.ifzq.gtimg.cn 被封时，
# 下面这些域名仍然可用。轮询可显著提高大批量抓取的成功率。
KLINE_HOSTS = [
    "https://ifzq.gtimg.cn/appstock/app/fqkline/get",
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
]


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


def _looks_like_json(text: str) -> bool:
    """腾讯反爬会返回 200/501 的 HTML 挑战页，需要按内容而非状态码判断。"""
    t = text.lstrip()
    return t.startswith("{") or t.startswith("[")


def fetch_kline_window_raw(
    sess: requests.Session, tcode: str, start: date, end: date
) -> pd.DataFrame:
    """依次尝试多个腾讯域名，任一成功即返回。"""
    param = f"{tcode},day,{start:%Y-%m-%d},{end:%Y-%m-%d},{_PAGE_LIMIT},qfq"
    errors = []
    for host in KLINE_HOSTS:
        try:
            r = _get(sess, f"{host}?param={param}", retries=2)
            if not _looks_like_json(r.text):
                errors.append(f"{host}: 反爬页")
                continue
            j = r.json()
            if j.get("code") not in (0, None):
                errors.append(f"{host}: code={j.get('code')}")
                continue
            return _parse_kline(j, tcode)
        except Exception as exc:  # noqa: BLE001 - 换域名继续
            errors.append(f"{host}: {type(exc).__name__}")
            continue
    raise RuntimeError("全部域名失败 | " + "; ".join(errors))


def _parse_kline(j: dict, tcode: str) -> pd.DataFrame:
    data = (j.get("data") or {}).get(tcode) or {}
    rows = data.get("qfqday") or data.get("day") or []
    recs = []
    for row in rows:
        if not row or len(row) < 6:
            continue
        d = str(row[0])
        if len(d) != 10 or d[4] != "-":
            continue  # 过滤混入的元数据行
        recs.append((
            d,
            _num(row[1]),   # open
            _num(row[2]),   # close
            _num(row[3]),   # high
            _num(row[4]),   # low
            _num(row[5]),   # volume（手）
        ))
    return pd.DataFrame(recs, columns=KLINE_COLS)


def fetch_kline_window(
    sess: requests.Session, tcode: str, start: date, end: date
) -> pd.DataFrame:
    """取 [start, end] 窗口内最多 640 根日线，前复权。"""
    return fetch_kline_window_raw(sess, tcode, start, end)


def _num(x) -> float:
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- #
# A 股主列表
# --------------------------------------------------------------------------- #
def _baostock_master() -> pd.DataFrame:
    import baostock as bs

    bs.login()
    rows: list[list[str]] = []
    fields: list[str] = []
    try:
        # 当日非交易日（周末/节假日）时逐日回退，直到拿到非空列表
        for back in range(0, 20):
            day = (datetime.now() - timedelta(days=back)).strftime("%Y-%m-%d")
            rs = bs.query_all_stock(day=day)
            if rs.error_code != "0":
                continue
            got = []
            while rs.next():
                got.append(rs.get_row_data())
            if got:
                rows, fields = got, rs.fields
                break
    finally:
        bs.logout()
    df = pd.DataFrame(rows, columns=fields) if rows else pd.DataFrame()
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
    STOCK_PREFIX = (
        "sh.600", "sh.601", "sh.603", "sh.605", "sh.688",
        "sz.000", "sz.001", "sz.002", "sz.003", "sz.300", "sz.301",
    )
    df = df[df["code"].str.startswith(STOCK_PREFIX)].copy()
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
        import baostock as bs

        bs.login()
        rows: list[list[str]] = []
        try:
            rs = bs.query_stock_industry()
            fields = rs.fields
            while rs.next():
                rows.append(rs.get_row_data())
        finally:
            bs.logout()
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
# 单只 K 线（前复权）
# --------------------------------------------------------------------------- #
def fetch_kline(
    code: str,
    cfg: Config,
    sess: requests.Session | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """分页向前回溯，取完整前复权日线历史。"""
    sess = sess or _session()
    tcode = to_tencent(code)
    start = start or date.fromisoformat(cfg.start_date)
    end = end or date.today()

    frames: list[pd.DataFrame] = []
    cursor = end
    guard = 0
    while cursor >= start and guard < 60:
        guard += 1
        part = fetch_kline_window(sess, tcode, start, cursor)
        if part.empty:
            break
        frames.append(part)
        oldest = date.fromisoformat(part["date"].min())
        if oldest <= start:
            break
        nxt = oldest - timedelta(days=1)
        if nxt >= cursor:  # 无进展，防止死循环
            break
        cursor = nxt

    if not frames:
        return pd.DataFrame(columns=KLINE_COLS + ["code"])
    df = pd.concat(frames, ignore_index=True)
    df = (
        df.drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["code"] = to_tencent(code)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(start)]
    return df


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
    return RAW_DIR / f"{to_tencent(code)}.csv"


def load_cached(code: str) -> pd.DataFrame | None:
    p = _cache_path(code)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["date"])
    except Exception:  # noqa: BLE001 - 损坏缓存视为缺失
        return None
    if df.empty:
        return None
    return df


def _download_one(code: str, cfg: Config, force: bool) -> tuple[str, int, str]:
    try:
        cached = None if force else load_cached(code)
        if cached is not None:
            last = cached["date"].max().date()
            if last >= date.today() - timedelta(days=7):
                return code, len(cached), "cached"
            inc = fetch_kline(code, cfg, start=last + timedelta(days=1))
            if not inc.empty:
                cached = (
                    pd.concat([cached, inc], ignore_index=True)
                    .drop_duplicates("date")
                    .sort_values("date")
                )
                cached.to_csv(_cache_path(code), index=False, encoding="utf-8-sig")
            return code, len(cached), "incremental"
        df = fetch_kline(code, cfg)
        if df.empty:
            return code, 0, "empty"
        df.to_csv(_cache_path(code), index=False, encoding="utf-8-sig")
        return code, len(df), "full"
    except Exception as exc:  # noqa: BLE001 - 单只失败不应中断整体
        return code, 0, f"error:{type(exc).__name__}"


def download_all(
    codes: Iterable[str], cfg: Config, force: bool = False, workers: int = 12
) -> pd.DataFrame:
    """并发下载并缓存，返回每只的结果统计。"""
    codes = list(dict.fromkeys(codes))
    recs = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, c, cfg, force): c for c in codes}
        for n, fut in enumerate(as_completed(futs), 1):
            code, rows, status = fut.result()
            recs.append({"code": code, "rows": rows, "status": status})
            if n % 200 == 0 or n == len(codes):
                log.info(
                    "下载进度 %d/%d  用时 %.0fs", n, len(codes), time.time() - t0
                )
    return pd.DataFrame(recs)


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
