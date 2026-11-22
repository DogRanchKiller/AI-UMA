"""外部增量数据：股东户数、业绩预告、龙虎榜。

这三类数据与量价因子相关性低，是真正独立的信息来源：
    * 股东户数  —— 筹码集中度变化，A股经典 alpha 方向（户数减少=筹码集中）
    * 业绩预告  —— 盈利预期的一次性跳变，公告即生效
    * 龙虎榜    —— 游资/机构席位行为

**前视防护是这里最重要的事。** 财务与股东数据都有「报告期」和「公告日」两个日期，
信号必须在公告日之后才可用。例如 2026-09-30 的股东户数要到 2026-11-13 才公告，
若按报告期对齐，等于提前 1.5 个月偷看未来，会造出漂亮的假 alpha。

因此本模块所有对外暴露的时间序列都以**公告日（或上榜日）的次一交易日**为起点。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR, Config

log = logging.getLogger(__name__)


def _cache(name: str) -> Path:
    return CACHE_DIR / f"ext_{name}.csv"


def _load(name: str) -> pd.DataFrame | None:
    p = _cache(name)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, dtype={"code": str})
    except Exception:  # noqa: BLE001 - 缓存损坏视为缺失
        return None


def _save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(_cache(name), index=False, encoding="utf-8-sig")


def _quarters(start_year: int = 2020, end: pd.Timestamp | None = None) -> list[str]:
    """生成季度报告期字符串，如 20240930。"""
    end = end or pd.Timestamp.today()
    out = []
    for y in range(start_year, end.year + 1):
        for md in ("0331", "0630", "0930", "1231"):
            q = f"{y}{md}"
            if pd.Timestamp(f"{y}-{md[:2]}-{md[2:]}") <= end:
                out.append(q)
    return out


def norm_code(x) -> str:
    """统一成腾讯风格六位代码：600000 -> sh600000。"""
    s = str(x).strip().split(".")[0].zfill(6)
    if s.startswith(("60", "68", "51", "58")):
        return "sh" + s
    if s.startswith(("00", "30", "15", "16")):
        return "sz" + s
    if s.startswith(("8", "4")):
        return "bj" + s
    return s


# --------------------------------------------------------------------------- #
# 股东户数
# --------------------------------------------------------------------------- #
def fetch_holder(cfg: Config, force: bool = False) -> pd.DataFrame:
    """股东户数变化。返回 code, report_date, ann_date, holders, chg_pct, avg_mv。"""
    if not force:
        cached = _load("holder")
        if cached is not None and len(cached):
            return cached

    import akshare as ak

    rows = []
    for q in _quarters(cfg.start_year):
        try:
            df = ak.stock_zh_a_gdhs(symbol=q)
        except Exception as exc:  # noqa: BLE001 - 单季失败不影响其余
            log.warning("股东户数 %s 失败: %s", q, type(exc).__name__)
            continue
        if df is None or df.empty:
            continue
        cols = {c: str(c) for c in df.columns}
        need = ["代码", "股东户数-本次", "股东户数-增减比例", "户均持股市值", "公告日期"]
        if not all(n in cols for n in need):
            log.warning("股东户数 %s 字段不全，跳过", q)
            continue
        part = pd.DataFrame({
            "code": df["代码"].map(norm_code),
            "report_date": q,
            "ann_date": pd.to_datetime(df["公告日期"], errors="coerce"),
            "holders": pd.to_numeric(df["股东户数-本次"], errors="coerce"),
            "holder_chg_pct": pd.to_numeric(df["股东户数-增减比例"], errors="coerce"),
            "avg_mv": pd.to_numeric(df["户均持股市值"], errors="coerce"),
        })
        rows.append(part.dropna(subset=["ann_date"]))
        time.sleep(0.15)

    if not rows:
        log.warning("股东户数全部失败，返回空")
        return pd.DataFrame(columns=["code", "report_date", "ann_date",
                                     "holders", "holder_chg_pct", "avg_mv"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["code", "report_date"])
    _save(out, "holder")
    log.info("股东户数：%d 条，%d 只", len(out), out["code"].nunique())
    return out


# --------------------------------------------------------------------------- #
# 业绩预告
# --------------------------------------------------------------------------- #
_FORECAST_SCORE = {
    "预增": 2, "略增": 1, "扭亏": 2, "续盈": 1,
    "预减": -2, "略减": -1, "首亏": -2, "续亏": -2, "增亏": -2, "减亏": 1,
    "不确定": 0,
}


def fetch_forecast(cfg: Config, force: bool = False) -> pd.DataFrame:
    """业绩预告。返回 code, report_date, ann_date, fc_type, fc_amp, fc_score。"""
    if not force:
        cached = _load("forecast")
        if cached is not None and len(cached):
            return cached

    import akshare as ak

    rows = []
    for q in _quarters(cfg.start_year):
        try:
            df = ak.stock_yjyg_em(date=q)
        except Exception as exc:  # noqa: BLE001
            log.warning("业绩预告 %s 失败: %s", q, type(exc).__name__)
            continue
        if df is None or df.empty:
            continue
        need = ["股票代码", "预告类型", "业绩变动幅度", "公告日期"]
        if not all(n in df.columns for n in need):
            continue
        part = pd.DataFrame({
            "code": df["股票代码"].map(norm_code),
            "report_date": q,
            "ann_date": pd.to_datetime(df["公告日期"], errors="coerce"),
            "fc_type": df["预告类型"].astype(str).str.strip(),
            "fc_amp": pd.to_numeric(df["业绩变动幅度"], errors="coerce"),
        })
        rows.append(part.dropna(subset=["ann_date"]))
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=["code", "report_date", "ann_date",
                                     "fc_type", "fc_amp", "fc_score"])
    out = pd.concat(rows, ignore_index=True)
    out["fc_score"] = out["fc_type"].map(_FORECAST_SCORE).fillna(0)
    # 同一公告日同股多条（多个指标）时取最强的一条
    out = (out.sort_values("fc_score", key=lambda s: s.abs(), ascending=False)
              .drop_duplicates(["code", "ann_date"]))
    _save(out, "forecast")
    log.info("业绩预告：%d 条，%d 只", len(out), out["code"].nunique())
    return out


# --------------------------------------------------------------------------- #
# 龙虎榜
# --------------------------------------------------------------------------- #
def fetch_lhb(cfg: Config, force: bool = False) -> pd.DataFrame:
    """龙虎榜。返回 code, trade_date, lhb_net, lhb_buy, lhb_sell。"""
    if not force:
        cached = _load("lhb")
        if cached is not None and len(cached):
            return cached

    import akshare as ak

    start = cfg.start_date.replace("-", "")
    end = (cfg.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
    df = ak.stock_lhb_detail_em(start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "trade_date", "lhb_net"])

    out = pd.DataFrame({
        "code": df["代码"].map(norm_code),
        "trade_date": pd.to_datetime(df["上榜日"], errors="coerce"),
        "lhb_net": pd.to_numeric(df.get("龙虎榜净买额"), errors="coerce"),
        "lhb_buy": pd.to_numeric(df.get("龙虎榜买入额"), errors="coerce"),
        "lhb_sell": pd.to_numeric(df.get("龙虎榜卖出额"), errors="coerce"),
    }).dropna(subset=["trade_date"])

    # 同一只票同一天可能因多个上榜原因重复出现，净买额只保留一条
    out = out.groupby(["code", "trade_date"], as_index=False).agg(
        lhb_net=("lhb_net", "sum"),
        lhb_buy=("lhb_buy", "max"),
        lhb_sell=("lhb_sell", "max"),
    )
    _save(out, "lhb")
    log.info("龙虎榜：%d 条，%d 只", len(out), out["code"].nunique())
    return out


# --------------------------------------------------------------------------- #
# 转成日频因子宽表（严格按公告日可用）
# --------------------------------------------------------------------------- #
def _to_daily(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    calendar: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """把「公告日 + 数值」的点事件转成日频宽表。

    规则：数据从公告日的**次一交易日**起可用（当天收盘后才看得到公告），
    并一直前向填充到下次更新。这样保证任意日期 t 的因子只用到 <= t 的信息。
    """
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return {v: pd.DataFrame(index=calendar) for v in value_cols}

    cal = pd.DatetimeIndex(calendar).sort_values()
    cal_np = cal.values.astype("datetime64[ns]")
    for vcol in value_cols:
        sub = df[["code", date_col, vcol]].dropna()
        if sub.empty:
            out[vcol] = pd.DataFrame(index=cal)
            continue
        sub = sub.copy()
        # CSV 读回的是 datetime 对象，必须先统一成 datetime64 才能与日历比较
        sub["avail"] = pd.to_datetime(sub[date_col], errors="coerce")
        sub = sub.dropna(subset=["avail"])
        if sub.empty:
            out[vcol] = pd.DataFrame(index=cal)
            continue
        # 找到公告日之后第一个交易日（side="right" 保证不含公告日当天）
        pos = np.searchsorted(cal_np, sub["avail"].values.astype("datetime64[ns]"),
                              side="right")
        ok = pos < len(cal)
        sub = sub.loc[ok].copy()
        sub["avail"] = cal_np[pos[ok]]

        wide = sub.pivot_table(index="avail", columns="code", values=vcol,
                               aggfunc="last")
        wide = wide.reindex(cal).ffill()
        out[vcol] = wide
    return out


def build_ext_factors(
    cfg: Config, calendar: pd.DatetimeIndex, force: bool = False
) -> dict[str, pd.DataFrame]:
    """构造外部数据因子宽表，列名与价格矩阵一致（腾讯风格代码）。"""
    holder = fetch_holder(cfg, force)
    forecast = fetch_forecast(cfg, force)
    lhb = fetch_lhb(cfg, force)

    fac: dict[str, pd.DataFrame] = {}

    h = _to_daily(holder, "ann_date",
                  ["holders", "holder_chg_pct", "avg_mv"], calendar)
    if "holder_chg_pct" in h and not h["holder_chg_pct"].empty:
        # 户数变化率取负：户数减少（筹码集中）→ 因子值更高
        fac["ext_holder_chg"] = (-h["holder_chg_pct"]).astype("float32")
        fac["ext_avg_mv"] = np.log1p(h["avg_mv"]).astype("float32")

    f = _to_daily(forecast, "ann_date", ["fc_score", "fc_amp"], calendar)
    if "fc_score" in f and not f["fc_score"].empty:
        fac["ext_fc_score"] = f["fc_score"].astype("float32")
        fac["ext_fc_amp"] = f["fc_amp"].astype("float32")

    # 龙虎榜：上榜日当天收盘后可见，次一交易日可用；做 20 日累计净买入
    l = _to_daily(lhb, "trade_date", ["lhb_net"], calendar)
    if "lhb_net" in l and not l["lhb_net"].empty:
        net = l["lhb_net"].fillna(0.0)
        fac["ext_lhb_net_20"] = net.rolling(20, min_periods=5).sum().astype("float32")
        cnt = (net != 0).astype("float32")
        fac["ext_lhb_cnt_20"] = cnt.rolling(20, min_periods=5).sum().astype("float32")

    return {k: v for k, v in fac.items() if not v.empty}
