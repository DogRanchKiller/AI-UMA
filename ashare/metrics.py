"""绩效指标计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 244


def _ann_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / TRADING_DAYS
    if years <= 0:
        return float("nan")
    return float(total ** (1 / years) - 1)


def _ann_vol(ret: pd.Series) -> float:
    ret = ret.dropna()
    if len(ret) < 2:
        return float("nan")
    return float(ret.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity / peak - 1.0).astype("float64")


def summarize(equity: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """核心指标汇总。equity 为净值序列（起点 1.0）。"""
    ret = equity.pct_change().dropna()
    mdd = max_drawdown(equity)
    ann = _ann_return(equity)
    vol = _ann_vol(ret)
    sharpe = ann / vol if vol and vol > 0 else float("nan")

    out = {
        "累计收益": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else float("nan"),
        "年化收益": ann,
        "年化波动": vol,
        "夏普比率": sharpe,
        "最大回撤": mdd,
        "卡玛比率": (ann / abs(mdd)) if mdd and mdd < 0 else float("nan"),
        "日胜率": float((ret > 0).mean()) if len(ret) else float("nan"),
        "交易日数": int(len(ret)),
    }
    if benchmark is not None and len(benchmark.dropna()) > 1:
        b = benchmark.reindex(equity.index).ffill()
        b_ret = b.pct_change().dropna()
        ex = ret.sub(b_ret.reindex(ret.index)).dropna()
        ex_ann = float((1 + ex).prod() ** (TRADING_DAYS / max(len(ex), 1)) - 1)
        te = float(ex.std() * np.sqrt(TRADING_DAYS)) if len(ex) > 1 else float("nan")
        out.update({
            "基准累计收益": float(b.iloc[-1] / b.iloc[0] - 1) if b.iloc[0] else float("nan"),
            "基准年化": _ann_return(b.dropna()),
            "超额年化": ex_ann,
            "信息比率": (ex_ann / te) if te and te > 0 else float("nan"),
            "月胜率": _monthly_win_rate(ret, b_ret),
            "最大回撤": mdd,
        })
    return out


def _monthly_win_rate(ret: pd.Series, b_ret: pd.Series) -> float:
    try:
        m = (1 + ret).resample("ME").prod() - 1
        bm = (1 + b_ret.reindex(ret.index)).resample("ME").prod() - 1
        both = pd.concat([m, bm], axis=1).dropna()
        if both.empty:
            return float("nan")
        return float((both.iloc[:, 0] > both.iloc[:, 1]).mean())
    except Exception:  # noqa: BLE001 - 时间序列无法重采样时不影响主指标
        return float("nan")


def ic_series(scores: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """每日横截面 Spearman IC。"""
    common = scores.index.intersection(fwd_ret.index)
    vals = {}
    for d in common:
        a = scores.loc[d]
        b = fwd_ret.loc[d]
        both = pd.concat([a, b], axis=1).dropna()
        if len(both) < 20:
            continue
        vals[d] = both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman")
    return pd.Series(vals, dtype="float64").sort_index()


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    if ic.empty:
        return {"IC均值": float("nan"), "ICIR": float("nan"), "IC胜率": float("nan")}
    mean = float(ic.mean())
    std = float(ic.std())
    return {
        "IC均值": mean,
        "ICIR": (mean / std) if std > 0 else float("nan"),
        "IC胜率": float((ic > 0).mean()),
        "IC样本数": int(len(ic)),
    }


def layered_returns(scores: pd.DataFrame, fwd_ret: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """按分数分层，返回各层平均未来收益，用于检验单调性。"""
    rows = {}
    for d in scores.index.intersection(fwd_ret.index):
        s, r = scores.loc[d], fwd_ret.loc[d]
        both = pd.concat([s.rename("s"), r.rename("r")], axis=1).dropna()
        if len(both) < n * 4:
            continue
        both["g"] = pd.qcut(both["s"].rank(method="first"), n, labels=False)
        rows[d] = both.groupby("g")["r"].mean()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T.sort_index()


def turnover_stats(weights: pd.DataFrame, every: int = 20) -> dict:
    """weights: date × code 的目标权重。

    年化换手按「每年调仓次数」折算，而不是按调仓行数折算——后者会把
    单次换手率错误地放大成几千个百分点。返回值中 1.0 表示 100% 换手。
    """
    empty = {"平均单边换手": float("nan"), "年化单边换手": float("nan")}
    if weights.empty or len(weights) < 2:
        return empty
    w = weights.fillna(0.0)
    diff = w.diff().abs().sum(axis=1) / 2.0
    diff = diff.iloc[1:]
    if diff.empty:
        return empty
    per = float(diff.mean())
    # 调仓频率：优先按实际日期跨度推算，退化时使用配置的调仓周期
    try:
        span_days = (w.index[-1] - w.index[0]).days
        n_years = span_days / 365.25
        rebalances_per_year = (len(diff) / n_years) if n_years > 0 else TRADING_DAYS / max(every, 1)
    except (TypeError, AttributeError):
        rebalances_per_year = TRADING_DAYS / max(every, 1)
    rebalances_per_year = float(min(rebalances_per_year, TRADING_DAYS))  # 防御异常输入
    return {
        "平均单边换手": per,
        "年化单边换手": per * rebalances_per_year,
        "年调仓次数": round(rebalances_per_year, 1),
    }
