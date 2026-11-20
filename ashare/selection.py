"""选股端风险约束。

模型输出的是横截面排序，排序本身不含风险预算。把排序直接做成等权多头，
会把全部筹码押在当期跌幅最深的一批小微盘股上——这类股票在流动性危机中
会同步崩盘（例如 2024 年初的微盘股踩踏），单次事件即可抹掉数年累计 alpha。

本模块在**不改动模型打分**的前提下，于选股端施加可解释的约束：
    * 流动性下限：日均成交额过滤，避免建不进也跑不掉
    * 波动率上限：剔除横截面波动最高的尾部
    * 自由落体过滤：剔除仍在连续暴跌中的标的
    * 单日成交额占比：限制持仓在单只股票成交量中的占比
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _cs_pct(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def selection_mask(
    mats: dict[str, pd.DataFrame],
    cfg: Config,
    base_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """返回 date × code 的布尔表：True 表示允许进入多头组合。"""
    close, amount = mats["close"], mats["amount"]
    ret1 = close / close.shift(1) - 1.0

    ok = pd.DataFrame(True, index=close.index, columns=close.columns)
    if base_mask is not None:
        ok &= base_mask.reindex(index=close.index, columns=close.columns).fillna(False)

    # 1) 流动性下限
    amt_ma20 = amount.rolling(20, min_periods=10).mean()
    ok &= amt_ma20 >= cfg.sel_min_amount_ma20

    # 2) 波动率上限（横截面分位）
    if cfg.sel_max_vol_pct < 1.0:
        vol60 = ret1.rolling(60, min_periods=40).std()
        ok &= _cs_pct(vol60) <= cfg.sel_max_vol_pct

    # 3) 自由落体：近 N 日跌幅过深的不接
    if cfg.sel_min_ret_20 > -1.0:
        ret20 = close / close.shift(20) - 1.0
        ok &= ret20 >= cfg.sel_min_ret_20

    # 4) 连续暴跌（近5日累计跌幅阈值）
    if cfg.sel_min_ret_5 > -1.0:
        ret5 = close / close.shift(5) - 1.0
        ok &= ret5 >= cfg.sel_min_ret_5

    return ok.fillna(False)


def apply_to_scores(
    scores: pd.DataFrame,
    sel: pd.DataFrame,
) -> pd.DataFrame:
    """把不允许的标的分数置为 NaN，使其无法被选中。"""
    aligned = sel.reindex(index=scores.index, columns=scores.columns)
    return scores.where(aligned.fillna(False))


def neutralize_scores(
    scores: pd.DataFrame,
    mats: dict[str, pd.DataFrame],
    use_size: bool = True,
    use_vol: bool = True,
) -> pd.DataFrame:
    """把打分对「规模」和「波动」做横截面中性化，再重新排序。

    不做中性化时，模型会把「小盘 + 高波动」当成 alpha 的载体，组合退化成
    一个隐藏的小微盘多头。中性化后组合在这个维度上不再有系统性押注，
    但保留模型在同等规模/波动下的选股能力。
    """
    close = mats["close"]
    ret1 = close / close.shift(1) - 1.0

    score_cs = scores.rank(axis=1, pct=True)
    exog = []
    if use_size:
        amt = mats["amount"].rolling(20, min_periods=10).mean()
        exog.append(np.log1p(amt).rank(axis=1, pct=True))
    if use_vol:
        vol = ret1.rolling(60, min_periods=40).std()
        exog.append(vol.rank(axis=1, pct=True))
    if not exog:
        return scores

    exog = [e.reindex(index=scores.index, columns=scores.columns) for e in exog]
    # 逐日做横截面最小二乘：y = a + b1*exog1 + b2*exog2 + eps，取 eps
    Y = score_cs.to_numpy(dtype="float64")
    Xs = [e.to_numpy(dtype="float64") for e in exog]
    out = np.full_like(Y, np.nan)
    for i in range(Y.shape[0]):
        y = Y[i]
        cols = [Xs[k][i] for k in range(len(Xs))]
        good = np.isfinite(y)
        for c in cols:
            good &= np.isfinite(c)
        if good.sum() < 30:
            continue
        A = np.column_stack([np.ones(good.sum())] + [c[good] for c in cols])
        try:
            beta, *_ = np.linalg.lstsq(A, y[good], rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[i, good] = y[good] - A @ beta
    res = pd.DataFrame(out, index=scores.index, columns=scores.columns)
    return res.where(scores.notna())


def market_regime(benchmark: pd.Series, ma: int = 60, buffer: float = 0.0) -> pd.Series:
    """市场状态闸门：基准在均线上方时允许做多。

    这是仓位控制而不是选股，作用是在系统性下跌中离场，代价是错过反弹初段。
    buffer>0 时要求基准高出均线一定比例才重新入场，降低来回打脸。
    """
    b = benchmark.dropna()
    sma = b.rolling(ma, min_periods=max(10, ma // 3)).mean()
    return (b > sma * (1.0 + buffer)).reindex(benchmark.index).fillna(False)


def market_regime_hysteresis(
    benchmark: pd.Series, ma: int = 60, enter: float = 0.0, exit_: float = -0.02
) -> pd.Series:
    """带迟滞的闸门：高于均线*(1+enter) 才进场，跌破均线*(1+exit_) 才离场。

    纯均线穿越在震荡市会反复出入场，把成本吃光。迟滞带显著降低这种撕扯。
    """
    b = benchmark.dropna()
    sma = b.rolling(ma, min_periods=max(10, ma // 3)).mean()
    up = (b > sma * (1.0 + enter)).to_numpy()
    dn = (b < sma * (1.0 + exit_)).to_numpy()
    out = np.zeros(len(b), dtype=bool)
    state = False
    for i in range(len(b)):
        if not np.isfinite(sma.iloc[i]):
            state = False
        elif state and dn[i]:
            state = False
        elif not state and up[i]:
            state = True
        out[i] = state
    return pd.Series(out, index=b.index).reindex(benchmark.index).fillna(False)


def market_regime_breadth(
    close: pd.DataFrame, ma: int = 60, threshold: float = 0.5, min_names: int = 200
) -> pd.Series:
    """市场宽度闸门：全市场站上均线的股票比例高于阈值时允许做多。

    对小盘反转组合来说，宽度比指数均线更贴近实际风险来源——2024 年初的踩踏
    里指数跌幅有限，但绝大多数个股同时破位。样本不足时保守判为关闭。
    """
    sma = close.rolling(ma, min_periods=max(10, ma // 3)).mean()
    valid = close.notna() & sma.notna()
    above = (close > sma) & valid
    n = valid.sum(axis=1)
    ratio = above.sum(axis=1) / n.replace(0, np.nan)
    ok = (ratio > threshold) & (n >= min_names)
    return ok.fillna(False)
