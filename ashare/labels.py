"""标签构造：与回测执行口径严格对齐的未来超额收益。

约定（重要）：
    在第 t 日收盘后用截至 t 的数据算出因子并给出信号，
    第 t+1 日开盘买入，持有 N 个交易日后第 t+1+N 日开盘卖出。

生产标签使用上述两个开盘时点的比例价（原价乘当日有效因子）计算股票收益，
再减去相同端点的指数收益，按截面分位作为训练目标。它是研究收益代理，
并非账户实际损益；持仓权益、可用日期、手续费与滑点另由回测计算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _forward_open(prices, horizon: int):
    """在原交易历上移动，不能先 dropna，否则缺失端点会被顺延。"""
    valid = prices.where(np.isfinite(prices) & (prices > 0))
    return (valid.shift(-(1 + horizon)) / valid.shift(-1) - 1.0).astype("float32")


def forward_return(open_: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """open-to-open 未来收益。"""
    return _forward_open(open_, horizon)


def forward_benchmark(benchmark: pd.Series, horizon: int) -> pd.Series:
    return _forward_open(benchmark.astype("float64"), horizon)


def build_label(
    open_: pd.DataFrame,
    benchmark: pd.Series,
    horizon: int,
    mask: pd.DataFrame | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """返回横截面分位化的未来超额收益标签。

    normalize=True 时输出 [0,1] 分位，回归目标对异常值不敏感；
    normalize=False 时输出原始超额收益（用于 IC 分析与展示）。
    """
    fr = forward_return(open_, horizon)
    fb = forward_benchmark(benchmark.reindex(open_.index), horizon)
    excess = fr.sub(fb, axis=0)
    if mask is not None:
        excess = excess.where(mask)
    if not normalize:
        return excess
    return rank_excess(excess)


def rank_excess(excess: pd.DataFrame) -> pd.DataFrame:
    return excess.rank(axis=1, pct=True).astype("float32")


def panel_excess(mats: dict[str, pd.DataFrame], benchmark_code: str, horizon: int,
                 mask: pd.DataFrame | None) -> pd.DataFrame:
    """生产与冒烟共用：从同一张开盘价表取股票和指数，端点无成交不计收益。"""
    volume = mats['volume']
    # 比例开盘价仅用于研究收益，不是成交报价或可提现的股利。
    opens = (mats['open'] * mats['adj_factor']).where(np.isfinite(volume) & (volume > 0))
    return build_label(opens, opens[benchmark_code], horizon, mask, normalize=False)


def realized_return(open_: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """原始未来收益，用于回测后的归因与展示。"""
    return forward_return(open_, horizon)
