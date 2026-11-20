"""标签构造：与回测执行口径严格对齐的未来超额收益。

约定（重要）：
    在第 t 日收盘后用截至 t 的数据算出因子并给出信号，
    第 t+1 日开盘买入，持有 N 个交易日后第 t+1+N 日开盘卖出。

因此标签为 open[t+1+N] / open[t+1] - 1，再减去同期基准收益，
最后做横截面分位标准化，作为回归目标。这样模型优化的目标与
回测实际能拿到的收益一致，不会因为用了收盘价而对不上。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(open_: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """open-to-open 未来收益。"""
    buy = open_.shift(-1)
    sell = open_.shift(-(1 + horizon))
    return (sell / buy.replace(0.0, np.nan) - 1.0).astype("float32")


def forward_benchmark(benchmark: pd.Series, horizon: int) -> pd.Series:
    b = benchmark.astype("float64")
    return (b.shift(-(1 + horizon)) / b.shift(-1) - 1.0).astype("float32")


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
    fb = forward_benchmark(benchmark, horizon)
    excess = fr.sub(fb, axis=0)
    if mask is not None:
        excess = excess.where(mask)
    if not normalize:
        return excess
    return excess.rank(axis=1, pct=True).astype("float32")


def realized_return(open_: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """原始未来收益，用于回测后的归因与展示。"""
    return forward_return(open_, horizon)
