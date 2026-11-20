"""标签测试：对齐口径 + 前视防护。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare import labels as lb


def _frames(n=30, m=20):
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = [f"sh6000{i:02d}" for i in range(m)]
    rng = np.random.default_rng(3)
    px = pd.DataFrame(
        10 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, m)), axis=0)),
        index=idx, columns=cols,
    ).astype("float32")
    bench = pd.Series(np.linspace(3000, 3200, n), index=idx)
    return px, bench


def test_label_matches_open_to_open_definition():
    """标签必须等于 open[t+1+H] / open[t+1] - 1 减基准同期收益。"""
    px, bench = _frames()
    H = 5
    excess = lb.build_label(px, bench, H, normalize=False)

    t = px.index[3]
    expect_stock = px.iloc[3 + 1 + H] / px.iloc[3 + 1] - 1
    expect_bench = bench.iloc[3 + 1 + H] / bench.iloc[3 + 1] - 1
    got = excess.loc[t]
    np.testing.assert_allclose(got.values, (expect_stock - expect_bench).values, rtol=1e-4)


def test_last_horizon_rows_are_nan():
    """信号日太靠后时标签必须缺失，不能悄悄用不存在的未来数据。"""
    px, bench = _frames()
    H = 5
    excess = lb.build_label(px, bench, H, normalize=False)
    tail = excess.iloc[-(H + 1):]
    assert tail.isna().all().all(), "末尾标签应为 NaN"
    assert excess.iloc[:-2].notna().any().any()


def test_normalized_label_is_percentile():
    px, bench = _frames()
    y = lb.build_label(px, bench, 5, normalize=True)
    s = y.stack().dropna()
    assert s.min() > 0 and s.max() <= 1.0
    # 横截面分位的理论均值为 (m+1)/(2m)，m=20 时约 0.525
    assert 0.45 < s.mean() < 0.60, f"分位均值异常 {s.mean()}"
    # 每个交易日的分位应均匀铺满 (0,1]
    per_day = y.dropna(how="all").iloc[0].dropna()
    assert len(per_day) == px.shape[1]
    assert per_day.nunique() == px.shape[1], "同日分位不应有重复"


def test_mask_removes_rows():
    px, bench = _frames()
    mask = pd.DataFrame(True, index=px.index, columns=px.columns)
    mask["sh600000"] = False
    y = lb.build_label(px, bench, 3, mask=mask, normalize=False)
    assert y["sh600000"].isna().all()


def test_benchmark_alignment_with_missing_days():
    """基准缺日时不应产生错位的标签。"""
    px, bench = _frames()
    bench = bench.drop(bench.index[5])
    excess = lb.build_label(px, bench, 4, normalize=False)
    assert excess.notna().any().any()
    assert excess.shape == px.shape
