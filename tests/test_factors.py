"""因子层测试：数值正确性 + 无前视泄漏。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare import factors as fc


def _toy_mats(n_days: int = 400, n_codes: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    cols = [f"sh6000{i:02d}" for i in range(n_codes)]
    ret = rng.normal(0.0004, 0.02, size=(n_days, n_codes))
    close = 10.0 * np.exp(np.cumsum(ret, axis=0))
    close = pd.DataFrame(close, index=idx, columns=cols).astype("float32")
    high = close * (1 + np.abs(rng.normal(0, 0.01, close.shape)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, close.shape)))
    open_ = close.shift(1).bfill() * (1 + rng.normal(0, 0.005, close.shape))
    vol = pd.DataFrame(
        rng.lognormal(12, 0.5, size=(n_days, n_codes)), index=idx, columns=cols
    ).astype("float32")
    return {
        "open": open_.astype("float32"),
        "high": high.astype("float32"),
        "low": low.astype("float32"),
        "close": close,
        "volume": vol,
        "amount": (close * vol * 100).astype("float32"),
    }


def test_all_declared_factors_produced():
    mats = _toy_mats()
    bench = mats["close"].mean(axis=1)
    fac = fc.compute_factors(mats, benchmark=bench)
    missing = [k for k in fc.FACTOR_META if k not in fac]
    assert not missing, f"以下因子未生成: {missing}"
    for k, v in fac.items():
        assert isinstance(v, pd.DataFrame), f"{k} 不是宽表"
        assert v.shape == mats["close"].shape, f"{k} 形状不符"
        assert str(v.dtypes.iloc[0]) == "float32", f"{k} 应为 float32"


def test_factors_have_no_lookahead():
    """核心测试：截断未来数据后，历史因子值必须完全不变。"""
    mats = _toy_mats()
    bench = mats["close"].mean(axis=1)
    full = fc.compute_factors(mats, benchmark=bench)

    cut = 320
    trunc_mats = {k: v.iloc[:cut] for k, v in mats.items()}
    trunc = fc.compute_factors(trunc_mats, benchmark=bench.iloc[:cut])

    for name, wide in full.items():
        a = wide.iloc[:cut]
        b = trunc[name]
        pd.testing.assert_frame_equal(
            a, b, check_exact=False, rtol=1e-5, atol=1e-6,
            obj=f"因子 {name} 存在前视泄漏",
        )


def test_ret_1_matches_manual_computation():
    mats = _toy_mats(n_days=50)
    fac = fc.compute_factors(mats, benchmark=mats["close"].mean(axis=1))
    close = mats["close"]
    expected = close.iloc[-1] / close.iloc[-2] - 1
    got = fac["ret_1"].iloc[-1]
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-5)


def test_cross_sectional_process_is_bounded():
    mats = _toy_mats()
    fac = fc.compute_factors(mats, benchmark=mats["close"].mean(axis=1))
    std = fc.standardize_panel(fac)
    for name, wide in std.items():
        s = wide.stack().dropna()
        assert len(s) > 0, f"{name} 全为空"
        assert s.min() >= 0.0 and s.max() <= 1.0, f"{name} 越界"
        # 分位数标准化的中位数应接近 0.5
        assert 0.4 < s.median() < 0.6, f"{name} 中位数异常 {s.median()}"


def test_streak_counts_consecutive_days():
    idx = pd.bdate_range("2024-01-01", periods=6)
    cond = pd.DataFrame({"a": [True, True, False, True, True, True]}, index=idx)
    got = fc._streak(cond)["a"].tolist()
    assert got == [1, 2, 0, 1, 2, 3], f"实际 {got}"


def test_standardization_handles_constant_column():
    """某一列全天不变时不应产生 NaN 或异常值。"""
    mats = _toy_mats(n_days=120, n_codes=5)
    mats["close"]["sh600000"] = 10.0
    fac = fc.compute_factors(mats, benchmark=mats["close"].mean(axis=1))
    std = fc.standardize_panel(fac)
    s = std["ret_20"].stack().dropna()
    assert len(s) > 0
    assert s.min() >= 0 and s.max() <= 1
