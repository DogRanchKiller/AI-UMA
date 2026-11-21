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
    ind = _industry_map(mats)
    fac = fc.compute_factors(mats, benchmark=bench, industry=ind)
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


# --------------------------------------------------------------------------- #
# 行业内相对强弱因子
# --------------------------------------------------------------------------- #
def _industry_map(mats, n_ind=3):
    cols = list(mats["close"].columns)
    return {c: f"行业{i % n_ind}" for i, c in enumerate(cols)}


def test_industry_factors_created_when_industry_given():
    mats = _toy_mats()
    ind = _industry_map(mats)
    fac = fc.compute_factors(mats, benchmark=mats["close"].mean(axis=1), industry=ind)
    for k in ("ind_rel_ret_20", "ind_rel_ret_5", "ind_rel_ret_60", "ind_rel_amount"):
        assert k in fac, f"缺少行业因子 {k}"
        assert fac[k].shape == mats["close"].shape


def test_industry_factors_absent_without_industry():
    mats = _toy_mats()
    fac = fc.compute_factors(mats, benchmark=mats["close"].mean(axis=1))
    assert "ind_rel_ret_20" not in fac
    # 但元数据里声明了，最终会被裁掉，不能报错
    assert "ret_20" in fac


def test_industry_relative_return_is_centered_within_group():
    """行业内相对收益在每一组内的中位数应接近 0（行业共同部分被减掉）。"""
    mats = _toy_mats(n_days=200, n_codes=30)
    ind = _industry_map(mats, n_ind=3)
    fac = fc.compute_factors(mats, benchmark=mats["close"].mean(axis=1), industry=ind)
    wide = fac["ind_rel_ret_20"].iloc[-1].dropna()
    groups = pd.Series({c: ind[c] for c in wide.index})
    med = wide.groupby(groups).median()
    assert (med.abs() < 1e-5).all(), f"组内中位数未归零: {med.to_dict()}"


def test_industry_relative_removes_industry_shock():
    """给整个行业加一个共同冲击，行业内相对收益不应受影响。"""
    mats = _toy_mats(n_days=150, n_codes=20)
    ind = _industry_map(mats, n_ind=2)
    base = fc.compute_factors(mats, benchmark=mats["close"].mean(axis=1), industry=ind)

    shocked = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in mats.items()}
    codes = [c for c in mats["close"].columns if ind[c] == "行业0"]
    for k in ("open", "high", "low", "close"):
        shocked[k][codes] = shocked[k][codes] * 1.5
    shocked["amount"] = (shocked["close"] * shocked["volume"] * 100).astype("float32")
    after = fc.compute_factors(shocked, benchmark=shocked["close"].mean(axis=1),
                               industry=ind)

    a = base["ind_rel_ret_20"].iloc[-1][codes].mean()
    b = after["ind_rel_ret_20"].iloc[-1][codes].mean()
    assert abs(b) < 0.05, f"行业整体冲击后组内相对收益应仍接近 0，实际 {b:.3f}"
    assert abs(a) < 0.05
