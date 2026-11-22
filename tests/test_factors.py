"""因子层测试：数值正确性 + 无前视泄漏。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare import factors as fc


# 独立冻结迁移前的公开契约；不能从生产元数据反推预期集合。
EXPECTED_FACTOR_NAMES = tuple("""
    ret_1 ret_5 ret_10 ret_20 ret_60 ret_120 mom_60_skip5 mom_250_skip20
    vol_20 vol_60 dvol_20 parkinson_20 atr_ratio dd_60 te_60
    turn_ratio amihud_20 log_amount amount_ratio vol_cv_20
    corr_pv_20 updown_vol_20 obv_slope_20 vwap_dev
    ma5_div ma20_div ma60_div ma_trend rsi_14 boll_pos high_250_dist low_250_dist
    breakout_60 consec_up limit_up_cnt_20 overnight_5 intraday_5 upper_shadow_20
    lower_shadow_20 beta_60 alpha_60 rel_ret_20 corr_bench_60
    ind_rel_ret_20 ind_rel_ret_5 ind_rel_ret_60 ind_rel_amount
""".split())


def _toy_panel(n_days: int = 400, n_codes: int = 12, seed: int = 7):
    """合成未复权日线：复权事件按生效日写入；成交额按模拟成交均价而非收盘报价生成。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    ret = rng.normal(0.0004, 0.02, size=(n_days, n_codes))
    close = 10.0 * np.exp(np.cumsum(ret, axis=0))
    open_ = np.vstack([np.full(n_codes, 10.), close[:-1]]) * (1 + rng.normal(0, .005, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, .01, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, .01, close.shape)))
    factor = np.ones(close.shape)
    factor[150:, ::2] *= 2.0
    factor[320:, ::3] *= 1.25
    shares = rng.integers(300_000, 3_000_000, size=close.shape)
    vwap = (open_ + close) / (2 * factor)  # 以两笔等量模拟成交定义样本的真实成交额
    return {
        f"sh6000{i:02d}": pd.DataFrame({
            "date": dates, "open": open_[:, i] / factor[:, i], "high": high[:, i] / factor[:, i],
            "low": low[:, i] / factor[:, i], "close": close[:, i] / factor[:, i],
            "volume": shares[:, i] / 100, "amount": vwap[:, i] * shares[:, i],
            "adj_factor": factor[:, i],
        }) for i in range(n_codes)
    }


def _toy_mats(**kwargs):
    return fc.clean_matrices(fc.build_matrices(_toy_panel(**kwargs)))


def _bars(close, adj_factor, *, vwap, hands=10.):
    """手算用合成样本。vwap 是给定的当日成交均价，不用报价代替。"""
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=len(close)),
        "open": vwap, "close": close, "high": np.maximum(close, vwap) * 1.01,
        "low": np.minimum(close, vwap) * .99, "volume": hands,
        "amount": np.asarray(vwap) * np.asarray(hands) * 100, "adj_factor": adj_factor,
    })


def test_all_declared_factors_produced():
    mats = _toy_mats()
    bench = mats["close"].mean(axis=1)
    ind = _industry_map(mats)
    fac = fc.compute_factors(mats, benchmark=bench, industry=ind)
    assert len(EXPECTED_FACTOR_NAMES) == 47
    assert tuple(fc.FACTOR_META) == tuple(fac) == EXPECTED_FACTOR_NAMES
    for k, v in fac.items():
        assert isinstance(v, pd.DataFrame), f"{k} 不是宽表"
        assert v.shape == mats["close"].shape, f"{k} 形状不符"
        assert str(v.dtypes.iloc[0]) == "float32", f"{k} 应为 float32"


def test_factors_have_no_lookahead():
    """核心测试：截断未来数据后，历史因子值必须完全不变。"""
    panel = _toy_panel()
    mats = fc.build_matrices(panel)
    bench = mats["close"].mean(axis=1)
    industry = _industry_map(mats)
    full = fc.compute_factors(mats, benchmark=bench, industry=industry)

    cut = 320  # 下一行部分股票第二次除权；必须从旧日期原行情独立建矩阵。
    prefix = {code: df.iloc[:cut].copy() for code, df in panel.items()}
    trunc_mats = fc.build_matrices(prefix)
    tail_mats = fc.build_matrices({code: df.iloc[cut:] for code, df in panel.items()})
    for name in trunc_mats:
        pd.testing.assert_frame_equal(trunc_mats[name], mats[name].iloc[:cut])
        # 追加中的股票已发生过除权，尾段不能重新基准化或补单位因子。
        pd.testing.assert_frame_equal(pd.concat([trunc_mats[name], tail_mats[name]]), mats[name])
    trunc = fc.compute_factors(trunc_mats, benchmark=bench.iloc[:cut], industry=industry)

    for name, wide in full.items():
        a = wide.iloc[:cut]
        b = trunc[name]
        pd.testing.assert_frame_equal(
            a, b, check_exact=False, rtol=1e-5, atol=1e-6,
            obj=f"因子 {name} 存在前视泄漏",
        )


def test_ma5_div_matches_original_audit_parameters():
    """审计给出三个数而非历史行情；构造满足末价和均价的五日窗口。"""
    shift, last, mean = 20.0, 30.0, 29.0
    close = np.array([28., 28., 29., 30., last])
    assert close.mean() == mean
    shifted = close - shift
    assert shifted[-1] / shifted.mean() - 1 == pytest.approx(1 / 9)

    bars = _bars(close, np.ones(5), vwap=close)
    factors = fc.compute_factors(fc.build_matrices({"X": bars}))
    actual = factors["ma5_div"]["X"].iloc[-1].item()
    expected = last / mean - 1  # 未平移参照值 1/29，不用舍入后的 3.45%
    assert abs(actual - expected) / abs(expected) < 1e-6


def test_reconstructed_audit_ma_div_with_split_in_window():
    """独立的跨除权重建样本，不是原始三参数算例。"""
    raw = np.array([28., 28., 29., 15., 15.])
    factors = [1, 1, 1, 2, 2]  # 第 4 日起比例变更；日后加法扣减再带入未来 8 元
    additive = raw - [22.5, 22.5, 22.5, 8., 8.]
    assert additive[-1] / additive.mean() - 1 == pytest.approx(1 / 9)  # +11.1%
    bars = _bars(raw, factors, vwap=raw - .1)
    mats = fc.build_matrices({"X": bars})
    fac = fc.compute_factors(mats)
    np.testing.assert_array_equal(mats["close"]["X"], raw)  # 交易用原价从未被改写
    # [28,28,29,30,30] 均价 29。若先把 30/29 转 float32 再减 1，相对误差达 1.19e-6。
    assert fac["ma5_div"]["X"].iloc[-1].item() == pytest.approx(1 / 29, rel=1e-6, abs=0)
    assert fac["ret_1"]["X"].iloc[3].item() == pytest.approx(1 / 29, rel=1e-6, abs=0)


def test_all_four_prices_share_one_basis_across_ex_date():
    # 前十日 1 股 20 元，后十日拆为 2 股、每股收盘 10 元。比例 OHLC 始终 19/21/18/20。
    split = np.repeat([1., 2.], 10)
    bars = _bars(20 / split, split, vwap=19 / split)
    bars["high"], bars["low"] = 21 / split, 18 / split
    mats = fc.build_matrices({"X": bars})
    fac = fc.compute_factors(mats)
    assert fac["atr_ratio"]["X"].iloc[-1] == pytest.approx(3 / 20)
    assert fac["overnight_5"]["X"].iloc[10] == pytest.approx(19 / 20 - 1)
    assert fac["intraday_5"]["X"].iloc[10] == pytest.approx(20 / 19 - 1)
    for field in ("open", "high", "low", "close"):
        np.testing.assert_array_equal(mats[field]["X"], bars[field])


def test_vwap_and_liquidity_across_ex_date():
    # 各10日：除权前成交均价20、10手，除权后成交均价11、30手；比例成交均价21.5。
    bars = _bars([21.] * 10 + [12.] * 10, [1.] * 10 + [2.] * 10,
                 vwap=[20.] * 10 + [11.] * 10, hands=[10.] * 10 + [30.] * 10)
    mats = fc.build_matrices({"X": bars})
    fac = fc.compute_factors(mats)
    np.testing.assert_array_equal(mats["amount"]["X"], [20_000] * 10 + [33_000] * 10)
    assert fac["vwap_dev"]["X"].iloc[-1] == pytest.approx(24 / 21.5 - 1)
    assert fac["log_amount"]["X"].iloc[-1] == pytest.approx(np.log1p(26_500))
    assert fac["amount_ratio"]["X"].iloc[-1] == pytest.approx(33_000 / 26_500)


@pytest.mark.parametrize("field", ["amount", "adj_factor"])
def test_legacy_bar_and_matrix_fail_closed(field):
    bars = _bars([15.], [1.], vwap=[14.])
    with pytest.raises(ValueError, match=field):
        fc.build_matrices({"X": bars.drop(columns=field)})
    mats = fc.build_matrices({"X": bars})
    del mats[field]
    with pytest.raises(ValueError, match=field):
        fc.compute_factors(mats)


@pytest.mark.parametrize(("field", "bad"), [("adj_factor", 0.), ("adj_factor", -1.),
    ("adj_factor", np.nan), ("adj_factor", np.inf), ("amount", np.nan), ("amount", -1.),
    ("volume", -1.), ("open", np.inf), ("close", 0.), ("amount", 0.)])
def test_invalid_traded_row_fails_closed(field, bad):
    bars = _bars([15.], [1.], vwap=[14.])
    mats = fc.build_matrices({"X": bars})
    bars.loc[0, field] = bad
    mats[field].iloc[0, 0] = bad
    with pytest.raises(ValueError, match=field):
        fc.build_matrices({"X": bars})
    with pytest.raises(ValueError, match=field):
        fc.compute_factors(mats)


def test_zero_turnover_is_not_imputed():
    bars = _bars([15.] * 20, [1.] * 20, vwap=14., hands=0.)
    fac = fc.compute_factors(fc.build_matrices({"X": bars}))
    assert fac["log_amount"]["X"].iloc[-1] == 0.
    assert np.isnan(fac["vwap_dev"]["X"].iloc[-1])


def test_duplicate_bars_and_nonchronological_matrix_are_rejected():
    bars = _bars([15., 15.], [1., 1.], vwap=14.)
    with pytest.raises(ValueError, match="日期"):
        fc.build_matrices({"X": pd.concat([bars, bars.iloc[:1]])})
    mats = fc.build_matrices({"X": bars})
    with pytest.raises(ValueError, match="日期"):
        fc.compute_factors({k: v.iloc[::-1] for k, v in mats.items()})


def test_unlisted_date_stays_missing_and_misaligned_factor_is_rejected():
    bars = _bars([15., 15., 15.], [1., 1., 1.], vwap=[14., 14., 14.])
    mats = fc.build_matrices({"X": bars.iloc[1:], "Y": bars})
    assert all(frame["X"].isna().iloc[0] for frame in mats.values())
    assert np.isnan(fc.compute_factors(mats)["ret_1"]["X"].iloc[1])
    mats["adj_factor"] = mats["adj_factor"].iloc[::-1]
    with pytest.raises(ValueError, match="adj_factor"):
        fc.compute_factors(mats)


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
    shocked["amount"][codes] = shocked["amount"][codes] * 1.5  # 成交价等比例变动，手数不变
    after = fc.compute_factors(shocked, benchmark=shocked["close"].mean(axis=1),
                               industry=ind)

    a = base["ind_rel_ret_20"].iloc[-1][codes].mean()
    b = after["ind_rel_ret_20"].iloc[-1][codes].mean()
    assert abs(b) < 0.05, f"行业整体冲击后组内相对收益应仍接近 0，实际 {b:.3f}"
    assert abs(a) < 0.05
