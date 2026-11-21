"""选股约束与仓位闸门测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare import selection as sl
from ashare.config import Config


def _mats(n=200, m=40, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = [f"sh6000{i:02d}" for i in range(m)]
    ret = rng.normal(0.0003, 0.02, (n, m))
    close = pd.DataFrame(10 * np.exp(np.cumsum(ret, axis=0)), index=idx, columns=cols)
    vol = pd.DataFrame(rng.lognormal(12, .4, (n, m)), index=idx, columns=cols)
    return {
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": vol, "amount": close * vol * 100,
    }


def test_selection_mask_disabled_allows_everything():
    mats = _mats()
    cfg = Config(sel_min_amount_ma20=0.0, sel_max_vol_pct=1.0,
                 sel_min_ret_20=-1.0, sel_min_ret_5=-1.0)
    m = sl.selection_mask(mats, cfg)
    assert m.iloc[-1].sum() == mats["close"].shape[1]


def test_selection_mask_liquidity():
    mats = _mats()
    cfg = Config(sel_min_amount_ma20=1e20)
    m = sl.selection_mask(mats, cfg)
    assert m.sum().sum() == 0


def test_selection_mask_vol_cap():
    mats = _mats()
    cfg = Config(sel_max_vol_pct=0.5)
    m = sl.selection_mask(mats, cfg)
    last = m.iloc[-1]
    assert last.sum() < mats["close"].shape[1], "应剔除一部分高波动标的"


def test_apply_to_scores_nans_excluded():
    mats = _mats()
    idx = mats["close"].index[-5:]
    cols = mats["close"].columns
    sc = pd.DataFrame(0.5, index=idx, columns=cols)
    sel = pd.DataFrame(True, index=idx, columns=cols)
    sel[cols[0]] = False
    out = sl.apply_to_scores(sc, sel)
    assert out[cols[0]].isna().all()
    assert out[cols[1]].notna().all()


def test_neutralize_removes_size_exposure():
    """中性化后，分数与规模的相关性应显著下降。

    注意：若构造的分数与规模完全共线，残差会恒等于 0，而对常数序列求
    Spearman 相关是未定义的（会返回随机值），所以这里用「规模 + 噪声」
    构造一个相关系数约 0.9 的分数来检验。
    """
    mats = _mats()
    amt = mats["amount"]
    idx = mats["close"].index[-1:]
    cols = mats["close"].columns
    # 中性化内部的规模代理是 20 日均额的对数，测试必须用同一口径比较
    size = np.log1p(amt.rolling(20, min_periods=10).mean().iloc[-1])
    rng = np.random.default_rng(5)
    noisy = size.rank(pct=True) * 2.0 + rng.normal(0, 0.12, len(cols))
    sc = pd.DataFrame([noisy.values], index=idx, columns=cols)

    before = sc.iloc[0].corr(size, method="spearman")
    neut = sl.neutralize_scores(sc, mats, use_size=True, use_vol=False)
    after = neut.iloc[0].corr(size, method="spearman")
    assert abs(before) > 0.6, f"构造的分数应与规模相关，实际 {before}"
    assert abs(after) < abs(before) * 0.5, f"中性化效果不足: {before} -> {after}"


def test_neutralize_perfect_collinear_gives_zero_residual():
    """完全共线时残差应被压到 0，而不是翻转成反向暴露。"""
    mats = _mats()
    amt = mats["amount"]
    idx = mats["close"].index[-1:]
    cols = mats["close"].columns
    size = np.log1p(amt.rolling(20, min_periods=10).mean().iloc[-1])
    sc = pd.DataFrame([size.values], index=idx, columns=cols)
    neut = sl.neutralize_scores(sc, mats, use_size=True, use_vol=False)
    assert neut.iloc[0].abs().max() < 1e-6, "完全共线时残差应约为 0"


def test_neutralize_preserves_nan():
    mats = _mats()
    sc = pd.DataFrame(np.random.rand(3, mats["close"].shape[1]),
                      index=mats["close"].index[-3:], columns=mats["close"].columns)
    sc.iloc[0, 0] = np.nan
    out = sl.neutralize_scores(sc, mats)
    assert pd.isna(out.iloc[0, 0])


def test_market_regime_flips_with_trend():
    idx = pd.bdate_range("2023-01-02", periods=200)
    up = pd.Series(np.linspace(3000, 4000, 200), index=idx)
    down = pd.Series(np.linspace(4000, 3000, 200), index=idx)
    r_up = sl.market_regime(up, ma=60)
    r_dn = sl.market_regime(down, ma=60)
    assert bool(r_up.iloc[-1]) is True, "上升趋势应开放"
    assert bool(r_dn.iloc[-1]) is False, "下降趋势应关闭"


def test_market_regime_buffer_requires_stronger_signal():
    idx = pd.bdate_range("2023-01-02", periods=200)
    flat = pd.Series(3000 + np.sin(np.arange(200) / 8) * 40, index=idx)
    weak = sl.market_regime(flat, ma=60, buffer=0.0)
    strong = sl.market_regime(flat, ma=60, buffer=0.10)
    assert weak.sum() >= strong.sum(), "缓冲越大，开放次数不应增加"


# --------------------------------------------------------------------------- #
# 迟滞闸门
# --------------------------------------------------------------------------- #
def test_hysteresis_switches_less_than_plain():
    """来回震荡的行情里，迟滞闸门的状态翻转次数应明显少于纯均线穿越。"""
    idx = pd.bdate_range("2023-01-02", periods=400)
    rng = np.random.default_rng(2)
    px = pd.Series(3000 + np.cumsum(rng.normal(0, 8, 400)), index=idx)
    plain = sl.market_regime(px, ma=60)
    hyst = sl.market_regime_hysteresis(px, ma=60, enter=0.0, exit_=-0.02)
    flips_plain = int((plain.astype(int).diff().abs() == 1).sum())
    flips_hyst = int((hyst.astype(int).diff().abs() == 1).sum())
    assert flips_hyst <= flips_plain, f"迟滞翻转 {flips_hyst} 应不多于纯穿越 {flips_plain}"


def test_hysteresis_follows_clear_trend():
    idx = pd.bdate_range("2023-01-02", periods=200)
    up = pd.Series(np.linspace(3000, 4200, 200), index=idx)
    dn = pd.Series(np.linspace(4200, 3000, 200), index=idx)
    assert bool(sl.market_regime_hysteresis(up, 60).iloc[-1]) is True
    assert bool(sl.market_regime_hysteresis(dn, 60).iloc[-1]) is False


def test_hysteresis_exit_band_delays_exit():
    """退出阈值放宽后，持有状态不应更早结束。"""
    idx = pd.bdate_range("2023-01-02", periods=300)
    px = pd.Series(np.concatenate([np.linspace(3000, 3600, 150),
                                   np.linspace(3600, 3400, 150)]), index=idx)
    tight = sl.market_regime_hysteresis(px, 60, enter=0.0, exit_=0.0)
    loose = sl.market_regime_hysteresis(px, 60, enter=0.0, exit_=-0.05)
    assert loose.sum() >= tight.sum(), "更宽的退出带应持有更久"


# --------------------------------------------------------------------------- #
# 宽度闸门
# --------------------------------------------------------------------------- #
def test_breadth_regime_requires_min_names():
    mats = _mats(n=120, m=10)
    r = sl.market_regime_breadth(mats["close"], ma=60, threshold=0.5, min_names=200)
    assert not r.any(), "样本不足时应保守判为关闭"


def test_breadth_regime_opens_in_broad_uptrend():
    idx = pd.bdate_range("2023-01-02", periods=150)
    cols = [f"c{i}" for i in range(260)]
    close = pd.DataFrame(np.linspace(10, 20, 150)[:, None].repeat(260, axis=1),
                         index=idx, columns=cols)
    r = sl.market_regime_breadth(close, ma=60, threshold=0.5, min_names=200)
    assert bool(r.iloc[-1]) is True, "全市场齐涨时应开放"


# --------------------------------------------------------------------------- #
# 连续仓位系数
# --------------------------------------------------------------------------- #
def test_regime_exposure_bounded_and_monotone():
    idx = pd.bdate_range("2023-01-02", periods=200)
    up = pd.Series(np.linspace(3000, 4200, 200), index=idx)
    dn = pd.Series(np.linspace(4200, 3000, 200), index=idx)

    e_up = sl.regime_exposure(up, ma=60, band=0.04)
    e_dn = sl.regime_exposure(dn, ma=60, band=0.04)

    assert e_up.min() >= 0 and e_up.max() <= 1
    assert e_dn.min() >= 0 and e_dn.max() <= 1
    assert e_up.iloc[-1] > 0.99, "单边上涨应接近满仓"
    assert e_dn.iloc[-1] < 0.01, "单边下跌应接近空仓"


def test_regime_exposure_wider_band_means_gentler():
    idx = pd.bdate_range("2023-01-02", periods=300)
    px = pd.Series(3000 + np.cumsum(np.random.default_rng(9).normal(0, 10, 300)), index=idx)
    narrow = sl.regime_exposure(px, ma=60, band=0.02)
    wide = sl.regime_exposure(px, ma=60, band=0.10)
    assert narrow.std() >= wide.std(), "带宽越窄，仓位变化越剧烈"


def test_regime_exposure_never_nan():
    idx = pd.bdate_range("2023-01-02", periods=120)
    px = pd.Series(np.linspace(3000, 3300, 120), index=idx)
    e = sl.regime_exposure(px, ma=60, band=0.04)
    assert e.notna().all()
    # 均线未形成时保守判为 0 仓位
    assert e.iloc[0] == 0.0
