"""回测引擎测试：资金守恒、涨跌停约束、费用计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare import backtest as bt
from ashare import metrics as mt
from ashare import universe as uv
from ashare.config import Config


def _scenario(opens, limit_up_idx=None, limit_down_idx=None):
    """构造 3 个交易日 × 2 只股票的确定性行情。"""
    idx = pd.bdate_range("2024-03-04", periods=3)
    cols = ["sh600000", "sh600001"]
    o = pd.DataFrame(opens, index=idx, columns=cols, dtype="float64")
    c = o.copy()
    mats = {"open": o, "close": c, "high": o, "low": o, "volume": o * 0 + 1e6,
            "amount": o * 1e8}
    mask = pd.DataFrame(True, index=idx, columns=cols)
    prev = c.shift(1)
    can_buy, can_sell = uv.tradeable_at_open(o, prev, cols)
    if limit_up_idx is not None:
        d, code = limit_up_idx
        can_buy.at[d, code] = False
    if limit_down_idx is not None:
        d, code = limit_down_idx
        can_sell.at[d, code] = False
    bench = pd.Series([3000.0, 3010.0, 3020.0], index=idx)
    return mats, mask, can_buy, can_sell, bench, idx, cols


def _cfg(**kw):
    cfg = Config()
    cfg.top_n = 2
    cfg.max_weight = 0.6
    cfg.init_cash = 100_000.0
    cfg.commission = 0.00025
    cfg.min_commission = 5.0
    cfg.stamp_tax = 0.0005
    cfg.slippage = 0.0          # 便于验证精确费用
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_no_signal_means_no_trade():
    mats, mask, cb, cs, bench, idx, cols = _scenario([[10, 20], [10, 20], [10, 20]])
    empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"), columns=cols, dtype="float64")
    with pytest.raises(ValueError):
        bt.run(empty, mats, mask, cb, cs, bench, _cfg())


def test_buy_then_value_tracks_price():
    """建仓后净值应随持仓价格变动，而不是停在 1.0。"""
    mats, mask, cb, cs, bench, idx, cols = _scenario([[10, 20], [10, 20], [12, 20]])
    scores = pd.DataFrame(
        [[0.9, 0.5], [np.nan, np.nan], [np.nan, np.nan]],
        index=idx, columns=cols,
    )
    res = bt.run(scores, mats, mask, cb, cs, bench, _cfg())
    # 第二天开盘建仓，第三天 sh600000 从 10 涨到 12
    assert res.diagnostics["成交笔数"] > 0, "应该有成交"
    assert res.equity.iloc[-1] > 1.0, "持仓股票上涨后净值应大于 1"
    assert res.equity.notna().all()


def test_cash_never_negative():
    mats, mask, cb, cs, bench, idx, cols = _scenario([[10, 20], [10, 20], [10, 20]])
    scores = pd.DataFrame(
        [[0.9, 0.8], [np.nan, np.nan], [np.nan, np.nan]], index=idx, columns=cols
    )
    cfg = _cfg()
    res = bt.run(scores, mats, mask, cb, cs, bench, cfg)
    assert res.equity.min() > 0


def test_limit_up_blocks_buy():
    mats, mask, cb, cs, bench, idx, cols = _scenario([[10, 20], [10, 20], [10, 20]])
    scores = pd.DataFrame(
        [[0.9, 0.5], [np.nan, np.nan], [np.nan, np.nan]], index=idx, columns=cols
    )
    res = bt.run(scores, mats, mask, cb, cs, bench, _cfg())
    bought_1 = res.trades[res.trades["code"] == "sh600000"]
    assert not bought_1.empty

    # 同样场景，但把最强的一只设为一字涨停
    mats2, mask2, cb2, cs2, bench2, idx2, cols2 = _scenario(
        [[10, 20], [10, 20], [10, 20]], limit_up_idx=(idx[1], "sh600000")
    )
    res2 = bt.run(scores, mats2, mask2, cb2, cs2, bench2, _cfg())
    assert res2.diagnostics["涨停未买入"] >= 1, "涨停股不应成交"
    assert (res2.trades["code"] == "sh600000").sum() == 0


def test_limit_down_blocks_sell():
    """跌停时持仓卖不出去，应计入诊断并继续持有。

    信号必须落在「倒数第二天」，这样才有下一交易日可供成交；
    把信号放在最后一天会因为不存在执行日而完全跳过，测不到任何东西。
    """
    mats, mask, cb, cs, bench, idx, cols = _scenario([[10, 20], [10, 20], [10, 20]])
    cfg = _cfg(top_n=1)
    # idx[0] 选 sh600000 -> 在 idx[1] 建仓
    # idx[1] 改为选 sh600001 -> 计划在 idx[2] 卖出 sh600000
    scores = pd.DataFrame(
        [[0.99, 0.01], [0.01, 0.99], [np.nan, np.nan]], index=idx, columns=cols
    )

    ok = bt.run(scores, mats, mask, cb, cs, bench, cfg)
    assert (ok.trades["code"] == "sh600000").sum() > 0, "应先建仓"
    assert (ok.trades["side"] == "sell").sum() > 0, "正常情形下应能卖出"

    # 同样流程，但 idx[2] 时 sh600000 跌停
    mats2, mask2, cb2, cs2, bench2, idx2, cols2 = _scenario(
        [[10, 20], [10, 20], [10, 20]], limit_down_idx=(idx[2], "sh600000")
    )
    blocked = bt.run(scores, mats2, mask2, cb2, cs2, bench2, cfg)
    assert blocked.diagnostics["跌停未卖出"] >= 1
    sells = blocked.trades[blocked.trades["side"] == "sell"]
    assert (sells["code"] == "sh600000").sum() == 0, "跌停股不应被卖出"


def test_fees_are_applied():
    """买入应扣除佣金，成本价高于开盘价（含滑点）。"""
    mats, mask, cb, cs, bench, idx, cols = _scenario([[10, 20], [10, 20], [10, 20]])
    scores = pd.DataFrame(
        [[0.9, 0.5], [np.nan, np.nan], [np.nan, np.nan]], index=idx, columns=cols
    )
    cfg = _cfg(slippage=0.001)
    res = bt.run(scores, mats, mask, cb, cs, bench, cfg)
    buy = res.trades[res.trades["side"] == "buy"].iloc[0]
    assert buy["price"] > 10.0, "含滑点的买入价应高于开盘价"
    assert buy["fee"] >= 5.0, "佣金应不低于最低佣金"


def test_turnover_annualization_is_reasonable():
    """年化换手不应出现几千倍这种荒谬数值。"""
    idx = pd.bdate_range("2023-01-02", periods=25, freq="20B")
    w = pd.DataFrame(0.1, index=idx, columns=[f"c{i}" for i in range(10)])
    # 每期完全换一批股票
    for k, d in enumerate(idx):
        w.loc[d] = 0.0
        for i in range(k * 10, k * 10 + 10):
            w.loc[d, f"c{i % 10}"] = 0.1
    st = mt.turnover_stats(w, every=20)
    assert st["年化单边换手"] < 60, f"年化换手异常: {st}"


def test_max_drawdown_sign():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1], index=pd.bdate_range("2024-01-01", periods=4))
    dd = mt.max_drawdown(eq)
    assert -0.26 < dd < -0.24, f"最大回撤算错: {dd}"


# --------------------------------------------------------------------------- #
# 权重方案与回撤刹车
# --------------------------------------------------------------------------- #
def _weights_for(cfg, vol=None, n=4):
    row = pd.Series({f"c{i}": 1.0 - i * 0.1 for i in range(n)})
    target = list(row.index)
    vr = pd.Series(vol) if vol is not None else None
    return bt.target_weights(row, target, vr, cfg)


def test_equal_weighting_sums_to_one():
    w = _weights_for(_cfg(max_weight=1.0))
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(abs(v - 0.25) < 1e-9 for v in w.values())


def test_inv_vol_favours_low_volatility():
    w = _weights_for(
        _cfg(weighting="inv_vol", max_weight=1.0),
        vol={"c0": 0.03, "c1": 0.01, "c2": 0.02, "c3": 0.04},
    )
    assert w["c1"] > w["c2"] > w["c0"] > w["c3"], f"逆波动权重排序错误: {w}"
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_rank_decay_weights_decrease_with_rank():
    w = _weights_for(_cfg(weighting="rank_decay", rank_decay=1.0, max_weight=1.0))
    vals = [w[c] for c in ["c0", "c1", "c2", "c3"]]
    assert vals == sorted(vals, reverse=True)
    assert abs(sum(vals) - 1.0) < 1e-9


def test_max_weight_cap_respected_and_renormalised():
    w = _weights_for(_cfg(max_weight=0.30), n=4)
    assert max(w.values()) <= 0.30 + 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_inv_vol_falls_back_when_vol_missing():
    w = _weights_for(_cfg(weighting="inv_vol", max_weight=1.0),
                     vol={"c0": None, "c1": None, "c2": None, "c3": None})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v > 0 for v in w.values())


def test_drawdown_brake_reduces_exposure():
    """触发刹车后应清仓观望，而不是一路持有到谷底。"""
    idx = pd.bdate_range("2024-03-04", periods=5)
    cols = ["sh600000", "sh600001"]
    o = pd.DataFrame([[10, 20], [10, 20], [8, 20], [7, 20], [6, 20]],
                     index=idx, columns=cols, dtype="float64")
    mats = {"open": o, "close": o, "high": o, "low": o,
            "volume": o * 0 + 1e6, "amount": o * 1e8}
    mask = pd.DataFrame(True, index=idx, columns=cols)
    cb, cs = uv.tradeable_at_open(o, o.shift(1), cols)
    bench = pd.Series([3000.0] * 5, index=idx)

    scores = pd.DataFrame(
        [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [np.nan, np.nan]],
        index=idx, columns=cols,
    )
    cfg = _cfg(top_n=1, drawdown_brake=0.05, drawdown_brake_scale=0.0,
               drawdown_recover=0.0, max_weight=1.0)
    res = bt.run(scores, mats, mask, cb, cs, bench, cfg)
    assert res.diagnostics["刹车清仓次数"] >= 1, "回撤应触发刹车"
    assert res.equity.min() > 0


def test_brake_off_by_default():
    mats, mask, cb, cs, bench, idx, cols = _scenario([[10, 20], [10, 20], [10, 20]])
    scores = pd.DataFrame(
        [[0.9, 0.5], [np.nan, np.nan], [np.nan, np.nan]], index=idx, columns=cols
    )
    res = bt.run(scores, mats, mask, cb, cs, bench, _cfg())
    assert res.diagnostics.get("刹车清仓次数", 0) == 0
