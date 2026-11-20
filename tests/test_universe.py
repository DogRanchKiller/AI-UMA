"""股票池与涨跌停判定测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare import universe as uv
from ashare.config import Config


def _mats(n=60, codes=("sh600000", "sz300001", "sh688001", "sz000001")):
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.DataFrame(10.0, index=idx, columns=list(codes))
    vol = pd.DataFrame(1e6, index=idx, columns=list(codes))
    return {
        "open": close.copy(), "high": close.copy(), "low": close.copy(),
        "close": close, "volume": vol,
        "amount": (close * vol * 100),
    }


def test_listed_days_counts_valid_bars():
    idx = pd.bdate_range("2024-01-01", periods=5)
    close = pd.DataFrame({"a": [10, np.nan, 10, 10, np.nan]}, index=idx)
    got = uv.listed_days(close)["a"].tolist()
    assert got == [1, 1, 2, 3, 3]


def test_new_stock_excluded():
    mats = _mats(n=60)
    names = {c: "测试" for c in mats["close"].columns}
    cfg = Config()
    cfg.min_list_days = 250
    mask = uv.build_mask(mats, names, cfg)
    assert mask.sum().sum() == 0, "只有 60 天历史，不该有可交易标的"


def test_st_and_star_excluded():
    mats = _mats(n=320)
    names = {"sh600000": "正常股份", "sz300001": "*ST某某",
             "sh688001": "科创某某", "sz000001": "平安银行"}
    cfg = Config()
    cfg.min_list_days = 250
    mask = uv.build_mask(mats, names, cfg)
    last = mask.iloc[-1]
    assert not last["sz300001"], "ST 应被剔除"
    assert not last["sh688001"], "科创板应被剔除"
    assert last["sh600000"] and last["sz000001"]


def test_liquidity_filter():
    mats = _mats(n=320)
    mats["amount"] = mats["amount"] * 0.001  # 成交额极低
    names = {c: "测试" for c in mats["close"].columns}
    cfg = Config()
    cfg.min_list_days = 250
    mask = uv.build_mask(mats, names, cfg)
    assert mask.iloc[-1].sum() == 0, "低流动性股票应被剔除"


def test_suspended_stock_excluded():
    mats = _mats(n=320)
    mats["volume"].iloc[-1] = 0
    names = {c: "测试" for c in mats["close"].columns}
    cfg = Config()
    cfg.min_list_days = 250
    mask = uv.build_mask(mats, names, cfg)
    assert mask.iloc[-1].sum() == 0, "停牌股应被剔除"


def test_index_columns_never_selectable():
    """基准指数混在面板里时，绝不能被当成可买股票。"""
    mats = _mats(n=320, codes=("sh600000", "sh000300", "sz000001"))
    names = {"sh600000": "正常股份", "sh000300": "沪深300", "sz000001": "平安银行"}
    cfg = Config()
    cfg.min_list_days = 250
    mask = uv.build_mask(mats, names, cfg, universe_codes=["sh600000", "sz000001"])
    assert not mask["sh000300"].any(), "指数不应出现在候选池"
    assert mask.iloc[-1]["sh600000"] and mask.iloc[-1]["sz000001"]


def test_universe_codes_none_keeps_backward_compat():
    mats = _mats(n=320)
    names = {c: "测试" for c in mats["close"].columns}
    cfg = Config()
    cfg.min_list_days = 250
    mask = uv.build_mask(mats, names, cfg)
    assert mask.iloc[-1].sum() > 0


def test_limit_pct_by_board():
    lim = uv.limit_pct_by_code(["sh600000", "sz000001", "sz300001", "sh688001"])
    assert lim["sh600000"] == 0.10
    assert lim["sz000001"] == 0.10
    assert lim["sz300001"] == 0.20
    assert lim["sh688001"] == 0.20


def test_limit_up_blocks_buy_and_limit_down_blocks_sell():
    idx = pd.bdate_range("2024-01-01", periods=2)
    cols = ["sh600000", "sz300001"]
    prev = pd.DataFrame([[10.0, 10.0], [10.0, 10.0]], index=idx, columns=cols)
    open_ = pd.DataFrame([[11.0, 10.0], [11.0, 12.0]], index=idx, columns=cols)
    can_buy, can_sell = uv.tradeable_at_open(open_, prev, cols)
    # sh600000 开盘 +10% = 主板涨停 -> 买不进
    assert not can_buy.loc[idx[0], "sh600000"]
    assert can_sell.loc[idx[0], "sh600000"]
    # sz300001 开盘 +20% = 创业板涨停 -> 买不进
    assert not can_buy.loc[idx[1], "sz300001"]


def test_suspended_not_tradeable():
    idx = pd.bdate_range("2024-01-01", periods=2)
    cols = ["sh600000"]
    prev = pd.DataFrame([[10.0], [10.0]], index=idx, columns=cols)
    open_ = pd.DataFrame([[np.nan], [10.0]], index=idx, columns=cols)
    can_buy, can_sell = uv.tradeable_at_open(open_, prev, cols)
    assert not can_buy.loc[idx[0], "sh600000"]
    assert not can_sell.loc[idx[0], "sh600000"]
