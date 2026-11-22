"""可交易股票池构建。

剔除规则（全部只用当日及以前的信息）：
  * 停牌 / 成交量为 0
  * 上市未满 min_list_days 个交易日
  * 股价低于 min_price
  * 20 日均成交额低于 min_amount_ma20（流动性不足，无法建仓）
  * 当前名称含 ST / *ST / 退
  * （可配置）科创板 688 段
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import Config

_ST_NAME_KEYS = ("ST", "*ST", "退", "S*ST", "SST")


def listed_days(close: pd.DataFrame) -> pd.DataFrame:
    """每只股票截至当日已有效交易的交易日数。"""
    valid = close.notna() & (close > 0)
    return valid.cumsum().astype("int32")


def build_mask(
    mats: dict[str, pd.DataFrame],
    names: dict[str, str],
    cfg: Config,
    universe_codes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """返回 date × code 的布尔表，True 表示当日可作为候选标的。

    universe_codes 用于把基准指数等非个股列从候选中彻底排除——面板里为了算
    beta/alpha 会同时载入指数，若不显式剔除，指数会被当成股票买入。
    """
    close, volume, amount = mats["close"], mats["volume"], mats["amount"]

    ok = close.notna() & (close > 0)
    ok &= volume.fillna(0) > 0

    # 上市时长
    ok &= listed_days(close) >= cfg.min_list_days

    # 价格下限
    ok &= close >= cfg.min_price

    # 流动性：20 日均成交额
    amt_ma20 = amount.rolling(20, min_periods=10).mean()
    ok &= amt_ma20 >= cfg.min_amount_ma20

    # 只允许真实个股（剔除指数、ETF 等)
    if universe_codes is not None:
        allowed = set(universe_codes)
        not_stock = [c for c in close.columns if c not in allowed]
        if not_stock:
            ok[not_stock] = False

    # ST / 退市风险（按最新证券名称静态剔除，见 README 局限性说明）
    if cfg.exclude_st:
        bad = [c for c in close.columns
               if any(k in str(names.get(c, "")) for k in _ST_NAME_KEYS)]
        if bad:
            ok[bad] = False

    # 科创板
    if cfg.exclude_star:
        star = [c for c in close.columns if c.startswith("sh688")]
        if star:
            ok[star] = False

    return ok.fillna(False)


def limit_pct_by_code(codes: list[str]) -> pd.Series:
    """按板块给出涨跌幅限制比例。创业板 300/301 与科创板 688 为 20%，其余 10%。"""
    def one(c: str) -> float:
        c = str(c)
        if c.startswith(("sz300", "sz301", "sh688")):
            return 0.20
        return 0.10
    return pd.Series({c: one(c) for c in codes}, dtype="float64")


def panel_tradeable_at_open(mats):
    """同一比例单位估算开盘幅度（非交易所精确参考价）；无昨收或无量不撮合。"""
    factor = mats['adj_factor']
    op = mats['open'] * factor
    prev = (mats['close'] * factor).shift(1)
    buy, sell = tradeable_at_open(op, prev, list(op.columns))
    valid = prev.gt(0) & mats['volume'].gt(0)
    return buy & valid, sell & valid


def tradeable_at_open(
    open_: pd.DataFrame, prev_close: pd.DataFrame, codes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """判断开盘时能否买入 / 卖出。

    返回 (can_buy, can_sell)：
      * 开盘价达到涨停价（含容差）则买不进
      * 开盘价达到跌停价（含容差）则卖不出
      * 停牌（开盘价缺失或 0）双向都不可交易
    """
    lim = limit_pct_by_code(codes)
    thresh = pd.DataFrame(
        np.repeat(lim.values.reshape(1, -1), len(open_), axis=0),
        index=open_.index, columns=open_.columns,
    )
    ret = open_ / prev_close.replace(0.0, np.nan) - 1.0
    alive = open_.notna() & (open_ > 0)
    at_limit_up = ret >= (thresh * 0.98)
    at_limit_down = ret <= -(thresh * 0.98)
    can_buy = alive & ~at_limit_up
    can_sell = alive & ~at_limit_down
    return can_buy.fillna(False), can_sell.fillna(False)
