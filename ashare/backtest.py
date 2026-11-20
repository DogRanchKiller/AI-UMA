"""T+1 事件式回测引擎。

执行口径（与 labels.py 严格一致）：
  信号在 t 日收盘后产生 → t+1 日开盘价成交 → 持有到下一调仓日开盘。

约束：
  * 开盘涨停买不进、开盘跌停卖不出、停牌不可交易
  * 佣金（含 5 元最低）、卖出印花税、双边滑点
  * 100 股整手
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from . import metrics

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    stats: dict
    diagnostics: dict = field(default_factory=dict)
    holdings_by_date: dict = field(default_factory=dict)

    def period_returns(self, freq: str = "ME") -> pd.Series:
        return (1 + self.equity.pct_change().fillna(0)).resample(freq).prod() - 1


def _rebalance_dates(signal_dates: list[pd.Timestamp], every: int) -> list[pd.Timestamp]:
    if not signal_dates:
        return []
    return [d for i, d in enumerate(signal_dates) if i % every == 0]


def run(
    scores: pd.DataFrame,
    mats: dict[str, pd.DataFrame],
    mask: pd.DataFrame,
    can_buy: pd.DataFrame,
    can_sell: pd.DataFrame,
    benchmark: pd.Series,
    cfg: Config,
    regime_ok: pd.Series | None = None,
) -> BacktestResult:
    """执行回测。scores 为宽表 date × code，仅包含调仓日。

    regime_ok 为市场状态闸门：该日给出 False 时清仓离场，不建新仓。
    """
    open_, close = mats["open"], mats["close"]
    dates = close.index
    score_dates = [d for d in scores.index if d in dates]
    if not score_dates:
        raise ValueError("没有可用的信号日期")

    # 信号日 -> 执行日（下一个交易日）
    date_pos = {d: i for i, d in enumerate(dates)}
    exec_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for d in score_dates:
        i = date_pos[d]
        if i + 1 < len(dates):
            exec_map[dates[i + 1]] = d

    cash = float(cfg.init_cash)
    shares: dict[str, float] = {}
    equity_vals: list[float] = []
    weight_rows: dict[pd.Timestamp, dict[str, float]] = {}
    holdings_log: dict = {}
    trades: list[dict] = []
    blocked = {"buy_limit_up": 0, "sell_limit_down": 0,
               "orphan_suspended": 0, "brake_off": 0}

    # 逆波动加权需要的 60 日波动率截面（只用历史，无前视）
    vol60 = (close / close.shift(1) - 1.0).rolling(60, min_periods=40).std()

    peak = 0.0
    brake_active = False

    for d in dates:
        sig = exec_map.get(d)
        if sig is not None:
            risk_on = True
            if regime_ok is not None:
                risk_on = bool(regime_ok.get(d, True))

            # 回撤刹车：以执行时点已知的净值回撤决定仓位乘数，不含未来信息
            gross_mult = 1.0
            if cfg.drawdown_brake > 0 and peak > 0:
                last_eq = equity_vals[-1] if equity_vals else cfg.init_cash
                dd = (peak - last_eq) / peak
                if not brake_active and dd >= cfg.drawdown_brake:
                    brake_active = True
                elif brake_active and dd <= cfg.drawdown_recover:
                    brake_active = False
                if brake_active:
                    gross_mult = cfg.drawdown_brake_scale

            cash = _execute(
                d, sig, scores, open_, close, mask, can_buy, can_sell,
                cash, shares, cfg, trades, blocked, weight_rows, risk_on,
                vol_row=vol60.loc[d] if d in vol60.index else None,
                gross_mult=gross_mult,
            )
        # 收盘估值
        cl = close.loc[d]
        val = cash
        for c, sh in shares.items():
            p = cl.get(c, np.nan)
            if pd.notna(p) and p > 0:
                val += sh * float(p)
            else:
                # 停牌用最近一次有效价格估值
                last = close[c].loc[:d].dropna()
                if len(last):
                    val += sh * float(last.iloc[-1])
                else:
                    val += 0.0
        equity_vals.append(val)
        if val > peak:
            peak = val
        holdings_log[d] = {c: sh for c, sh in shares.items() if sh > 0}

    raw_equity = pd.Series(equity_vals, index=dates, name="equity").astype("float64")
    cash_ratio = cash / raw_equity.iloc[-1] if raw_equity.iloc[-1] else float("nan")
    equity = raw_equity / raw_equity.iloc[0]

    bench = benchmark.reindex(dates).ffill()
    bench = bench / bench.dropna().iloc[0] if bench.notna().any() else bench

    weights = pd.DataFrame(weight_rows).T.fillna(0.0)
    trade_df = pd.DataFrame(trades)

    stats = metrics.summarize(equity, bench)
    if not weights.empty:
        stats.update(metrics.turnover_stats(weights, cfg.rebalance_every))

    diag = {
        "执行次数": len(exec_map),
        "成交笔数": int(len(trade_df)) if not trade_df.empty else 0,
        "涨停未买入": blocked["buy_limit_up"],
        "跌停未卖出": blocked["sell_limit_down"],
        "闸门离场次数": blocked.get("regime_off", 0),
        "刹车清仓次数": blocked.get("brake_off", 0),
        "期末持仓数": len([c for c, s in shares.items() if s > 0]),
        "期末现金占比": float(cash_ratio),
    }
    return BacktestResult(
        equity=equity, benchmark=bench, weights=weights, trades=trade_df,
        stats=stats, diagnostics=diag, holdings_by_date=holdings_log,
    )


def _flag(df: pd.DataFrame, d, c) -> bool:
    """安全读取某日某股的可交易标记，缺失视为不可交易。"""
    if c not in df.columns or d not in df.index:
        return False
    v = df.at[d, c]
    return bool(v) if pd.notna(v) else False


def target_weights(
    row: pd.Series,
    target: list[str],
    vol_row: pd.Series | None,
    cfg: Config,
) -> dict[str, float]:
    """按配置的权重方案给出目标权重，并把总和归一到 1、单票不超过上限。"""
    n = len(target)
    if n == 0:
        return {}
    ordered = [c for c in row.sort_values(ascending=False).index if c in set(target)]
    if not ordered:
        return {}

    if cfg.weighting == "inv_vol" and vol_row is not None:
        raw = {}
        for c in ordered:
            v = vol_row.get(c, np.nan)
            # 波动率缺失或异常时退回等权，避免整份组合失真
            raw[c] = (1.0 / float(v)) if (pd.notna(v) and v and v > 1e-6) else np.nan
        if all(pd.isna(x) for x in raw.values()):
            raw = {c: 1.0 for c in ordered}
    elif cfg.weighting == "rank_decay":
        k = cfg.rank_decay
        raw = {c: 1.0 / ((i + 1) ** k) for i, c in enumerate(ordered)}
    else:
        raw = {c: 1.0 for c in ordered}

    s = np.nansum(list(raw.values()))
    if not np.isfinite(s) or s <= 0:
        raw = {c: 1.0 for c in ordered}
        s = float(len(ordered))
    w = {c: (float(v) / s if pd.notna(v) else 0.0) for c, v in raw.items()}

    # 单票上限后重新归一，迭代几次即可收敛
    for _ in range(4):
        over = {c: v for c, v in w.items() if v > cfg.max_weight}
        if not over:
            break
        for c in over:
            w[c] = cfg.max_weight
        rest = [c for c in w if c not in over]
        free = 1.0 - cfg.max_weight * len(over)
        tot = sum(w[c] for c in rest)
        if tot <= 0 or free <= 0:
            break
        for c in rest:
            w[c] = w[c] / tot * free
    return w


def _execute(
    d, sig, scores, open_, close, mask, can_buy, can_sell,
    cash, shares, cfg, trades, blocked, weight_rows, risk_on: bool = True,
    vol_row: pd.Series | None = None, gross_mult: float = 1.0,
) -> float:
    """在 d 日开盘执行 sig 日的信号，返回执行后的现金。

    vol_row   : 当日的 60 日波动率截面，供逆波动加权使用
    gross_mult: 仓位乘数（回撤刹车触发时小于 1 或为 0）
    """
    op = open_.loc[d]
    if not risk_on or gross_mult <= 0:
        # 闸门关闭或刹车归零：清仓离场，本期不建新仓
        for c in list(shares):
            px = op.get(c, np.nan)
            if pd.isna(px) or px <= 0 or not _flag(can_sell, d, c):
                blocked["sell_limit_down"] += 1
                continue
            sh = shares.pop(c)
            exec_px = float(px) * (1 - cfg.slippage)
            gross = sh * exec_px
            fee = max(cfg.min_commission, gross * cfg.commission) + gross * cfg.stamp_tax
            cash += gross - fee
            trades.append({"date": d, "code": c, "side": "sell", "shares": sh,
                           "price": exec_px, "fee": fee})
        if not risk_on:
            blocked["regime_off"] = blocked.get("regime_off", 0) + 1
        else:
            blocked["brake_off"] = blocked.get("brake_off", 0) + 1
        return cash

    row = scores.loc[sig].dropna()
    if row.empty:
        return cash

    # 只在可交易池内选股
    if d in mask.index:
        avail = mask.loc[d]
        row = row[row.index.isin(avail.index[avail.fillna(False).astype(bool)])]
    if row.empty:
        return cash

    n_hold = max(1, min(cfg.top_n, len(row)))
    top = row.sort_values(ascending=False).head(n_hold).index.tolist()

    # 换手缓冲：已持有的股票只要还排在前 top_n*1.5 内就继续持有，降低换手
    rank_all = row.rank(ascending=False)
    buffer_rank = int(cfg.top_n * 1.5)
    keep = [c for c in shares if c in rank_all.index and rank_all[c] <= buffer_rank]
    target = list(dict.fromkeys(top + keep))
    target_set = set(target)

    pos_value = 0.0
    for c, sh in shares.items():
        p = op.get(c, np.nan)
        pos_value += sh * float(p) if pd.notna(p) and p > 0 else 0.0
    equity_open = cash + pos_value
    if equity_open <= 0:
        return cash

    wmap = target_weights(row, target, vol_row, cfg)

    # ---- 1. 卖出不在目标内的持仓（跌停/停牌则被迫继续持有） ----
    for c in list(shares):
        if c in target_set:
            continue
        px = op.get(c, np.nan)
        if pd.isna(px) or px <= 0 or not _flag(can_sell, d, c):
            blocked["sell_limit_down"] += 1
            continue
        sh = shares.pop(c)
        exec_px = float(px) * (1 - cfg.slippage)
        gross = sh * exec_px
        fee = max(cfg.min_commission, gross * cfg.commission) + gross * cfg.stamp_tax
        cash += gross - fee
        trades.append({"date": d, "code": c, "side": "sell", "shares": sh,
                       "price": exec_px, "fee": fee})

    # ---- 2. 买入目标内权重不足的股票，按分数从高到低 ----
    for c in row.sort_values(ascending=False).index:
        if c not in target_set:
            continue
        px = op.get(c, np.nan)
        if pd.isna(px) or px <= 0:
            continue
        tgt_val = equity_open * wmap.get(c, 0.0) * gross_mult
        if tgt_val <= 0:
            continue
        cur_val = shares.get(c, 0.0) * float(px)
        if cur_val >= tgt_val * 0.7:
            continue  # 已接近目标权重
        budget = min(tgt_val - cur_val, cash)
        if budget <= 0:
            continue
        if not _flag(can_buy, d, c):
            blocked["buy_limit_up"] += 1
            continue
        exec_px = float(px) * (1 + cfg.slippage)
        # 预留最低佣金，避免最后一手导致现金为负
        usable = cash - cfg.min_commission
        qty = int(min(budget, usable) / exec_px // 100) * 100
        if qty < 100:
            continue
        gross = qty * exec_px
        fee = max(cfg.min_commission, gross * cfg.commission)
        if gross + fee > cash:
            continue
        cash -= gross + fee
        shares[c] = shares.get(c, 0.0) + qty
        trades.append({"date": d, "code": c, "side": "buy", "shares": qty,
                       "price": exec_px, "fee": fee})

    # ---- 3. 记录收盘权重（用收盘价估值，便于与净值口径一致） ----
    cl = close.loc[d]
    total = cash
    for c, sh in shares.items():
        p = cl.get(c, np.nan)
        if pd.notna(p) and p > 0:
            total += sh * float(p)
    if total > 0:
        weight_rows[d] = {
            c: (sh * float(cl.get(c, 0) or 0)) / total
            for c, sh in shares.items()
            if sh > 0 and pd.notna(cl.get(c, np.nan))
        }
    return cash
