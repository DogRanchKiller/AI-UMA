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
import re
from decimal import Decimal
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


def volatility60(mats):
    """历史收盘比例收益的波动率；无因子的旧调用方提供已连续的价格。"""
    close = mats["close"] * mats.get("adj_factor", 1.)
    return close.pct_change(fill_method=None).rolling(60, min_periods=40).std()


_EVENT_FIELDS = ('code', 'dividPreNoticeDate', 'dividAgmPumDate', 'dividPlanAnnounceDate',
                 'dividPlanDate', 'dividRegistDate', 'dividOperateDate', 'dividPayDate',
                 'dividStockMarketDate', 'dividCashPsBeforeTax', 'dividCashPsAfterTax',
                 'dividStocksPs', 'dividCashStock', 'dividReserveToStockPs')


def _event_index(rows):
    """仅保留逐字段完全相同的重复记录；有冲突的键不能授权记账。"""
    if rows is None or rows.empty:
        return {}
    if set(rows.columns) != set(_EVENT_FIELDS) or not rows.columns.is_unique:
        raise ValueError('公司行为原始字段不完整')
    grouped = {}
    for row in rows.drop_duplicates().to_dict('records'):
        if (not all(isinstance(v, str) for v in row.values()) or not re.fullmatch(r'(sh|sz)\.\d{6}', row['code'])
                or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', row['dividOperateDate'])):
            raise ValueError('公司行为必须保留 SDK 原始串、代码和生效日')
        code = row['code'].replace('.', '')
        key = (pd.Timestamp(row['dividOperateDate']), code)
        if key in grouped:
            raise ValueError(f'{key[0].date()} {code} 公司行为重复冲突，拒绝估值')
        grouped[key] = row
    return grouped


def _terms(row, code, d, reg, previous_close, f0, f1):
    """由独立公告条款给出税前现金和新增股份；乘数仅用于对账。"""
    try:
        if not row or pd.Timestamp(row['dividRegistDate']) != reg or not pd.Timestamp(row['dividPlanDate']) <= reg:
            raise ValueError('缺少独立记录、重复冲突或登记/公告日期不符')
        # 只接受明确的每十股方案；不接受配股、不明附加权益或数值歧义。
        plan = re.sub(r'[（(]含税(?:[，,]扣税后[\d.]+(?:或[\d.]+)*元?)?[）)]$', '', row['dividCashStock'])
        parts = re.findall(r'(送|转增?|派)(\d+(?:\.\d+)?)(股|元)?', plan)
        if not parts or plan != '10' + ''.join(''.join(part) for part in parts):
            raise ValueError('未支持的方案')
        terms = {}
        for kind, amount, unit in parts:
            kind = kind[0]
            if kind in terms or (unit and unit != ('元' if kind == '派' else '股')):
                raise ValueError('方案条款重复或单位无效')
            terms[kind] = Decimal(amount) / 10
        for kind, field in [('派', 'dividCashPsBeforeTax'), ('送', 'dividStocksPs'), ('转', 'dividReserveToStockPs')]:
            v = row[field]
            if pd.isna(v) or v == '':
                if kind in terms:
                    raise ValueError(f'{field} 缺少独立数值')
            elif Decimal(str(v)) != terms.get(kind, 0):
                raise ValueError('每股字段与公告方案不一致')
        bonus, reserve, cash = (terms.get(k, Decimal('0')) for k in ('送', '转', '派'))
        if not (cash or bonus or reserve):
            raise ValueError('无可核对权益')
        pay = pd.Timestamp(row['dividPayDate']) if cash else None
        listing = pd.Timestamp(row['dividStockMarketDate']) if bonus or reserve else None
        if (cash and not pay >= d) or ((bonus or reserve) and not listing >= d):
            raise ValueError('缺少发放或上市日')
        reference = (previous_close - float(cash)) / (1 + float(bonus + reserve))
        if not np.isfinite(reference) or reference <= 0 or abs(reference - previous_close * f0 / f1) > .005001:
            raise ValueError('条款与累计因子折算参考价不符')
        return cash, bonus + reserve, pay, listing
    except (ValueError, ArithmeticError, TypeError) as exc:
        raise ValueError(f'{d.date()} 持仓跨除权 {code}：{exc}，拒绝估值') from exc


class _Entitlements:
    """不可用的应收款与待上市股份分开估值；从不把乘数本身兑换成持仓或现金。"""
    def __init__(self, rows):
        self.by_date = {}
        for (d, c), record in _event_index(rows).items():
            self.by_date.setdefault(d, {})[c] = record
        self.cash_due = []
        self.stock_due = []
        self.applied = 0

    def stock_positions(self):
        qty = {}
        for _, c, n in self.stock_due:
            qty[c] = qty.get(c, 0) + n
        return qty

    def at_open(self, d, prior_date, mats, changed, shares):
        records = self.by_date.get(d, {})
        locked = self.stock_positions()
        for c in shares.keys() | locked.keys():
            has_jump = bool(changed.at[d, c]) if changed is not None else False
            if not has_jump and c not in records:
                continue
            if not has_jump or c in locked or prior_date is None:
                raise ValueError(f'{d.date()} 持仓跨除权 {c}：条款无可匹配因子或有未上市的重叠权益，拒绝估值')
            row = records.get(c)
            f = mats['adj_factor']
            cash_ps, shares_ps, pay, listing = _terms(row, c, d, prior_date, mats['close'].at[prior_date, c],
                                                       f.at[prior_date, c], f.at[d, c])
            qty = Decimal(str(shares[c]))  # 今日开盘前的实际股数，即登记日收市股数。
            bonus = qty * shares_ps
            if bonus != bonus.to_integral_value():
                raise ValueError(f'{d.date()} 持仓跨除权 {c}：零碎股分配规则未知，拒绝估值')
            if bonus:
                self.stock_due.append((listing, c, int(bonus)))
            if cash_ps:
                self.cash_due.append((pay, float(qty * cash_ps)))
            self.applied += 1
        # 公告的股票上市日才能卖出；除权日新买者不享有此前已登记权益。
        for listing, c, n in self.stock_due:
            if listing <= d:
                shares[c] = shares.get(c, 0) + n
        self.stock_due = [x for x in self.stock_due if x[0] > d]

    def settle_cash(self, d):
        """发放日收市才入可用现金；未假设当日开盘已到账。"""
        paid = sum(value for date, value in self.cash_due if date <= d)
        self.cash_due = [x for x in self.cash_due if x[0] > d]
        return paid

    def restricted(self, d, px):
        value = sum(amount for _, amount in self.cash_due)
        for c, qty in self.stock_positions().items():
            valid = px[c].loc[:d].dropna()
            if valid.empty:
                raise ValueError(f'{d.date()} 待上市权益 {c} 无可用报价，拒绝估值')
            value += qty * valid.iloc[-1]
        return value


def run(
    scores: pd.DataFrame,
    mats: dict[str, pd.DataFrame],
    mask: pd.DataFrame,
    can_buy: pd.DataFrame,
    can_sell: pd.DataFrame,
    benchmark: pd.Series,
    cfg: Config,
    regime_ok: pd.Series | None = None,
    industry: dict[str, str] | None = None,
    *, corporate_actions: pd.DataFrame | None = None,
) -> BacktestResult:
    """执行回测。scores 为宽表 date × code，仅包含调仓日。

    mask 和 regime_ok 按信号日收盘取值，调用方无需平移。regime_ok 为 None 表示不设闸门；
    若已提供闸门，信号日取值缺失视为关闭。开盘价、can_buy 和 can_sell 按执行日取值。
    industry 为代码到行业名的映射，用于限制单一行业入选数量。
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
    vol60 = volatility60(mats)
    # 停牌空行不产生事件；复牌对比上次可见因子，不能漏掉停牌期间的除权。
    actions = None
    if "adj_factor" in mats:
        factor = mats["adj_factor"]
        prior = factor.ffill().shift(1)
        actions = factor.notna() & prior.notna() & factor.ne(prior)

    peak = 0.0
    brake_active = False

    ledger = _Entitlements(corporate_actions)
    for i, d in enumerate(dates):
        ledger.at_open(d, dates[i - 1] if i else None, mats, actions, shares)
        sig = exec_map.get(d)
        if sig is not None:
            # regime_ok 可以是布尔（二值闸门）或 [0,1] 浮点（连续仓位系数）
            exposure = 1.0
            if regime_ok is not None:
                v = regime_ok.get(sig, 0.0)
                exposure = float(v) if pd.notna(v) else 0.0
                exposure = min(max(exposure, 0.0), 1.0)
            risk_on = exposure > 1e-3

            # 回撤刹车：以执行时点已知的净值回撤决定仓位乘数，不含未来信息
            gross_mult = exposure
            if cfg.drawdown_brake > 0 and peak > 0:
                last_eq = equity_vals[-1] if equity_vals else cfg.init_cash
                dd = (peak - last_eq) / peak
                if not brake_active and dd >= cfg.drawdown_brake:
                    brake_active = True
                elif brake_active and dd <= cfg.drawdown_recover:
                    brake_active = False
                if brake_active:
                    gross_mult = exposure * cfg.drawdown_brake_scale

            cash = _execute(
                d, sig, scores, open_, close, mask, can_buy, can_sell,
                cash, shares, cfg, trades, blocked, weight_rows, risk_on,
                vol_row=vol60.loc[sig],
                gross_mult=gross_mult,
                industry=industry, restricted_open=ledger.restricted(d, open_),
                restricted_close=ledger.restricted(d, close), locked=ledger.stock_positions(),
            )
        # 收盘估值
        cl = close.loc[d]
        cash += ledger.settle_cash(d)
        val = cash + ledger.restricted(d, close)
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
        "公司行为口径": "独立公告税前权益（不含红利个税）；发放日收市入现金、股份上市日起可卖",
        "已登记持仓权益事件数": ledger.applied,
        "期末应收税前现金": sum(x[1] for x in ledger.cash_due),
        "期末待上市股份": ledger.stock_positions(),
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
    industry: dict[str, str] | None = None,
    restricted_open: float = 0., restricted_close: float = 0., locked: dict | None = None,
) -> float:
    """在 d 日开盘执行 sig 日的信号，返回执行后的现金。

    vol_row   : 截至信号日收盘的 60 日波动率截面，供逆波动加权使用
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

    # 候选池含收盘价、全天成交量等字段，只能使用信号日截面。
    if sig not in mask.index:
        return cash
    row = scores.loc[sig].dropna()
    avail = mask.loc[sig].fillna(False).astype(bool)
    row = row[row.index.isin(avail.index[avail])]
    if row.empty:
        return cash

    n_hold = max(1, min(cfg.top_n, len(row)))
    ranked = row.sort_values(ascending=False)

    # 选股：按分数从高到低，可选限制单一行业的入选数量
    if cfg.max_per_industry > 0 and industry:
        top = []
        per_ind: dict[str, int] = {}
        for c in ranked.index:
            ind = industry.get(c, "")
            if ind and per_ind.get(ind, 0) >= cfg.max_per_industry:
                continue
            top.append(c)
            if ind:
                per_ind[ind] = per_ind.get(ind, 0) + 1
            if len(top) >= n_hold:
                break
        # 行业分类缺失或过度稀薄时，宁可退回不限制也不要选不满
        if len(top) < n_hold:
            chosen = set(top)
            for c in ranked.index:
                if c not in chosen:
                    top.append(c)
                    if len(top) >= n_hold:
                        break
    else:
        top = ranked.head(n_hold).index.tolist()

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
    equity_open = cash + pos_value + restricted_open
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
        cur_val = (shares.get(c, 0.0) + (locked or {}).get(c, 0)) * float(px)
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
    total = cash + restricted_close
    for c, sh in shares.items():
        p = cl.get(c, np.nan)
        if pd.notna(p) and p > 0:
            total += sh * float(p)
    if total > 0:
        weight_rows[d] = {
            c: ((shares.get(c, 0) + (locked or {}).get(c, 0)) * float(cl[c])) / total
            for c in shares.keys() | (locked or {}).keys()
            if pd.notna(cl.get(c, np.nan))
        }
    return cash
