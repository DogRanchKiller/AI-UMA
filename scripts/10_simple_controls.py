"""固定 43 因子 vs 20 日反转 vs 沪深300 ETF；只回放，不训练。见实验目录的 protocol.md。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ashare import backtest, factors, metrics, portfolio, universe
from ashare.config import Config, OUT_DIR

EXPERIMENT = OUT_DIR / "simple_controls_20261121"
FROZEN = {"preds.csv": "6e559ad94ee75fe6e671e74a2b4924edf30ae211a2309263f259422914892072",
          "config_used.json": "d8e7dc43a1a8d1092ed52bad33bac4acf5bba7942bfedd57295f7c17ad0238be"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reversal_scores(close, scores):
    """原始收益取负，不做缩尾（避免把尾部候选变成并列分数），不填停牌缺口。"""
    ret = close.loc[scores.index] / close.shift(20).loc[scores.index] - 1
    return (-ret).where(scores.notna() & np.isfinite(ret))


def annual_returns(equity):
    # 第一行是首笔交易前的本金；先算日收益再按年复合，不丢跨年首日收益。
    returns = equity.pct_change(fill_method=None).iloc[1:]
    return (1 + returns).groupby(returns.index.year).prod() - 1


def etf_ledger(bars, events, decisions, cfg):
    """510300 未复权现金账本。decisions 按执行日索引，只来自上个交易日的信号。"""
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ValueError('ETF 交易日历重复或乱序')
    if not decisions.index.isin(bars.index[1:]).all() or decisions.index.has_duplicates or decisions.isna().any():
        raise ValueError('ETF 调仓日不在交易窗口或存在空闸门')
    if (not np.isfinite(bars[['open', 'close', 'volume']]).all().all()
            or bars[['open', 'close', 'volume']].le(0).any().any()):
        raise ValueError('数据验收失败：ETF 不是完整、有效的成交日线；不代填停牌日')
    events = events[['record_date', 'ex_date', 'pay_date', 'cash_per_unit']]
    # 该基金的场内登记日在除息之前；不同市场的同日登记事件不能套用这一规则。
    if (events.isna().any().any() or not ((events.record_date < events.ex_date)
            & (events.ex_date <= events.pay_date) & (events.cash_per_unit > 0)).all()):
        raise ValueError('不支持或缺失的 ETF 公司行动日期/金额')
    relevant = events.loc[(events.record_date >= bars.index[0]) & (events.record_date <= bars.index[-1])]
    for column in ('record_date', 'ex_date', 'pay_date'):
        within = relevant[column].between(bars.index[0], bars.index[-1])
        if not relevant.loc[within, column].isin(bars.index).all():
            raise ValueError(f'分红 {column} 未匹配交易日')
    ex_cash = events.groupby('ex_date').cash_per_unit.sum().reindex(bars.index, fill_value=0.0)
    ref_price = bars.close.shift(1) - ex_cash
    can_buy, can_sell = universe.tradeable_at_open(bars.open.to_frame('sh510300'),
                                                  ref_price.to_frame('sh510300'), ['sh510300'])
    due_by_ex, due_by_pay = {}, {}
    shares, cash, owing = 0, float(cfg.init_cash), 0.0
    rows, trades = [], []
    for day, bar in bars.iterrows():
        owing += due_by_ex.get(day, 0.0)
        if day in decisions.index:
            risk_on = bool(decisions.at[day])
            px = bar.open * (1 + cfg.slippage if risk_on else 1 - cfg.slippage)
            qty = 0
            if risk_on and can_buy.at[day, 'sh510300']:
                budget = max(0, min(cash - cfg.min_commission, cash / (1 + cfg.commission)))
                qty = int(budget / (px * 100)) * 100
            elif not risk_on and can_sell.at[day, 'sh510300']:
                qty = -shares
            if qty:
                fee = max(cfg.min_commission, abs(qty) * px * cfg.commission)
                cash -= qty * px + fee  # ETF 不缴股票卖出印花税
                shares += qty
                trades.append((day, 'sh510300', 'buy' if qty > 0 else 'sell', abs(qty), px, fee))
        for event in relevant.loc[relevant.record_date == day].itertuples():
            earned = shares * event.cash_per_unit
            due_by_ex[event.ex_date] = due_by_ex.get(event.ex_date, 0.0) + earned
            due_by_pay[event.pay_date] = due_by_pay.get(event.pay_date, 0.0) + earned
        dividend = due_by_pay.get(day, 0.0)  # 保守：派息日收盘到账，当日开盘不可花用
        cash += dividend
        owing -= dividend
        if cash < -1e-6 or owing < -1e-6 or shares < 0:
            raise AssertionError('ETF 现金、应收或持仓为负')
        rows.append((cash, shares, owing, dividend, (cash + shares * bar.close + owing) / cfg.init_cash))
    ledger = pd.DataFrame(rows, index=bars.index, columns=['cash', 'shares', 'receivable', 'dividend', 'equity'])
    return ledger, pd.DataFrame(trades, columns=['date', 'code', 'side', 'shares', 'price', 'fee'])


def run(out):
    if out.exists():
        raise FileExistsError(f'结果目录必须是新目录，以免覆盖已有实验: {out}')
    for name, checksum in FROZEN.items():
        if digest(OUT_DIR / name) != checksum:
            raise ValueError(f'{name} 与冻结样本不一致')
    cfg = Config.load(OUT_DIR / 'config_used.json')
    cfg.use_ext_factors = False
    load = run_path(str(ROOT / 'scripts' / '02_build.py'))['load_inputs']
    _, names, panel = load(cfg)
    mats = factors.clean_matrices(factors.build_matrices(panel))
    del panel
    close = mats['close']
    bench = close[cfg.benchmark].dropna()
    mask = universe.build_mask(mats, names, cfg, universe_codes=list(names))
    cb, cs = universe.tradeable_at_open(mats['open'], close.shift(1), list(close))
    pred = pd.read_csv(OUT_DIR / 'preds.csv', parse_dates=['date'], dtype={'score': 'float32'})
    scores = pred.pivot(index='date', columns='code', values='score').reindex(columns=close.columns)
    if int(scores.notna().sum().sum()) != len(pred) or (scores.notna() & ~mask.loc[scores.index]).any().any():
        raise ValueError('冻结预测与原始股票池不一致')
    model, gate = portfolio.build_target_scores(scores, mats, mask, bench, cfg)
    rev, _ = portfolio.build_target_scores(reversal_scores(close, scores), mats, mask, bench, cfg)
    pos = close.index.get_indexer(scores.index)
    assert (pos >= 0).all() and (pos[:-1] + 1 < len(close)).all() and pos[-1] == len(close) - 1
    signals, exec_days = scores.index[:-1], close.index[pos[:-1] + 1]
    allowed = gate.loc[signals]
    assert not allowed.isna().any() and allowed.isin([True, False]).all()
    decisions = pd.Series(allowed.to_numpy(), index=exec_days)
    calendar = close.index[pos[0]:]
    missing = (scores.notna() & rev.isna()).sum(axis=1)
    raw_file = EXPERIMENT / 'inputs' / 'raw.csv'
    div_file = EXPERIMENT / 'inputs' / 'dividends.csv'
    bars = pd.read_csv(raw_file, parse_dates=['date']).set_index('date')
    if not calendar.isin(bars.index).all():
        raise ValueError('ETF 原始行情覆盖不足，不能用指数或前向填充代替')
    bars = bars.loc[calendar]
    events = pd.read_csv(div_file, parse_dates=['record_date', 'ex_date', 'pay_date'])
    ref = pd.read_csv(OUT_DIR / 'ext_ab_20261121' / 'formal_gate_replay.csv').set_index(['model', 'gate'])
    out.mkdir(parents=True)
    schedule = pd.DataFrame({'signal': signals, 'execution': exec_days, 'allowed': allowed.to_numpy(),
                             'scored': scores.notna().sum(axis=1).loc[signals].to_numpy(),
                             'no_ret20': missing.loc[signals].to_numpy()})
    schedule.to_csv(out / 'schedule.csv', index=False)
    navs, summary, ledger_csv, all_trades = {}, [], {}, []
    for mult in (1, 2):
        costs = replace(cfg, slippage=cfg.slippage * mult)
        ranking = {'model43': model, 'reversal20': rev}
        if missing.loc[signals].any():
            ranking['model43_matched'] = model.where(rev.notna())
        for arm, score in ranking.items():
            res = backtest.run(score, mats, mask, cb, cs, bench, costs, regime_ok=gate)
            assert res.diagnostics['执行次数'] == len(exec_days)
            if mult == 1 and arm == 'model43':
                np.testing.assert_allclose([res.stats['年化收益'], res.stats['最大回撤']],
                                           ref.loc[('base', True), ['annual', 'drawdown']].to_numpy(float),
                                           rtol=0, atol=1e-10)
            navs[mult, arm] = res.equity.loc[calendar]
            all_trades.append(res.trades.assign(slippage_mult=mult, arm=arm))
        ledger, etrades = etf_ledger(bars, events, decisions, costs)
        ledger_csv[mult] = ledger
        navs[mult, 'etf510300'] = ledger.equity
        all_trades.append(etrades.assign(slippage_mult=mult, arm='etf510300'))
    trades = pd.concat(all_trades, ignore_index=True)
    years = []
    for (mult, arm), eq in navs.items():
        assert eq.index.equals(calendar) and np.isclose(eq.iloc[0], 1.0) and (eq > 0).all()
        yr = annual_returns(eq)
        np.testing.assert_allclose((1 + yr).prod(), eq.iloc[-1], rtol=1e-11)
        n = len(eq) - 1
        activity = trades[(trades.arm == arm) & (trades.slippage_mult == mult)]
        summary.append((mult, arm, eq.iloc[-1] - 1, eq.iloc[-1] ** (metrics.TRADING_DAYS / n) - 1,
                        metrics.max_drawdown(eq), n, len(activity), activity.fee.sum()))
        years.extend((mult, arm, year, val) for year, val in yr.items())
    s = pd.DataFrame(summary, columns=['slippage_mult', 'arm', 'total_return', 'annual_return', 'max_drawdown',
                                      'trading_days', 'trades', 'cash_fees'])
    y = pd.DataFrame(years, columns=['slippage_mult', 'arm', 'year', 'return']).pivot(
        index=['slippage_mult', 'year'], columns='arm', values='return')
    y['model_minus_reversal'] = y.model43 - y.reversal20
    y['model_minus_etf'] = y.model43 - y.etf510300
    s.to_csv(out / 'summary.csv', index=False)
    y.to_csv(out / 'yearly.csv')
    pd.DataFrame({f'{arm}_x{mult}': eq for (mult, arm), eq in navs.items()}).to_csv(out / 'equity.csv', index_label='date')
    trades.to_csv(out / 'trades.csv', index=False)
    pd.concat(ledger_csv, names=['slippage_mult', 'date']).to_csv(out / 'etf_cash.csv')
    manifest = {'predictions': len(pred), 'prediction_dates': len(scores), 'first_signal': str(signals[0].date()),
                'first_execution': str(exec_days[0].date()), 'last_mark': str(calendar[-1].date()),
                'gate_on': int(decisions.sum()), 'executions': len(decisions), 'tradable_days': len(calendar)-1,
                'missing_ret20_executable': int(missing.loc[signals].sum()), 'sha256': dict(FROZEN)}
    for path in (raw_file, div_file, ROOT/'data/cache/master_list.csv', ROOT/'data/raw/sh000300.csv',
                 ROOT/'ashare/backtest.py', Path(__file__).resolve()):
        manifest['sha256'][path.name] = digest(path)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(s.to_string(index=False)); print(y.to_string()); print(manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EXPERIMENT / 'results', help='新目录；绝不覆盖已有结果')
    run(parser.parse_args().out)
