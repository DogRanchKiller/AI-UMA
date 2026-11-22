"""M2：比例收益用于研究，不能充当股数、分红或实际成交价。"""
import numpy as np
import pandas as pd
import pytest

from ashare import backtest, labels, universe
from ashare.config import Config


def market():
    dates = pd.bdate_range('2024-03-04', periods=5)
    prices = pd.DataFrame({'sh600000': [20., 20., 20., 10., 10.5],
                           'sh000300': [100., 100., 100., 100., 102.]}, index=dates)
    factor = prices * 0 + 1
    factor.iloc[3:, 0] = 2.
    return dict(open=prices, close=prices.copy(), adj_factor=factor, volume=prices * 1000)


def test_open_excess_uses_effective_factor_at_each_endpoint():
    mats = market()
    values = labels.panel_excess(mats, 'sh000300', 2, None)
    assert values.iloc[0, 0] == pytest.approx(0.)
    assert values.iloc[1, 0] == pytest.approx(.05 - .02)


@pytest.mark.parametrize('quote,expected', [(10., (True, True)), (11., (False, True)), (9., (True, False))])
def test_ex_date_limits_use_comparable_prices(quote, expected):
    mats = market()
    mats['open'].iloc[3, 0] = quote
    flags = universe.panel_tradeable_at_open(mats)
    assert tuple(x.iloc[3, 0] for x in flags) == expected


def replay(mats, signal_positions, *, code='sh600000', cfg=None, **kwargs):
    dates = mats['close'].index
    cfg = cfg or Config(top_n=1, max_weight=.5, init_cash=100000., slippage=0.)
    scores = pd.DataFrame({code: 1.}, index=dates[signal_positions])
    return backtest.run(scores, mats, mats['close'].notna(), *universe.panel_tradeable_at_open(mats),
                        mats['close']['sh000300'], cfg, **kwargs)


@pytest.mark.parametrize('rebalance,gap', [(False, False), (True, False), (False, True)])
def test_held_action_aborts_before_execution_or_mark(rebalance, gap):
    mats = market()
    if gap:
        for frame in mats.values():
            frame.iloc[2, 0] = np.nan
    with pytest.raises(ValueError, match='2024-03-07.*sh600000'):
        replay(mats, [0, 2] if rebalance else [0])


def test_entry_after_action_uses_raw_price_and_real_lots():
    mats = market()
    result = replay(mats, [2])
    order = result.trades.iloc[0]
    assert order.price == 10 and order.shares == 5000
    assert result.equity.iloc[-1] == pytest.approx((100000 - order.fee + 5000 * .5) / 100000)


def test_inverse_volatility_ignores_ex_date_price_step():
    dates = pd.bdate_range('2024-03-04', periods=62)
    moves = np.resize([.01, -.01], len(dates))
    continuous = pd.DataFrame({'stock': 20 * (1 + moves).cumprod()}, index=dates)
    factor = continuous * 0 + 1
    factor.iloc[30:] = 2
    vol = backtest.volatility60({'close': continuous / factor, 'adj_factor': factor})
    assert vol.iloc[-1, 0] == pytest.approx(np.std(moves[-60:], ddof=1))


def event(**changes):
    row = dict(code='sh.600000', dividPreNoticeDate='', dividAgmPumDate='',
               dividPlanAnnounceDate='2024-03-01', dividPlanDate='2024-03-04',
               dividRegistDate='2024-03-06', dividOperateDate='2024-03-07',
               dividPayDate='2024-03-08', dividStockMarketDate='2024-03-08',
               dividCashPsBeforeTax='2', dividCashPsAfterTax='1.8或2',
               dividStocksPs='.3', dividReserveToStockPs='.5', dividCashStock='10转5送3派20元（含税）')
    return dict(row, **changes)


def test_announced_gross_entitlements_are_not_credited_twice_or_tradable_early():
    mats = market()  # 理论参考价=(20-2)/(1+.3+.5)=10
    dates = mats['close'].index
    rows = pd.DataFrame([event()] * 6)
    gate = pd.Series([True, False, False], index=dates[[0, 2, 3]])
    result = replay(mats, [0, 2, 3], corporate_actions=rows, regime_ok=gate)
    first, sell, listing_sale = result.trades.to_dict('records')
    assert (first['shares'], sell['shares'], listing_sale['shares']) == (2500, 2500, 2000)
    assert (sell['date'], listing_sale['date']) == (dates[3], dates[4])
    assert result.equity.iloc[3] == pytest.approx((100000 - first['fee'] - sell['fee']) / 100000)
    assert result.equity.iloc[4] == pytest.approx((101000 - sum(t['fee'] for t in (first, sell, listing_sale))) / 100000)
    assert result.diagnostics['已登记持仓权益事件数'] == 1
    assert result.diagnostics['期末现金占比'] == pytest.approx(1.)
    assert result.holdings_by_date[dates[3]] == {}  # 次日上市股份不冒充可用持仓
    truncated = replay({k: v.iloc[:-1] for k, v in mats.items()}, [0], corporate_actions=rows)
    assert truncated.diagnostics['期末应收税前现金'] == 5000
    assert truncated.diagnostics['期末待上市股份'] == {'sh600000': 2000}
    new = replay(mats, [2], corporate_actions=rows)
    assert new.diagnostics['已登记持仓权益事件数'] == 0


@pytest.mark.parametrize('field,value', [('dividPlanDate', '2024-03-07'), ('dividRegistDate', '2024-03-05'),
    ('dividCashPsBeforeTax', ''), ('dividStocksPs', '.8'), ('dividCashStock', '10配3股'),
    ('dividPayDate', ''), ('dividStockMarketDate', ''), ('dividStockMarketDate', '2024-03-06')])
def test_unknown_or_inconsistent_terms_reject_held_action(field, value):
    with pytest.raises(ValueError, match='2024-03-07.*sh600000'):
        replay(market(), [0], corporate_actions=pd.DataFrame([event(**{field: value})]))


def test_conflicting_records_and_wrong_factor_do_not_authorize_entitlements():
    with pytest.raises(ValueError, match='重复冲突'):
        replay(market(), [0], corporate_actions=pd.DataFrame([event(), event(dividCashPsBeforeTax='.4')]))
    mats = market()
    mats['adj_factor'].iloc[3:, 0] = 1.7
    with pytest.raises(ValueError, match='累计因子折算参考价不符'):
        replay(mats, [0], corporate_actions=pd.DataFrame([event()]))


def test_pingan_gross_receivable_matches_independent_announced_cash_not_price_ratio():
    # 真实 sz000001，BaoStock 未复权报价、query_adjust_factor 和 query_dividend_data
    # 独立取得；与项目本机实采目录无关。成交用 6/12 的原价 10.85，登记日为 6/13。
    dates = pd.to_datetime(['2024-06-11', '2024-06-12', '2024-06-13', '2024-06-14', '2024-06-17'])
    op = pd.DataFrame({'sz000001': [10.98, 10.85, 10.89, 10.15, 10.13], 'sh000300': 100.}, index=dates)
    cl, factor = op.copy(), op * 0 + 1
    cl['sz000001'] = [10.87, 10.88, 10.80, 10.18, 10.10]
    factor['sz000001'] = [104.876880] * 3 + [112.368136] * 2
    mats = dict(open=op, close=cl, adj_factor=factor, volume=op * 1000)
    row = event(code='sz.000001', dividPlanDate='2024-06-06', dividRegistDate='2024-06-13',
                dividOperateDate='2024-06-14', dividPayDate='2024-06-14', dividStockMarketDate='',
                dividCashPsBeforeTax='0.719', dividCashPsAfterTax='0.6471或0.719', dividStocksPs='0.000000',
                dividReserveToStockPs='', dividCashStock='10派7.19元（含税，扣税后6.471或7.19元）')
    res = replay(mats, [0], code='sz000001', corporate_actions=pd.DataFrame([row]))
    order = res.trades.iloc[0]
    assert order.price == 10.85 and order.shares == 4600
    assert res.holdings_by_date[dates[3]] == {'sz000001': 4600}
    assert res.equity.loc[dates[3]] == pytest.approx((100000 - order.fee + 4600 * (10.18 - 10.85 + .719)) / 100000)
    assert res.diagnostics['期末应收税前现金'] == 0


def test_pay_day_close_credit_cannot_fund_opening_order():
    mats = market()
    dates = mats['open'].index
    for frame in (mats['open'], mats['close']):
        frame.loc[dates[3]:, 'sh600000'] = 18.
        frame['sh600001'] = 10.
    mats['adj_factor'].loc[dates[3]:, 'sh600000'] = 20 / 18
    mats['adj_factor']['sh600001'] = 1.
    mats['volume']['sh600001'] = 100000.
    row = event(dividPayDate='2024-03-07', dividCashPsBeforeTax='2', dividCashStock='10派20元（含税）',
                dividStocksPs='0', dividReserveToStockPs='', dividStockMarketDate='')
    scores = pd.DataFrame({'sh600000': [1., 0.], 'sh600001': [0., 1.]}, index=dates[[0, 2]])
    cfg = Config(top_n=1, max_weight=1., init_cash=100000., slippage=0., commission=0., min_commission=0., stamp_tax=0.)
    result = backtest.run(scores, mats, mats['close'].notna(), *universe.panel_tradeable_at_open(mats),
                          mats['close']['sh000300'], cfg, corporate_actions=pd.DataFrame([row]))
    assert result.trades.iloc[-1].code == 'sh600001'
    assert result.trades.iloc[-1].shares == 9000  # 开盘只能用旧股卖出的 90,000，不能用应收 10,000。
    assert result.equity.iloc[-1] == pytest.approx(1.)
    assert result.diagnostics['期末现金占比'] == pytest.approx(.1)
