"""同闸门简单策略研究的日期、排序和 ETF 现金流不变量。"""
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

from ashare.config import Config

RESEARCH = run_path(str(Path(__file__).resolve().parents[1] / 'scripts' / '10_simple_controls.py'))


def test_reversal_has_correct_sign_lookback_and_common_scores():
    days = pd.bdate_range('2023-01-02', periods=22)
    codes = ['winner', 'loser', 'not_scored', 'missing_endpoint']
    close = pd.DataFrame(10., index=days, columns=codes)
    close.loc[days[0], 'missing_endpoint'] = np.nan
    close.loc[days[20]] = [11., 9., 8., 10.]
    scores = pd.DataFrame([[1., .1, np.nan, .7]], index=days[20:21], columns=codes)
    actual = RESEARCH['reversal_scores'](close, scores)
    np.testing.assert_allclose(actual.iloc[0], [-.1, .1, np.nan, np.nan], equal_nan=True)
    close.loc[days[21]] *= 2
    pd.testing.assert_frame_equal(actual, RESEARCH['reversal_scores'](close, scores))


def test_cross_year_return_preserves_first_trade_and_new_year_open():
    eq = pd.Series([1., .99, 1.0, .90], index=pd.to_datetime(['2022-12-29', '2022-12-30', '2023-01-03', '2023-01-04']))
    yearly = RESEARCH['annual_returns'](eq)
    np.testing.assert_allclose(yearly, [-.01, .9 / .99 - 1])
    assert (1 + yearly).prod() == pytest.approx(.9)


def etf_inputs():
    days = pd.bdate_range('2024-03-04', periods=6)
    bars = pd.DataFrame({'open': [10., 10., 9., 9., 9., 9.],
                         'close': [10., 10., 9., 9., 9., 9.], 'volume': 10_000}, index=days)
    events = pd.DataFrame({'record_date': [days[1]], 'ex_date': [days[2]],
                           'pay_date': [days[4]], 'cash_per_unit': [1.]})
    return days, bars, events, Config(init_cash=10500., commission=.001, min_commission=5., slippage=0.)


def test_holder_receives_dividend_after_ex_day_sale_and_fees_are_fund_specific():
    days, bars, events, cfg = etf_inputs()
    # 除息日的未复权价格降 10%，不是跌停；该日应能按实际开盘卖出。
    ledger, trades = RESEARCH['etf_ledger'](bars, events, pd.Series([True, False], index=days[1:3]), cfg)
    assert list(trades.shares) == [1000, 1000]
    np.testing.assert_allclose(trades.fee, [10., 9.])  # 未扣股票 cfg.stamp_tax
    assert list(ledger.loc[days[[2, 3, 4]], 'receivable']) == [1000, 1000, 0]
    assert ledger.loc[days[4], 'cash'] == pytest.approx(10500 - 19)
    assert ledger.loc[days[2], 'equity'] == pytest.approx(ledger.loc[days[4], 'equity'])


def test_ex_day_buyer_gets_no_dividend():
    days, bars, events, cfg = etf_inputs()
    ledger, trades = RESEARCH['etf_ledger'](bars, events, pd.Series([True], index=days[2:3]), cfg)
    assert len(trades) == 1 and trades.iloc[0].date == days[2]
    assert ledger[['receivable', 'dividend']].eq(0).all().all()


def test_dividend_cannot_fund_pay_day_open_but_can_be_used_next_time():
    days, bars, events, cfg = etf_inputs()
    ledger, trades = RESEARCH['etf_ledger'](bars, events, pd.Series(True, index=days[[1, 4, 5]]), cfg)
    assert trades.date.tolist() == [days[1], days[5]]
    assert list(trades.shares) == [1000, 100]
    assert list(trades.fee) == [10, 5]  # 第二次买入适用最低佣金
    assert ledger.cash.min() >= 0


@pytest.mark.parametrize('sell_blocked', [False, True])
def test_etf_limit_blocks_open_trade(sell_blocked):
    days, bars, events, cfg = etf_inputs()
    events = events.iloc[:0]
    if sell_blocked:
        bars.loc[days[2], 'open'] = 9.0
        decisions = pd.Series([True, False], index=days[1:3])
    else:
        bars.loc[days[1], 'open'] = 11.
        decisions = pd.Series([True], index=days[1:2])
    _, trades = RESEARCH['etf_ledger'](bars, events, decisions, cfg)
    assert len(trades) == int(sell_blocked) and not (trades.side == 'sell').any()


@pytest.mark.parametrize('bad_input', ['calendar', 'schedule', 'bar', 'event', 'holiday'])
def test_invalid_etf_inputs_are_rejected_not_forward_filled(bad_input):
    days, bars, events, cfg = etf_inputs()
    decisions = pd.Series([True], index=days[1:2])
    if bad_input == 'calendar':
        bars = pd.concat([bars, bars.iloc[-1:]])
    elif bad_input == 'schedule':
        decisions.index = days[:1]
    elif bad_input == 'bar':
        bars.loc[days[2], 'close'] = np.nan
    elif bad_input == 'event':
        events['ex_date'] = events['record_date']
    else:
        events['pay_date'] = days[-1] - pd.Timedelta(days=1)  # 周日，日历中不存在
    with pytest.raises(ValueError):
        RESEARCH['etf_ledger'](bars, events, decisions, cfg)
