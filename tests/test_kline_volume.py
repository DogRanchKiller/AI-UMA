"""只测试冻结研究逻辑；不读取/写入正式研究结果或交易日志。"""
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

K = run_path(str(Path(__file__).resolve().parents[1] / 'scripts/13_kline_volume.py'))


def bars(n=76):
    days = pd.bdate_range('2024-01-02', periods=n)
    return days, pd.DataFrame({'open': 98., 'high': 100., 'low': 95., 'close': 97., 'volume': 100.}, index=days)


@pytest.mark.parametrize('close,expected', [(97., ('expand','outside','outside')),
    (96.99999, ('outside','outside','outside')), (99.9, ('expand','expand','outside')),
    (99.8999, ('expand','outside','outside')), (100., ('expand','expand','outside'))])
def test_three_definitions_are_exact_and_independent(close, expected):
    days, frame = bars(); t = days[60]
    frame.loc[t, ['close', 'volume']] = [close, 120]
    result = K['classify'](frame, days, t)
    assert tuple(result[s] for s in K['SPECS']) == expected
    assert result['high60'] == result['prior_high60'] == 100.
    assert result['volume_ratio'] == pytest.approx(1.2)


@pytest.mark.parametrize('high,edge,spec', [(6.9,6.693,'near3'),(2.7,2.6973,'near01')])
def test_exact_decimal_edges_that_binary_multiplication_would_misclassify(high, edge, spec):
    days, f = bars(); t = days[60]
    f['high'] = high; f['low'] = high/2; f['open'] = f['close'] = edge; f.loc[t, 'volume'] = 120.
    assert K['classify'](f, days, t)[spec] == 'expand'
    f.loc[t, 'close'] = edge - .000001
    assert K['classify'](f, days, t)[spec] == 'outside'


def test_per_date_equal_weight_not_pooled_and_masks_follow_mature_calendar():
    dates = pd.date_range('2023-12-29', periods=5)
    rows = []
    for spec in K['SPECS']:
        for i, (n, ha, hb, gate, mature) in enumerate([(1,1,0,True,True),(4,0,1,False,True),(0,0,0,True,True),
                                                     (2,1,0,True,True),(0,0,0,True,False)]):
            stats = dict(n_complete=n, n_missing=0, n_pending=0, n_pairs=n, mature=mature, up_expand=int(n*ha),
                         up_contract=int(n*hb), date=dates[i], gate=gate, spec=spec)
            for col in ['ret_expand', 'ret_contract','ret_index','ret_diff','excess_expand','excess_contract']:
                stats[col] = 0 if n else np.nan
            rows.append(dict(stats, hit_diff=ha-hb if n else np.nan, hit_expand=ha if n else np.nan, hit_contract=hb if n else np.nan))
    table = K['subgroup_summaries'](pd.DataFrame(rows)).set_index(['spec','period','regime'])
    overall = table.loc[('near3','all','all')]
    assert (overall.effective_dates, overall.mature_dates, overall.score_dates, overall.up_expand,
            overall.up_contract, overall.pairs_complete) == (3,4,5,3,4,7)
    assert overall.hit_diff == pytest.approx(1/3)  # 合并则为 -1/7，方向相反。
    assert pd.isna(table.loc[('near3','2024','all'),'hit_diff_ci_low'])


def test_prior_break_does_not_require_closeness_to_todays_spike():
    dates, f = bars(); t = dates[60]
    f.loc[t] = [101., 110., 97., 101., 50.]
    out = K['classify'](f, dates, t)
    assert (out['near3'], out['near01'], out['prior_break']) == ('outside','outside','contract')


def test_mean_and_reference_high_use_exact_trading_days_and_no_future():
    days, frame = bars(); t = days[60]
    frame.loc[days[0], 'high'] = 120.  # 仅属于“不含今日”的此前60日
    frame.loc[days[39], 'volume'] = 10000.  # 在前20日以外
    frame.loc[days[40], 'volume'] = 120.
    frame.loc[t, 'volume'] = 101.
    old = K['classify'](frame, days, t)
    assert (old['high60'], old['prior_high60'], old['volume_mean20'], old['near3']) == (100., 120., 101., 'flat')
    edited = frame.copy(); edited.loc[days[61]:] = -999999.
    assert old == K['classify'](edited, days, t) == K['classify'](frame.loc[:t], days, t)


@pytest.mark.parametrize('column,bad,back', [('high', np.inf, 30), ('volume', 0., 30), ('high', 0., 5),
    ('volume', np.nan, 10), ('low', 99., 0), ('open', 0., 0), ('close', 101., 0)])
def test_faulty_reference_or_ohlcv_is_unknown_not_filled(column, bad, back):
    days, f = bars(); t = days[60]; f.loc[days[60-back], column] = bad
    assert K['classify'](f, days, t)['near3'] == 'unclassifiable'


def test_missing_session_and_59_vs_60_vs_61_rows():
    days, f = bars()
    assert K['classify'](f, days, days[58])['near3'] == 'unclassifiable'
    r = K['classify'](f, days, days[59]); assert r['near3'] == 'flat' and r['prior_break'] == 'unclassifiable'
    g = f.drop(days[0]); assert K['classify'](g, days, days[60])['near3'] == 'flat'
    assert K['classify'](g, days, days[60])['prior_break'] == 'unclassifiable'
    assert K['classify'](f.drop(days[55]), days, days[60])['near3'] == 'unclassifiable'


def sample(dates):
    rows = []
    for day in dates:
        for rank, (near, tight, prior) in enumerate([('expand','outside','outside'),('contract','expand','outside'),
                   ('contract','contract','flat'),('unclassifiable','unclassifiable','unclassifiable')], 1):
            rows.append(dict(date=day, gate=True, code=f'sz{rank:06d}', rank=rank, state='strong' if rank%2 else 'unconfirmed',
                             near3=near, near01=tight, prior_break=prior))
    return pd.DataFrame(rows)


def test_plan_is_independent_per_spec_and_ignores_old_paired_or_outcome_fields():
    days = pd.bdate_range('2024-02-19', periods=16); pool = sample(days[[0, -1]])
    plan, dates = K['make_plans'](pool)
    assert len(dates) == 6 and len(plan) == 4
    assert set(plan.loc[plan.spec == 'near3','expand_rank']) == {1}
    assert set(plan.loc[plan.spec == 'near01','expand_rank']) == {2}
    assert dates.loc[dates.spec == 'prior_break','n_pairs'].eq(0).all()
    changed = pool.assign(paired=np.arange(len(pool))%2 == 0, hit_diff=-77, next_return=1000)
    pd.testing.assert_frame_equal(plan, K['make_plans'](changed)[0])


def test_lock_is_verified_before_prices_and_all_legs_share_the_same_endpoints(tmp_path):
    days = pd.bdate_range('2024-02-19', periods=16); pool = sample([days[0], days[-1]])
    plan, coverage = K['make_plans'](pool); K['lock_plans'](tmp_path, plan, coverage)
    calls = []
    def price(code, day):
        calls.append((code, day))
        return (11. if code == 'sz000001' else 10.) if day == days[11] else 10.
    pairs, daily, summary = K['evaluate_locked'](tmp_path, days, price, 'idx')
    assert set(d for _, d in calls) == {days[1], days[11]}
    snapshots = pairs[pairs.spec == 'near3'].reset_index(drop=True)
    assert list(snapshots.status) == ['complete', 'pending']
    assert snapshots.loc[0, 'expand_return'] == pytest.approx(.1) and snapshots.loc[0, 'contract_return'] == 0
    assert len(daily) == 6
    small = summary[(summary.spec == 'near3') & (summary.period == 'all') & (summary.regime == 'all')].iloc[0]
    assert small.effective_dates == 1 and small.pairs_complete == 1 and small.up_expand == 1 and small.up_contract == 0
    assert small.small_sample and pd.isna(small.hit_diff_ci_low)
    path = tmp_path / 'plan.csv'; path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='指纹'):
        K['evaluate_locked'](tmp_path, days, lambda *_: pytest.fail('read a price after tampering'), 'idx')


def test_no_pairs_retains_all_rows_and_schema_without_touching_outcomes(tmp_path):
    days = pd.bdate_range('2024-02-19', periods=16); pool = sample([days[0], days[-1]])
    for spec in K['SPECS']:
        pool[spec] = 'expand'
    plan, coverage = K['make_plans'](pool); K['lock_plans'](tmp_path, plan, coverage)
    pair, daily, summary = K['evaluate_locked'](tmp_path, days, lambda *_: pytest.fail('empty should not read price'), 'idx')
    assert pair.empty and {'status','expand_entry_open','contract_exit_open','index_return'}.issubset(pair)
    assert len(daily) == 6 and daily.n_complete.eq(0).all() and daily.hit_diff.isna().all()
    assert summary.effective_dates.eq(0).all() and summary.hit_diff.isna().all() and summary.small_sample.all()


def test_missing_one_leg_excludes_both_and_index_from_daily_denominator(tmp_path):
    days = pd.bdate_range('2024-02-19', periods=16)
    plan, coverage = K['make_plans'](sample([days[0]])); K['lock_plans'](tmp_path, plan, coverage)
    out, daily, _ = K['evaluate_locked'](tmp_path, days, lambda c,d: np.nan if c == 'sz000002' and d == days[11] else 10., 'idx')
    assert out.status.eq('missing').all() and out['index_return'].eq(0).all()
    assert daily.n_complete.eq(0).all() and daily['ret_index'].isna().all()


def test_run_rejects_preexisting_output_without_reading_or_replacing_input(tmp_path):
    file = tmp_path / 'unchanged'; file.write_bytes(b'unchanged')
    with pytest.raises(FileExistsError):
        K['run'](tmp_path)
    assert file.read_bytes() == b'unchanged'
