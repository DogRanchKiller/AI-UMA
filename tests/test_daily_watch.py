"""观察名单按信号日信息生成，不等于策略买单。"""
from datetime import datetime, timezone
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

from ashare.config import Config

M = run_path(str(Path(__file__).resolve().parents[1] / 'scripts/11_daily_watch.py'))


@pytest.fixture
def market():
    days = pd.bdate_range('2023-01-02', periods=85)
    codes = [f'sh60{i:04d}' for i in range(120)]
    close = np.linspace(5, 10, len(days))
    bar = pd.DataFrame({'close': close, 'open': close, 'high': close + .1, 'low': close - .1,
                        'volume': 100_000.0}, index=days)
    idx = np.linspace(2000, 1500, len(days))
    bench = pd.DataFrame({'open': idx, 'close': idx, 'low': idx - 1, 'high': idx + 1,
                          'volume': 100_000.0}, index=days)
    names = {c: f'样本{i}' for i, c in enumerate(codes)}
    pred = pd.DataFrame({'date': days[-3], 'code': codes, 'score': np.linspace(1, .01, len(codes))})
    return days, codes, names, pred, bar, bench, Config(min_list_days=40, min_amount_ma20=0)


def test_freshness_is_in_sessions_and_future_data_not_visible(market):
    days, codes, names, pred, bar, bench, cfg = market
    # 调仓后第二个交易日的观察仍可用旧排名，但不冒充新预测。
    d = days[-2]
    def scan(p, b, x):
        return M['make_snapshot'](d, p, names, b, lambda _: x, cfg, pinned=(codes[-1],))
    initial = scan(pred, bench, bar)
    future = pred.copy(); future['date'] = days[-1]; future['score'] = -future.score
    altered = bar.copy(); altered.loc[days[-1], 'close'] *= 5
    b = bench.copy(); b.loc[days[-1], 'close'] *= 5
    assert initial == scan(pd.concat([pred, future]), b, altered)
    assert initial['score_date'] == str(days[-3].date()) and initial['score_age'] == 1
    assert not initial['gate']  # 关闸仍然可以有观察，不能变成买单
    assert [r['code'] for r in initial['matches']] == codes[:6]
    assert initial['pinned'][0]['rank'] == 120


def test_historical_preview_does_not_backdate_fixed_observers(market):
    days, _, names, pred, bar, bench, cfg = market
    snap = M['make_snapshot'](days[-1], pred, names, bench, lambda _: bar, cfg)
    assert snap['pinned'] == [] and len(snap['matches']) == 6


def test_score_expires_on_twentieth_trading_day(market):
    days, _, _, pred, *_ = market
    pred['date'] = days[0]
    assert M['pick_scores'](pred, days[:20], 20)[2] == 19
    with pytest.raises(ValueError, match='20 个交易日'):
        M['pick_scores'](pred, days[:21], 20)


@pytest.mark.parametrize('case', ['none', 'off_calendar', 'duplicate', 'bad_code', 'small', 'nan'])
def test_invalid_score_snapshot_rejected(market, case):
    days, _, _, pred, *_ = market
    if case == 'none':
        pred['date'] = days[-1] + pd.Timedelta(days=5)
    elif case == 'off_calendar':
        pred['date'] = days[0] - pd.Timedelta(days=1)
    elif case == 'duplicate':
        pred = pd.concat([pred,pred.iloc[:1]])
    elif case == 'bad_code':
        pred.loc[0, 'code'] = '../foreign'
    elif case == 'small':
        pred = pred.iloc[:10]
    else:
        pred.loc[0, 'score'] = np.nan
    with pytest.raises(ValueError):
        M['pick_scores'](pred, days, 20)


def test_unknown_and_ineligible_are_not_weak_trend(market):
    days, codes, names, pred, bar, bench, cfg = market
    a, b, c, d, e, f = codes[:6]
    no_bar = bar.drop(days[-1]); sparse = bar.drop(days[-10]); halted = bar.copy(); halted.iloc[-1,-1]=0
    flat = bar.copy(); flat[['open', 'low', 'high', 'close']] = 7.; names[d] = '*ST样本'
    samples = {a:no_bar, b:sparse, c:halted, e:flat}
    snap = M['make_snapshot'](days[-1], pred, names, bench, lambda code: samples.get(code,bar), cfg, pinned=(c,d,e))
    assert snap['unknown'] == [a,b,c]
    assert [x['state'] for x in snap['pinned']] == ['unknown','ineligible','unconfirmed']
    assert [x['code'] for x in snap['matches']] == [f]


@pytest.mark.parametrize('field,value', [('open',np.nan),('open',0.),('volume',0.),('high',.01),('low',np.inf)])
def test_broken_session_bar_never_publishes_as_strong(market, field, value):
    days, codes, names, pred, bar, bench, cfg = market
    broken = bar.copy(); broken.loc[days[-1],field] = value
    snap = M['make_snapshot'](days[-1],pred,names,bench,lambda c:broken if c==codes[0] else bar,cfg,pinned=(codes[0],))
    assert codes[0] in snap['unknown'] and snap['pinned'][0]['state'] == 'unknown'
    assert codes[0] not in {r['code'] for r in snap['matches']}
    bench.loc[days[-1],field] = value
    with pytest.raises(ValueError, match='指数'):
        M['make_snapshot'](days[-1],pred,names,bench,lambda _:bar,cfg,pinned=())


def test_all_data_missing_and_bad_index_cannot_be_empty_report(market):
    days, _, names, pred, bar, bench, cfg = market
    with pytest.raises(ValueError, match='候选池行情均缺失'):
        M['make_snapshot'](days[-1],pred,names,bench,lambda _: bar.iloc[:-1],cfg,pinned=())
    bench.loc[days[-1],'close'] = np.nan
    with pytest.raises(ValueError, match='指数'):
        M['make_snapshot'](days[-1],pred,names,bench,lambda _: bar,cfg,pinned=())


def test_only_three_highlights_but_all_matching_codes_survive(market, capsys):
    days,codes,names,pred,bar,bench,cfg = market
    snap = M['make_snapshot'](days[-1],pred,names,bench,lambda _:bar,cfg,pinned=())
    diff = {'initial':codes[:6], 'new':[], 'kept':[], 'lost':[], 'unknown':[]}
    M['render'](snap,diff,None)
    text = capsys.readouterr().out
    assert '达标 6 只' in text and codes[2] in text and codes[3] not in text
    assert '沿用旧评分' in text and '关，仅观察' in text
    diff.update(initial=[], new=[codes[5]], kept=codes[:5])
    M['render'](snap, diff, str(days[-2].date()))
    assert f'[新进] {codes[5]}' in capsys.readouterr().out


def test_score_date_cannot_regress_or_be_silently_revised():
    check = M['validate_score_continuity']
    old = {'score_date': '2026-11-05', 'score_sha256': 'prior'}
    check([], old)
    check([old], old)
    check([old], dict(old, score_date='2026-11-20', score_sha256='new'))
    with pytest.raises(ValueError, match='早于'):
        check([old], dict(old, score_date='2026-10-09'))
    with pytest.raises(ValueError, match='改写'):
        check([old], dict(old, score_sha256='different'))


def test_calendar_gap_is_not_treated_as_a_holiday(market):
    days, _, _, _, _, bench, _ = market
    M['verify_calendar'](bench, bench.iloc[:-1], days[-2])  # 观察日以后的缺口无关
    with pytest.raises(ValueError, match='交易日历'):
        M['verify_calendar'](bench.drop(days[-4]), bench, days[-2])


def test_date_cannot_silently_backfill_or_publish_before_close():
    dates = pd.to_datetime(['2026-11-19','2026-11-20'])
    b = pd.DataFrame({'close':[10,11]},index=dates)
    # UTC 06:59 是上海 14:59。当日报价即使已落盘也不能登记为已收盘。
    before = datetime(2026,11,20,6,59,tzinfo=timezone.utc)
    with pytest.raises(ValueError,match='无已收盘'):
        M['select_day'](b,None,'record',before)
    assert M['select_day'](b,None,None,before) == dates[0]
    after = datetime(2026,11,20,7,1,tzinfo=timezone.utc)
    assert M['select_day'](b,None,'record',after) == dates[1]
    weekend = datetime(2026,11,21,8,tzinfo=timezone.utc)
    assert M['select_day'](b,None,'baseline',weekend) == dates[1]
    with pytest.raises(ValueError, match='无已收盘'):
        M['select_day'](b,None,'record',weekend)
    with pytest.raises(ValueError, match='倒填'):
        M['select_day'](b,'2026-11-19','baseline',weekend)
    with pytest.raises(ValueError, match='上海当日'):
        M['select_day'](b,'2026-11-20','record',weekend)
    with pytest.raises(ValueError,match='没有已收盘'):
        M['select_day'](b.iloc[0:0],None,None,before)


def test_quote_files_never_resolve_external_codes_and_detect_corruption(tmp_path):
    assert M['read_bars'](tmp_path,'sh600795').empty
    with pytest.raises(ValueError,match='有效'):
        M['read_bars'](tmp_path,'../unexpected')
    data = 'date,open,close,low,high,volume,code\n2026-11-20,1,2,1,2,3,sh601963\n'
    (tmp_path/'sh600795.csv').write_text(data,encoding='utf8')
    with pytest.raises(ValueError,match='代码不符'):
        M['read_bars'](tmp_path,'sh600795')
    data = data.replace('sh601963','sh600795')
    (tmp_path/'sh600795.csv').write_text(data + data.splitlines()[1]+'\n',encoding='utf8')
    with pytest.raises(ValueError,match='重复'):
        M['read_bars'](tmp_path,'sh600795')
