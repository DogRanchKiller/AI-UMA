"""冻结配对研究：只用事前名次选对，空样本不是零收益，统计单位为评分日。"""
from itertools import combinations, permutations
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

M = run_path(str(Path(__file__).resolve().parents[1] / 'scripts/12_trend_pairs.py'))


def test_neighbor_matching_maximizes_count_then_minimizes_distance():
    # 5先贪心取6，会让10无对照；应宁取距离4的1以保住第二对。
    assert M['match_ranks']([5, 10], [6, 1]) == [(5, 1), (10, 6)]
    # 全枚举的小规模双边指派（包括交叉及弃配），独立验证优化目标。
    rng = np.random.default_rng(314159)
    for _ in range(24):
        values = rng.choice(range(1, 30), 8, replace=False).tolist()
        a, b = values[:4], values[4:]
        feasible = [(0, 0)]
        for size in range(1, 5):
            for left in combinations(a, size):
                for right in permutations(b, size):
                    gaps = [abs(x-y) for x,y in zip(left, right)]
                    if max(gaps) <= 5:
                        feasible.append((size, -sum(gaps)))
        got = M['match_ranks'](a, b)
        assert (len(got), -sum(abs(x-y) for x,y in got)) == max(feasible)
        assert len(set(x for x,_ in got)) == len(set(y for _,y in got)) == len(got)
        assert got == sorted(got) and [y for _,y in got] == sorted(y for _,y in got)


def test_matching_caliper_ties_order_and_empty_groups():
    match = M['match_ranks']
    assert match([10], [4, 15]) == [(10, 15)]
    assert match([10], [4, 16]) == []
    assert match([2, 4], [3]) == [(2, 3)]
    assert match([3], [4, 2]) == [(3, 2)]
    assert match([], [1]) == match([1], []) == []
    with pytest.raises(ValueError, match='名次'):
        match([1, 3], [3, 2])


def test_holiday_and_observation_endpoint_never_moves_or_looks_ahead():
    days = pd.to_datetime(['2024-02-08', '2024-02-19', '2024-02-20', '2024-02-21', '2024-02-22',
                           '2024-02-23', '2024-02-26', '2024-02-27', '2024-02-28', '2024-02-29',
                           '2024-03-01', '2024-03-04', '2024-03-05'])
    entry, end = M['observation_dates'](days, days[0])
    assert (entry, end) == (days[1], days[11])
    pair = dict(strong_code='a', control_code='b', strong_rank=9, control_rank=8)
    def price(code, when):
        assert when in (entry, end)
        return {'a': (20, 22), 'b': (10, 9), 'i': (100, 105)}[code][int(when == end)]
    complete = M['outcome'](pair, entry, end, price, 'i')
    assert complete['status'] == 'complete'
    assert [complete[k+'_return'] for k in ('strong','control','index')] == pytest.approx([.1, -.1, .05])
    missing = M['outcome'](pair, entry, end, lambda c,d: np.nan if (c,d)==('b',end) else price(c,d), 'i')
    assert missing['status'] == 'missing' and np.isnan(missing['control_return'])
    assert missing['control_code'] == 'b' and missing['strong_return'] == complete['strong_return']
    entry2, end2 = M['observation_dates'](days[:11], days[0])
    assert entry2 == entry and end2 is None
    pending = M['outcome'](pair, entry2, end2, lambda *_: pytest.fail('不应读任何未来价格'), 'i')
    assert pending['status'] == 'pending' and np.isnan(pending['index_return'])
    assert M['observation_dates'](days[-1:], days[-1]) == (None, None)
    with pytest.raises(ValueError, match='端点'):
        M['outcome'](pair, end, entry, price, 'i')


@pytest.mark.parametrize('op,volume', [(0, 1), (np.nan, 1), (np.inf, 1), (11, 0), (11, np.inf)])
def test_no_open_fill_or_nontrading_surrogates(op, volume):
    days = pd.date_range('2025-04-01', periods=3)
    frame = pd.DataFrame({'open': [10,op], 'volume': [12,volume]}, index=days[:2])
    assert M['quote_open'](frame, days[0]) == 10
    for day in days[1:]:
        assert np.isnan(M['quote_open'](frame, day))


def test_zero_is_not_up_and_same_complete_sample_for_both_legs():
    fn = M['daily_values']
    frame = pd.DataFrame(dict(strong_return=[.1,0.],control_return=[-.05,.03],index_return=[.015,.015]))
    result = fn(frame)
    assert result['hit_strong'] == .5 and result['hit_control'] == .5 and result['hit_diff'] == 0
    assert result['ret_diff'] == pytest.approx(.06) and result['excess_strong'] == pytest.approx(.035)
    assert all(np.isnan(v) for v in fn(frame.iloc[:0]).values())
    frame.loc[1,'control_return'] = np.nan
    with pytest.raises(ValueError, match='完整对'):
        fn(frame)


def test_date_weights_dont_pool_pairs_or_include_empty_or_pending_days():
    rows = []
    for i,(n,diff,gate,mature) in enumerate([(1,1.,False,True), (30,-1.,False,True),
                                               (0,np.nan,False,True), (1,.5,True,True), (0,np.nan,False,False)]):
        row = {m: diff for m in M['METRICS']}
        rows.append(dict(row, date=pd.Timestamp('2022-01-01')+pd.Timedelta(days=i), gate=gate,
                         n_pairs=max(1,n), n_complete=n, mature=mature))
    table = M['summaries'](pd.DataFrame(rows)).set_index(['period','regime'])
    off = table.loc[('all','off')]
    assert (off.mature_dates, off.effective_dates, off.pairs_complete) == (3,2,31)
    assert off.hit_diff == 0 and table.loc[('all','all'),'hit_diff'] == pytest.approx(1/6)
    assert np.isnan(table.loc[('2022','off'),'hit_diff_ci_low'])


def test_bootstrap_uses_three_date_blocks_on_full_calendar():
    # 6 个日期，每次正好两个长度3的块。标记不符的日期不可先移除。
    vals = np.array([np.nan, 0.8, 0.0, np.nan, -.1, -.6])
    starts = np.random.default_rng(20261121).integers(0,6,size=(10_000,2))
    circular = np.r_[vals, vals[:2]]
    sums = np.array([np.nansum(circular[i:i+3]) for i in range(6)])
    valid = np.array([np.isfinite(circular[i:i+3]).sum() for i in range(6)])
    draws = sums[starts].sum(axis=1) / valid[starts].sum(axis=1)
    assert M['block_hit_interval'](vals) == pytest.approx(np.percentile(draws,[2.5,97.5]))
    assert all(np.isnan(v) for v in M['block_hit_interval']([np.nan,.1]))


def test_frozen_input_mismatch_and_existing_output_fail_closed(tmp_path):
    path = tmp_path/'data.csv'
    path.write_text('first', encoding='utf-8')
    digest = M['sha256'](path)
    M['check_input'](path, digest)
    path.write_text('replacement', encoding='utf-8')
    with pytest.raises(ValueError, match='指纹'):
        M['check_input'](path, digest)
    with pytest.raises(FileExistsError):
        M['run'](tmp_path)
    assert path.read_text(encoding='utf-8') == 'replacement'
