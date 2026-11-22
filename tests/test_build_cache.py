"""构建缓存：同样的输入可复用；日期、行情或训练输入发生变化就不能沿用旧打分。"""
import os
import pickle
from dataclasses import replace
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

from ashare import build_cache as cache
from ashare.config import Config


def matrices(n=18):
    dates = pd.bdate_range('2024-08-01', periods=n)
    codes = ['sh600000', 'sh000300']
    close = pd.DataFrame(np.arange(n * 2).reshape(n, 2) / 5 + 10., index=dates, columns=codes)
    return dict(open=close * .99, close=close, high=close * 1.01, low=close * .98,
                volume=close * 8000, amount=close * close * 800000, adj_factor=close * 0 + 1.)


def factor_key(mats, cfg=None, industry=None, extra=None):
    return cache.factor_key(mats, cfg or Config(), industry or {}, extra or {})


@pytest.mark.parametrize('field', ['open', 'high', 'low', 'close', 'volume', 'adj_factor'])
def test_changed_middle_of_same_dated_feed_invalidates_both_layers(field):
    m = matrices()
    before = factor_key(m)
    m[field].iloc[5, 0] += 1
    after = factor_key(m)
    mask = m['volume'] > 0
    days = mask.index[::4]
    assert before != after
    assert cache.prediction_key(before, mask, days, Config()) != cache.prediction_key(after, mask, days, Config())


def test_keys_cover_axes_optional_sources_and_the_actual_model_inputs():
    m, cfg = matrices(), Config()
    key = factor_key(m)
    assert factor_key({name: frame.copy() for name, frame in reversed(list(m.items()))}) == key
    assert key != factor_key(matrices(19))
    assert key != factor_key({k: v.rename(columns={'sh600000': 'sz000001'}) for k, v in m.items()})
    assert key != factor_key({k: v.set_axis(v.index + pd.Timedelta(days=1)) for k, v in m.items()})
    assert key != factor_key(m, replace(cfg, benchmark='sh000905'))
    ind = replace(cfg, use_industry_factors=True)
    assert factor_key(m, ind, {'sh600000': '行业甲'}) != factor_key(m, ind, {'sh600000': '行业乙'})
    ext = replace(cfg, use_ext_factors=True)
    assert factor_key(m, ext, extra={'example': m['open']}) != factor_key(m, ext, extra={'example': m['close']})
    mask = m['volume'] > 0
    dates = mask.index[::3]
    base = cache.prediction_key(key, mask, dates, cfg)
    assert base == cache.prediction_key(key, mask.copy(), dates, replace(cfg, top_n=10))
    assert base != cache.prediction_key(key, mask, dates[:-1], cfg)
    assert base != cache.prediction_key(key, mask, dates, replace(cfg, label_horizon=5))
    assert base != cache.prediction_key(key, mask, dates, replace(cfg, lgb_params=dict(cfg.lgb_params, seed=1)))
    mask.iloc[2, 0] = False
    assert base != cache.prediction_key(key, mask, dates, cfg)


def test_cache_migration_and_failed_write_preserve_last_coherent_pair(tmp_path):
    p = tmp_path / 'preds.pkl'
    key, changed = 'b' * 64, 'c' * 64
    assert cache.read_cache(p, key) is None
    p.write_bytes(pickle.dumps({'old': True}))  # 旧 sidecar 即使存在，也不是输入证据
    (tmp_path / 'preds_key.json').write_text('null', encoding='utf8')
    assert cache.read_cache(p, key) is None
    cache.write_cache(p, key, {'value': 10})
    # 改 mtime 不失效；原有 pickle 读者仍可读取有效负载。
    os.utime(p, (100, 100))
    assert cache.read_cache(p, key) == {'value': 10} and cache.read_cache(p, changed) is None
    with p.open('rb') as source:
        assert pickle.load(source) == {'value': 10}
    before = p.read_bytes()
    with pytest.raises((AttributeError, pickle.PicklingError)):
        cache.write_cache(p, changed, lambda: 1)
    assert p.read_bytes() == before and cache.read_cache(p, key) == {'value': 10}
    p.write_bytes(b'!' + before[1:])  # 键在，内容损坏，不作为命中
    assert cache.read_cache(p, key) is None


@pytest.mark.parametrize('issue', ['stock_extra', 'reference_gap', 'duplicate', 'unordered', 'no_reference'])
def test_panel_cannot_count_nontrading_days_or_compress_index_gaps(issue):
    fn = run_path(str(Path(__file__).resolve().parents[1] / 'scripts/02_build.py'))['check_market_dates']
    dates = pd.to_datetime(['2026-11-13', '2026-11-16', '2026-11-17'])
    panels = {code: pd.DataFrame({'date': dates}) for code in ('sh000300', 'sh000905')}
    fn(panels, dates, 'sh000300')
    if issue == 'stock_extra':
        dates = dates.insert(1, pd.Timestamp('2026-11-14'))
    elif issue == 'reference_gap':
        panels['sh000905'] = panels['sh000905'].iloc[[0, 2]]
    elif issue == 'duplicate':
        panels['sh000300'] = panels['sh000300'].iloc[[0, 1, 1, 2]]
    elif issue == 'unordered':
        panels['sh000300'] = panels['sh000300'].iloc[::-1]
    else:
        del panels['sh000905']
    with pytest.raises(SystemExit, match='交易日历'):
        fn(panels, dates, 'sh000300')


def test_production_cache_flow_for_new_day_and_same_date_revision(tmp_path, monkeypatch):
    build = run_path(str(Path(__file__).resolve().parents[1] / 'scripts/02_build.py'))['build_pipeline']
    env = build.__globals__
    state = {'mats': matrices()}
    calls = {'factors': 0, 'fit': 0}
    class StopAfterScoring(Exception):
        pass
    def compute(mats, **kwargs):
        calls['factors'] += 1
        return {'ret_20': mats['close']}
    def train(*_args):
        calls['fit'] += 1
        return env['md'].ModelBundle(pd.DataFrame({'score': [0.5]}))
    def stop(*_args):
        raise StopAfterScoring
    monkeypatch.setitem(env, 'CACHE_DIR', tmp_path)
    def load(_):
        dates = pd.DataFrame({'date': state['mats']['close'].index})
        return None, {'sh600000': '股票'}, {'sh000300': dates, 'sh000905': dates}
    monkeypatch.setitem(env, 'load_inputs', load)
    monkeypatch.setattr(env['fc'], 'build_matrices', lambda _: state['mats'])
    monkeypatch.setattr(env['fc'], 'compute_factors', compute)
    monkeypatch.setattr(env['md'], 'rolling_train', train)
    monkeypatch.setattr(env['md'], 'baseline_composite', stop)
    cfg = Config(warmup_days=2, train_days=3, min_list_days=1, min_amount_ma20=0)
    def score():
        with pytest.raises(StopAfterScoring):
            build(cfg)
        return (calls['factors'], calls['fit'])
    assert score() == (1, 1)
    assert score() == (1, 1)
    state['mats']['open'].iloc[4, 0] -= 0.5
    assert score() == (2, 2)
    state['mats'] = matrices(19)
    assert score() == (3, 3)
    cfg.label_horizon = 5
    assert score() == (3, 4)
    state['mats']['open'].iloc[-1, 1] = np.nan  # 个股到新日但基准未齐；不覆盖已有缓存
    previous = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(SystemExit, match='缺少完整基准日线'):
        build(cfg)
    assert previous == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
