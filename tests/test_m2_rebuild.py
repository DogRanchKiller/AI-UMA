"""隔离入口的契约测试；临时构造的行情不用于衡量收益。"""
import csv
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ashare import datasource as ds
from ashare.config import Config

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / '14_m2_rebuild.py'
spec = importlib.util.spec_from_file_location('m2_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_exact_frozen_input_config_and_universe():
    values = Config().to_dict()
    values.pop('use_ext_factors')
    values.pop('start_year')
    values.update(top_n=15, num_boost_round=311)
    old = {'config': values, 'universe': {'date_range': ['2019-01-02', '2026-11-20']}}
    cfg = runner.effective_config(values, Config, old['universe']['date_range'][1])
    assert cfg.end_date == '2026-11-20'
    assert not cfg.use_ext_factors
    assert {k: v for k, v in cfg.to_dict().items() if k in old['config'] and k != 'end_date'} == {
        k: v for k, v in old['config'].items() if k != 'end_date'}
    with pytest.raises(ValueError, match='配置键'):
        runner.effective_config({**old['config'], 'unknown': 1}, Config, cfg.end_date)


def test_missing_w2_never_calls_build_or_writes_legacy(tmp_path, monkeypatch):
    base = tmp_path / 'data' / 'm2'
    base.mkdir(parents=True)
    monkeypatch.setattr(runner, 'M2', base)
    old = tmp_path / 'legacy'
    old.mkdir()
    sentinel = old / 'run_summary.json'
    sentinel.write_bytes(b'old')
    def missing(dest, build):
        raise ValueError('W2 逐只证据缺失')
    monkeypatch.setattr(runner, 'execute', missing)
    assert runner.main(['--build', '--run-id', 'audit']) == 2
    dest = base / 'runs' / 'audit'
    status = (dest / 'status.json').read_bytes()
    assert json.loads(status)['status'] == 'blocked'
    assert not (dest / 'output').exists() and not (dest / 'cache').exists()
    assert runner.main(['--run-id', 'audit']) == 2
    assert (dest / 'status.json').read_bytes() == status
    assert sentinel.read_bytes() == b'old'
    for invalid in ('../legacy', '/absolute', 'a/../b', 'a.b', 'A:cache'):
        assert runner.main(['--run-id', invalid]) == 2


def test_pin_rejects_changed_bytes(tmp_path):
    (tmp_path / 'file').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        runner.pinned(tmp_path, {'file': runner.digest(b'original')})


def frame(code, dates, volumes):
    df = pd.DataFrame({'code': code, 'date': dates, 'open': 10, 'high': 10, 'close': 10, 'low': 10,
                       'volume': volumes, 'amount': [v * 1000 for v in volumes], 'adj_factor': 1.0})
    return ds._dated_bars(df, code)


@pytest.mark.parametrize('code,extra_volume,witness,allowed', [
    ('sz000001', 0, True, True), ('sz000001', 0, False, False),
    ('sz000001', 1, True, False), ('sh000300', 0, True, False)])
def test_halts_require_exact_stock_day_and_zero_amounts(tmp_path, code, extra_volume, witness, allowed):
    old = ['2019-01-02']
    extra = '2019-01-03'
    f = frame(code, [*old, extra], [2, extra_volume])
    frozen = SimpleNamespace(code=code, rows=1, dates_sha256=runner.date_hash(old))
    name = 'halt-evidence/' + extra + '.json'
    path = tmp_path / name
    path.parent.mkdir()
    raw = dict(source=runner.SOURCE_ID, day=extra, fields=['code', 'tradeStatus', 'code_name'],
               rows=[['sz.000001', '0' if witness else '1', 'example']])
    path.write_text(json.dumps(raw))
    sha = runner.digest(path.read_bytes())
    ix = f'day,raw_path,raw_sha256\n{extra},{name},{sha}\n'.encode()
    proof = runner.halt_checker(tmp_path, ix, {extra})
    omitted = []
    if allowed:
        p = runner.project_dates(f, frozen, old, proof, omitted)
        assert p.date.dt.strftime('%Y-%m-%d').tolist() == old
        assert len(f) == 2 and omitted == [{'code': code, 'date': extra, 'volume': 0, 'amount': 0,
                                            'tradeStatus': '0', 'halt_raw_sha256': sha}]
    else:
        with pytest.raises(ValueError):
            runner.project_dates(f, frozen, old, proof, omitted)


def test_same_row_count_cannot_substitute_different_date():
    code = 'sz000001'
    old = ['2019-01-02']
    frozen = SimpleNamespace(code=code, rows=1, dates_sha256=runner.date_hash(old))
    with pytest.raises(ValueError, match='缺旧日'):
        runner.project_dates(frame(code, ['2019-01-03'], [0]), frozen, old, lambda *a: True, [])


@pytest.mark.parametrize('case', ['short', 'crossed', 'halted', 'bad_flag', 'exact_120'])
def test_short_nonmember_must_prove_dates_before_120_row_filter(case):
    code = 'sz000001'
    days = pd.bdate_range('2019-01-02', periods=120).strftime('%Y-%m-%d').tolist()
    old = days if case == 'exact_120' else days[:-1]
    spec = SimpleNamespace(code=code, rows=len(old), panel=int(case in ('bad_flag', 'exact_120')),
                           dates_sha256=runner.date_hash(old))
    have = days if case in ('crossed', 'halted') else old
    volumes = [2] * len(have)
    if case == 'halted':
        volumes[-1] = 0
    exclusions = []
    args = frame(code, have, volumes), spec, old, lambda c, d: '0' * 64, exclusions
    if case in ('crossed', 'bad_flag'):
        with pytest.raises(ValueError, match='额外日期|入池标志'):
            runner.panel_view(*args)
    elif case == 'exact_120':
        assert len(runner.panel_view(*args)) == 120
    else:
        assert runner.panel_view(*args) is None
        assert len(exclusions) == int(case == 'halted')


def csv_blob(columns, rows):
    f = io.StringIO(newline='')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows(rows)
    return f.getvalue().encode()


@pytest.mark.parametrize('variant', ['ok', 'wrong_year', 'conflict', 'missing', 'corrupt', 'wrong_code', 'fake_empty', 'bad_event_table'])
def test_corporate_action_coverage_requires_real_year_and_identical_raw_rows(tmp_path, variant):
    code = 'sz000001'
    row = [''] * len(runner.DIV)
    row[0] = 'sz.000001' if variant != 'wrong_code' else 'sh.000001'
    row[6] = '2026-06-12' if variant != 'wrong_year' else '2025-06-12'
    row[9] = '0.12345'
    second = row.copy()
    if variant == 'conflict':
        second[1] = '2026-02-01'
    cols = ['code', 'year', 'status', 'rows', 'raw_path', 'raw_sha256', 'detail']
    recs = []
    for year, rows in [('2025', []), ('2026', [row, second])]:
        name = f'dividend-queries/{code}/{year}.json'
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        status = 'queried' if rows and variant != 'fake_empty' else 'empty'
        q = {'source': runner.SOURCE_ID, 'code': 'sz.000001', 'year': year, 'yearType': 'operate',
             'fields': runner.DIV, 'rows': rows, 'status': status}
        path.write_text(json.dumps(q))
        sha = runner.digest(path.read_bytes()) if variant != 'corrupt' else '0' * 64
        recs.append([code, year, status, len(rows), name, sha, ''])
    if variant == 'missing':
        recs.pop(0)
    events = csv_blob(runner.DIV, [row] if variant != 'bad_event_table' else [])
    args = tmp_path, csv_blob(cols, recs), events, {code}, [2025, 2026]
    if variant != 'ok':
        with pytest.raises(ValueError):
            runner.read_actions(*args)
    else:
        data, keys = runner.read_actions(*args)
        assert keys == {(code, row[6])} and data.values.tolist() == [row]


def test_ready_missing_or_corrupt_halt_hash(tmp_path):
    ready = {'version': 'm2-acquisition-v2', 'status': 'complete', 'contract_sha256': runner.META_HASH['CONTRACT-v1.md'],
             'bars_provenance_sha256': runner.META_HASH['BARS-PROVENANCE-v2.md'],
             'frozen_meta_sha256': runner.META_HASH['frozen-meta.csv']}
    for key, name in runner.READY_FILES.items():
        blob = name.encode()
        (tmp_path / name).write_bytes(blob)
        ready[key] = runner.digest(blob)
    marker = tmp_path / 'READY.json'
    marker.write_text(json.dumps(ready))
    files, facts = runner.release(tmp_path)
    assert len(files) == 4 and facts['ready_sha256'] == runner.digest(marker.read_bytes())
    marker.write_text(json.dumps({**ready, 'version': 'm2-acquisition-v1'}))
    with pytest.raises(ValueError, match='version'):
        runner.release(tmp_path)
    marker.write_text(json.dumps(ready))
    (tmp_path / 'halt-evidence.csv').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA256'):
        runner.release(tmp_path)
    del ready['halt_evidence_sha256']
    marker.write_text(json.dumps(ready))
    with pytest.raises(KeyError):
        runner.release(tmp_path)


@pytest.mark.parametrize('variant', ['ok', 'missing_terms', 'old_projection', 'wrong_factor', 'blank_volume', 'wrong_stock'])
def test_sdk_raw_five_fields_and_cumulative_factor(variant):
    code, day = 'sz000001', '2019-01-02'
    bars = frame(code, [day], [2])
    bars['adj_factor'] = 1.25 if variant != 'wrong_factor' else 0.8
    factor = {'fields': runner.SDK_FACTORS, 'rows': [['sz.000001', day, '0.8', '1.25', '1.125']]}
    if variant == 'old_projection':
        factor = {'fields': ['code', 'dividOperateDate', 'backAdjustFactor'], 'rows': [['sz.000001', day, '1.25']]}
    raw = [day, 'sz.000002' if variant == 'wrong_stock' else 'sz.000001', '10', '10', '10', '10',
           '' if variant == 'blank_volume' else '200', '2000', '3']
    packet = {'source': runner.SOURCE_ID, 'code': code, 'start': runner.START, 'end': runner.END,
              'adjust_start': '1990-01-01', 'queries': [{'fields': runner.SDK_BARS, 'rows': [raw]}, factor]}
    args = bars, packet, code, set() if variant == 'missing_terms' else {(code, day)}, True
    if variant == 'ok':
        runner.sdk_match(*args)
    else:
        with pytest.raises(ValueError):
            runner.sdk_match(*args)


def test_run_build_isolates_and_rejects_unsupported_event(tmp_path):
    module = SimpleNamespace()
    actions, inputs, industry = object(), object(), {'a': 'b'}
    cfg = Config()
    def fail(config, *, use_cache, corporate_actions):
        assert config is cfg and not use_cache and corporate_actions is actions
        assert module.load_inputs(cfg) is inputs and module.ds.industry_map(cfg) == industry
        assert module.CACHE_DIR == tmp_path / 'cache' and module.OUT_DIR == tmp_path / 'unvalidated'
        (module.CACHE_DIR / 'factors.pkl').write_bytes(b'temporary')
        raise ValueError('持仓除权无法估值')
    module.build_pipeline = fail
    with pytest.raises(ValueError, match='持仓除权'):
        runner.run_build(module, tmp_path, {}, {}, cfg, inputs, industry, actions)
    assert not (tmp_path / 'output').exists() and not list(tmp_path.rglob('comparison.json'))


@pytest.mark.parametrize('variant', ['ok', 'wrong_factor', 'wrong_day', 'missing_output', 'nan'])
def test_verify_real_output_files_before_comparing(tmp_path, variant):
    cfg = Config()
    dates = ['2019-01-02', '2019-01-03']
    blobs = {'backtest_equity.csv': csv_blob(['date'], [[d] for d in dates]),
             'importance.csv': csv_blob(['factor'], [['f']]), 'preds.csv': csv_blob(['date'], [[dates[1]]])}
    old = {'model': {'n_factors': 1}, 'universe': {'stocks_in_panel': 2, 'date_range': dates},
           'performance': {'交易日数': 1}, 'diagnostics': {'成交笔数': 0}, 'config': cfg.to_dict()}
    for group in ('performance_raw_signal', 'performance_baseline_factor', 'ic', 'ic_baseline_factor'):
        old[group] = {'indicator': 0.1}
    new = json.loads(json.dumps(old))
    new['ic']['indicator'] = 0.2 if variant != 'nan' else float('nan')
    for name in ('signal.json', 'factors_meta.json', 'weights.csv', 'layers.csv', 'ic.csv'):
        (tmp_path / name).write_text('x')
    for name, content in blobs.items():
        (tmp_path / name).write_bytes(content)
    if variant == 'wrong_factor':
        (tmp_path / 'importance.csv').write_bytes(csv_blob(['factor'], [['g']]))
    if variant == 'wrong_day':
        (tmp_path / 'preds.csv').write_bytes(csv_blob(['date'], [[dates[0]]]))
    if variant == 'missing_output':
        (tmp_path / 'layers.csv').unlink()
    (tmp_path / 'run_summary.json').write_text(json.dumps(new))
    if variant != 'ok':
        with pytest.raises(ValueError):
            runner.verify_output(old, blobs, cfg, new, tmp_path)
    else:
        value = runner.verify_output(old, blobs, cfg, new, tmp_path)
        assert value['ic']['indicator'] == {'old': 0.1, 'new': 0.2, 'delta': 0.1}
