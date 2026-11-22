"""真实交易日驱动的行情同步；所有行情与网络边界均注入，写入仅在 tmp_path。"""
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, get_ident
from time import sleep
from datetime import date, datetime
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ashare import datasource as ds
from ashare.config import Config

FRI, MON, TUE = map(date.fromisoformat, ['2026-11-20', '2026-11-23', '2026-11-24'])
STOCK, INDEX = 'sh600795', 'sh000300'


def bars(code, days, prices=None):
    px = prices if prices is not None else [10.] * len(days)
    return pd.DataFrame({'date': pd.to_datetime(days), 'open': px, 'close': px, 'high': px,
                         'low': px, 'volume': 32891., 'amount': 33000000., 'adj_factor': 1., 'code': code})


class Query:
    def __init__(self, fields, rows, error='0'):
        self.fields, self.rows, self.error_code = fields, iter(rows), error
        self.error_msg = 'mock'

    def next(self):
        self.row = next(self.rows, None)
        return self.row is not None

    def get_row_data(self):
        return self.row


def mock_calendar(monkeypatch, mutate=lambda rows: rows, err='0'):
    calls = []
    closed = {'2024-02-09', '2024-02-12', '2024-02-13', '2024-02-14', '2024-02-15', '2024-02-16'}

    def trade_dates(start_date, end_date):
        calls.append((start_date, end_date))
        days = pd.date_range(start_date, end_date)
        rows = [(d.date().isoformat(), '1' if d.weekday() < 5 and str(d.date()) not in closed else '0') for d in days]
        return Query(['calendar_date', 'is_trading_day'], mutate(rows), err)

    fake = SimpleNamespace(login=lambda: SimpleNamespace(error_code='0'), logout=lambda: None,
                           query_trade_dates=trade_dates,
                           query_all_stock=lambda day: Query(['code', 'tradeStatus', 'code_name'], [('sh.600795', '1', '国电电力')]))
    monkeypatch.setitem(sys.modules, 'baostock', fake)
    return calls


@pytest.mark.parametrize('now,expect', [
    ('2026-11-23T06:59:59+00:00', FRI), ('2026-11-23T07:00:00+00:00', MON),
    ('2026-11-21T18:00:00+08:00', FRI), ('2024-02-15T16:00:00+08:00', date(2024, 2, 8)),
    ('2024-02-19T14:59:00+08:00', date(2024, 2, 8)), ('2024-02-19T15:00:00+08:00', date(2024, 2, 19)),
])
def test_target_is_last_closed_session_in_authoritative_calendar(monkeypatch, now, expect):
    calls = mock_calendar(monkeypatch)
    cal = ds.resolve_market_calendar(Config(start_date='2024-02-01'), now=datetime.fromisoformat(now))
    assert cal.target == expect and len(calls) == 1


@pytest.mark.parametrize('mutation,err', [(lambda rows: rows[:-1], '0'), (lambda rows: rows[:1] + rows, '0'),
                                        (lambda rows: rows[:-1] + [(rows[-1][0], 'x')], '0'),
                                        (lambda rows: [], '0'), (lambda rows: rows, '1')])
def test_incomplete_or_bad_calendar_fails_instead_of_guessing(monkeypatch, mutation, err):
    mock_calendar(monkeypatch, mutation, err)
    with pytest.raises((ValueError, RuntimeError)):
        ds.resolve_market_calendar(Config(start_date='2026-11-18'), now=datetime.fromisoformat('2026-11-23T16:00:00+08:00'))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, 'RAW_DIR', tmp_path)
    directory = ds._cache_path(STOCK).parent
    directory.mkdir()
    return Config(start_date='2026-11-20'), directory


def simulate(monkeypatch, producer):
    calls = []

    def fetch(code, cfg, start=None, end=None):
        calls.append((code, start, end))
        return producer(ds.to_tencent(code), start, end)

    monkeypatch.setattr(ds, 'fetch_kline', fetch)
    return calls


def sync(code, cfg, days=(FRI, MON), force=False, suspended=()):
    cal = ds.MarketCalendar(pd.DatetimeIndex(days), {days[-1]: frozenset(suspended)})
    return ds.download_all([code], cfg, force=force, workers=1, calendar=cal).iloc[0]


def seed(path, code=STOCK, days=(FRI,)):
    f = path / f'{code}.csv'
    bars(code, days).to_csv(f, index=False)
    return f


def test_friday_to_monday_appends_with_overlap_then_no_price_call(env, monkeypatch):
    cfg, path = env
    f = seed(path)
    calls = simulate(monkeypatch, lambda c, *_: bars(c, [FRI, MON, TUE], [10., 11., 99.]))
    first = sync('sh.600795', cfg)
    assert first['status'] == 'incremental' and first['fresh'] and first['last_date'] == str(MON)
    assert calls == [('sh600795', FRI, MON)]
    assert pd.read_csv(f)['date'].tolist() == [str(FRI), str(MON)]
    original = f.read_bytes()
    assert sync(STOCK, cfg)['status'] == 'cached'
    assert f.read_bytes() == original and len(calls) == 1


@pytest.mark.parametrize('problem,expected', [('empty', 'missing_target'), ('old', 'missing_target'),
                                             ('invalid', 'invalid'), ('zero', 'no_trade_unconfirmed'), ('timeout', 'error')])
def test_bad_download_never_overwrites_or_reports_old_rows_as_fresh(env, monkeypatch, problem, expected):
    cfg, path = env
    f = seed(path); before = f.read_bytes()

    def quote(c, *args):
        if problem == 'timeout':
            raise TimeoutError('mock timeout')
        if problem in ('empty', 'old'):
            return bars(c, [] if problem == 'empty' else [FRI])
        df = bars(c, [FRI, MON]); df.loc[1, 'open' if problem == 'invalid' else 'volume'] = np.nan if problem == 'invalid' else 0
        return df

    simulate(monkeypatch, quote)
    row = sync(STOCK, cfg)
    assert (row.status, row.fresh, row.rows, row.last_date) == (expected, False, 1, str(FRI))
    assert f.read_bytes() == before


def test_only_positive_security_specific_halt_evidence_is_excused(env, monkeypatch):
    cfg, path = env
    f = seed(path); before = f.read_bytes()
    simulate(monkeypatch, lambda c, *_: bars(c, []))
    halted = sync(STOCK, cfg, suspended=[STOCK]); unknown = sync('sz000001', cfg, suspended=[STOCK])
    index = sync(INDEX, cfg, suspended=[INDEX])
    assert halted.status == 'suspended' and not halted.fresh and f.read_bytes() == before
    assert unknown.status == index.status == 'missing_target' and not index.fresh
    assert not (path / f'{INDEX}.csv').exists()


def test_overlap_price_rebased_rebuilds_before_any_replace(env, monkeypatch):
    cfg, path = env
    past = date(2026, 11, 16)
    days = [past, date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19), FRI, MON]
    f = seed(path, days=days[:-1]); before = f.read_bytes()
    src = bars(STOCK, days, [5., 5., 5., 5., 5., 5.2])
    calls = simulate(monkeypatch, lambda c, start, end: src[src.date >= pd.Timestamp(start)].copy())
    result = sync(STOCK, cfg, days=days)
    assert len(calls) == 2 and calls[-1][1:] == (past, MON)
    assert result.status == 'rebased' and result.fresh and f.read_bytes() != before
    assert pd.read_csv(f)['close'].tolist() == [5., 5., 5., 5., 5., 5.2]


@pytest.mark.parametrize('force', [False, True])
def test_incomplete_force_or_rebase_cannot_shrink_histories(env, monkeypatch, force):
    cfg, path = env
    past = date(2026, 11, 19)
    f = seed(path, days=[past, FRI]); before = f.read_bytes()
    simulate(monkeypatch, lambda c, *_: bars(c, [FRI, MON], [5., 5.2]))
    row = sync(STOCK, cfg, days=[past, FRI, MON], force=force)
    assert not row.fresh and f.read_bytes() == before and row.rows == 2


def test_incremental_cannot_change_a_previously_traded_anchor_to_no_trade(env, monkeypatch):
    cfg, path = env
    f = seed(path); before = f.read_bytes()
    def changed(c, *_):
        df = bars(c, [FRI, MON]); df.loc[0, 'volume'] = 0
        return df
    simulate(monkeypatch, changed)
    row = sync(STOCK, cfg)
    assert not row.fresh and row.status == 'invalid' and f.read_bytes() == before


def test_cached_stock_with_a_bar_on_official_closed_day_is_not_fresh(env, monkeypatch):
    cfg, path = env
    saturday = date(2026, 11, 21)
    f = seed(path, days=[FRI, saturday, MON]); before = f.read_bytes()
    simulate(monkeypatch, lambda *_: pytest.fail('must not refresh invalid cache silently'))
    row = sync(STOCK, cfg)
    assert not row.fresh and row.status == 'invalid' and f.read_bytes() == before


def test_mandatory_index_must_cover_each_new_session(env, monkeypatch):
    cfg, path = env
    f = seed(path, INDEX); before = f.read_bytes()
    simulate(monkeypatch, lambda c, *_: bars(c, [FRI, TUE]))
    row = sync(INDEX, cfg, days=[FRI, MON, TUE])
    assert row.status == 'missing_sessions' and not row.fresh and f.read_bytes() == before


def test_confirmed_intermediate_halt_does_not_require_a_fictitious_bar(env, monkeypatch):
    cfg, path = env
    seed(path)
    cal = ds.MarketCalendar(pd.DatetimeIndex([FRI, MON, TUE]), {MON: frozenset({STOCK})})
    simulate(monkeypatch, lambda c, *_: bars(c, [FRI, TUE]))
    row = ds.download_all([STOCK], cfg, workers=1, calendar=cal).iloc[0]
    assert row.status == 'incremental' and row.fresh and row.rows == 2


def test_new_history_and_force_validate_full_coverage_and_stop_before_tomorrow(env, monkeypatch):
    cfg, path = env
    calls = simulate(monkeypatch, lambda c, *_: bars(c, [FRI, MON, TUE]))
    assert sync(INDEX, cfg).status == 'full'
    exact = pd.read_csv(path / f'{INDEX}.csv')
    assert exact.date.tolist() == [str(FRI), str(MON)]
    assert sync(INDEX, cfg, force=True).status == 'full'
    assert len(calls) == 2 and all(call[1:] == (FRI, MON) for call in calls)


def test_atomic_replace_failure_preserves_file_and_cleans_only_staged_data(env, monkeypatch):
    cfg, path = env
    f = seed(path); before = f.read_bytes()
    simulate(monkeypatch, lambda c, *_: bars(c, [FRI, MON]))
    def cannot_replace(*_):
        raise OSError('read-only destination')
    monkeypatch.setattr(ds.os, 'replace', cannot_replace)
    row = sync(STOCK, cfg)
    assert row.status == 'error' and not row.fresh and f.read_bytes() == before
    assert set(path.iterdir()) == {f}


def test_calendar_also_rejects_holiday_at_start_before_first_actual_session(env, monkeypatch):
    cfg, path = env
    holiday = date(2024, 1, 1); first = date(2024, 1, 2)
    f = seed(path, days=[holiday, first]); before = f.read_bytes()
    cal = ds.MarketCalendar(pd.DatetimeIndex([first]), covered_from=holiday)
    row = ds.download_all([STOCK], cfg, workers=1, calendar=cal).iloc[0]
    assert row.status == 'invalid' and not row.fresh and f.read_bytes() == before


def test_bad_target_volume_is_not_excused_by_a_halt_flag(env, monkeypatch):
    cfg, path = env
    f = seed(path); before = f.read_bytes()
    def negative(c, *_):
        data = bars(c, [FRI, MON]); data.loc[1, 'volume'] = -1
        return data
    simulate(monkeypatch, negative)
    row = sync(STOCK, cfg, suspended=[STOCK])
    assert row.status == 'invalid' and not row.fresh and f.read_bytes() == before


def quote_sdk(monkeypatch, events, *, fail_page=None, row_flag='3'):
    def response(kind, fields, rows):
        q = Query(fields, rows)
        q.adjustflag = ''  # 真实 SDK 的结果头为空；不复权证据在每行字段。
        if kind == fail_page:  # 0.9.40 网络断页会停留在满页，且不修改 error_code
            q.data, q.per_page_count = rows, str(len(rows))
        return q

    def quotes(code, fields, start_date, end_date, frequency, adjustflag):
        assert (frequency, adjustflag) == ('d', '3')
        rows = []
        for day in (FRI, MON, TUE):
            raw = dict(date=str(day), code=code, open='9', high='11', low='8', close='10',
                       volume='12345', amount='116700.52', adjustflag=row_flag)
            if start_date <= str(day) <= end_date:
                rows.append([raw[k] for k in fields.split(',')])
        return response('quotes', fields.split(','), rows)

    def factors(code, start_date, end_date):
        assert start_date == '1990-01-01'
        assert code != 'sh.000300', '指数不查询复权因子'
        return response('events', ['code', 'dividOperateDate', 'backAdjustFactor'],
                        [(code, d, v) for d, v in events])

    monkeypatch.setitem(sys.modules, 'baostock', SimpleNamespace(
        login=lambda: SimpleNamespace(error_code='0'), logout=lambda: None,
        query_history_k_data_plus=quotes, query_adjust_factor=factors))


@pytest.mark.parametrize('code,expected', [(STOCK, [1.5, 2.]), (INDEX, [1., 1.])])
def test_sdk_to_disk_to_panel_causal_factor_and_real_turnover(env, monkeypatch, code, expected):
    cfg, _ = env
    quote_sdk(monkeypatch, [('2001-01-01', '1.5'), (str(MON), '2'), (str(TUE), '3')])
    assert sync(code, cfg).fresh
    panel = ds.load_panel([code], cfg, min_rows=1)[code]
    assert panel.adj_factor.tolist() == expected
    assert panel.open.tolist() == [9, 9] and panel.close.tolist() == [10, 10]
    assert panel.amount.tolist() == [116700.52] * 2
    assert panel.volume.tolist() == [123.45] * 2


def test_adjusted_sdk_row_rejected_even_if_request_was_raw(env, monkeypatch):
    cfg, path = env
    f = seed(path); before = f.read_bytes()
    quote_sdk(monkeypatch, [], row_flag='2')
    row = sync(STOCK, cfg)
    assert not row.fresh and '未返回不复权' in row.detail and f.read_bytes() == before


def test_sdk_session_is_exclusive_across_quotes_calendar_and_halts(env, monkeypatch):
    cfg, _ = env
    mock_calendar(monkeypatch)
    calendar_api = sys.modules['baostock']
    quote_sdk(monkeypatch, [])
    sdk = sys.modules['baostock']
    sdk.query_all_stock = calendar_api.query_all_stock
    sdk.query_trade_dates = calendar_api.query_trade_dates
    owner = []; gate = Barrier(4)

    def login():
        assert not owner, '另一个线程的 socket 仍在使用'
        owner.append(get_ident())
        sleep(.01)  # 在 SDK 调用边界让出线程，强制重叠请求竞争同一会话。
        return SimpleNamespace(error_code='0')

    def logout():
        assert owner.pop() == get_ident()

    sdk.login, sdk.logout = login, logout

    def run(action):
        gate.wait(timeout=5)
        return action()

    actions = [lambda: ds.fetch_kline(STOCK, cfg, start=FRI, end=MON),
               lambda: ds.fetch_kline(INDEX, cfg, start=FRI, end=MON),
               lambda: ds.resolve_market_calendar(cfg, now=datetime(2026, 11, 23, 16)),
               lambda: ds._historical_halts(FRI)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, actions))
    assert len(results[0]) == len(results[1]) == 2
    assert results[2].target == MON and results[3] == frozenset() and not owner


@pytest.mark.parametrize('where', ['quotes', 'events'])
def test_silent_sdk_page_interrupt_does_not_replace_cache(env, monkeypatch, where):
    cfg, path = env
    f = seed(path); before = f.read_bytes()
    quote_sdk(monkeypatch, [('2001-01-01', '1.5')], fail_page=where)
    row = sync(STOCK, cfg)
    assert not row.fresh and '分页中断' in row.detail and f.read_bytes() == before


def test_cli_bounded_scope_checks_reference_indices_and_returns_nonzero(tmp_path, monkeypatch, caplog):
    module = run_path(str(Path(__file__).resolve().parents[1] / 'scripts/01_fetch.py'))
    main = module['main']; monkeypatch.setitem(main.__globals__, 'CACHE_DIR', tmp_path)
    cal = ds.MarketCalendar(pd.DatetimeIndex([FRI, MON]), stocks={'sh.600795': '国电电力', 'sz.000001': '平安银行'}); calls = []
    monkeypatch.setattr(ds, 'resolve_market_calendar', lambda cfg: (calls.append(cfg) or cal))
    monkeypatch.setattr(ds, 'RAW_DIR', tmp_path)
    simulate(monkeypatch, lambda c, *_: bars(c, [] if c == INDEX else [FRI, MON]))
    monkeypatch.setattr(sys, 'argv', ['01_fetch.py', '--limit', '1', '--workers', '1'])
    assert main() != 0 and len(calls) == 1
    stats = pd.read_csv(tmp_path / 'fetch_stats.csv')
    assert set(stats.code) == {STOCK, INDEX, 'sh000905', 'sh000852', 'sh000001'}
    assert stats.loc[stats.code == INDEX, 'status'].item() == 'missing_target'
    assert 'record' in caplog.text


@pytest.mark.parametrize('mode,expected,invoked', [
    ('default', 0, False), ('success', 17, True), ('halted', 17, True), ('missing', 1, False),
    ('index_late', 1, False), ('old_target', 1, False), ('midnight', 1, False), ('different_names', 1, False),
    ('changed_names', 1, False), ('no_roster', 1, False), ('no_master', 1, False),
])
def test_record_only_on_request_for_verified_whole_market_today(tmp_path, monkeypatch, mode, expected, invoked):
    main = run_path(str(Path(__file__).resolve().parents[1] / 'scripts/01_fetch.py'))['main']
    actual_calls = []; close_checks = []
    def closed(day):
        close_checks.append(day)
        return mode != 'old_target' and not (mode == 'midnight' and len(close_checks) > 1)
    def record():
        actual_calls.append('manually requested'); return 17
    monkeypatch.setitem(main.__globals__, 'CACHE_DIR', tmp_path)
    monkeypatch.setitem(main.__globals__, 'is_today_closed', closed)
    monkeypatch.setitem(main.__globals__, 'run_record', record)
    monkeypatch.setattr(ds, 'RAW_DIR', tmp_path)
    f = tmp_path / 'master_list.csv'
    def master(name):
        pd.DataFrame([{'code': 'sh.600795', 'name': name, 'kind': 'stock'}]).to_csv(f, index=False)
    if mode != 'no_master':
        master('过期名称' if mode == 'different_names' else '国电电力')
    cal = ds.MarketCalendar(pd.DatetimeIndex([FRI, MON]),
                            {MON: frozenset({STOCK} if mode == 'halted' else ())},
                            stocks={} if mode == 'no_roster' else {'sh.600795':'国电电力'})
    monkeypatch.setattr(ds, 'resolve_market_calendar', lambda cfg: cal)
    def fetch(c, *_):
        if mode == 'changed_names':
            master('下载时被修改')
        missing = (c == STOCK and mode in ('halted', 'missing')) or (c == INDEX and mode == 'index_late')
        return bars(c, [FRI] if missing else [FRI, MON])
    simulate(monkeypatch, fetch)
    args = ['01_fetch.py', '--workers', '1'] + ([] if mode == 'default' else ['--record'])
    monkeypatch.setattr(sys, 'argv', args)
    assert main() == expected and bool(actual_calls) == invoked


def test_limit_and_record_are_mutually_exclusive_before_any_work(monkeypatch):
    main = run_path(str(Path(__file__).resolve().parents[1] / 'scripts/01_fetch.py'))['main']
    monkeypatch.setattr(ds, 'resolve_market_calendar', lambda *_: pytest.fail('must reject conflicting options first'))
    monkeypatch.setattr(sys, 'argv', ['01_fetch.py', '--limit', '1', '--record'])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code != 0


def test_legacy_negative_qfq_file_is_preserved_and_never_served(env, monkeypatch):
    cfg, path = env
    legacy = path.parent / f'{STOCK}.csv'
    bars(STOCK, [FRI], [-2.919]).drop(columns=['amount', 'adj_factor']).to_csv(legacy, index=False)
    before = legacy.read_bytes()
    with pytest.raises(ValueError, match='旧腾讯缓存'):
        ds.load_panel([STOCK], cfg, min_rows=1)
    quote_sdk(monkeypatch, [])
    assert sync(STOCK, cfg).status == 'full'
    assert ds.load_cached(STOCK).open.tolist() == [9., 9.]
    assert legacy.read_bytes() == before


@pytest.mark.parametrize('problem', ['negative', 'legacy', 'no_amount', 'factor'])
def test_new_cache_rejects_invalid_deep_history_at_load(env, problem):
    cfg, path = env
    days = pd.bdate_range(end=str(FRI), periods=10)
    df = bars(STOCK, days)
    if problem == 'legacy':
        df = df.drop(columns=['amount', 'adj_factor'])
    else:
        field, val = {'negative': ('low', -1), 'no_amount': ('amount', np.nan), 'factor': ('adj_factor', 0)}[problem]
        df.loc[0, field] = val
    f = path / f'{STOCK}.csv'; df.to_csv(f, index=False); before = f.read_bytes()
    with pytest.raises(ValueError):
        ds.load_cached(STOCK)
    assert not sync(STOCK, cfg).fresh and f.read_bytes() == before


def test_negative_value_in_current_equity_anchor_is_not_tradable(env, monkeypatch):
    cfg, path = env
    f = seed(path); before = f.read_bytes()
    simulate(monkeypatch, lambda c, *_: bars(c, [FRI, MON], [-2.9, 5.]))
    row = sync(STOCK, cfg)
    assert row.status == 'invalid' and not row.fresh and f.read_bytes() == before
