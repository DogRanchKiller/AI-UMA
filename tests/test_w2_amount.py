"""重放真实 SDK 行，经下载、落盘、读回、矩阵后逐条与交易所独立日报比对。"""
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ashare import datasource as ds
from ashare.config import Config
from ashare.factors import build_matrices


FIXTURES = Path(__file__).parent / 'fixtures' / 'w2_amount'
CODES = ('000001', '000002', '000063', '000333', '000651')
DATES = tuple(f'2024-03-0{n}' for n in range(4, 9))


def records(name):
    with (FIXTURES / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


class Result:
    error_code = '0'
    error_msg = 'success'
    adjustflag = ''  # SDK 0.9.4 的结果级属性为空；有效标记在查询返回的每一行。

    def __init__(self, fields, rows):
        self.fields = fields
        self.rows = iter(rows)

    def next(self):
        self.row = next(self.rows, None)
        return self.row is not None

    def get_row_data(self):
        return [self.row[f] for f in self.fields]


def test_each_cached_matrix_amount_matches_szse_daily_turnover(tmp_path, monkeypatch):
    bars, events = records('baostock.csv'), records('adjust.csv')
    reference = records('szse.csv')
    pairs = {(code, day) for code in CODES for day in DATES}
    assert len(bars) == len(reference) == len(pairs) == 25
    assert {(r['code'], r['date']) for r in bars} == {('sz.' + c, d) for c, d in pairs}
    assert {(r['zqdm'], r['jyrq']) for r in reference} == pairs

    def daily(code, fields, start_date, end_date, frequency, adjustflag):
        assert (start_date, end_date, frequency, adjustflag) == (DATES[0], DATES[-1], 'd', '3')
        return Result(fields.split(','), (r for r in bars if r['code'] == code))

    def adjustments(code, start_date, end_date):
        assert end_date == DATES[-1]
        return Result(list(events[0]), (r for r in events if r['code'] == code
                                        and start_date <= r['dividOperateDate'] <= end_date))

    sdk = SimpleNamespace(login=lambda: SimpleNamespace(error_code='0'), logout=lambda: None,
                          query_history_k_data_plus=daily, query_adjust_factor=adjustments)
    monkeypatch.setitem(sys.modules, 'baostock', sdk)
    monkeypatch.setattr(ds, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(ds.requests.Session, 'request', lambda *_a, **_kw: pytest.fail('offline replay'))
    cfg = Config(start_date=DATES[0], end_date=DATES[-1])
    codes = ['sz' + c for c in CODES]
    calendar = ds.MarketCalendar(pd.DatetimeIndex(DATES))
    statuses = ds.download_all(codes, cfg, workers=1, calendar=calendar)
    assert statuses['fresh'].all(), statuses.to_string(index=False)
    mats = build_matrices(ds.load_panel(codes, cfg, min_rows=5))

    for r in bars:
        key = (pd.Timestamp(r['date']), r['code'].replace('.', ''))
        # 卡住现成矩阵浮点精度以内的股→手及金额原值；不能改用收盘价近似。
        assert float(mats['volume'].at[key]) == pytest.approx(int(r['volume']) / 100, rel=1e-7)
        assert float(mats['amount'].at[key]) == pytest.approx(float(r['amount']), rel=1e-7)

    errors = []
    for r in reference:
        code, day = r['zqdm'], r['jyrq']
        published = float(r['cjje'].replace(',', '')) * 10_000  # 深交所成交金额(万元) → 元
        actual = float(mats['amount'].at[pd.Timestamp(day), 'sz' + code])
        error = abs(actual - published) / published
        assert error < 0.01, (code, day, actual, published, error)
        errors.append((error, code, day, actual, published))
    error, code, day, actual, published = max(errors)
    print(f'25 pairs; max relative error={error:.9%} ({code} {day}, matrix={actual}, SZSE={published})')
