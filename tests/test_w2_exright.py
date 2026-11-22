"""真实除息窗口的独立价格参照；不是审计所指的 +11.1% / +3.45% 原例。"""
from decimal import Decimal

import pandas as pd
import pytest

from ashare.factors import build_matrices, clean_matrices, compute_factors


def test_pingan_ex_date_matches_baostock_backward_close():
    # 2026-11-22 取自 BaoStock。可复查 sz.000001，2024-06-11 至 06-17：
    # query_history_k_data_plus(..., frequency='d', adjustflag='3')，量为股、额为元。
    bars = pd.DataFrame([
        ('2024-06-11', 10.98, 11.02, 10.79, 10.87, 133762854, 1452841956.89),
        ('2024-06-12', 10.85, 10.91, 10.78, 10.88, 95742635, 1037927497.39),
        ('2024-06-13', 10.89, 10.93, 10.80, 10.80, 120350779, 1304403407.05),
        ('2024-06-14', 10.15, 10.23, 9.99, 10.18, 163799047, 1657433954.94),
        ('2024-06-17', 10.13, 10.16, 10.08, 10.10, 68121702, 688669914.12),
    ], columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    bars['date'] = pd.to_datetime(bars['date'])
    bars['volume'] /= 100  # 因子入口的单位为手
    # query_adjust_factor(..., '1990-01-01', '2024-06-17')：最近两个生效日
    # 2023-06-14、2024-06-14 的 backAdjustFactor；只在生效日向后使用。
    bars['adj_factor'] = [104.876880] * 3 + [112.368136] * 2
    code = 'sz000001'
    mats = clean_matrices(build_matrices({code: bars}))
    factors = compute_factors(mats)

    # 另查同一供应商 adjustflag='1' 的收盘价，不从生产因子或上面的乘数反算预期。
    closes = list(map(Decimal, ('1140.0116856000', '1141.0604544000', '1132.6703040000',
                                '1143.9076244800', '1134.9181736000')))
    ma5_div = closes[-1] / (sum(closes) / 5) - 1
    ret_1 = closes[-2] / closes[-3] - 1  # 除息日真实比例收益为正，原价比却是负数
    assert factors['ma5_div'].at[pd.Timestamp('2024-06-17'), code] == pytest.approx(float(ma5_div), rel=1e-6, abs=0)
    assert factors['ret_1'].at[pd.Timestamp('2024-06-14'), code] == pytest.approx(float(ret_1), rel=1e-6, abs=0)
