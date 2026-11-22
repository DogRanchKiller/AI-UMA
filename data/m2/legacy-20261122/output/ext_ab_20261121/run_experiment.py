"""Run from repository root; uses archived baseline config and local data."""
import importlib.util
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location('build', 'scripts/02_build.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
root = Path(__file__).resolve().parent
b.OUT_DIR = b.CACHE_DIR = root
original = b.ed.build_ext_factors


def event_adjusted(cfg, cal):
    factors = original(cfg, cal)
    events = b.ed._load('lhb')
    positions = cal.searchsorted(pd.to_datetime(events.trade_date), side='right')
    events = events.loc[positions < len(cal)].copy()
    events['avail'] = cal[positions[positions < len(cal)]]
    groups = events.groupby(['avail', 'code'])
    # Use the price universe, not the list of stocks that ever appear on the board.
    codes = b.ds.master_list(cfg).code.map(b.ds.to_tencent).unique()
    for key, series in [('ext_lhb_net_20', groups.lhb_net.sum()),
                        ('ext_lhb_cnt_20', groups.size())]:
        daily = series.unstack().reindex(index=cal, columns=codes).fillna(0)
        factors[key] = daily.rolling(20, min_periods=5).sum().astype('float32')
    assert len(factors) == 6
    return factors


b.ed.build_ext_factors = event_adjusted
cfg = b.Config.load(Path('output/config_used.json'))
cfg.use_ext_factors = True
_, summary = b.build_pipeline(cfg, use_cache=False)
a = pd.read_csv('output/preds.csv', usecols=['date', 'code'])
c = pd.read_csv(root / 'preds.csv', usecols=['date', 'code'])
assert a.equals(c), 'unmatched prediction rows'
assert summary['model']['n_factors'] == 49
print('Paired predictions:', len(a), flush=True)
