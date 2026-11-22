"""高点附近的放量/缩量：严格按事先冻结的三种口径，离线复核，不更新模型或名单。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = 'output/kline_volume_20261122'
PROTOCOL = EXP + '/比较口径.md'
OLD = 'output/trend_pairs_20261121/results'
ENGINE = 'scripts/12_trend_pairs.py'
FROZEN = {
    PROTOCOL: '69ca2ac04ffd2bcc3233b8e8d19e91f3462e65fdc8c2cd6c691b83b62a3ad9f9',
    OLD + '/manifest.json': '7577dc3d89d877f1bf4bad2a3e1bdd0708b4b991a5ed1e4f5fabf520d7f13036',
    ENGINE: 'afc28c1b65598be840096067b5bdeab3d295440947cf32e1205e3856ef5d1778',
}
SPECS = ('near3', 'near01', 'prior_break')
ELIGIBLE = ('strong', 'unconfirmed')
BAR = ('open', 'high', 'low', 'close', 'volume')
PRICE_STATS = (*BAR, 'high60', 'prior_high60', 'volume_mean20', 'volume_ratio')
POOL_COLS = ['date', 'gate', 'code', 'name', 'rank', 'state']
PLAN = ['spec', 'date', 'gate', 'expand_code', 'expand_rank', 'contract_code', 'contract_rank', 'rank_gap']
LEG = [f'{prefix}_{attr}' for prefix in ('expand', 'contract', 'index') for attr in ('entry_open', 'exit_open', 'return')]
COVERAGE = ['spec','role','date','gate','n_pool','n_strong','n_unconfirmed','n_unknown','n_ineligible',
            'n_unclassifiable','n_outside','n_flat','n_expand','n_contract','n_pairs']
LOCK = ('plan.csv', 'plan_dates.csv')
PLAN_STAMP = 'plan.sha256'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f'冻结文件指纹不符，停止：{path.name}')
    return data  # 校验和解析同一字节，不接受二次打开期间换源。


@lru_cache(maxsize=1)
def engine():
    checked(ROOT / ENGINE, FROZEN[ENGINE])
    return run_path(str(ROOT / ENGINE))


def role(spec):
    return 'primary' if spec == 'near3' else 'secondary'


def classify(frame, calendar, day):
    """仅取该日 OHLCV，以及本日/此前完整窗口的 high/volume；永不压缩交易日。"""
    pos = calendar.get_loc(day)
    window = frame.reindex(calendar[max(0, pos-60):pos+1])
    today = window.iloc[-1]
    result = {c: float(today[c]) for c in BAR}
    result.update(dict.fromkeys(PRICE_STATS[len(BAR):], np.nan))
    result.update(dict.fromkeys(SPECS, 'unclassifiable'))
    vals = today[list(BAR)].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or not (vals > 0).all() or not (today.low <= min(today.open, today.close)
                                                                 <= max(today.open, today.close) <= today.high):
        return result

    def high(window60):
        values = window60[['high', 'volume']].to_numpy(dtype=float)
        return (float(values[:,0].max()) if len(values) == 60 and np.isfinite(values).all()
                and (values > 0).all() else np.nan)

    h = high(window.iloc[-60:]); prev = high(window.iloc[:-1])
    result.update(high60=h, prior_high60=prev)
    volumes = window.volume.iloc[-21:-1].to_numpy(dtype=float)
    if len(volumes) != 20 or not np.isfinite(volumes).all() or not (volumes > 0).all():
        return result
    # 十进制近边界：不用二进制浮点的阈值乘积/相除决定“恰等”；不引入容差带。
    total = sum(Decimal(str(v)) for v in volumes)
    now20 = Decimal(str(today.volume)) * 20
    group = 'expand' if now20 > total else 'contract' if now20 < total else 'flat'
    result.update(volume_mean20=float(total/20), volume_ratio=float(now20/total))
    cl = Decimal(str(today.close))
    if np.isfinite(h):
        top = Decimal(str(h))
        result.update(near3=group if cl >= Decimal('.97')*top else 'outside',
                      near01=group if cl >= Decimal('.999')*top else 'outside')
    if np.isfinite(prev):
        result['prior_break'] = group if cl > Decimal(str(prev)) else 'outside'
    return result


class Quotes:
    """保持两个阶段独立：规划阶段只解析收盘前所需 high/volume 和评分日自身 OHLC。"""
    def __init__(self, hashes):
        self.hashes, self.used = hashes, {}

    def _bytes(self, code, phase):
        if code not in self.hashes or not re.fullmatch(r'(?:sh|sz)[0-9]{6}', code):
            raise ValueError(f'代码不在冻结行情清单：{code!r}')
        data = checked(ROOT / 'data/raw' / f'{code}.csv', self.hashes[code])
        self.used.setdefault(code, set()).add(phase)
        return data

    @staticmethod
    def _dates(data, code):
        table = pd.read_csv(io.BytesIO(data), usecols=['date','code'], parse_dates=['date'])
        idx = pd.DatetimeIndex(table.date)
        if (not table.code.eq(code).all() or idx.empty or idx.hasnans or not idx.is_unique
                or not idx.is_monotonic_increasing or not idx.equals(idx.normalize())):
            raise ValueError(f'{code} 日历或证券代码异常')
        return idx

    def calendar(self, code):
        return self._dates(self._bytes(code, 'calendar'), code)

    def signal_bars(self, code, calendar, scores):
        data = self._bytes(code, 'signals')
        raw_dates = self._dates(data, code)
        wanted = set()
        for t in calendar.get_indexer(scores):
            if t < 0:
                raise ValueError('评分日缺少冻结指数的交易日期')
            wanted.update(calendar[max(0,t-60):t+1])

        def project(fields, dates):
            skip = np.flatnonzero(~raw_dates.isin(dates)) + 1  # CSV 有一行表头。
            return pd.read_csv(io.BytesIO(data), usecols=['date', *fields], parse_dates=['date'], skiprows=skip,
                               float_precision='round_trip').set_index('date')

        # 同一原文件只开一次。仅评分日读取 open/low/close，不把随后端点传入分组函数。
        return project(['high','volume'], wanted).join(project(['open','low','close'], scores), how='left')

    @lru_cache(maxsize=192)
    def _outcome_bars(self, code):
        data = self._bytes(code, 'endpoints')
        self._dates(data, code)
        return pd.read_csv(io.BytesIO(data), usecols=['date','open','volume'], parse_dates=['date'],
                           float_precision='round_trip').set_index('date')

    def open_at(self, code, day):
        return engine()['quote_open'](self._outcome_bars(code), day)


def audit_pool(base, predictions):
    if (not set(base.state).issubset({*ELIGIBLE, 'unknown','ineligible'}) or len(base) != 12085
            or predictions.date.nunique() != 72 or set(base.date) != set(predictions.date)):
        raise ValueError('输入不符：必须是已核验的72个评分日、12085行原池')
    grouped = base.groupby('date')
    for day, cross in predictions.groupby('date'):
        ordered = cross.sort_values(['score','code'], ascending=[False, True]).iloc[:len(cross)//20]
        pool = grouped.get_group(day).sort_values('rank')
        if (pool['rank'].tolist() != list(range(1, len(ordered)+1)) or pool.code.tolist() != ordered.code.tolist()
                or pool.gate.nunique(dropna=False) != 1 or pd.isna(pool.gate.iloc[0])):
            raise ValueError(f'{day.date()} 既有候选行不对应完整前5%原名次或当日市场状态')


def classify_pool(base, calendar, quotes):
    records = []
    for code, samples in base[POOL_COLS].groupby('code', sort=True):
        frame = quotes.signal_bars(code, calendar, pd.DatetimeIndex(samples.date))
        for row in samples.to_dict('records'):
            obs = classify(frame, calendar, row['date'])
            if row['state'] not in ELIGIBLE:
                obs.update(dict.fromkeys(SPECS, row['state']))
            records.append({**row, **obs})
    return pd.DataFrame(records, columns=[*POOL_COLS,*PRICE_STATS,*SPECS]).sort_values(['date','rank'])


def make_plans(pool):
    """同一规格、同日，用原唯一名次做双边不放回匹配；不读取旧的 paired 或任何结果列。"""
    plans, coverage = [], []
    for day, cross in pool.groupby('date', sort=True):
        gate = bool(cross.gate.iloc[0]); by_rank = cross.set_index('rank')
        state = cross.state.value_counts()
        for spec in SPECS:
            valid = cross.loc[cross.state.isin(ELIGIBLE)]
            labels = valid[spec].value_counts()
            ranks = engine()['match_ranks'](valid.loc[valid[spec] == 'expand','rank'],
                                            valid.loc[valid[spec] == 'contract','rank'])
            for a, b in ranks:
                plans.append([spec, day, gate, by_rank.at[a,'code'], a, by_rank.at[b,'code'], b, abs(a-b)])
            counts = [state.get(k,0) for k in ('strong','unconfirmed','unknown','ineligible')]
            groups = [labels.get(k,0) for k in ('unclassifiable','outside','flat','expand','contract')]
            if sum(groups) != len(valid):
                raise ValueError(f'{spec} 的合格池出现未定义组别')
            coverage.append([spec,role(spec),day,gate,len(cross),*counts,*groups,len(ranks)])
    return pd.DataFrame(plans, columns=PLAN), pd.DataFrame(coverage, columns=COVERAGE)


def write_table(out, name, tbl):
    tbl.to_csv(out/name, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d', lineterminator='\n', mode='x')


def lock_plans(out, plans, dates):
    for name, table in zip(LOCK, (plans[PLAN], dates[COVERAGE])):
        write_table(out, name, table)
    hashes = {n:digest((out/n).read_bytes()) for n in LOCK}
    with (out/PLAN_STAMP).open('x', encoding='utf8') as handle:
        handle.write(''.join(f'{sha}  {name}\n' for name,sha in hashes.items()))
    return hashes


def load_plans(out):
    lines = (out/PLAN_STAMP).read_text(encoding='utf8').splitlines()
    if len(lines) != len(LOCK):
        raise ValueError('计划指纹不完整')
    tables = []
    for name, expected_cols, line in zip(LOCK, (PLAN, COVERAGE), lines):
        sha, fname = line.split('  ')
        if fname != name or not re.fullmatch('[a-f0-9]{64}', sha):
            raise ValueError('计划指纹或文件名不符')
        table = pd.read_csv(io.BytesIO(checked(out/name,sha)), parse_dates=['date'])
        if table.columns.tolist() != expected_cols:
            raise ValueError(f'{name} 不是本研究的无收益计划')
        tables.append(table)
    return tables


def daily_values(good):
    expanded = good.rename(columns={'expand_return':'strong_return','contract_return':'control_return'})
    metrics = engine()['daily_values'](expanded)
    return {k.replace('strong','expand').replace('control','contract'): v for k,v in metrics.items()}


def subgroup_summaries(daily):
    results = []
    metrics = list(daily_values(pd.DataFrame()))
    for spec in SPECS:
        dates = daily[daily.spec == spec].sort_values('date')
        mature = dates[dates.mature]
        for period in ['all', *sorted(set(dates.date.dt.year.astype(str)))]:
            for regime in ('all','on','off'):
                def keep(frame):
                    year_ok = frame.date.notna() if period == 'all' else frame.date.dt.year.eq(int(period))
                    gate_ok = frame.gate.notna() if regime == 'all' else frame.gate.eq(regime == 'on')
                    return year_ok & gate_ok
                sub = dates[keep(dates)]; good = sub[sub.n_complete > 0]
                lo, hi = engine()['block_hit_interval'](mature.hit_diff.where(keep(mature)))
                complete = int(sub.n_complete.sum())
                row = dict(spec=spec, role=role(spec), period=period, regime=regime, score_dates=len(sub),
                           mature_dates=int(sub.mature.sum()), effective_dates=len(good), pairs_planned=int(sub.n_pairs.sum()),
                           pairs_complete=complete, pairs_missing=int(sub.n_missing.sum()), pairs_pending=int(sub.n_pending.sum()),
                           up_expand=int(sub.up_expand.sum()), up_contract=int(sub.up_contract.sum()),
                           small_sample=len(good) < 10 or complete < 30, hit_diff_ci_low=lo, hit_diff_ci_high=hi)
                row.update(good[metrics].mean().to_dict())
                results.append(row)
    return pd.DataFrame(results)


def evaluate_locked(out, calendar, open_at, benchmark):
    plan, coverage = load_plans(out)  # 严格在读取/核验全部规格计划与日期轴之后才允许取开盘。
    schedule = {day:engine()['observation_dates'](calendar, day) for day in coverage.date.unique()}
    evaluated = []
    for pair in plan.to_dict('records'):
        day = pair['date']
        old_keys = {'strong_code':pair['expand_code'], 'control_code':pair['contract_code']}
        obs = engine()['outcome'](old_keys, *schedule[day], open_at, benchmark)
        values = {k.replace('strong_', 'expand_').replace('control_', 'contract_'):v for k,v in obs.items()}
        evaluated.append({**pair, **values})
    pairs = pd.DataFrame(evaluated, columns=[*PLAN,'entry','exit','status',*LEG])
    groups = {(s,d):g for (s,d),g in pairs.groupby(['spec','date'])}; records = []
    for row in coverage.to_dict('records'):
        s,d = row['spec'],row['date']; g = groups.get((s,d), pairs.iloc[:0]); good = g[g.status == 'complete']
        if len(g) != row['n_pairs']:
            raise ValueError('计划与逐日配对数不符')
        entry,end = schedule[d]
        records.append(dict(row, entry=entry, exit=end, mature=end is not None, n_complete=len(good),
                            n_missing=int(g.status.eq('missing').sum()), n_pending=int(g.status.eq('pending').sum()),
                            up_expand=int(good.expand_return.gt(0).sum()), up_contract=int(good.contract_return.gt(0).sum()),
                            **daily_values(good)))
    daily = pd.DataFrame(records)
    return pairs, daily, subgroup_summaries(daily)


def run(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'不覆盖已有研究；需要一个尚不存在的输出目录：{out}')
    old = json.loads(checked(ROOT / (OLD+'/manifest.json'), FROZEN[OLD+'/manifest.json']))
    if old['runner_sha256'] != FROZEN[ENGINE]:
        raise ValueError('冻结配对工具与原清单不一致')
    inputs = {**old['input_sha256'], **FROZEN, OLD+'/pool.csv':old['output_sha256']['pool.csv']}
    sources = {name:checked(ROOT/name, fingerprint) for name,fingerprint in inputs.items()}
    tool = engine()
    pred = pd.read_csv(io.BytesIO(sources[tool['PRED']]), parse_dates=['date'])
    pool = pd.read_csv(io.BytesIO(sources[OLD+'/pool.csv']), usecols=POOL_COLS, parse_dates=['date'])
    audit_pool(pool, pred)
    quote_index = pd.read_csv(io.BytesIO(sources[tool['QUOTE_INDEX']])).set_index('code', verify_integrity=True)
    quotes = Quotes(quote_index.sha256.to_dict())
    cfg = json.loads(sources[tool['CFG']]); benchmark = cfg['benchmark']
    calendar = quotes.calendar(benchmark)
    if (not calendar.equals(quotes.calendar('sh000905')) or str(calendar[-1].date()) != old['last_close']
            or cfg['label_horizon'] != tool['HORIZON'] or not pool.date.isin(calendar).all()):
        raise ValueError('冻结行情日期/基准/期限不符')
    derived = classify_pool(pool, calendar, quotes)
    plan, coverage = make_plans(derived)
    out.mkdir(parents=True, exist_ok=False)
    write_table(out, 'pool.csv', derived)
    locks = lock_plans(out, plan, coverage)
    pairs, daily, summary = evaluate_locked(out, calendar, quotes.open_at, benchmark)
    raw = pd.DataFrame([dict(code=c, sha256=quotes.hashes[c], phases='|'.join(sorted(p))) for c,p in sorted(quotes.used.items())])
    for name, table in [('pairs.csv', pairs), ('dates.csv', daily), ('summary.csv', summary), ('quote_hashes.csv', raw)]:
        write_table(out, name, table)
    names = ['pool.csv',*LOCK,PLAN_STAMP,'pairs.csv','dates.csv','summary.csv','quote_hashes.csv']
    manifest = dict(protocol=PROTOCOL, input_sha256=inputs, runner_sha256=digest(Path(__file__).read_bytes()),
                    numpy=np.__version__, pandas=pd.__version__, benchmark=benchmark, calendar_cross_check='sh000905',
                    quote_hash_index=tool['QUOTE_INDEX'], checked_quote_files=len(quotes.used), quote_sources_by_phase='quote_hashes.csv',
                    predictions=len(pred), score_dates=int(pool.date.nunique()), first_score=str(pool.date.min().date()),
                    last_score=str(pool.date.max().date()), last_close=old['last_close'], original_pool_rows=len(pool),
                    original_state_counts={k:int(v) for k,v in pool.state.value_counts().items()},
                    rank_caliper=tool['MAX_GAP'], holding_intervals=tool['HORIZON'], bootstrap_seed=tool['SEED'],
                    bootstrap_block=tool['BLOCK'], bootstrap_draws=tool['DRAWS'], fee_deducted=False,
                    plan_sha256=locks, plan_stage='all dates and specifications sealed before outcomes',
                    summary_weight='equal complete-pair dates; annual and regime masks applied after whole mature-axis sampling',
                    units='returns, hit rates and differences are fractions (0.01 difference = 1 percentage point)',
                    output_sha256={name:digest((out/name).read_bytes()) for name in names})
    with (out/'manifest.json').open('x', encoding='utf8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/EXP/'results', help='全新输出目录；已存在则停止')
    table = run(parser.parse_args().out)
    print(table[(table.period == 'all') & (table.regime == 'all')].to_string(index=False))
