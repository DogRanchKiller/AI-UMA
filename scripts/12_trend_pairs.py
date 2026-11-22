"""冻结研究：同一评分日、相差至多5名的趋势达标/未达标股票，随后十个区间的开盘价比较。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ashare.config import Config

EXP = ROOT / 'output' / 'trend_pairs_20261121'
WATCH = 'scripts/11_daily_watch.py'
QUOTE_INDEX = 'output/simple_controls_20261121/results/stock_cache_sha256.csv'
PRED = 'output/preds.csv'
CFG = 'output/config_used.json'
MASTER = 'data/cache/master_list.csv'
FROZEN = {
    WATCH: '217cae8200a4cd171ac762fdfc9ade5f89b785c952cb9a554a54d0ad8f6ec009',
    QUOTE_INDEX: '7cb27d3894f1f037812af55e24c613d3054daebdd5bdda810a99f9176d63a86a',
    PRED: '6e559ad94ee75fe6e671e74a2b4924edf30ae211a2309263f259422914892072',
    CFG: 'd8e7dc43a1a8d1092ed52bad33bac4acf5bba7942bfedd57295f7c17ad0238be',
    MASTER: '2c3f05e2cb332e002ea21ccd6080beb8638751a9fbca1fa5521d345880511f7c',
    'output/trend_pairs_20261121/比较口径.md': '2e2adc252ccba7b7aa256f257cb2ce7640540f8ad814a4fd9ac91e85b9c1ddfd',
    'ashare/selection.py': '0590286e58f713c5b4428d19186b65abd92887e49ae0edf637c5eccc0a6ff23a',
    'ashare/universe.py': '4499686f7e8e0c8c365b0d83315ac587550a042b7f5c70656e3271051d73369e',
}
MAX_GAP = 5
HORIZON = 10
SEED = 20261121
BLOCK = 3
DRAWS = 10_000
METRICS = ('hit_strong', 'hit_control', 'hit_diff', 'ret_strong', 'ret_control', 'ret_diff',
           'ret_index', 'excess_strong', 'excess_control')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_input(path, expected):
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f'冻结输入缺失或指纹不符，停止：{path.name}')


def match_ranks(strong, control):
    """不交叉的最优1:1匹配；只读名次，不接触收益、交易端点或名称。"""
    a, b = sorted(strong), sorted(control)
    if len(set(a + b)) != len(a) + len(b):
        raise ValueError('当日的模型名次须唯一')
    n, m = len(a), len(b)
    # 单调绝对距离：交叉的两对互换后不会增距，也不会突破原有最大距离。
    # 约束内每多配一对的奖励严格大于最大可能总距离，因此先最大基数，再最小总距。
    bonus = min(n, m) * MAX_GAP + 1
    value = [[0] * (m + 1) for _ in range(n + 1)]
    action = np.zeros((n, m), dtype=np.int8)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            gap = abs(a[i] - b[j])
            pair = bonus - gap + value[i+1][j+1] if gap <= MAX_GAP else -1
            # 同目标值按「配当前→跳对照→跳达标」取第一项。
            choices = (pair, value[i][j+1], value[i+1][j])
            value[i][j] = max(choices)
            action[i, j] = choices.index(value[i][j])
    pairs = []
    i = j = 0
    while i < n and j < m:
        choice = action[i, j]
        if choice == 0:
            pairs.append((a[i], b[j]))
            i += 1
            j += 1
        elif choice == 1:
            j += 1
        else:
            i += 1
    return pairs


def observation_dates(calendar, day):
    i = calendar.get_loc(day)
    def offset(k):
        return calendar[i+k] if i+k < len(calendar) else None
    return offset(1), offset(1 + HORIZON)


def quote_open(frame, day):
    if day not in frame.index:
        return np.nan
    op, volume = frame.loc[day, ['open', 'volume']]
    return float(op) if np.isfinite(op) and op > 0 and np.isfinite(volume) and volume > 0 else np.nan


def outcome(pair, entry, end, open_at, benchmark):
    row = dict(pair, entry=entry, exit=end, status='pending')
    for prefix in ('strong', 'control', 'index'):
        row.update({prefix + '_entry_open': np.nan, prefix + '_exit_open': np.nan, prefix + '_return': np.nan})
    if end is None:
        return row  # 未成熟不消费后来行情；保留先前配定的两个代码
    if entry is None or entry >= end:
        raise ValueError('复核交易端点次序异常')
    for prefix, code in (('strong', pair['strong_code']), ('control', pair['control_code']), ('index', benchmark)):
        start, stop = open_at(code, entry), open_at(code, end)
        row[prefix + '_entry_open'] = start
        row[prefix + '_exit_open'] = stop
        if np.isfinite(start) and start > 0 and np.isfinite(stop) and stop > 0:
            row[prefix + '_return'] = stop / start - 1
    row['status'] = 'complete' if all(np.isfinite(row[p + '_return']) for p in ('strong','control','index')) else 'missing'
    return row


def daily_values(complete):
    if complete.empty:
        return dict.fromkeys(METRICS, np.nan)
    a, b, ix = (complete[c].to_numpy(dtype=float) for c in ('strong_return','control_return','index_return'))
    if not np.isfinite([a, b, ix]).all():
        raise ValueError('完整对含有缺失端点，不能用不同分母比较')
    ha, hb = np.mean(a > 0), np.mean(b > 0)
    return dict(zip(METRICS, [ha, hb, ha - hb, a.mean(), b.mean(), (a-b).mean(),
                              ix.mean(), (a-ix).mean(), (b-ix).mean()]))


def block_hit_interval(values):
    """非该状态、无有效配对的成熟日期为 NaN，不能先删再抽块。"""
    x = np.asarray(values, dtype=float)
    if np.isfinite(x).sum() < 2:
        return np.nan, np.nan
    n = len(x)
    starts = np.random.default_rng(SEED).integers(0, n, size=(DRAWS, (n + BLOCK - 1) // BLOCK))
    pos = ((starts[:, :, None] + np.arange(BLOCK)) % n).reshape(DRAWS, -1)[:, :n]
    draws = x[pos]
    counts = np.isfinite(draws).sum(axis=1)
    usable = counts > 0
    means = np.nansum(draws[usable], axis=1) / counts[usable]
    return tuple(np.quantile(means, [.025, .975]))


def summaries(daily):
    timeline = daily.loc[daily.mature].sort_values('date')
    periods = [('all', timeline)] + [(str(y), rows) for y,rows in timeline.groupby(timeline.date.dt.year)]
    result = []
    for period, days in periods:
        for regime, keep in (('off', ~days.gate), ('on', days.gate), ('all', days.gate.notna())):
            members = days.loc[keep]
            good = members.loc[members.n_complete > 0]
            row = {'period': period, 'regime': regime, 'mature_dates': len(members), 'effective_dates': len(good),
                   'pairs_planned': int(members.n_pairs.sum()), 'pairs_complete': int(members.n_complete.sum()),
                   'days_hit_diff_positive': int((good.hit_diff > 0).sum()),
                   'days_hit_diff_negative': int((good.hit_diff < 0).sum())}
            row.update(good[list(METRICS)].mean().to_dict())
            row['hit_diff_ci_low'], row['hit_diff_ci_high'] = (block_hit_interval(days.hit_diff.where(keep))
                                                               if period == 'all' else (np.nan, np.nan))
            result.append(row)
    return pd.DataFrame(result)


def run(out):
    if out.exists():
        raise FileExistsError(f'请指定尚不存在的新输出目录：{out}')
    for name, fingerprint in FROZEN.items():
        check_input(ROOT / name, fingerprint)
    w = run_path(str(ROOT / WATCH))
    cfg = Config.load(ROOT / CFG)
    pred = pd.read_csv(ROOT / PRED, parse_dates=['date'])
    master = pd.read_csv(ROOT / MASTER).dropna(subset=['name'])
    names = {w['to_tencent'](r.code): r.name for r in master.itertuples() if r.kind == 'stock'}
    stock_hashes = pd.read_csv(ROOT / QUOTE_INDEX).set_index('code').sha256.to_dict()
    read_codes = set()
    @lru_cache(maxsize=1024)
    def load(code):
        if code not in stock_hashes or not w['CODE'].fullmatch(code):
            raise ValueError(f'行情输入不在冻结清单中：{code}')
        check_input(ROOT / 'data/raw' / f'{code}.csv', stock_hashes[code])
        read_codes.add(code)
        return w['read_bars'](ROOT / 'data/raw', code)
    bench = load(cfg.benchmark)
    calendar = bench.index
    w['verify_calendar'](bench, load('sh000905'), calendar[-1])
    plans, pool_records, days = [], [], []
    # 阶段一：本地既有观察脚本返回全部高分池的状态；这里的 pinned 参数仅要求全量展示已有的排名股票，
    # 不改变筛选或纳入人工关注名单。任何后续区间回报都不传给配对器。
    for day, cross in pred.groupby('date', sort=True):
        ordered, _, _ = w['pick_scores'](cross, calendar[calendar <= day], cfg.rebalance_every)
        pool = list(ordered.code.iloc[:len(ordered)//20])
        snap = w['make_snapshot'](day, cross, names, bench, load, cfg, pinned=pool)
        by_rank = {r['rank']: r for r in snap['pinned']}
        strong = [k for k, r in by_rank.items() if r['state'] == 'strong']
        control = [k for k, r in by_rank.items() if r['state'] == 'unconfirmed']
        pairings = match_ranks(strong, control)
        used = {v for pair in pairings for v in pair}
        for k, r in by_rank.items():
            pool_records.append(dict(date=day, gate=snap['gate'], paired=k in used, **r))
        for a, b in pairings:
            plans.append(dict(date=day, gate=snap['gate'], strong_code=by_rank[a]['code'], strong_rank=a,
                              control_code=by_rank[b]['code'], control_rank=b, rank_gap=abs(a-b)))
        entry, end = observation_dates(calendar, day)
        days.append(dict(date=day, gate=snap['gate'], score_sha256=snap['score_sha256'], entry=entry, exit=end,
                         mature=end is not None, n_ranked=len(ordered), n_pool=len(pool), n_strong=len(strong),
                         n_control=len(control), n_unknown=len(snap['unknown']),
                         n_ineligible=len(pool)-len(strong)-len(control)-len(snap['unknown']), n_pairs=len(pairings)))
    # 阶段二：已成型的配对只允许标记完整/缺失/未到期，不可反向更换标的。
    schedule = {d['date']: (d['entry'], d['exit']) for d in days}
    @lru_cache(maxsize=None)
    def open_at(code, day):
        return quote_open(load(code), day)
    outcomes = [outcome(pair, *schedule[pair['date']], open_at, cfg.benchmark) for pair in plans]
    pairs = pd.DataFrame(outcomes)
    for day in days:
        subset = pairs.loc[pairs.date == day['date']]
        good = subset.loc[subset.status == 'complete']
        day.update(n_complete=len(good), n_missing=int((subset.status == 'missing').sum()),
                   n_pending=int((subset.status == 'pending').sum()), **daily_values(good))
    daily = pd.DataFrame(days)
    summary = summaries(daily)
    out.mkdir(parents=True, exist_ok=False)
    for name, table in [('pool', pd.DataFrame(pool_records)), ('pairs', pairs), ('dates', daily), ('summary', summary)]:
        table.to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
    manifest = {'input_sha256': FROZEN, 'runner_sha256': sha256(Path(__file__)), 'numpy': np.__version__,
                'pandas': pd.__version__, 'checked_quote_files': len(read_codes), 'quote_hash_index': QUOTE_INDEX,
                'predictions': len(pred), 'dates': len(days), 'last_close': calendar[-1].date().isoformat(),
                'rank_caliper': MAX_GAP, 'holding_intervals': HORIZON, 'bootstrap_seed': SEED,
                'bootstrap_block': BLOCK, 'bootstrap_draws': DRAWS,
                'output_sha256': {f'{n}.csv': sha256(out / f'{n}.csv') for n in ['pool','pairs','dates','summary']}}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EXP / 'results')
    args = parser.parse_args()
    result = run(args.out)
    print(result.loc[result.period == 'all'].to_string(index=False))
