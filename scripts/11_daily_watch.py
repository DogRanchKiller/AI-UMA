"""离线收盘观察，不拉行情、不生成模型打分、不交易。默认只预览。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ashare import selection, universe
from ashare.config import CACHE_DIR, OUT_DIR, RAW_DIR, Config
from ashare.datasource import to_tencent

SHANGHAI = timezone(timedelta(hours=8))
PINNED = ('sh600795', 'sh601963')  # 已登记的人工观察标的，不占筛选名额
CODE = re.compile(r'(?:sh|sz)[0-9]{6}')
BAR_FIELDS = ['open', 'close', 'low', 'high', 'volume']


def valid_session(bar):
    values = bar.reindex(BAR_FIELDS).to_numpy(dtype=float)
    op, cl, low, high, _ = values
    return np.isfinite(values).all() and (values > 0).all() and low <= min(op, cl) <= max(op, cl) <= high


def read_bars(raw, code):
    if not CODE.fullmatch(code):
        raise ValueError(f'不是有效的行情代码：{code}')
    path = raw / f'{code}.csv'
    if not path.is_file():
        return pd.DataFrame(columns=BAR_FIELDS, index=pd.DatetimeIndex([]), dtype=float)
    frame = pd.read_csv(path, usecols=['date', *BAR_FIELDS, 'code'], parse_dates=['date'])
    if not frame.code.eq(code).all() or frame.date.duplicated().any():
        raise ValueError(f'{code} 行情文件代码不符或出现重复交易日')
    return frame.drop(columns='code').set_index('date').sort_index().astype(float)


def select_day(benchmark, date, mode, now):
    today = pd.Timestamp(now.astimezone(SHANGHAI).date())
    closed = benchmark.loc[:today]
    if now.astimezone(SHANGHAI).hour < 15:
        closed = closed.loc[closed.index < today]
    if closed.empty:
        raise ValueError('没有已收盘的指数行情')
    latest = closed.index[-1]
    day = (today if mode == 'record' else latest) if date is None else pd.Timestamp(date)
    if day not in closed.index:
        raise ValueError(f'{day.date()} 无已收盘的指数日线，最新为 {latest.date()}；不沿用旧行情发新报告')
    if mode == 'record' and day != today:
        raise ValueError('正式观察只能登记上海当日的收盘行情；历史请预览')
    if mode == 'baseline' and day != latest:
        raise ValueError('基线只能使用最新已收盘的缓存日，不能倒填历史观察')
    return day


def verify_calendar(benchmark, reference, day):
    """以另一条缓存指数序列交叉检查交易日，防止漏掉单日后把 t+11 错挪一天。"""
    dates = benchmark.loc[:day].index
    if dates.empty or not dates.equals(reference.loc[dates[0]:day].index):
        raise ValueError('沪深300与中证500本地交易日历不一致；先补齐日期再做观察')


def pick_scores(predictions, calendar, ttl):
    asof = calendar[-1]
    available = predictions.loc[predictions.date <= asof]
    if available.empty:
        raise ValueError('该日没有可用的历史模型评分')
    score_date = available.date.max()
    if score_date not in calendar:
        raise ValueError('评分日期不在指数交易日历，无法确认评分时效')
    age = len(calendar) - 1 - calendar.get_loc(score_date)
    if age >= ttl:
        raise ValueError(f'最近评分为 {score_date.date()}、已过去 {age} 个交易日；达到 {ttl} 日更新间隔，暂停筛选')
    row = available.loc[available.date == score_date, ['code', 'score']]
    if (row.code.duplicated().any() or not row.code.str.fullmatch(CODE, na=False).all()
            or not np.isfinite(row.score).all()):
        raise ValueError('评分截面有重复、空值或无效代码')
    if len(row) < 20:
        raise ValueError('评分截面不足20只，前5%没有有效比较范围；请核对预测缓存')
    return row.sort_values(['score', 'code'], ascending=[False, True]).reset_index(drop=True), score_date, age


def make_snapshot(day, pred, names, bench, load, cfg, pinned=None):
    # 人工卡以 11/20 收盘为起点。不能在更早的历史预览里倒推两只事后才选的关注股。
    if pinned is None:
        pinned = PINNED if day >= pd.Timestamp('2026-11-20') else ()
    if not cfg.use_regime_filter or cfg.regime_mode != 'hysteresis':
        raise ValueError('此观察口径只支持冻结的迟滞市场开关配置')
    b = bench.loc[:day, 'close']
    if (len(b) < max(21, cfg.regime_ma) or not b.index.is_unique or not b.index.is_monotonic_increasing
            or b.index[-1] != day or not np.isfinite(b).all() or not (b > 0).all()
            or not valid_session(bench.loc[day])):
        raise ValueError('指数日历/收盘不完整，不能确定观察和市场状态')
    scores, score_date, age = pick_scores(pred, b.index, cfg.rebalance_every)
    ranked = {code: i for i, code in enumerate(scores.code, 1)}
    pool = list(scores.code.iloc[:len(scores) // 20])  # 与观察卡一致，只取前 5% 向下取整
    codes = list(dict.fromkeys([*pool, *pinned]))
    frames = {code: load(code).reindex(b.index) for code in codes}
    close = pd.DataFrame({code: df.close for code, df in frames.items()}, index=b.index)
    volume = pd.DataFrame({code: df.volume for code, df in frames.items()}, index=b.index)
    close = close.where(np.isfinite(close) & (close > 0))
    volume = volume.where(np.isfinite(volume) & (volume >= 0))
    mask = universe.build_mask({'close': close, 'volume': volume, 'amount': close * volume * 100},
                               names, cfg, universe_codes=list(names)).iloc[-1]
    rows, unknown = {}, []
    for code in codes:
        window = close[code].tail(21)
        row = {'code': code, 'name': names.get(code, code), 'rank': ranked.get(code), 'state': 'unknown'}
        rows[code] = row
        if window.isna().any() or not valid_session(frames[code].loc[day]):
            if code in pool:
                unknown.append(code)  # 行情缺口不是已证实的走弱或退选
            continue
        last = float(window.iloc[-1])
        mean = float(window.iloc[1:].mean())
        r5, r20 = last / float(window.iloc[-6]) - 1, last / float(window.iloc[0]) - 1
        # 日线报价精度有限；避免浮点误差把恰好等于均线的平盘写成站上。
        strong = last > mean + 1e-10 and r5 > 0 and r20 > 0
        row.update(close=last, ma20=mean, r5=r5, r20=r20,
                   state=('ineligible' if not mask[code] else 'strong' if strong else 'unconfirmed'))
    if pool and len(unknown) == len(pool):
        raise ValueError('候选池行情均缺失或不完整；不能将断更登记为空名单')
    gate = selection.market_regime_hysteresis(b, cfg.regime_ma, cfg.regime_enter, cfg.regime_exit)
    score_bytes = scores.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode()
    return {'rule_version': 1, 'score_sha256': hashlib.sha256(score_bytes).hexdigest(),
            'date': day.date().isoformat(), 'score_date': score_date.date().isoformat(),
            'score_age': int(age), 'ranked_count': len(scores), 'screened_count': len(pool),
            'gate': bool(gate.iloc[-1]), 'index_close': float(b.iloc[-1]),
            'index_ma': float(b.tail(cfg.regime_ma).mean()),
            'matches': [rows[c] for c in pool if rows[c]['state'] == 'strong'],
            'unknown': unknown, 'pinned': [rows[c] for c in pinned]}


def validate_score_continuity(earlier, snapshot):
    """上游预测文件更新可能丢掉上次临时末日或重算同日，不能让排名悄悄倒退/替换。"""
    if not earlier:
        return
    previous = earlier[-1]
    if snapshot['score_date'] < previous['score_date']:
        raise ValueError('可用评分日期早于上次登记；请核实预测缓存，暂停发布')
    if (snapshot['score_date'] == previous['score_date']
            and snapshot['score_sha256'] != previous['score_sha256']):
        raise ValueError('已使用的同日评分截面被改写，不能当作原评分继续发布')


def render(snapshot, diff, previous):
    ranked = snapshot['ranked_count']
    reuse = '当日评分' if not snapshot['score_age'] else '沿用旧评分，不是今日新预测'
    gate = '开（不是买入指令）' if snapshot['gate'] else '关，仅观察、不新建仓'
    print(f"收盘观察 {snapshot['date']} | 评分 {snapshot['score_date']}，已过 {snapshot['score_age']} 个交易日（{reuse}）")
    print(f"市场开关：{gate}。{ranked} 个有分数的标的 → 前5% {snapshot['screened_count']} 只 → "
          f"达标 {len(snapshot['matches'])} 只，行情不足/待核 {len(snapshot['unknown'])} 只。")
    print('对照：' + (f'上次实际登记 {previous}（非逐日补算）' if previous else '没有更早的登记；首批不是新入选'))
    labels = {'initial': '首批', 'new': '新进', 'kept': '持续', 'lost': '失效/退出高分池', 'unknown': '待核'}
    print('；'.join(f'{v} {len(diff[k])}' for k, v in labels.items()))
    # 新进优先展示，组内保留分数排名；全部达标者都入账，不受三行版面限制。
    new_codes = set(diff['new'])
    highlights = sorted(snapshot['matches'], key=lambda row: row['code'] not in new_codes)
    for row in highlights[:3]:
        tag = next((v for k, v in labels.items() if row['code'] in diff[k]), '观察')
        print(f"  [{tag}] {row['code']} {row['name']}，第 {row['rank']} 名，收 {row['close']:.2f}，"
              f"MA20 {row['ma20']:.2f}，5日 {row['r5']:+.2%}，20日 {row['r20']:+.2%}")
    if not snapshot['matches']:
        print('没有已确认的趋势候选，不凑数。')
    if diff['lost']:
        print('已离开条件（前3，完整代码见 --json）：' + '、'.join(diff['lost'][:3]))
    states = {'strong': '走势偏强', 'unconfirmed': '转强未确认', 'unknown': '行情不足/待核', 'ineligible': '目前不在合格股票池'}
    if snapshot['pinned']:
        print('原观察卡（不占筛选名额）：' + '；'.join(f"{r['name']}：{states[r['state']]}" for r in snapshot['pinned']))


def main():
    from ashare import watch_journal as journal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', help='观察截至 YYYY-MM-DD；默认最新已收盘日，record 模式默认上海当天')
    parser.add_argument('--db', type=Path, default=OUT_DIR/'daily_watch'/'observations.sqlite')
    parser.add_argument('--json', action='store_true', help='打印全量名单/变化或复核事件')
    group = parser.add_mutually_exclusive_group()
    for opt, meaning in [('baseline', '在空库登记最新缓存，明确标作基线'), ('record', '仅登记上海当天收盘后观察'),
                         ('review', '仅读取已登记事件并按截至日复核，不回填历史')]:
        group.add_argument(f'--{opt}', action='store_const', const=opt, dest='mode', help=meaning)
    args = parser.parse_args()
    cfg = Config.load(OUT_DIR/'config_used.json')
    load = lru_cache(maxsize=512)(lambda code: read_bars(RAW_DIR, code))
    bench = load(cfg.benchmark)
    now = datetime.now(SHANGHAI)
    day = select_day(bench, args.date, args.mode, now)
    verify_calendar(bench, load('sh000905'), day)
    history = [s for s in journal.load_history(args.db) if s['date'] <= day.date().isoformat()]
    if args.mode == 'review':
        def open_at(code, when):
            bar = load(code)
            if when not in bar.index or not bar.at[when, 'volume'] > 0:
                return float('nan')
            return float(bar.at[when, 'open'])
        events = journal.review_events(history, bench.loc[:day].index, open_at, cfg.benchmark, horizon=cfg.label_horizon)
        if args.json:
            print(json.dumps(events, ensure_ascii=False, allow_nan=False, indent=2))
        else:
            print(f'截至 {day.date()}，观察事件 {len(events)}：' + '；'.join(
                f'{name} {sum(e["status"] == k for e in events)}' for k, name in
                [('pending', '未到期'), ('complete', '已复核'), ('missing', '端点缺行情')]))
            print('用 --review --json 查看每笔股价涨跌、指数涨跌、差额及观察性质；不代表扣费组合收益。')
        return
    preds = pd.read_csv(OUT_DIR/'preds.csv', parse_dates=['date'], usecols=['date', 'code', 'score'])
    master = pd.read_csv(CACHE_DIR/'master_list.csv', dtype={'code': str}).dropna(subset=['name'])
    names = {to_tencent(r.code): r.name for r in master.itertuples() if r.kind == 'stock'}
    snap = make_snapshot(day, preds, names, bench, load, cfg)
    # 当日已有登记时仍与上一笔更早的观察比较，不把自己当作「持续」。
    earlier = [s for s in history if s['date'] < snap['date']]
    validate_score_continuity(earlier, snap)
    diff = journal.changes(earlier, snap)
    result = 'preview'
    if args.mode in ('baseline', 'record'):
        result = journal.save_snapshot(args.db, snap, baseline=args.mode == 'baseline', now=now)
    previous = earlier[-1]['date'] if earlier else None
    if args.json:
        print(json.dumps({'mode': args.mode or 'preview', 'result': result, 'previous': previous,
                          'snapshot': snap, 'changes': diff}, ensure_ascii=False, allow_nan=False, indent=2))
    else:
        print(f'模式：{args.mode or "preview（只读预览）"}；{result}')
        render(snap, diff, previous)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(f'未生成报告：{exc}')
