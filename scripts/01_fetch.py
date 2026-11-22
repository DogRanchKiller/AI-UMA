"""按上海市场最近已收盘的真实交易日同步行情。默认只更新缓存，不登记观察。

    python scripts/01_fetch.py             # 已达目标日则不重复下载
    python scripts/01_fetch.py --force     # 全量重取；不接受截断已有历史
    python scripts/01_fetch.py --limit 50  # 仅子集与必需指数，不表示全市场就绪
    python scripts/01_fetch.py --record    # 仅全市场/当日/名册一致且核验通过，调用原观察入口
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from ashare import datasource as ds
from ashare.config import CACHE_DIR, Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger('fetch')


def run_record() -> int:
    """不在取数器内实现第二套名单逻辑；原观察脚本仍可因评分过期等原因拒绝。"""
    return subprocess.run([sys.executable, str(Path(__file__).with_name('11_daily_watch.py')), '--record'], check=False).returncode


def is_today_closed(target) -> bool:
    now = datetime.now(ds.SHANGHAI)
    return now.date() == target and now.time() >= time(15)


def local_names(path):
    """与观察脚本相同的来源、代码规范和 stock 行；歧义名册不用于正式登记。"""
    df = pd.read_csv(path, dtype={'code': str})
    stocks = df.loc[df.kind == 'stock', ['code', 'name']].copy()
    if stocks.empty or stocks.isna().any().any():
        raise ValueError('本地股票主表为空或缺代码/名称')
    stocks['code'] = stocks.code.map(ds.to_tencent)
    if stocks.code.duplicated().any():
        raise ValueError('本地主表含重复股票代码')
    return dict(zip(stocks.code, stocks.name))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument('--limit', type=int, help='仅处理前 N 只及必要指数，不能用于登记')
    scope.add_argument('--record', action='store_true', help='全市场合格后登记今日收盘观察（默认不登记）')
    ap.add_argument('--force', action='store_true', help='全量重取；须保留全部旧历史覆盖')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--start', default='', help='覆盖开始日期，如 2019-01-01')
    args = ap.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        ap.error('--workers / --limit 须为正整数')
    cfg = Config()
    if args.start:
        cfg.start_date = args.start
    try:
        cal = ds.resolve_market_calendar(cfg)
        master = CACHE_DIR / 'master_list.csv'
        names = {ds.to_tencent(c): n for c, n in cal.stocks.items()}
        if args.record:
            if not is_today_closed(cal.target):
                raise ValueError('目标不是上海当天已收盘交易日；不登记历史或盘中行情')
            if not names or not master.is_file() or local_names(master) != names:
                raise ValueError('目标日证券名册与本地观察主表不一致或无法核实。请单独核对、刷新主表后重试；不登记')
        if names:
            codes = sorted(names)
        elif master.is_file():
            log.warning('无法核实目标日全市场名册；仅尝试更新本地主表覆盖，不视为全市场就绪')
            codes = sorted(local_names(master))
        else:
            raise ValueError('目标日名册和本地主表均不可用，不能确定下载范围')
        indices = {ds.to_tencent(cfg.benchmark), *ds.REFERENCE_INDICES}
        if args.limit:
            codes = codes[:args.limit]
        requested = set(codes) | indices
        log.info('目标 %s；股票 %d 只、必需指数 %d 个，范围=%s', cal.target, len(codes), len(indices),
                 '子集（不可登记）' if args.limit else ('目标日全市场' if names else '本地缓存名册'))
        stat = ds.download_all(sorted(requested), cfg, force=args.force, workers=args.workers, calendar=cal)
        if set(stat.code) != requested or stat.code.duplicated().any() or not stat.target_date.eq(str(cal.target)).all():
            raise ValueError('下载统计未逐项覆盖本批证券、日期；不能报告完成')
        stat['scope'] = 'subset' if args.limit else ('market' if names else 'cached_roster')
        ds.save_fetch_table(stat, CACHE_DIR / 'fetch_stats.csv')
        accepted_halt = stat.status.eq('suspended') & ~stat.code.isin(indices)
        fresh = stat.fresh.eq(True) & stat.last_date.eq(str(cal.target))
        bad = stat.loc[~(fresh | accepted_halt)]
        log.info('目标 %s，实际行情已新 %d/%d，确证停牌无行情 %d；状态分布 %s', cal.target,
                 fresh.sum(), len(stat), accepted_halt.sum(), stat.status.value_counts().to_dict())
        if not bad.empty:
            sample = bad[['code', 'status', 'detail']].head(8).to_dict('records')
            log.error('未完成 %d 项，停止后续 record；各证券保留最后一次可用缓存。示例：%s', len(bad), sample)
            return 1
        if not names:
            log.warning('仅完成已知本地名单，不能确认全市场；停止后续 record')
            return 1
        if args.record:
            if not is_today_closed(cal.target) or local_names(master) != names:
                raise ValueError('下载期间日期或股票名册发生变化，拒绝登记')
            return run_record()
        if not args.limit and not master.exists():
            # 只在首次全量行情确认后初始化；不改写任何已有历史主表。
            ds.save_fetch_table(pd.DataFrame([{'code': c, 'name': n, 'kind': 'stock'} for c, n in cal.stocks.items()]), master)
        return 0
    except Exception as exc:
        log.error('本批无法确认完整：%s；停止后续 record', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
