"""步骤 9：拉取并缓存外部增量数据（股东户数 / 业绩预告 / 龙虎榜）。

用法：
    python scripts/09_fetch_ext.py            # 增量（已缓存则跳过）
    python scripts/09_fetch_ext.py --force    # 强制全量重拉

说明：
    * 三类数据都带公告日/上榜日，模块内严格按「次一交易日」可用对齐，防前视。
    * 北向资金个股持股自 2024-08 起已停止日频披露，不在接入范围。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from ashare.config import CACHE_DIR, Config
from ashare import extdata as ed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
log = logging.getLogger("fetch_ext")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    t0 = time.time()

    log.info("起始年份 %d，季度数 %d", cfg.start_year, len(ed._quarters(cfg.start_year)))

    holder = ed.fetch_holder(cfg, force=args.force)
    log.info("股东户数 %d 条 / %d 只，公告日 %s ~ %s",
             len(holder), holder["code"].nunique() if len(holder) else 0,
             holder["ann_date"].min() if len(holder) else "—",
             holder["ann_date"].max() if len(holder) else "—")

    fc = ed.fetch_forecast(cfg, force=args.force)
    log.info("业绩预告 %d 条 / %d 只，公告日 %s ~ %s",
             len(fc), fc["code"].nunique() if len(fc) else 0,
             fc["ann_date"].min() if len(fc) else "—",
             fc["ann_date"].max() if len(fc) else "—")
    if len(fc):
        log.info("预告类型分布：%s", fc["fc_type"].value_counts().head(8).to_dict())

    lhb = ed.fetch_lhb(cfg, force=args.force)
    log.info("龙虎榜 %d 条 / %d 只，日期 %s ~ %s",
             len(lhb), lhb["code"].nunique() if len(lhb) else 0,
             lhb["trade_date"].min() if len(lhb) else "—",
             lhb["trade_date"].max() if len(lhb) else "—")

    log.info("总耗时 %.0fs，缓存目录 %s", time.time() - t0, CACHE_DIR)

    # 前视体检：公告日必须不早于报告期
    if len(holder):
        rep = pd.to_datetime(holder["report_date"], format="%Y%m%d", errors="coerce")
        bad = (holder["ann_date"] < rep).sum()
        log.info("股东户数：公告日早于报告期的异常记录 %d 条（应为 0）", int(bad))
    if len(fc):
        rep = pd.to_datetime(fc["report_date"], format="%Y%m%d", errors="coerce")
        bad = (fc["ann_date"] < rep).sum()
        log.info("业绩预告：公告日早于报告期的异常记录 %d 条（应为 0）", int(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
