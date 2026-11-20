"""步骤 1：下载并缓存全市场前复权日线。

用法：
    python scripts/01_fetch.py            # 增量下载（已缓存则只补最新）
    python scripts/01_fetch.py --force    # 全量重下
    python scripts/01_fetch.py --limit 50 # 只下前 50 只（冒烟测试）
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
from ashare import datasource as ds

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
log = logging.getLogger("fetch")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="忽略缓存全量重下")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--start", default="", help="覆盖开始日期，如 2019-01-01")
    args = ap.parse_args()

    cfg = Config()
    if args.start:
        cfg.start_date = args.start

    codes = ds.master_list(cfg)["code"].tolist()
    idx_codes = [cfg.benchmark, "sh000905", "sh000852", "sh000001"]
    if args.limit:
        codes = codes[: args.limit]
    log.info("目标 %d 只股票 + %d 个指数，起始 %s", len(codes), len(idx_codes), cfg.start_date)

    t0 = time.time()
    stat = ds.download_all(codes + idx_codes, cfg, force=args.force, workers=args.workers)
    cost = time.time() - t0

    stat_path = CACHE_DIR / "fetch_stats.csv"
    stat.to_csv(stat_path, index=False, encoding="utf-8-sig")

    ok = stat[stat["rows"] > 0]
    log.info("完成 %d/%d，用时 %.0fs，成功率 %.1f%%", len(ok), len(stat), cost,
             100 * len(ok) / max(len(stat), 1))
    log.info("状态分布：%s", stat["status"].str.split(":").str[0].value_counts().to_dict())
    if len(ok):
        log.info("总行数 %.1f 万，中位每股 %d 行", ok["rows"].sum() / 1e4, int(ok["rows"].median()))
    bad = stat[stat["rows"] == 0]
    if len(bad):
        log.warning("失败 %d 只，示例：%s", len(bad), bad["code"].head(8).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
