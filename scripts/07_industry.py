"""行业约束实验：限制单一行业入选数量。

动机：组合的风险来源不只是个股，还有产业链集中。A股在特定主题行情里，
同一条产业链的股票相关系数极高，15 只票可能有 8 只都在讲同一个故事。
限制单行业入选数是压制这类风格暴露最直接的手段。

用法：
    python scripts/07_industry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ashare.config import Config
from ashare import datasource as ds
from ashare import factors as fc
from ashare import universe as uv
from ashare import selection as sl
from ashare import backtest as bt

RAW = dict(sel_min_amount_ma20=0.0, sel_max_vol_pct=1.0,
           sel_min_ret_20=-1.0, sel_min_ret_5=-1.0)


def load():
    cfg = Config()
    master = ds.master_list(cfg)
    names = {ds.to_tencent(c): n for c, n in zip(master["code"], master["name"])}
    industry = ds.industry_map(cfg)
    print(f"行业映射：{len(industry)} 只，{len(set(industry.values()))} 个行业")
    panel = ds.load_panel(master["code"].tolist() + [cfg.benchmark], cfg, min_rows=120)
    mats = fc.clean_matrices(fc.build_matrices(panel))
    mask = uv.build_mask(mats, names, cfg, universe_codes=list(names))
    bench = mats["close"][cfg.benchmark].dropna()
    cb, cs = uv.tradeable_at_open(mats["open"], mats["close"].shift(1), list(mats["close"].columns))
    preds = pd.read_csv("output/preds.csv", parse_dates=["date"])
    sc = preds.pivot(index="date", columns="code", values="score").reindex(columns=mats["close"].columns)
    return mats, mask, bench, cb, cs, sc, industry, names


def make(**kw):
    c = Config()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def concentration(sig_codes, industry) -> dict:
    inds = [industry.get(c, "未知") for c in sig_codes]
    vc = pd.Series(inds).value_counts()
    if vc.empty:
        return {"最大行业占比": np.nan, "行业数": 0, "赫芬达尔": np.nan}
    shares = vc / vc.sum()
    return {
        "最大行业占比": round(float(shares.iloc[0]), 3),
        "行业数": int(len(vc)),
        "赫芬达尔": round(float((shares ** 2).sum()), 3),
    }


def main() -> int:
    mats, mask, bench, cb, cs, sc, industry, names = load()
    close = mats["close"]
    g = sl.market_regime_hysteresis(bench, 60, 0.0, -0.02)

    rows = []
    print("\n" + "=" * 92)
    print(f"{'行业上限':>8} {'年化%':>8} {'回撤%':>8} {'夏普':>7} {'超额%':>8} "
          f"{'最大行业占比':>13} {'行业数':>7} {'赫芬达尔':>9}")
    for cap in (0, 1, 2, 3, 5):
        cfg = make(**RAW)
        cfg.max_per_industry = cap
        r = bt.run(sc, mats, mask, cb, cs, bench, cfg, regime_ok=g, industry=industry)

        # 用最后一个调仓期的实际持仓衡量集中度
        w = r.weights
        last = w.iloc[-1] if not w.empty else pd.Series(dtype=float)
        held = [c for c, v in last.items() if v > 0.001]
        conc = concentration(held, industry)
        p = r.stats
        tag = "不限制" if cap == 0 else f"{cap} 只/行业"
        rows.append({
            "行业上限": tag, "年化%": round(p["年化收益"] * 100, 2),
            "回撤%": round(p["最大回撤"] * 100, 1), "夏普": round(p["夏普比率"], 2),
            "超额年化%": round(p["超额年化"] * 100, 2),
            **conc,
        })
        print(f"{tag:>9} {p['年化收益']*100:8.2f} {p['最大回撤']*100:8.1f} "
              f"{p['夏普比率']:7.2f} {p['超额年化']*100:8.2f} "
              f"{conc['最大行业占比']:13.3f} {conc['行业数']:7d} {conc['赫芬达尔']:9.3f}")

    df = pd.DataFrame(rows)
    df.to_csv("output/industry_experiment.csv", index=False, encoding="utf-8-sig")
    print("\n结论看两点：①集中度是否真的下降 ②收益/回撤是否值得这个代价")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
