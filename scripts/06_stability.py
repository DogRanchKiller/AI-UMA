"""分年度稳定性检验。

在同一段历史上挑参数必然引入选择偏差。这里的做法是：对候选方案逐年拆解，
只有当优势在多数年份都成立、而不是由某一年单独贡献时，才认为它可信。

用法：
    python scripts/06_stability.py [preds文件]
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


def load(pred_file: str):
    cfg = Config()
    master = ds.master_list(cfg)
    names = {ds.to_tencent(c): n for c, n in zip(master["code"], master["name"])}
    panel = ds.load_panel(master["code"].tolist() + [cfg.benchmark], cfg, min_rows=120)
    mats = fc.clean_matrices(fc.build_matrices(panel))
    mask = uv.build_mask(mats, names, cfg, universe_codes=list(names))
    close = mats["close"]
    bench = close[cfg.benchmark].dropna()
    cb, cs = uv.tradeable_at_open(mats["open"], close.shift(1), list(close.columns))
    preds = pd.read_csv(pred_file, parse_dates=["date"])
    sc = preds.pivot(index="date", columns="code", values="score").reindex(columns=close.columns)
    return mats, mask, bench, cb, cs, sc


def make(**kw):
    c = Config()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def yearly(eq: pd.Series, bm: pd.Series) -> pd.DataFrame:
    """分年度收益与超额。

    注意：区间第一年通常还没开始交易（滚动训练需要足够历史），策略收益为 0，
    这一年的「超额」恒等于负的基准收益，对所有方案都一样，不具区分度。
    因此这里显式剔除首个不完整年份，避免把空仓年计入胜率。
    """
    e = eq.resample("YE").last()
    b = bm.reindex(eq.index).ffill().resample("YE").last()
    er = e.pct_change()
    br = b.pct_change()
    er.iloc[0] = e.iloc[0] / eq.iloc[0] - 1
    br.iloc[0] = b.iloc[0] / bm.dropna().iloc[0] - 1
    df = pd.DataFrame({"策略": (er * 100).round(1), "基准": (br * 100).round(1)})
    df["超额"] = (df["策略"] - df["基准"]).round(1)
    # 首个不完整年（策略几乎全程空仓）不参与稳定性统计
    first_real = df.index[df["策略"].abs() > 1]
    if len(first_real):
        df = df[df.index >= first_real[0]]
    return df


def main() -> int:
    pred_file = sys.argv[1] if len(sys.argv) > 1 else "output/preds.csv"
    mats, mask, bench, cb, cs, sc = load(pred_file)
    close = mats["close"]
    print(f"预测文件: {pred_file} · 信号日 {len(sc)}")

    g60 = sl.market_regime(bench, 60)
    g60h = sl.market_regime_hysteresis(bench, 60, enter=0.0, exit_=-0.02)

    cands = [
        ("对照:无闸门+equal+20", None, make(**dict(RAW, top_n=20))),
        ("旧默认:MA60+equal+20", g60, make(**dict(RAW, top_n=20))),
        ("新默认:MA60迟滞+inv_vol+15", g60h,
         make(**dict(RAW, top_n=15, weighting="inv_vol"))),
        ("MA60迟滞+inv_vol+10", g60h,
         make(**dict(RAW, top_n=10, weighting="inv_vol"))),
        ("MA60迟滞+inv_vol+20", g60h,
         make(**dict(RAW, top_n=20, weighting="inv_vol"))),
        ("MA60迟滞+inv_vol+30", g60h,
         make(**dict(RAW, top_n=30, weighting="inv_vol"))),
    ]

    summary, per_year = [], {}
    for tag, reg, cfg in cands:
        r = bt.run(sc, mats, mask, cb, cs, bench, cfg, regime_ok=reg)
        yr = yearly(r.equity, r.benchmark)
        per_year[tag] = yr["超额"]
        p = r.stats
        summary.append({
            "方案": tag,
            "年化%": round(p["年化收益"] * 100, 2),
            "回撤%": round(p["最大回撤"] * 100, 1),
            "夏普": round(p["夏普比率"], 2),
            "超额年化%": round(p["超额年化"] * 100, 2),
            "超额>0年数": int((yr["超额"] > 0).sum()),
            "总年数": int(len(yr)),
            "最差年超额": round(float(yr["超额"].min()), 1),
        })

    print("\n=== 汇总 ===")
    print(pd.DataFrame(summary).to_string(index=False))

    py = pd.DataFrame(per_year)
    print("\n=== 分年度超额（%）===")
    print(py.to_string())
    print("\n各年最优方案：")
    for year, row in py.iterrows():
        print(f"  {year}: {row.idxmax():26s} {row.max():6.1f}%")

    print("\n稳定性评分（超额为正的年份占比 × 平均超额）：")
    score = (py > 0).mean() * py.mean()
    for k, v in score.sort_values(ascending=False).items():
        print(f"  {k:26s} 正年份占比 {(py[k]>0).mean():5.0%}  平均超额 {py[k].mean():6.1f}%  得分 {v:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
