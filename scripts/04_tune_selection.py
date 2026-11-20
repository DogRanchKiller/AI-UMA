"""选股约束 / 中性化 / 仓位闸门 的对照实验。

固定模型打分（output/preds.csv），只改动组合构建方式，可复现地比较各方案。
结论用于设定 config.py 的默认值，也是对「信号有效但组合亏钱」这一现象的
实测解释——不是凭直觉拍参数。

用法：
    python scripts/04_tune_selection.py
输出：
    output/selection_experiment.csv
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
           sel_min_ret_20=-1.0, sel_min_ret_5=-1.0,
           neutralize_size=False, neutralize_vol=False)


def load():
    cfg = Config()
    master = ds.master_list(cfg)
    names = {ds.to_tencent(c): n for c, n in zip(master["code"], master["name"])}
    panel = ds.load_panel(master["code"].tolist() + [cfg.benchmark], cfg, min_rows=120)
    mats = fc.clean_matrices(fc.build_matrices(panel))
    mask = uv.build_mask(mats, names, cfg, universe_codes=list(names))
    close = mats["close"]
    bench = close[cfg.benchmark].dropna()
    cb, cs = uv.tradeable_at_open(mats["open"], close.shift(1), list(close.columns))
    preds = pd.read_csv("output/preds.csv", parse_dates=["date"])
    sc = preds.pivot(index="date", columns="code", values="score").reindex(columns=close.columns)
    return mats, mask, bench, cb, cs, sc


def make(**kw):
    c = Config()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def evaluate(sc, mats, mask, bench, cb, cs, cfg, tag, regime=None):
    r = bt.run(sc, mats, mask, cb, cs, bench, cfg, regime_ok=regime)
    p = r.stats
    return {
        "方案": tag,
        "年化%": round(p["年化收益"] * 100, 2),
        "累计%": round(p["累计收益"] * 100, 1),
        "夏普": round(p["夏普比率"], 2),
        "回撤%": round(p["最大回撤"] * 100, 1),
        "超额年化%": round(p["超额年化"] * 100, 2),
        "信息比": round(p["信息比率"], 2),
        "换手": round(p.get("年化单边换手", 0), 1),
    }


def main() -> int:
    mats, mask, bench, cb, cs, sc = load()
    print(f"信号日 {len(sc)} 个，评分列 {sc.shape[1]} 只")

    variants = []          # (tag, scores, cfg, regime)

    # ---------- A. 单项选股约束 ----------
    variants.append(("基线(完全裸奔)", sc, make(**RAW), None))

    c = make(**dict(RAW, sel_max_vol_pct=0.85))
    variants.append(("波动率上限85%", sl.apply_to_scores(sc, sl.selection_mask(mats, c, mask)), c, None))

    c = make(**dict(RAW, sel_min_amount_ma20=1e8))
    variants.append(("流动性1亿", sl.apply_to_scores(sc, sl.selection_mask(mats, c, mask)), c, None))

    c = make(**dict(RAW, sel_min_ret_20=-0.30))
    variants.append(("剔除20日跌超30%", sl.apply_to_scores(sc, sl.selection_mask(mats, c, mask)), c, None))

    # ---------- B. 中性化 ----------
    neuts = {
        "规模中性化": sl.neutralize_scores(sc, mats, use_size=True, use_vol=False),
        "波动中性化": sl.neutralize_scores(sc, mats, use_size=False, use_vol=True),
        "规模+波动中性化": sl.neutralize_scores(sc, mats, use_size=True, use_vol=True),
    }
    for tag, s in neuts.items():
        variants.append((tag, s, make(**RAW), None))

    # ---------- C. 仓位闸门 ----------
    for ma in (20, 60, 120):
        for buf in (0.0, 0.02):
            reg = sl.market_regime(bench, ma=ma, buffer=buf)
            tag = f"闸门MA{ma}" + (f"+{buf:.0%}缓冲" if buf else "")
            variants.append((tag, sc, make(**RAW), reg))

    # ---------- D. 中性化 + 闸门 ----------
    neut_sv = neuts["规模+波动中性化"]
    for ma, buf in ((60, 0.0), (120, 0.0), (60, 0.02)):
        reg = sl.market_regime(bench, ma=ma, buffer=buf)
        sfx = f"MA{ma}" + (f"+{buf:.0%}" if buf else "")
        variants.append((f"中性化+闸门{sfx}", neut_sv, make(**RAW), reg))
        for k in (30, 40):
            variants.append((f"中性化+闸门{sfx}+{k}只", neut_sv, make(**dict(RAW, top_n=k)), reg))

    rows = []
    for tag, s, c, reg in variants:
        try:
            r = evaluate(s, mats, mask, bench, cb, cs, c, tag, reg)
            rows.append(r)
            print(f"完成 {tag:26s} 年化{r['年化%']:7.2f}%  回撤{r['回撤%']:6.1f}%  超额{r['超额年化%']:7.2f}%")
        except Exception as exc:  # noqa: BLE001 - 单方案失败不影响其余
            print(f"{tag} 失败: {type(exc).__name__}: {exc}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(df.to_string(index=False))
    df.to_csv("output/selection_experiment.csv", index=False, encoding="utf-8-sig")
    print("\n按超额年化排序前 6：")
    print(df.sort_values("超额年化%", ascending=False).head(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
