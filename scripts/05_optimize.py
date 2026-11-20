"""组合层优化实验：权重方案、持仓数、择时闸门、回撤刹车。

固定模型打分（output/preds.csv），只改组合构建方式，因此结果可复现且与重训无关。
选优原则不是「回测收益最高」，而是**风险调整后稳健**：同时看年化、回撤、超额、
夏普与换手，并优先选择参数逻辑更简单、更少自由度的方案。

用法：
    python scripts/05_optimize.py
输出：
    output/optimize_experiment.csv
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


def run_one(sc, mats, mask, bench, cb, cs, cfg, regime=None):
    r = bt.run(sc, mats, mask, cb, cs, bench, cfg, regime_ok=regime)
    p = r.stats
    return {
        "年化%": round(p["年化收益"] * 100, 2),
        "累计%": round(p["累计收益"] * 100, 1),
        "夏普": round(p["夏普比率"], 2),
        "回撤%": round(p["最大回撤"] * 100, 1),
        "卡玛": round(p["卡玛比率"], 2),
        "超额年化%": round(p["超额年化"] * 100, 2),
        "信息比": round(p["信息比率"], 2),
        "换手": round(p.get("年化单边换手", 0), 1),
        "exec": r.diagnostics.get("执行次数"),
        "brake": r.diagnostics.get("刹车清仓次数"),
    }


def main() -> int:
    mats, mask, bench, cb, cs, sc = load()
    close = mats["close"]
    print(f"信号日 {len(sc)} · 评分列 {sc.shape[1]}")

    # 预置闸门
    gates = {
        "无闸门": None,
        "MA20": sl.market_regime(bench, 20),
        "MA60": sl.market_regime(bench, 60),
        "MA60迟滞": sl.market_regime_hysteresis(bench, 60, enter=0.0, exit_=-0.02),
        "MA20迟滞": sl.market_regime_hysteresis(bench, 20, enter=0.0, exit_=-0.02),
        "宽度50%": sl.market_regime_breadth(close, 60, 0.5),
        "宽度55%": sl.market_regime_breadth(close, 60, 0.55),
        "宽度60%": sl.market_regime_breadth(close, 60, 0.60),
    }
    print("\n闸门开放比例：")
    for k, v in gates.items():
        if v is None:
            print(f"  {k:10s} —")
        else:
            days = v.reindex(close.index).fillna(False)
            print(f"  {k:10s} {days.mean():6.1%}")

    rows = []

    def rec(stage, tag, **kw):
        cfg = make(**dict(RAW, **kw.pop("cfg", {})))
        reg = kw.pop("regime", None)
        try:
            st = run_one(sc, mats, mask, bench, cb, cs, cfg, reg)
            rows.append({"阶段": stage, "方案": tag, **st})
            print(f"  {tag:30s} 年化{st['年化%']:7.2f}%  回撤{st['回撤%']:6.1f}%  "
                  f"超额{st['超额年化%']:7.2f}%  夏普{st['夏普']:5.2f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag} 失败: {type(exc).__name__}: {exc}")

    print("\n[A] 权重方案（沿用当前 MA60 闸门）")
    for w in ("equal", "inv_vol", "rank_decay"):
        rec("A权重", f"MA60+{w}", regime=gates["MA60"], cfg={"weighting": w})

    print("\n[B] 持仓只数")
    for n in (10, 20, 30, 50):
        rec("B持仓", f"MA60+inv_vol+{n}只", regime=gates["MA60"],
            cfg={"weighting": "inv_vol", "top_n": n})

    print("\n[C] 择时闸门对比（inv_vol+30只）")
    base = {"weighting": "inv_vol", "top_n": 30}
    for gname, g in gates.items():
        rec("C闸门", f"{gname}", regime=g, cfg=base)

    print("\n[D] 回撤刹车（在 C 的最优闸门上叠加）")
    for gname in ("MA60", "MA60迟滞", "宽度50%"):
        for brk in (0.0, 0.10, 0.15):
            tag = f"{gname}+刹车{brk:.0%}" if brk else f"{gname}+无刹车"
            rec("D刹车", tag, regime=gates[gname],
                cfg={**base, "drawdown_brake": brk, "drawdown_brake_scale": 0.5,
                     "drawdown_recover": 0.05})

    df = pd.DataFrame(rows)
    df.to_csv("output/optimize_experiment.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 108)
    print(df.to_string(index=False))

    print("\n按夏普排序前 8：")
    print(df.sort_values("夏普", ascending=False).head(8).to_string(index=False))
    print("\n按卡玛排序前 8：")
    print(df.sort_values("卡玛", ascending=False).head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
