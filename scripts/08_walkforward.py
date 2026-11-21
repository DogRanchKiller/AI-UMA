"""嵌套 walk-forward 验证：用真正样本外的方式检验组合层参数。

问题：top_n / weighting / 闸门参数都是在完整历史上挑的，样本内表现天然乐观。
分年度稳定性检验只能说明"不是靠某一年"，无法量化过拟合的代价。

做法（嵌套选择）：
    对每个评估年份 Y：
        1. 只用 Y 年之前的信号日，在候选配置里选出样本内夏普最高的那个；
        2. 用该配置交易 Y 年；
        3. 把所有年份的样本外收益拼成一条曲线。

因为回测本身是因果的（t 时点只用到 t 及之前的信息），所以某配置在 [起点, Y) 上的
净值路径不依赖 Y 年之后的数据，可以直接切出来当作样本内评估。

输出三类曲线做对照：
    A. 固定配置     —— 全程用当前默认参数（就是"以全历史选参"的做法，偏乐观）
    B. 嵌套选择     —— 每年只用过去数据选参（真实可得的表现）
    C. 事后最优     —— 每年用当年表现最好的配置（不可能实现，作为上界参考）

用法：
    python scripts/08_walkforward.py
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
from ashare import metrics as mt

RAW = dict(sel_min_amount_ma20=0.0, sel_max_vol_pct=1.0,
           sel_min_ret_20=-1.0, sel_min_ret_5=-1.0)
TD = 244


def load():
    cfg = Config()
    master = ds.master_list(cfg)
    industry = ds.industry_map(cfg)
    # 矩阵用腾讯风格代码（sh600000），universe_codes 必须用同一套命名，
    # 否则 build_mask 会把所有股票判为不可交易，回测静默空跑。
    names = {ds.to_tencent(c): n for c, n in zip(master["code"], master["name"])}
    panel = ds.load_panel(master["code"].tolist() + [cfg.benchmark], cfg, min_rows=120)
    mats = fc.clean_matrices(fc.build_matrices(panel))
    mask = uv.build_mask(mats, names, cfg, universe_codes=list(names))
    if int(mask.sum().sum()) == 0:
        raise SystemExit("可交易池为空，检查 universe_codes 的代码风格是否与矩阵一致")
    bench = mats["close"][cfg.benchmark].dropna()
    cb, cs = uv.tradeable_at_open(mats["open"], mats["close"].shift(1),
                                  list(mats["close"].columns))
    preds = pd.read_csv("output/preds.csv", parse_dates=["date"])
    sc = preds.pivot(index="date", columns="code", values="score") \
               .reindex(columns=mats["close"].columns)
    return mats, mask, bench, cb, cs, sc, industry


def make(**kw):
    c = Config()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def sharpe(eq: pd.Series) -> float:
    r = eq.pct_change().dropna()
    if len(r) < 60 or r.std() == 0:
        return -np.inf
    return float(r.mean() / r.std() * np.sqrt(TD))


def main() -> int:
    mats, mask, bench, cb, cs, sc, industry = load()
    close = mats["close"]

    # 候选闸门：把均线周期、是否迟滞、是否用闸门全部交给嵌套选择去决定。
    # 这一点很关键——如果只在已选定的 MA60 内部比较，就无法检验"为什么是 MA60"
    # 这个本身也来自全历史观察的决定。
    gates: dict[str, pd.Series | None] = {"无闸门": None}
    for ma in (20, 60, 120):
        gates[f"MA{ma}"] = sl.market_regime(bench, ma)
        gates[f"MA{ma}迟滞"] = sl.market_regime_hysteresis(bench, ma, 0.0, -0.02)

    # 候选配置网格
    cands: list[tuple[str, object, object]] = []
    for tn in (10, 15, 25):
        for w in ("equal", "inv_vol"):
            for gname, g in gates.items():
                cands.append((
                    f"n{tn}|{w}|{gname}",
                    g,
                    make(**dict(RAW, top_n=tn, weighting=w)),
                ))
    print(f"候选配置 {len(cands)} 个，信号日 {len(sc)}")

    # 一次性跑完全部候选（回测因果，可安全切片）
    eqs: dict[str, pd.Series] = {}
    for tag, g, cfg in cands:
        r = bt.run(sc, mats, mask, cb, cs, bench, cfg, regime_ok=g, industry=industry)
        # 只保留实际开始交易的区间之后
        eq = r.equity
        start = eq.index[eq.ne(1.0).cumsum() >= 1][0] if (eq.ne(1.0)).any() else eq.index[0]
        eqs[tag] = eq.loc[start:]
    idx = sorted(set().union(*[set(e.index) for e in eqs.values()]))
    idx = pd.DatetimeIndex(idx)
    eqdf = pd.DataFrame({k: v.reindex(idx) for k, v in eqs.items()}).ffill()
    print(f"净值矩阵 {eqdf.shape}")

    years = sorted({d.year for d in idx})
    eval_years = [y for y in years if len(idx[idx.year < y]) >= 2 * TD]
    print(f"评估年份（前置至少 2 年样本内）：{eval_years}")

    rows = []
    rets_nested, rets_fixed, rets_oracle = [], [], []
    rets_ens3, rets_ens5 = [], []
    fixed_tag = "n15|inv_vol|MA60迟滞"

    for y in eval_years:
        is_end = pd.Timestamp(f"{y}-01-01")
        in_idx = idx[idx < is_end]
        y_idx = idx[idx.year == y]
        if len(y_idx) == 0:
            continue

        is_sr = {t: sharpe(eqdf.loc[in_idx, t]) for t in eqdf.columns}
        ranked_cfg = sorted(is_sr, key=is_sr.get, reverse=True)
        best = ranked_cfg[0]
        best_hindsight = max((t for t in eqdf.columns), key=lambda t: sharpe(eqdf.loc[y_idx, t]))

        def yearly_ret(tag):
            seg = eqdf.loc[y_idx, tag]
            if seg.isna().all() or seg.iloc[0] <= 0:
                return np.nan
            return float(seg.iloc[-1] / seg.iloc[0] - 1)

        # 配置集成：把资金等分给样本内前 k 名配置，各自独立运行后收益取平均
        def blend(k: int) -> float:
            vals = [yearly_ret(t) for t in ranked_cfg[:k]]
            vals = [v for v in vals if pd.notna(v)]
            return float(np.mean(vals)) if vals else np.nan

        rn, rf, ro = yearly_ret(best), yearly_ret(fixed_tag), yearly_ret(best_hindsight)
        e3, e5 = blend(3), blend(5)
        rets_nested.append(rn); rets_fixed.append(rf); rets_oracle.append(ro)
        rets_ens3.append(e3); rets_ens5.append(e5)
        rows.append({
            "年份": y,
            "样本内选出": best,
            "样本内夏普": round(is_sr[best], 2),
            "单个收益%": round(rn * 100, 1) if pd.notna(rn) else None,
            "前3集成%": round(e3 * 100, 1) if pd.notna(e3) else None,
            "前5集成%": round(e5 * 100, 1) if pd.notna(e5) else None,
            "固定配置收益%": round(rf * 100, 1) if pd.notna(rf) else None,
            "事后最优配置": best_hindsight,
            "事后最优收益%": round(ro * 100, 1) if pd.notna(ro) else None,
        })
        print(f"  {y}: 选 {best:24s} 样本内{is_sr[best]:5.2f} | "
              f"单个{rn*100:7.1f}%  前3集成{e3*100:7.1f}%  前5集成{e5*100:7.1f}%  "
              f"固定{rf*100:7.1f}%  事后最优{ro*100:7.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv("output/walkforward.csv", index=False, encoding="utf-8-sig")

    # 拼接成净值曲线
    bench_y = {}
    for y in eval_years:
        seg = bench.reindex(idx[idx.year == y])
        if len(seg) > 1:
            bench_y[y] = float(seg.iloc[-1] / seg.iloc[0] - 1)

    def compound(rets):
        v = 1.0
        for r in rets:
            if pd.notna(r):
                v *= (1 + r)
        return v

    print("\n" + "=" * 78)
    n_years = len(eval_years)
    print(f"样本外年份数: {n_years}")
    for label, rets in (("单个最优(嵌套选择)", rets_nested),
                        ("前3配置集成", rets_ens3),
                        ("前5配置集成", rets_ens5),
                        ("固定配置(偏乐观)", rets_fixed),
                        ("事后最优(不可实现)", rets_oracle)):
        arr = np.array([r for r in rets if pd.notna(r)])
        if len(arr) == 0:
            continue
        total = compound(rets)
        ann = total ** (1 / (len(arr))) - 1
        print(f"  {label:22s} 累计 {total-1:8.1%}  年化 {ann:7.2%}  "
              f"胜年 {int((arr>0).sum())}/{len(arr)}  最差年 {arr.min():7.1%}  "
              f"年化波动 {arr.std():6.1%}")
    b_total = compound(list(bench_y.values()))
    print(f"  {'基准(沪深300)':20s} 累计 {b_total-1:8.1%}  "
          f"年化 {b_total**(1/len(bench_y))-1:7.2%}")

    # 选择一致性：每年选出来的配置有多少次相同
    picks = df["样本内选出"].value_counts()
    print(f"\n嵌套选择被选中的配置分布（共 {n_years} 次）：")
    for k, v in picks.items():
        print(f"  {k:26s} {v} 次")
    print("\n选择越分散，说明样本内优势越不稳定；若每年都选同一个，"
          "则该配置相对稳健。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
