"""全链路冒烟测试：用少量股票快速验证各模块能跑通且数值合理。"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ashare.config import Config
from ashare import datasource as ds
from ashare import factors as fc
from ashare import labels as lb
from ashare import universe as uv
from ashare import model as md
from ashare import backtest as bt
from ashare import metrics as mt
from ashare import signals as sg

N = 200


def step(name):
    print(f"\n{'='*20} {name} {'='*20}")


def main() -> int:
    cfg = Config()
    cfg.train_days = 400
    cfg.top_n = 15

    step("载入面板")
    master = ds.master_list(cfg)
    codes = master["code"].tolist()[:N]
    names = {ds.to_tencent(c): n for c, n in zip(master["code"], master["name"])}
    panel = ds.load_panel(codes + [cfg.benchmark], cfg, min_rows=120)
    print(f"面板 {len(panel)} 只")
    assert len(panel) > 50, "面板太小"

    step("矩阵与股票池")
    mats = fc.clean_matrices(fc.build_matrices(panel))
    close = mats["close"]
    print(f"矩阵 {close.shape}, 日期 {close.index.min().date()} ~ {close.index.max().date()}")
    mask = uv.build_mask(mats, names, cfg)
    print(f"日均可选 {mask.sum(axis=1).mean():.0f} 只 / 共 {close.shape[1]} 只")

    step("因子计算")
    bench = close[cfg.benchmark].dropna()
    # 行业因子属于可选项：默认配置下不参与计算，因此冒烟测试按默认口径校验，
    # 并单独确认"关闭时确实不生成、开启时确实生成"。
    fac = fc.compute_factors(mats, benchmark=bench)
    print(f"因子 {len(fac)} 个（默认不含行业因子）")
    if cfg.use_industry_factors:
        expected = [k for k in fc.FACTOR_META if not k.startswith("ind_rel_")]
    else:
        expected = [k for k in fc.FACTOR_META if not k.startswith("ind_rel_")]
    missing = [k for k in expected if k not in fac]
    assert not missing, f"缺失因子: {missing}"
    if not cfg.use_industry_factors:
        assert not any(k.startswith("ind_rel_") for k in fac), \
            "行业因子关闭时不应出现在结果里"
    std = fc.standardize_panel(fac)
    for k in ["ret_20", "vol_20", "rsi_14", "beta_60"]:
        v = std[k].stack().dropna()
        print(f"  {k:14s} 分位区间 [{v.min():.3f}, {v.max():.3f}] 中位 {v.median():.3f}")
        assert -0.01 <= v.min() <= 1.01 and -0.01 <= v.max() <= 1.01, f"{k} 未归一化"
        assert 0.35 < v.median() < 0.65, f"{k} 中位异常"

    step("标签")
    y_raw = lb.build_label(mats["open"], bench, cfg.label_horizon, mask, normalize=False)
    y_rank = lb.build_label(mats["open"], bench, cfg.label_horizon, mask, normalize=True)
    s = y_raw.stack().dropna()
    print(f"超额收益: 均值 {s.mean():.4f} 标准差 {s.std():.4f} min {s.min():.3f} max {s.max():.3f}")
    sr = y_rank.stack().dropna()
    assert 0.45 < sr.median() < 0.55, "标签分位中位应接近 0.5"

    step("长表与数据集")
    X = fc.to_long(std, mask)
    y = y_rank.stack(future_stack=True).reindex(X.index).astype("float32")
    X = X.loc[y.index]
    print(f"样本 {len(X):,} 行 × {X.shape[1]} 因子")
    dsv = md.Dataset(X, y)
    print(f"唯一日期 {len(dsv.unique_dates)}")

    step("滚动训练（少量信号日）")
    dates = close.index
    # 从有足够历史的位置开始取信号日，且必须留出下一个交易日供成交
    pred_dates = list(dates[520::40])[:8]
    if dates[-1] not in pred_dates:
        pred_dates.append(dates[-1])
    bundle = md.rolling_train(X, y, pred_dates, cfg)
    print(f"预测 {len(bundle.predictions)} 条，训练日志：")
    print(bundle.train_log.to_string())
    assert not bundle.predictions.empty, "预测为空"

    step("回测")
    sc = bundle.predictions.pivot(index="date", columns="code", values="score")
    sc = sc.reindex(columns=close.columns)
    cb, cs = uv.tradeable_at_open(mats["open"], close.shift(1), list(close.columns))
    res = bt.run(sc, mats, mask, cb, cs, bench, cfg)
    print(f"净值样本 {len(res.equity)}，期末 {res.equity.iloc[-1]:.4f}")
    print("诊断:", res.diagnostics)
    print("指标:", {k: round(v, 4) for k, v in res.stats.items() if isinstance(v, float)})
    assert res.equity.notna().all(), "净值含 NaN"
    assert res.equity.iloc[-1] > 0, "净值非正"

    step("IC 与分层")
    ic = mt.ic_series(sc, y_raw.reindex(sc.index))
    print("IC:", mt.ic_summary(ic))
    layers = mt.layered_returns(sc, y_raw.reindex(sc.index), n=5)
    print("分层:\n", layers.mean().round(4).to_string())

    step("信号")
    sig = sg.build_signal(sc, close, std, names, sc.index.max(), top_n=cfg.top_n)
    print(sg.render_text(sig))

    print("\n全部冒烟测试通过")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
