"""步骤 2：构建因子、训练模型、回测，并落地全部结果。

用法：
    python scripts/02_build.py                 # 默认参数跑一遍
    python scripts/02_build.py --top-n 30      # 覆盖参数
    python scripts/02_build.py --no-cache      # 强制重算因子

输出（output/ 下）：
    factors_meta.json   因子清单与分组
    preds.csv           每个调仓日的全市场打分
    backtest_equity.csv 策略与基准净值
    backtest_trades.csv 成交明细
    metrics.json        绩效指标
    importance.csv      因子重要性
    ic.csv              逐日 IC
    layers.csv          分层收益
    signal.json         最新一期买卖建议
    run_summary.json    运行摘要（供前端读取）
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ashare.config import CACHE_DIR, OUT_DIR, Config
from ashare import datasource as ds
from ashare import factors as fc
from ashare import labels as lb
from ashare import metrics as mt
from ashare import model as md
from ashare import portfolio as po
from ashare import signals as sg
from ashare import universe as uv
from ashare import backtest as bt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
log = logging.getLogger("build")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return str(o)


def dump(obj, name: str) -> None:
    (OUT_DIR / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def load_inputs(cfg: Config):
    master = ds.master_list(cfg)
    names = {
        ds.to_tencent(c): n
        for c, n in zip(master["code"], master["name"])
    }
    codes = master["code"].tolist() + [cfg.benchmark, "sh000905", "sh000852", "sh000001"]
    panel = ds.load_panel(codes, cfg, min_rows=120)
    log.info("有效面板：%d 只", len(panel))
    if len(panel) < 100:
        raise SystemExit("可用数据太少，请先运行 scripts/01_fetch.py")
    return master, names, panel


def build_pipeline(cfg: Config, use_cache: bool = True, top_n: int | None = None):
    if top_n:
        cfg.top_n = top_n
    t0 = time.time()

    master, names, panel = load_inputs(cfg)
    mats = fc.clean_matrices(fc.build_matrices(panel))
    close, open_ = mats["close"], mats["open"]
    log.info("行情矩阵：%s，耗时 %.0fs", close.shape, time.time() - t0)

    bench = close[cfg.benchmark].dropna() if cfg.benchmark in close.columns else None
    if bench is None:
        raise SystemExit(f"缺少基准 {cfg.benchmark}")

    mask = uv.build_mask(mats, names, cfg, universe_codes=list(names))
    log.info("可交易标的日均 %d 只", int(mask.sum(axis=1).mean()))

    cache_f = CACHE_DIR / "factors.pkl"
    if use_cache and cache_f.exists():
        factors = pickle.loads(cache_f.read_bytes())
        log.info("因子命中缓存")
    else:
        factors = fc.compute_factors(mats, benchmark=bench)
        factors = fc.standardize_panel(factors)
        cache_f.write_bytes(pickle.dumps(factors))
        log.info("因子计算完成：%d 个，耗时 %.0fs", len(factors), time.time() - t0)

    # 长表
    X = fc.to_long(factors, mask)
    y_raw = lb.build_label(open_, bench, cfg.label_horizon, mask, normalize=False)
    y_rank = lb.build_label(open_, bench, cfg.label_horizon, mask, normalize=True)
    y = y_rank.stack(future_stack=True).reindex(X.index).astype("float32")
    X = X.loc[y.index]
    log.info("训练样本：%s 行 × %d 因子", f"{len(X):,}", X.shape[1])

    # 调仓日
    dates = close.index
    usable = [d for d in dates if d in factors["ret_20"].index]
    first_pos = cfg.warmup_days
    if len(usable) <= first_pos + cfg.train_days:
        raise SystemExit("历史长度不足，无法完成滚动训练")
    pred_dates = usable[first_pos:: cfg.rebalance_every]
    if usable[-1] not in pred_dates:
        pred_dates.append(usable[-1])
    log.info("信号日 %d 个：%s ~ %s", len(pred_dates), pred_dates[0].date(), pred_dates[-1].date())

    # 滚动训练（带缓存：模型训练是确定性的，重复跑只会浪费时间）
    cache_p = CACHE_DIR / "preds.pkl"
    cache_k = CACHE_DIR / "preds_key.json"
    key = {"v": 2, "h": cfg.label_horizon, "td": cfg.train_days, "re": cfg.retrain_every,
           "nbr": cfg.num_boost_round, "start": cfg.start_date,
           "factors": str(cache_f.stat().st_mtime) if cache_f.exists() else "0"}
    hit = False
    if use_cache and cache_p.exists() and cache_k.exists():
        try:
            if json.loads(cache_k.read_text(encoding="utf-8")) == key:
                bundle = pickle.loads(cache_p.read_bytes())
                hit = True
                log.info("预测命中缓存")
        except Exception:  # noqa: BLE001 - 缓存损坏则重训
            hit = False
    if not hit:
        bundle = md.rolling_train(X, y, pred_dates, cfg)
        cache_p.write_bytes(pickle.dumps(bundle))
        cache_k.write_text(json.dumps(key), encoding="utf-8")
    preds = bundle.predictions
    log.info("预测完成：%d 条，耗时 %.0fs", len(preds), time.time() - t0)

    # 等权因子基线（同口径对照）
    base = md.baseline_composite(X, cfg)
    base = base[base["date"].isin(pred_dates)]

    # 打分宽表：裸信号
    sc_raw = preds.pivot(index="date", columns="code", values="score").reindex(columns=close.columns)
    sc_base = base.pivot(index="date", columns="code", values="score").reindex(columns=close.columns)

    # 组合构建：中性化 / 选股约束 / 仓位闸门
    sc, regime = po.build_target_scores(sc_raw, mats, mask, bench, cfg)
    sc_base, _ = po.build_target_scores(sc_base, mats, mask, bench, cfg)

    # 交易约束矩阵
    prev_close = close.shift(1)
    can_buy, can_sell = uv.tradeable_at_open(open_, prev_close, list(close.columns))

    # 回测
    log.info("开始回测……闸门=%s 周期=%s", cfg.use_regime_filter, cfg.regime_ma)
    res = bt.run(sc, mats, mask, can_buy, can_sell, bench, cfg, regime_ok=regime)
    res_base = bt.run(sc_base, mats, mask, can_buy, can_sell, bench, cfg, regime_ok=regime)
    # 裸信号对照：不加任何组合约束，用于说明约束层的实际价值
    res_raw = bt.run(sc_raw, mats, mask, can_buy, can_sell, bench, cfg)
    log.info("回测完成，耗时 %.0fs", time.time() - t0)

    # 评估：IC / 分层（只评估有已实现标签的信号日）
    realized_excess = y_raw.reindex(sc_raw.index)
    ic = mt.ic_series(sc_raw, realized_excess)
    ic_base = mt.ic_series(sc_base, realized_excess)
    layers = mt.layered_returns(sc_raw, realized_excess, n=5)

    imp = bundle.importance
    if not imp.empty:
        imp["name"] = imp["factor"].map(lambda k: fc.FACTOR_META.get(k, (k, ""))[0])
        imp["group"] = imp["factor"].map(lambda k: fc.FACTOR_META.get(k, (k, ""))[1])

    # 最新信号
    log.info("生成最新信号……")
    last = sc.index.max()
    sig = sg.build_signal(sc, close, factors, names, last, top_n=cfg.top_n)
    # 闸门状态：未入场时信号只作观察，不应直接照做
    if regime is not None:
        near = regime.reindex(regime.index[regime.index <= last])
        risk_on = bool(near.iloc[-1]) if len(near) else True
        sig["risk_on"] = risk_on
        sig["regime_ma"] = cfg.regime_ma
        if not risk_on:
            sig["notice"] = (
                f"市场闸门关闭：基准位于 {cfg.regime_ma} 日均线下方，"
                "系统当前建议空仓观望，不应新建仓。"
            )

    # ---------------- 落地 ----------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUT_DIR / "preds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"equity": res.equity, "benchmark": res.benchmark}).to_csv(
        OUT_DIR / "backtest_equity.csv", index_label="date", encoding="utf-8-sig"
    )
    if not res.trades.empty:
        res.trades.to_csv(OUT_DIR / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    res.weights.to_csv(OUT_DIR / "weights.csv", index_label="date", encoding="utf-8-sig")
    imp.to_csv(OUT_DIR / "importance.csv", index=False, encoding="utf-8-sig")
    ic.rename("ic").to_csv(OUT_DIR / "ic.csv", index_label="date", encoding="utf-8-sig")
    if not layers.empty:
        layers.to_csv(OUT_DIR / "layers.csv", index_label="date", encoding="utf-8-sig")
    dump(sig, "signal.json")
    dump(
        {"factors": {k: {"name": v[0], "group": v[1]} for k, v in fc.FACTOR_META.items()},
         "groups": fc.FACTOR_GROUPS},
        "factors_meta.json",
    )

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg.to_dict(),
        "universe": {
            "stocks_in_panel": len(panel),
            "avg_daily_tradable": int(mask.sum(axis=1).mean()),
            "date_range": [str(close.index.min().date()), str(close.index.max().date())],
        },
        "model": {
            "n_factors": len(fc.FACTOR_META),
            "n_train_rows": int(len(X)),
            "n_signals": int(len(preds["date"].unique())),
            "last_signal_date": str(last.date()),
        },
        "performance": res.stats,
        "performance_raw_signal": res_raw.stats,
        "performance_baseline_factor": res_base.stats,
        "diagnostics": res.diagnostics,
        "ic": mt.ic_summary(ic),
        "ic_baseline_factor": mt.ic_summary(ic_base),
        "run_seconds": round(time.time() - t0, 1),
    }
    dump(summary, "run_summary.json")

    log.info("=" * 60)
    for k, v in res.stats.items():
        if isinstance(v, float):
            log.info("%-12s %.4f", k, v)
        else:
            log.info("%-12s %s", k, v)
    log.info("-" * 60)
    log.info("IC均值 %.4f  ICIR %.2f  IC胜率 %.2f",
             mt.ic_summary(ic).get("IC均值", float("nan")),
             mt.ic_summary(ic).get("ICIR", float("nan")),
             mt.ic_summary(ic).get("IC胜率", float("nan")))
    log.info("总耗时 %.0fs", time.time() - t0)
    return res, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--start", default="")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--rebalance", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.start:
        cfg.start_date = args.start
    if args.horizon:
        cfg.label_horizon = args.horizon
    if args.rebalance:
        cfg.rebalance_every = args.rebalance
    cfg.save(OUT_DIR / "config_used.json")
    build_pipeline(cfg, use_cache=not args.no_cache, top_n=args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
