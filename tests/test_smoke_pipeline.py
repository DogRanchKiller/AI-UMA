"""在临时目录从逐股日线 CSV 运行正式构建，不读取已有行情、模型或报告。"""

import json
import pickle
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

from ashare import datasource as ds
from ashare.config import Config


def test_offline_raw_bars_to_trades_metrics_and_latest_signal(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    raw = tmp_path / "raw"
    out = tmp_path / "output"
    cache.mkdir()
    monkeypatch.setattr(ds, "CACHE_DIR", cache)
    monkeypatch.setattr(ds, "RAW_DIR", raw)
    monkeypatch.setattr(ds, "_bs_session", lambda: pytest.fail("冒烟测试不应访问行情服务"))
    build = run_path(str(Path(__file__).resolve().parents[1] / "scripts/02_build.py"))["build_pipeline"]
    monkeypatch.setitem(build.__globals__, "CACHE_DIR", cache)
    monkeypatch.setitem(build.__globals__, "OUT_DIR", out)

    # 100 只个股跨过正式构建的面板门槛；其中 1 只 ST 不可入池。
    codes = [f"sh{600000 + i}" for i in range(100)]
    names = [f"合成{i:02d}" for i in range(99)] + ["ST合成"]
    pd.DataFrame({"code": codes, "name": names, "kind": "stock"}).to_csv(cache / "master_list.csv", index=False)
    pd.DataFrame({"code": codes, "industry": "合成行业"}).to_csv(cache / "industry.csv", index=False)
    cfg = Config(warmup_days=270, train_days=60, rebalance_every=60, label_horizon=5,
                 min_list_days=10, num_boost_round=8, top_n=4, weighting="equal")
    cfg.lgb_params.update(num_threads=1, feature_fraction=1, bagging_fraction=1)
    dates = pd.bdate_range("2022-01-03", periods=332)
    t = np.arange(len(dates))
    opens, closes = {}, {}
    for i, code in enumerate([*codes, cfg.benchmark, "sh000905"]):
        # 个股有不同漂移和相位；指数缓升使实际仓位闸门开放。开盘和收盘不等。
        drift = (i - 45) / 40000 if code in codes else .0007
        op = 20 * np.exp(drift * t + .002 * np.sin(t / 9 + i))
        cl = op * (1 + .003 * np.cos(t / 7 + i))
        hands = 60000 + 100 * (t % 21) + i
        opens[code], closes[code] = op, cl
        bars = pd.DataFrame({"code": code, "date": dates, "open": op, "close": cl,
                             "high": np.maximum(op, cl) * 1.01, "low": np.minimum(op, cl) * .99,
                             "volume": hands, "amount": hands * 100 * (op + cl) / 2, "adj_factor": 1.0})
        path = ds._cache_path(code)
        path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(path, index=False)

    result, summary = build(cfg, use_cache=False)
    first, last = dates[270], dates[-1]
    preds = pd.read_csv(out / "preds.csv", parse_dates=["date"]).pivot(index="date", columns="code", values="score")
    assert list(preds.index) == [first, dates[330], last]
    assert set(preds.columns) == set(codes[:-1])  # ST 和两个指数均不得有预测
    assert preds.notna().all().all() and preds.nunique(axis=1).min() > cfg.top_n
    with (cache / "preds.pkl").open("rb") as f:  # 仅本用例刚生成的缓存
        fitted = pickle.load(f)
    assert fitted.train_log.retrained.all() and fitted.train_log.status.eq("ok").all()
    assert fitted.train_log.n_train.tolist() == [cfg.train_days * 99] * len(preds)
    assert fitted.importance.gain.sum() > 0
    with (cache / "factors.pkl").open("rb") as f:
        factors = pickle.load(f)
    momentum = pd.Series({c: cl[270] / cl[250] - 1 for c, cl in closes.items()})
    clipped = momentum.clip(momentum.quantile(.01), momentum.quantile(.99))
    np.testing.assert_allclose(factors["ret_20"].loc[first, momentum.index], clipped.rank(pct=True), atol=1e-6)

    # 第一笔建仓只允许在下一交易日；用实际开盘、整手、手续费和收盘重算净值。
    day = dates[271]
    assert result.equity.loc[:first].eq(1).all()
    assert result.trades.date.min() == day
    buys = result.trades.loc[result.trades.date.eq(day)].set_index("code")
    assert len(buys) == cfg.top_n and buys.side.eq("buy").all()
    assert (buys.shares > 0).all() and (buys.shares % 100 == 0).all()
    assert preds.loc[first, buys.index].min() >= preds.loc[first].drop(buys.index).max()
    cash = cfg.init_cash
    value = 0.0
    for c, order in buys.iterrows():
        price = opens[c][271] * (1 + cfg.slippage)
        gross = price * order.shares
        fee = max(cfg.min_commission, gross * cfg.commission)
        assert order.price == pytest.approx(price) and order.fee == pytest.approx(fee)
        cash -= gross + fee
        value += closes[c][271] * order.shares
    assert result.equity.at[day] == pytest.approx((cash + value) / cfg.init_cash)

    # 只有首期已有到期开盘；独立从合成原始价重算两腿超额，再对照 IC 和五层均值。
    exit_pos = 271 + cfg.label_horizon
    bench = opens[cfg.benchmark]
    excess = pd.Series({c: opens[c][exit_pos] / opens[c][271] - bench[exit_pos] / bench[271]
                        for c in preds.columns})
    scores = preds.loc[first]
    ic = pd.read_csv(out / "ic.csv", index_col="date", parse_dates=True)
    layers = pd.read_csv(out / "layers.csv", index_col="date", parse_dates=True)
    assert list(ic.index) == list(layers.index) == [first]
    assert ic.at[first, "ic"] == pytest.approx(scores.rank().corr(excess.rank()), abs=1e-6)
    groups = pd.qcut(scores.rank(method="first"), 5, labels=False)
    np.testing.assert_allclose(layers.loc[first], excess.groupby(groups).mean(), atol=2e-7)
    assert summary["ic"]["IC样本数"] == 1

    signal = json.loads((out / "signal.json").read_text(encoding="utf-8"))
    assert signal["market_date"] == signal["score_date"] == str(last.date())
    assert signal["risk_on"] is True
    picks = signal["buy"]
    assert len({p["code"] for p in picks}) == len(picks) == cfg.top_n
    latest = preds.loc[last]
    expected = latest.nlargest(cfg.top_n).to_numpy()
    np.testing.assert_allclose([latest[p["code"]] for p in picks], expected)
    for pick in picks:
        c = pick["code"]
        assert pick["score"] == pytest.approx(latest[c], abs=5.1e-5)
        assert pick["price"] == round(closes[c][-1], 2)
        assert pick["name"] == names[codes.index(c)] and pick["industry"] == "合成行业"
