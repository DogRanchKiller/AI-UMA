"""Verify the production engine against the frozen 'known' (signal-close) replay."""
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parent
OUTPUT = EXPERIMENT.parent
METRICS = ["annual", "drawdown", "sharpe", "2025"]


def main():
    build = run_path(str(OUTPUT.parent / "scripts" / "02_build.py"))
    cfg = build["Config"].load(OUTPUT / "config_used.json")
    _, names, panel = build["load_inputs"](cfg)
    fc, uv = build["fc"], build["uv"]
    mats = fc.clean_matrices(fc.build_matrices(panel))
    del panel
    close = mats["close"]
    bench = close[cfg.benchmark].dropna()
    mask = uv.build_mask(mats, names, cfg, universe_codes=list(names))
    buy, sell = uv.tradeable_at_open(mats["open"], close.shift(), list(close))
    industry = build["ds"].industry_map(cfg)
    ref = pd.read_csv(EXPERIMENT / "gate_replay.csv").query("mode == 'known'")
    ref = ref.set_index(["model", "gate"])[METRICS].sort_index()
    periods = pd.read_csv(EXPERIMENT / "gate_periods.csv", parse_dates=["execution"])
    periods = periods.query("mode == 'known'").set_index(["gate", "execution"]).sort_index()
    records, runs = [], {}
    baseline_index = None
    for model, path in [("base", OUTPUT), ("ext", EXPERIMENT)]:
        preds = pd.read_csv(path / "preds.csv", parse_dates=["date"], dtype={"score": "float32"})
        scores = preds.set_index(["date", "code"])["score"].sort_index()
        if baseline_index is None:
            baseline_index = scores.index
        else:
            pd.testing.assert_index_equal(baseline_index, scores.index)
        scores = scores.unstack().reindex(columns=close.columns)
        scores, gate = build["po"].build_target_scores(scores, mats, mask, bench, cfg)
        for gated in (False, True):
            # Raw signal-dated inputs: no shifts, patches or retraining.
            res = build["bt"].run(scores, mats, mask, buy, sell, bench, cfg,
                                  regime_ok=gate if gated else None, industry=industry)
            eq = res.equity
            ret_2025 = eq.loc["2025"].iloc[-1] / eq.loc[:"2024"].iloc[-1] - 1
            records.append([model, gated, res.stats["年化收益"], res.stats["最大回撤"],
                            res.stats["夏普比率"], ret_2025])
            runs[model, gated] = res

    result = pd.DataFrame(records, columns=["model", "gate", *METRICS])
    actual = result.set_index(["model", "gate"]).sort_index()
    pd.testing.assert_index_equal(actual.index, ref.index)
    np.testing.assert_allclose(actual, ref, rtol=0, atol=1e-10)
    for gated in (False, True):
        base, ext = runs["base", gated], runs["ext", gated]
        relative = np.log(ext.equity / base.equity).diff().fillna(0)
        dates = close.index
        indices = dates.get_indexer(scores.index)
        assert (indices >= 0).all()
        starts = dates[indices[indices + 1 < len(dates)] + 1]
        archived = periods.loc[gated]
        assert starts.equals(archived.index)
        for i, start in enumerate(starts):
            stop = starts[i + 1] if i + 1 < len(starts) else dates[-1] + pd.Timedelta(days=1)
            ref_row = archived.loc[start]
            base_codes, ext_codes = base.holdings_by_date[start], ext.holdings_by_date[start]
            assert (len(base_codes), len(ext_codes), len(base_codes.keys() & ext_codes.keys())) == (
                ref_row.base_n, ref_row.ext_n, ref_row.common)
            log_relative = relative.loc[(dates >= start) & (dates < stop)].sum()
            np.testing.assert_allclose(log_relative, ref_row.log_relative, rtol=0, atol=1e-10)
    result.to_csv(EXPERIMENT / "formal_gate_replay.csv", index=False)
    print(actual.to_string())
    print(f"Matched {len(baseline_index):,} prediction keys, {len(periods)} paired periods, "
          f"{actual.size} summary values; max summary error: {(actual - ref).abs().max().max():.3g}")


if __name__ == "__main__":
    main()
