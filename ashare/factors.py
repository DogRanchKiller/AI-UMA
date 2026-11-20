"""因子库：全部由前复权 OHLCV 派生，不含任何未来信息。

因子组织方式为宽表（index=交易日, columns=股票代码），便于向量化横截面运算。
所有因子在返回前不做标准化，标准化由 `cross_sectional_process` 统一完成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 因子中文名 -> 分组，用于前端展示与归因
FACTOR_META: dict[str, tuple[str, str]] = {
    "ret_1": ("1日收益", "动量反转"),
    "ret_5": ("5日收益", "动量反转"),
    "ret_10": ("10日收益", "动量反转"),
    "ret_20": ("20日收益", "动量反转"),
    "ret_60": ("60日收益", "动量反转"),
    "ret_120": ("120日收益", "动量反转"),
    "mom_60_skip5": ("60日动量(剔除近5日)", "动量反转"),
    "mom_250_skip20": ("250日动量(剔除近20日)", "动量反转"),
    "vol_20": ("20日波动率", "波动风险"),
    "vol_60": ("60日波动率", "波动风险"),
    "dvol_20": ("20日下行波动率", "波动风险"),
    "parkinson_20": ("20日Parkinson波动", "波动风险"),
    "atr_ratio": ("ATR占价比", "波动风险"),
    "dd_60": ("60日距高点回撤", "波动风险"),
    "te_60": ("60日跟踪误差", "波动风险"),
    "turn_ratio": ("量比(5/20日)", "流动性"),
    "amihud_20": ("Amihud非流动性", "流动性"),
    "log_amount": ("对数成交额", "流动性"),
    "amount_ratio": ("成交额放大率(5/20日)", "流动性"),
    "vol_cv_20": ("成交量变异系数", "流动性"),
    "corr_pv_20": ("价量相关性", "量价关系"),
    "updown_vol_20": ("涨跌放量比", "量价关系"),
    "obv_slope_20": ("OBV斜率", "量价关系"),
    "vwap_dev": ("VWAP偏离度", "量价关系"),
    "ma5_div": ("5日线乖离", "技术形态"),
    "ma20_div": ("20日线乖离", "技术形态"),
    "ma60_div": ("60日线乖离", "技术形态"),
    "ma_trend": ("20/60日线趋势", "技术形态"),
    "rsi_14": ("RSI(14)", "技术形态"),
    "boll_pos": ("布林带位置", "技术形态"),
    "high_250_dist": ("距250日高点", "技术形态"),
    "low_250_dist": ("距250日低点", "技术形态"),
    "breakout_60": ("60日突破", "技术形态"),
    "consec_up": ("连续上涨天数", "技术形态"),
    "limit_up_cnt_20": ("20日强势天数", "市场情绪"),
    "overnight_5": ("5日隔夜收益", "日内结构"),
    "intraday_5": ("5日日内收益", "日内结构"),
    "upper_shadow_20": ("20日上影线占比", "日内结构"),
    "lower_shadow_20": ("20日下影线占比", "日内结构"),
    "beta_60": ("60日Beta", "市场关联"),
    "alpha_60": ("60日Alpha", "市场关联"),
    "rel_ret_20": ("20日相对基准", "市场关联"),
    "corr_bench_60": ("60日与基准相关性", "市场关联"),
}

FACTOR_GROUPS = sorted({v[1] for v in FACTOR_META.values()})


def factor_names() -> list[str]:
    return list(FACTOR_META.keys())


# --------------------------------------------------------------------------- #
# 宽表构建
# --------------------------------------------------------------------------- #
def build_matrices(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """把逐股 DataFrame 汇总为字段宽表。"""
    fields = ["open", "high", "low", "close", "volume"]
    out: dict[str, pd.DataFrame] = {f: {} for f in fields}
    for code, df in panel.items():
        if df is None or df.empty:
            continue
        s = df.set_index("date")
        for f in fields:
            out[f][code] = s[f]
    mats = {}
    for f in fields:
        m = pd.DataFrame(out[f]).sort_index()
        mats[f] = m.astype("float32")
    # 成交额（元）：volume 单位为手(100股)，近似成交额 = 收盘价 * 成交量 * 100
    mats["amount"] = (mats["close"] * mats["volume"] * 100.0).astype("float32")
    return mats


def clean_matrices(mats: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """剔除全零/负价等异常，保证后续除法安全。"""
    close = mats["close"].where(mats["close"] > 0)
    out = dict(mats)
    out["close"] = close
    for f in ("open", "high", "low"):
        out[f] = mats[f].where(mats[f] > 0)
    out["volume"] = mats["volume"].where(mats["volume"] >= 0)
    out["amount"] = mats["amount"].where(mats["amount"] >= 0)
    return out


# --------------------------------------------------------------------------- #
# 因子计算
# --------------------------------------------------------------------------- #
def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return (a / b.replace(0.0, np.nan)).astype("float32")


def _pct(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return _safe_div(close, close.shift(n)) - 1.0


def _streak(cond: pd.DataFrame) -> pd.DataFrame:
    """连续为真的天数（今天为真则计入，断掉归零）。纯 numpy 实现，避免逐行循环。"""
    arr = cond.fillna(False).to_numpy(dtype=bool)
    n, m = arr.shape
    rows = np.arange(n)[:, None]
    last_false = np.maximum.accumulate(np.where(~arr, rows, -1), axis=0)
    streak = rows - last_false
    streak[~arr] = 0
    return pd.DataFrame(streak, index=cond.index, columns=cond.columns).astype("float32")


def compute_factors(
    mats: dict[str, pd.DataFrame], benchmark: pd.Series | None = None
) -> dict[str, pd.DataFrame]:
    """计算全部因子，返回 {因子名: 宽表}。"""
    close, open_, high, low = mats["close"], mats["open"], mats["high"], mats["low"]
    vol, amount = mats["volume"], mats["amount"]

    ret1 = _pct(close, 1)
    f: dict[str, pd.DataFrame] = {}

    # ---------------- 动量 / 反转 ----------------
    for n in (1, 5, 10, 20, 60, 120):
        f[f"ret_{n}"] = _pct(close, n)
    f["mom_60_skip5"] = _safe_div(close.shift(5), close.shift(65)) - 1.0
    f["mom_250_skip20"] = _safe_div(close.shift(20), close.shift(270)) - 1.0

    # ---------------- 波动风险 ----------------
    f["vol_20"] = ret1.rolling(20, min_periods=15).std()
    f["vol_60"] = ret1.rolling(60, min_periods=40).std()
    f["dvol_20"] = ret1.where(ret1 < 0).rolling(20, min_periods=10).std()
    hl = np.log(_safe_div(high, low)).pow(2)
    f["parkinson_20"] = hl.rolling(20, min_periods=15).mean().pow(0.5) / (4 * np.log(2)) ** 0.5
    tr = pd.concat(
        [(high - low).stack(), (high - close.shift(1)).abs().stack(),
         (low - close.shift(1)).abs().stack()], axis=1
    ).max(axis=1).unstack()
    f["atr_ratio"] = _safe_div(tr.rolling(14, min_periods=10).mean(), close)
    f["dd_60"] = _safe_div(close, close.rolling(60, min_periods=40).max()) - 1.0

    # ---------------- 流动性 ----------------
    vma5 = vol.rolling(5, min_periods=3).mean()
    vma20 = vol.rolling(20, min_periods=15).mean()
    f["turn_ratio"] = _safe_div(vma5, vma20)
    f["amihud_20"] = _safe_div(ret1.abs(), amount).rolling(20, min_periods=15).mean() * 1e8
    f["log_amount"] = np.log1p(amount.rolling(20, min_periods=15).mean())
    ama5 = amount.rolling(5, min_periods=3).mean()
    ama20 = amount.rolling(20, min_periods=15).mean()
    f["amount_ratio"] = _safe_div(ama5, ama20)
    f["vol_cv_20"] = _safe_div(vol.rolling(20, min_periods=15).std(), vma20)

    # ---------------- 量价关系 ----------------
    f["corr_pv_20"] = ret1.rolling(20, min_periods=15).corr(vol)
    up_mask = (ret1 > 0)
    up_vol = vol.where(up_mask).rolling(20, min_periods=8).mean()
    dn_vol = vol.where(ret1 < 0).rolling(20, min_periods=8).mean()
    f["updown_vol_20"] = _safe_div(up_vol, dn_vol)
    signed_vol = vol * np.sign(ret1.fillna(0.0))
    obv = signed_vol.cumsum()
    f["obv_slope_20"] = _safe_div(obv - obv.shift(20), vma20 * 20)
    vwap20 = _safe_div(amount.rolling(20, min_periods=15).sum(),
                       (vol.rolling(20, min_periods=15).sum() * 100.0))
    f["vwap_dev"] = _safe_div(close, vwap20) - 1.0

    # ---------------- 技术形态 ----------------
    for n in (5, 20, 60):
        ma = close.rolling(n, min_periods=max(3, n // 2)).mean()
        f[f"ma{n}_div"] = _safe_div(close, ma) - 1.0
    ma20 = close.rolling(20, min_periods=15).mean()
    ma60 = close.rolling(60, min_periods=40).mean()
    f["ma_trend"] = _safe_div(ma20, ma60) - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=10).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=10).mean()
    f["rsi_14"] = 100.0 - 100.0 / (1.0 + _safe_div(gain, loss))

    sd20 = close.rolling(20, min_periods=15).std()
    f["boll_pos"] = _safe_div(close - ma20, 2 * sd20)

    hmax = high.rolling(250, min_periods=60).max()
    lmin = low.rolling(250, min_periods=60).min()
    f["high_250_dist"] = _safe_div(close, hmax) - 1.0
    f["low_250_dist"] = _safe_div(close, lmin) - 1.0
    hmax60 = high.rolling(60, min_periods=40).max()
    f["breakout_60"] = (close >= hmax60 * 0.999).astype("float32").where(hmax60.notna())

    f["consec_up"] = _streak((ret1 > 0))

    # ---------------- 市场情绪 ----------------
    strong = (ret1 > 0.095).astype("float32")
    f["limit_up_cnt_20"] = strong.rolling(20, min_periods=10).sum()

    # ---------------- 日内结构 ----------------
    overnight = _safe_div(open_, close.shift(1)) - 1.0
    intraday = _safe_div(close, open_) - 1.0
    f["overnight_5"] = overnight.rolling(5, min_periods=3).mean()
    f["intraday_5"] = intraday.rolling(5, min_periods=3).mean()
    rng = (high - low).replace(0.0, np.nan)
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    f["upper_shadow_20"] = _safe_div(high - body_hi, rng).rolling(20, min_periods=10).mean()
    f["lower_shadow_20"] = _safe_div(body_lo - low, rng).rolling(20, min_periods=10).mean()

    # ---------------- 市场关联 ----------------
    if benchmark is not None and len(benchmark) > 0:
        b = benchmark.reindex(close.index).astype("float32").ffill()
        bret = b / b.shift(1) - 1.0
        # 注意：cov 是「日期×股票」的宽表，var 是按日期的 Series，
        # 必须用 div(..., axis=0) 按行对齐，直接相除会错误地按列对齐而全成 NaN。
        cov = ret1.rolling(60, min_periods=40).cov(bret)
        var = bret.rolling(60, min_periods=40).var()
        beta = cov.div(var.replace(0.0, np.nan), axis=0)
        f["beta_60"] = beta.astype("float32")
        bench_ret60 = _pct(b.to_frame("b"), 60)["b"]
        f["alpha_60"] = f["ret_60"].sub(
            beta.mul(bench_ret60, axis=0).astype("float32")
        )
        f["rel_ret_20"] = f["ret_20"].sub(_pct(b.to_frame("b"), 20)["b"], axis=0)
        f["corr_bench_60"] = ret1.rolling(60, min_periods=40).corr(bret)
        te_core = ret1.sub(beta.mul(bret, axis=0))
        f["te_60"] = te_core.rolling(60, min_periods=40).std()

    # 统一裁掉不在元数据里的临时键，并转 float32
    out = {}
    for k in FACTOR_META:
        if k in f:
            out[k] = f[k].astype("float32")
    return out


# --------------------------------------------------------------------------- #
# 横截面处理
# --------------------------------------------------------------------------- #
def cross_sectional_process(
    fac: pd.DataFrame, winsor: float = 0.01
) -> pd.DataFrame:
    """按日做去极值 + 分位标准化，输出 [0,1] 区间的百分位分数。

    A 股风格轮动剧烈，横截面分位比原始值更稳健；同时让不同量纲的因子
    可以在同一模型内可比，也让因子得分可以直接做等权合成。
    """
    if fac.empty:
        return fac
    lo = fac.quantile(winsor, axis=1)
    hi = fac.quantile(1 - winsor, axis=1)
    clipped = fac.clip(lower=lo, upper=hi, axis=0)
    ranked = clipped.rank(axis=1, pct=True)
    return ranked.astype("float32")


def standardize_panel(
    factors: dict[str, pd.DataFrame], winsor: float = 0.01
) -> dict[str, pd.DataFrame]:
    return {k: cross_sectional_process(v, winsor) for k, v in factors.items()}


def to_long(
    factors: dict[str, pd.DataFrame], keep_mask: pd.DataFrame | None = None
) -> pd.DataFrame:
    """宽表因子 -> 长表（MultiIndex: date, code），列名为因子名。"""
    cols = {}
    for name, wide in factors.items():
        s = wide.stack(future_stack=True)
        cols[name] = s
    long = pd.DataFrame(cols)
    long.index = long.index.set_names(["date", "code"])
    if keep_mask is not None:
        mask = keep_mask.stack(future_stack=True).reindex(long.index).fillna(False)
        long = long[mask.astype(bool)]
    return long.astype("float32")
