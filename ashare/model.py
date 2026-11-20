"""滚动训练与打分。

采用扩展窗口的滚动重训（walk-forward）：每次只用信号日之前**已经实现完毕**的
标签训练，避免任何前视泄漏。标签需要在 t+1+H 日才能知道，因此训练集的上界
取 t - H 个交易日。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """保存一次滚动训练的全部产物，供回测与信号使用。"""

    predictions: pd.DataFrame            # 长表 date, code, score
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_log: pd.DataFrame = field(default_factory=pd.DataFrame)


class Dataset:
    """把长表因子与标签打包成可直接切片的数组，避免反复重建 DataFrame。"""

    def __init__(self, features: pd.DataFrame, target: pd.Series):
        feats = list(features.columns)
        df = features.copy()
        df["__y__"] = target.reindex(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        self.feature_names = feats
        self.X = df[feats].to_numpy(dtype="float32", copy=False)
        self.y = df["__y__"].to_numpy(dtype="float32", copy=False)
        self.index = df.index
        self.dates = df.index.get_level_values(0)
        self.codes = df.index.get_level_values(1)
        self._uniq_dates = self.dates.unique()

    def _bounds(self, d) -> tuple[int, int]:
        lo = int(np.searchsorted(self.dates, d, "left"))
        hi = int(np.searchsorted(self.dates, d, "right"))
        return lo, hi

    def slice_between(self, d0, d1) -> tuple[np.ndarray, np.ndarray]:
        lo = int(np.searchsorted(self.dates, d0, "left"))
        hi = int(np.searchsorted(self.dates, d1, "right"))
        return self.X[lo:hi], self.y[lo:hi]

    def slice_on(self, d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = self._bounds(d)
        return self.X[lo:hi], self.y[lo:hi], self.codes[lo:hi]

    @property
    def unique_dates(self) -> np.ndarray:
        return self._uniq_dates


def _fit_predict(
    ds: Dataset, pred_dates: list, cfg: Config, importance_acc: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb

    recs = []
    logs = []
    params = dict(cfg.lgb_params)
    horizon = cfg.label_horizon

    for i, d in enumerate(pred_dates):
        pos = int(np.searchsorted(ds.unique_dates, d, "left"))
        # 训练集下界：历史不足一个完整窗口时用扩展窗口（用尽已有数据），
        # 达到 train_days 之后转为滚动窗口。只有样本实在太少才跳过。
        min_train = max(cfg.label_horizon + 20, 250)
        if pos < min_train:
            logs.append({"date": d, "status": "skip:insufficient_history",
                         "n_train": 0, "retrained": False})
            continue
        # 训练集上界：标签必须在 d 日之前已实现完毕，否则就是前视
        d_train_end = ds.unique_dates[pos - horizon - 1]
        d_train_start = ds.unique_dates[max(pos - cfg.train_days - horizon, 0)]

        Xtr, ytr = ds.slice_between(d_train_start, d_train_end)
        ok = np.isfinite(ytr)
        Xtr, ytr = Xtr[ok], ytr[ok]
        if len(ytr) < 5000:
            logs.append({"date": d, "status": "skip:few_samples",
                         "n_train": int(len(ytr)), "retrained": False})
            continue

        model = lgb.train(
            params,
            # 必须显式传 feature_name：Dataset 内部转成 numpy 后列名会丢失，
            # 否则 feature_importance 只能给出 Column_0/Column_1 这类无意义编号。
            lgb.Dataset(Xtr, ytr, free_raw_data=True, feature_name=ds.feature_names),
            num_boost_round=cfg.num_boost_round,
        )
        for name, gain in zip(model.feature_name(), model.feature_importance("gain")):
            importance_acc[name] = importance_acc.get(name, 0.0) + float(gain)

        Xp, _, codes = ds.slice_on(d)
        if len(codes) == 0:
            continue
        score = model.predict(Xp)
        recs.append(pd.DataFrame({"date": d, "code": codes, "score": score}))
        logs.append({"date": d, "status": "ok", "n_train": int(len(ytr)),
                     "n_pred": int(len(codes)), "retrained": True})
        if (i + 1) % 10 == 0:
            log.info("滚动训练 %d/%d，最近训练样本 %d", i + 1, len(pred_dates), len(ytr))

    preds = (pd.concat(recs, ignore_index=True) if recs
             else pd.DataFrame(columns=["date", "code", "score"]))
    if not preds.empty:
        preds["score"] = preds.groupby("date")["score"].rank(pct=True).astype("float32")
    return preds, pd.DataFrame(logs)


def baseline_composite(
    features: pd.DataFrame, cfg: Config, orientation: dict[str, float] | None = None
) -> pd.DataFrame:
    """等权因子合成基线：所有因子分位等权平均。

    这是一个诚实的对照——如果 LightGBM 跑不赢简单等权，就说明模型没学到东西。
    """
    oriented = features.copy()
    if orientation:
        for k, sign in orientation.items():
            if k in oriented.columns and sign < 0:
                oriented[k] = 1.0 - oriented[k]
    score = oriented.mean(axis=1, skipna=True)
    out = score.reset_index()
    out.columns = ["date", "code", "score"]
    out["score"] = out.groupby("date")["score"].rank(pct=True).astype("float32")
    return out


def rolling_train(
    features: pd.DataFrame,
    target: pd.Series,
    pred_dates: list,
    cfg: Config,
) -> ModelBundle:
    ds = Dataset(features, target)
    acc: dict[str, float] = {}
    preds, logs = _fit_predict(ds, pred_dates, cfg, acc)
    imp = pd.DataFrame(
        {"factor": list(acc.keys()), "gain": list(acc.values())}
    ).sort_values("gain", ascending=False).reset_index(drop=True)
    if not imp.empty:
        imp["share"] = (imp["gain"] / imp["gain"].sum()).round(4)
    return ModelBundle(predictions=preds, importance=imp, train_log=logs)
