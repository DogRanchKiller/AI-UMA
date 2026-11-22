"""在生产数据交界处验证防前视：收盘训练所见的标签、公告转日频后的可见值。"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from ashare.config import Config
from ashare.extdata import _to_daily
from ashare.labels import forward_return
from ashare.model import rolling_train


def test_training_uses_only_labels_realized_by_signal_close(monkeypatch):
    # 合成交易历；足够的日期和股票使真实训练路径越过预热期及最低样本数。
    dates = pd.bdate_range(end="2024-03-15", periods=265, name="date")
    codes = pd.Index([f"sh{600000 + i}" for i in range(24)], name="code")
    row_ids = np.arange(len(dates) * len(codes))
    opens = pd.DataFrame(100.0 + row_ids.reshape(len(dates), -1), index=dates, columns=codes)
    cfg = Config(label_horizon=3, train_days=len(dates))
    labels = forward_return(opens, cfg.label_horizon).stack(future_stack=True)
    features = pd.DataFrame({"row_id": row_ids}, index=labels.index)

    # 3/11（周一）收盘预测。3/5 的标签：3/6 开盘进、3/11 开盘出；
    # 3/6 的标签要到 3/12 才兑现。完整行情给两者都赋了值，不能靠过滤 NaN 过关。
    signal_date = pd.Timestamp("2024-03-11")
    known_labels = labels.loc[:pd.Timestamp("2024-03-05")]

    class ReachedFit(Exception):
        """在拟合入口停下，不训练树。"""

    def inspect_fit(params, train_set, **kwargs):
        np.testing.assert_array_equal(train_set.data[:, 0], features.loc[known_labels.index, "row_id"])
        np.testing.assert_array_equal(train_set.get_label(), known_labels.to_numpy())
        raise ReachedFit

    monkeypatch.setattr(lgb, "train", inspect_fit)
    with pytest.raises(ReachedFit):
        rolling_train(features, labels, [signal_date], cfg)


def test_announcements_become_visible_on_the_next_trading_session():
    # 4/3 收盘后至 4/8 开盘之间无交易日；4/6 是周六。
    calendar = pd.to_datetime(["2024-04-02", "2024-04-03", "2024-04-08",
                               "2024-04-09", "2024-04-10", "2024-04-11"])
    announcements = pd.DataFrame([
        ("sz000001", "2024-04-02", 10.0),
        ("sz000001", "2024-04-03", 12.0),
        ("sz000002", "2024-04-06", 20.0),
        ("sz000001", "2024-04-10", 18.0),
    ], columns=["code", "ann_date", "holders"])

    actual = _to_daily(announcements, "ann_date", ["holders"], calendar)["holders"]
    expected = pd.DataFrame({
        "sz000001": [np.nan, 10.0, 12.0, 12.0, 12.0, 18.0],
        "sz000002": [np.nan, np.nan, 20.0, 20.0, 20.0, 20.0],
    }, index=calendar).rename_axis(columns="code")
    pd.testing.assert_frame_equal(actual, expected)
