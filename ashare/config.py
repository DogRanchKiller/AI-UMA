"""全局配置。所有可调参数集中在此，便于回测对比。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = ROOT / "output"

for _d in (DATA_DIR, RAW_DIR, CACHE_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # ---------- 取数 ----------
    start_date: str = "2019-01-01"      # 历史起点（留足因子预热）
    end_date: str = ""                  # 空 = 今天
    benchmark: str = "sh000300"         # 基准：沪深300

    # ---------- 股票池 ----------
    min_list_days: int = 250            # 上市满 250 个交易日后才可交易
    min_price: float = 2.0              # 剔除低价股
    min_amount_ma20: float = 3e7        # 20日均成交额下限（元），约 3000 万
    exclude_st: bool = True             # 剔除 ST / *ST
    exclude_star: bool = True           # 剔除科创板(688)，涨跌幅规则不同
    exclude_bse: bool = True            # 剔除北交所

    # ---------- 因子 / 标签 ----------
    label_horizon: int = 10             # 预测未来 N 个交易日超额收益
    warmup_days: int = 260              # 因子预热期（最长回看窗口）

    # ---------- 模型 ----------
    train_days: int = 750               # 滚动训练窗口（交易日）
    retrain_every: int = 20             # 每 N 个交易日重训一次
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": 0,
        "seed": 42,
    })
    num_boost_round: int = 300

    # ---------- 组合 ----------
    top_n: int = 20                     # 每期持仓只数
    rebalance_every: int = 20           # 调仓周期（交易日）
    max_weight: float = 0.10            # 单票权重上限
    min_score_pct: float = 0.0          # 入选最低分位要求（0 = 不限制）

    # ---------- 选股端风险约束（不改动模型打分） ----------
    # 实测：额外的流动性下限与自由落体过滤会把 alpha 的来源一起滤掉
    # （见 scripts/04_tune_selection.py 与 output/selection_experiment.csv），
    # 因此默认关闭，仅在研究时显式打开。
    sel_min_amount_ma20: float = 0.0    # 0 = 不额外限制（基础池已有 3000 万门槛）
    sel_max_vol_pct: float = 1.0        # 1.0 = 不剔除
    sel_min_ret_20: float = -1.0        # -1.0 = 不过滤
    sel_min_ret_5: float = -1.0

    # ---------- 中性化与仓位闸门 ----------
    # 实测有效的是仓位闸门；中性化单独使用收益改善有限且不回撤，
    # 与闸门叠加后反而更差，故默认关闭。
    neutralize_size: bool = False
    neutralize_vol: bool = False
    use_regime_filter: bool = True      # 启用市场状态闸门
    regime_ma: int = 60                 # 闸门均线周期
    regime_buffer: float = 0.0          # 需高出均线的比例才入场

    # ---------- 交易成本 ----------
    commission: float = 0.00025         # 佣金，双边
    min_commission: float = 5.0         # 单笔最低佣金（元）
    stamp_tax: float = 0.0005           # 印花税，卖出单边
    slippage: float = 0.001             # 滑点，双边

    # ---------- 资金 ----------
    init_cash: float = 1_000_000.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


CONFIG = Config()
