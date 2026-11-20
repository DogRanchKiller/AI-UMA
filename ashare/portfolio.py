"""组合构建：把模型打分转成实际可执行的目标持仓。

这一层不改动模型，只决定「用什么价格约束、在什么市场环境下承担风险」。
实测依据见 scripts/04_tune_selection.py。
"""

from __future__ import annotations

import pandas as pd

from .config import Config
from . import selection as sl


def build_target_scores(
    scores: pd.DataFrame,
    mats: dict[str, pd.DataFrame],
    mask: pd.DataFrame,
    benchmark: pd.Series,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """返回 (可用于选股的打分矩阵, 仓位闸门序列)。"""
    out = scores

    # 1) 中性化（默认关闭）
    if cfg.neutralize_size or cfg.neutralize_vol:
        out = sl.neutralize_scores(
            out, mats, use_size=cfg.neutralize_size, use_vol=cfg.neutralize_vol
        )

    # 2) 选股端硬约束（默认关闭）
    need_sel = (
        cfg.sel_min_amount_ma20 > 0
        or cfg.sel_max_vol_pct < 1.0
        or cfg.sel_min_ret_20 > -1.0
        or cfg.sel_min_ret_5 > -1.0
    )
    if need_sel:
        sel = sl.selection_mask(mats, cfg, mask)
        out = sl.apply_to_scores(out, sel)

    # 3) 仓位闸门
    regime = None
    if cfg.use_regime_filter:
        regime = sl.market_regime(benchmark, ma=cfg.regime_ma, buffer=cfg.regime_buffer)

    return out, regime
