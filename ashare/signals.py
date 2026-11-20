"""每日信号生成：买入清单、卖出清单与个股归因。

归因方法：对入选股票，用「该股在各因子上的横截面分位 - 全市场平均分位」
找出贡献最大的正向与负向因子，给出可读的理由。这是解释模型打分，不是在
宣称这些因子导致了未来收益。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import FACTOR_META


def factor_reasons(
    factors: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    code: str,
    top_k: int = 3,
) -> tuple[list[str], list[str]]:
    """返回 (正向理由, 负向理由)，用中文因子名描述。"""
    pos: list[tuple[float, str]] = []
    neg: list[tuple[float, str]] = []
    for name, wide in factors.items():
        if date not in wide.index or code not in wide.columns:
            continue
        val = wide.at[date, code]
        if pd.isna(val):
            continue
        dev = float(val) - 0.5  # 分位偏离中位
        label = FACTOR_META.get(name, (name, ""))[0]
        if dev > 0:
            pos.append((dev, label))
        else:
            neg.append((dev, label))
    pos.sort(reverse=True)
    neg.sort()
    return [x[1] for x in pos[:top_k]], [x[1] for x in neg[:top_k]]


def build_signal(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    names: dict[str, str],
    date: pd.Timestamp,
    top_n: int = 20,
    current_holdings: list[str] | None = None,
    industry: dict[str, str] | None = None,
) -> dict:
    """生成某个信号日的完整建议。"""
    if date not in scores.index:
        # 允许传入最近的一个信号日
        prior = scores.index[scores.index <= date]
        if len(prior) == 0:
            return {"date": None, "buy": [], "sell": [], "hold": []}
        date = prior[-1]

    row = scores.loc[date].dropna().sort_values(ascending=False)
    if row.empty:
        return {"date": str(date.date()), "buy": [], "sell": [], "hold": []}

    # 分位分数转成百分位排名，便于阅读
    pct = row.rank(pct=True) * 100
    picks = row.head(top_n).index.tolist()
    holdings = list(current_holdings or [])

    buys = []
    for c in picks:
        pos_r, neg_r = factor_reasons(factors, date, c)
        px = close.at[date, c] if c in close.columns else np.nan
        buys.append({
            "code": c,
            "name": names.get(c, c),
            "score": round(float(row[c]), 4),
            "percentile": round(float(pct[c]), 1),
            "price": round(float(px), 2) if pd.notna(px) else None,
            "industry": (industry or {}).get(c, ""),
            "reasons": pos_r,
            "risks": neg_r,
        })

    sells = [
        {
            "code": c,
            "name": names.get(c, c),
            "percentile": round(float(pct[c]), 1) if c in pct.index else None,
            "reason": "评分跌出候选池" if c not in picks else "权重再平衡",
        }
        for c in holdings
        if c not in picks
    ]
    holds = [
        {"code": c, "name": names.get(c, c),
         "percentile": round(float(pct[c]), 1) if c in pct.index else None}
        for c in holdings if c in picks
    ]

    return {
        "date": str(date.date()),
        "buy": buys,
        "sell": sells,
        "hold": holds,
        "empty": len(buys) == 0,
    }


def render_text(sig: dict) -> str:
    """把信号渲染成可读文本，方便命令行查看。"""
    lines = [f"信号日期：{sig.get('date')}", ""]
    buys = sig.get("buy") or []
    if not buys:
        lines.append("本期无满足条件的标的。")
        return "\n".join(lines)
    lines.append(f"建议买入（{len(buys)} 只，等权）：")
    for i, b in enumerate(buys, 1):
        rs = "、".join(b["reasons"]) or "—"
        lines.append(
            f"  {i:2d}. {b['code']} {b['name']:<6s} 分位{b['percentile']:5.1f}%  "
            f"参考价{b['price']}  逻辑：{rs}"
        )
    if sig.get("sell"):
        lines.append("")
        lines.append("建议卖出：")
        for s in sig["sell"]:
            lines.append(f"  - {s['code']} {s['name']}  {s['reason']}")
    return "\n".join(lines)
