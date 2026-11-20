"""A股量化信号引擎。

模块划分：
    config      —— 全局配置与超参数
    datasource  —— 行情/快照数据获取与本地缓存
    universe    —— 可交易股票池构建（剔除ST/停牌/新股/低流动性）
    factors     —— 量价、波动、流动性、技术类因子
    labels      —— 未来超额收益标签
    model       —— LightGBM 滚动训练与预测
    backtest    —— T+1 回测引擎（含费用、滑点、涨跌停约束）
    metrics     —— 绩效指标
    signals     —— 每日买卖信号与归因
"""

__version__ = "1.0.0"
