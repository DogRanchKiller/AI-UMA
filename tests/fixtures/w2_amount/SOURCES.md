# 成交额核对样本：深市 5 股 × 5 个交易日

固定股票 `000001、000002、000063、000333、000651`；固定日期 2024-03-04 至 2024-03-08（均为上海本地交易日）。两侧分别查询，以证券代码和交易日期配对，不按误差替换证券或日期。深交所日报和 BaoStock 事件实采于 2026-11-22 04:36:39–04:36:47（UTC+08:00）；日线在 04:40:28–04:40:29 再查询一次，加入返回列 `adjustflag`，并确认此前 25 行已有字段全部未变。

**生产输入**：[baostock.csv](baostock.csv) 是 BaoStock Python SDK 0.9.4 返回的 25 行，保留原字段和小数。登录后对每个 `code='sz.'+六位代码` 调用：

```python
bs.query_history_k_data_plus(code, 'date,code,open,high,low,close,volume,amount,adjustflag',
                             start_date='2024-03-04', end_date='2024-03-08',
                             frequency='d', adjustflag='3')
```

OHLC 为未复权元/股；`volume` 为股；`amount` 为全日成交额（元），不从价格、成交量推算。为让生产代码还原有效因子，[adjust.csv](adjust.csv) 另存五条 SDK 事件：`bs.query_adjust_factor(code, start_date='1990-01-01', end_date='2024-03-08')`，每股保留起始日当时已生效的最后一条 `dividOperateDate,backAdjustFactor`（周内没有新事件）；这不是金额参照。

**独立参考**：[szse.csv](szse.csv) 来自深圳证券交易所「股票行情」历史日报。每个代码、日期各作一次 `GET https://www.szse.cn/api/report/ShowReport/data`，参数为 `SHOWTYPE=JSON&CATALOGID=1815_stock&TABKEY=tab1&txtDMorJC=<六位代码>&txtBeginDate=<YYYY-MM-DD>`。从返回的第一个报表 `data` 中提取匹配的 `jyrq,zqdm,cjje`，保留原逗号和精度。报表列名 `cjje=成交金额(万元)`，表注为「表中成交量、成交金额为所有交易方式汇总值」；不是席位额。例：[000651，2024-03-04 官方记录](https://www.szse.cn/api/report/ShowReport/data?SHOWTYPE=JSON&CATALOGID=1815_stock&TABKEY=tab1&txtDMorJC=000651&txtBeginDate=2024-03-04)。

**离线复核**：项目根目录执行 `python -m pytest tests/test_w2_amount.py -q -s`。只回放 SDK 行，生产代码执行股→手转换、临时目录缓存写读和宽表构建。每个格子分别检查 `abs(矩阵成交额 - 官方万元 × 10000) / (官方万元 × 10000) < 0.01`，不取平均；打印最大误差及对应股票、交易日。原始两来源差异最大的一格是 000651 在 2024-03-04：`2,069,842,376.9400` 元 vs `207,675.44` 万元，差 **0.332828141%**；保留差异，不用任何一侧填补另一侧。另查[该股当日深交所大宗交易披露](https://www.szse.cn/api/report/ShowReport/data?SHOWTYPE=JSON&CATALOGID=1932_dzjyzqjy_after&TABKEY=tab1&txtDMorJC=000651&txtStart=2024-03-04&txtEnd=2024-03-04)为 691.21 万元，与两侧差额 691.202306 万元在报表精度内相符；该数不参与测试。本例仅核对此 25 笔，不保证服务商所有成交额均等同交易所。
