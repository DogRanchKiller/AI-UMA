# 510300.SH 场内成交与现金分红｜2026-11-20 收盘快照

华泰柏瑞沪深300交易型开放式指数证券投资基金，上交所代码 **510300**；不是 000300 指数、基金单位净值、场外联接基金或全收益指数。日期为上海交易日，截止 **2026-11-20**。资料取于 **2026-11-21 上海时间**，具体采集时间、URL 和 HTTP 日期见下。

## 可直接使用的两个 CSV

| 文件 | 字段和口径 |
|---|---|
| [raw.csv](raw.csv) | `date,open,close,high,low,volume`，ISO 日期、有序唯一；场内未复权，人民币元/份，保留交易价的 3 位小数；成交量为整数**份**，不是手、成交额或流通份额。无前填、插值、净值替代或除息回填。 |
| [dividends.csv](dividends.csv) | `record_date,ex_date,pay_date,cash_per_unit`。分别为场内权益登记日、除息日、现金红利发放日和每份现金红利（元/份）；公告的“元/10份”除以 10。按实际事件日期纳入，而非按年报利润年度归属。 |

- 2019-01-02 至 2026-11-20 合计 **1,912 个交易日**；2021-01-21 至 2026-11-20（含首尾）**1,412 日**，之前 **500 日**仅可用作预热。1,912 日均有正成交量。
- 主窗口首日开 5.483、收 5.562；末日开 4.014、收 4.018、高 4.035、低 3.970，成交 4,541,716,993 份。相同日期的 NAV 与场内成交价不是同一数量。
- 输入含 8 笔现金分红，5 笔在主窗口内登记，金额合计 0.419 元/份（仅当每次均有持仓，不是收益率）。另有一笔 0.072 元/份在 2021-01-21 发放，属于 2021-01-15 登记、2021-01-18 除息的权益；**主窗口首日才买入的账户不得领取**。
- 八次公告中，场内登记日均在除息日前一交易日。按登记日收市持有份数锁定权益；除息日新买不取得本次红利，已获权者除息日卖出不丧失收款权；现金应在发放日才可动用。分红不包含在原始价格里；持仓市值按原始价格估值时，除息与现金到账之间的应收红利不可漏记或重复计入。未假设分红当日自动再投资。

## 行情来源及核验

1. 上海证券交易所：[ETF 列表/品种页](https://www.sse.com.cn/assortment/fund/list/etfinfo/basic/index.shtml?FUNDID=510300)、[官方日线，按 20190101–20261120 查询](https://yunhq.sse.com.cn:32042/v1/sh1/dayk/510300?select=date,open,high,low,close,volume,amount&begin=20190101&end=20261120)、[当日收市行情快照](https://yunhq.sse.com.cn:32042/v1/sh1/snap/510300?select=name,last,open,high,low,volume,amount)。交易所传回代码 510300，全程 1,912 条 `date,open,high,low,close,volume,amount`。收市快照报告日期 20261120，时间字段 162904；并非将 16:29:04 当作执行成交时间。完整 HTTP 返回[日线](evidence/sse_dayk_510300.json)和[最新快照](evidence/sse_snap_510300.json)已保留，[请求与采集时间](evidence/sse_urls.json)：2026-11-21 14:54:05–06（+08:00）。
2. CSV 从 BaoStock 0.9.4 的 [日 K 接口说明](https://www.baostock.com/mainContent?file=stockKData.md) 对应查询生成：`query_history_k_data_plus('sh.510300', ..., start_date='2019-01-01', end_date='2026-11-20', frequency='d', adjustflag='3')`。按日期重排字段、消除多余的尾零，不改变数值。全程证券代码、`adjustflag=3`（未复权）、`tradestatus=1`；**与上交所原始 OHLC、成交份数和金额逐日逐字段完全一致**，只保留整数份数。见[原始行情](evidence/baostock_510300_kline.csv)、[证券名称/上市日](evidence/baostock_basic.csv)、[全日历](evidence/baostock_calendar.csv)、[调用参数与采集时间](evidence/baostock_query.json)（14:51:42）。与 BaoStock 交易日历完全一致，另同本项目现有沪深300指数文件的交易日期逐行比对一致；**未取其价格、量、收益或产生任何替代行情**。
3. [新浪财经 510300 日线](https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sh510300&scale=240&ma=no&datalen=2000)：提取采集时返回的近 2,000 条中的同一区间，1,912 日 OHLCV 完全相同。另用[腾讯行情的无复权日线](https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh510300,day,2026-01-01,2026-11-20,640,)核对 2026 年全部 213 日（响应 `day`，并非 `qfqday` 或 `hfqday`），OHLC 一致；腾讯成交量为整手，乘 100 后与逐份记录相差最多 49 份，**因此交付不使用腾讯量**。原始[新浪](evidence/sina_kline.json)、[腾讯](evidence/tencent_2026_day.json)、[URL 和 HTTP 日期](evidence/http_sources.json)均留存。

各站展示可能使用相同交易所底层数据；多端一致是传输、标的、复权与字段口径的复核，不代表数据提供方彼此独立采样。历史数据源没有公开每根日 K 最初投递时刻；此为事后冻结快照，不是带逐字段首次发布时间的历史数据仓库。行情发布时间仅定位到各交易日收市，采集时刻另列。

## 分红及份额折算

- 基金管理人华泰柏瑞基金管理有限公司是分红公告发布方；[产品信息](https://www.huatai-pb.com/products/zhishu/510300/index.html)。[逐笔公告索引](evidence/dividend_notices.csv)给出 8 个公告日期、原公告每 10 份金额、所有事件日期、归一单位金额及**上交所原 PDF** 的 URL 和本地路径；同名 TXT 由本机已有的 `pdftotext -raw -enc UTF-8` 提取。逐公告核验基金主代码、发布者、登记/除息/派息列、金额和收市登记资格。2019-01 还保留巨潮披露副本。精确请求时间存于[公告批次](evidence/notice_urls.json)和[补充的 2019 上交所公告](evidence/notice_urls_2019_sse.json)。
- 与[东方财富/天天基金 F10（数据来源东方财富 Choice）](https://fundf10.eastmoney.com/fhsp_510300.html)和[同花顺研究中心基金 F10](https://basic.10jqka.com.cn/510300/bonus.html)各自的完整列表比对：两者的 14 笔历史分红在三日期、单位金额上全部吻合；其中该快照窗 8 笔与上交所披露的管理人公告相符。见[东财原页](evidence/eastmoney_f10.html)、[同花顺原页（GBK）](evidence/ths_f10.html)，两者均于 2026-11-21 14:51:44 采集。F10 页面上的“分红基准日”或净值字段未作为持仓登记、除息、派息或成交数据。
- 东财 F10 拆分表从成立起仅列 **2012-05-11** 份额折算 1:0.3709。[华泰柏瑞《基金份额折算结果的公告》，2012-05-15 巨潮披露](https://static.cninfo.com.cn/finalpage/2012-05-15/cninfo_oefund60998698.html)写明：每 1 份折为 **0.37094933 份**，属于 2019 预热之前；[原始网页](evidence/cninfo_split_notice_2012.html)与[元信息](evidence/split_url.json)已存档。**窗口内未检出拆并份，无需调整持仓份数。** 进一步用 BaoStock 的[调整因子记录](evidence/baostock_adjust_events.csv)和[查询说明](evidence/baostock_adjust_query.json)检验：2018 事件只作起始因子锚；2019–2026 仅有 8 个变更日期，恰为上述 8 个除息日。每次变更的前复权因子比值均可由 `登记日收盘价/(登记日收盘价−每份现金红利)` 解释，最大绝对残差 < 0.0000005（因子保留 6 位小数）。**因子仅用于核查事件，不作为交易价输入。**

## 离线复核

这两个 CSV 均为 UTF-8 无 BOM、逗号分隔、LF 行尾。记录 SHA-256（以文件字节计）：

| 输入 | SHA-256 |
|---|---|
| [raw.csv](raw.csv) | `ff45a4063e679ee150608698a79a065abc76fc167e8959991b5640acd1ab9a47` |
| [dividends.csv](dividends.csv) | `c92c976ea33e47ddea7ba288cae733bb58c9ea9286b08c92598348428d64d3ab` |

全目录文件（清单本身除外）见 [SHA256SUMS.txt](SHA256SUMS.txt)。在项目根目录运行下列只读检查，用既有 Python 与 Beautiful Soup、本地证据完成，无网络、无需安装：

```powershell
python -B -X utf8 output/simple_controls_20261121/inputs/verify_snapshot.py
```

检查器逐行核验交易资格、日历、OHLCV、金额、未复权标记、两份 F10、八份公告提取、因子变化与哈希。不基于原始行情直接推算或猜测现金分红。本目录是研究输入；不生成买卖指令、不改原项目正式数据。
