"""Read-only, offline checks for the frozen 510300 input snapshot (Python + bs4)."""
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path

from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parent
EVIDENCE = BASE / "evidence"
START = "2019-01-02"
SCORE_START = "2021-01-21"
END = "2026-11-20"
D = Decimal
PRICE_FIELDS = ("open", "close", "high", "low")
QUOTE_FIELDS = ("date", *PRICE_FIELDS, "volume")
DIV_FIELDS = ("record_date", "ex_date", "pay_date", "cash_per_unit")


def require(ok, description):
    if not ok:
        raise ValueError(description)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(name):
    return json.loads((EVIDENCE / name).read_bytes(), parse_float=D)


def table_rows(name, encoding, heading):
    html = (EVIDENCE / name).read_bytes().decode(encoding)
    soup = BeautifulSoup(html, "html.parser")
    tables = [t for t in soup.find_all("table") if heading in t.get_text()]
    require(len(tables) == 1, f"ambiguous or absent F10 table: {name}, {heading}")
    return [[c.get_text(strip=True) for c in r.find_all("td")]
            for r in tables[0].find_all("tr") if r.find("td")]


def check_notices(divs):
    notice_rows = read_csv(EVIDENCE / "dividend_notices.csv")
    require(len(notice_rows) == len(divs) == 8, "unexpected number of event disclosures")
    for n, v in zip(notice_rows, divs, strict=True):
        require(tuple(n[k] for k in DIV_FIELDS) == tuple(v[k] for k in DIV_FIELDS),
                f"delivery differs from notice table on {v['ex_date']}")
        # Use the locally extracted -raw reading order, NOT table-layout reflow.
        pdf = (BASE / n["pdf"]).resolve()
        require(pdf.parent == EVIDENCE and pdf.read_bytes().startswith(b"%PDF-"), "invalid PDF path/content")
        text = re.sub(r"\s+", "", pdf.with_suffix(".txt").read_text(encoding="utf-8"))
        require("基金主代码510300" in text and "华泰柏瑞基金管理有限公司" in text
                and "权益登记日下午上海证券交易所交易结束后" in text, f"wrong issuer or record terms: {pdf.name}")
        for key, heading in (("announcement_date", "公告送出日期[：:]"), ("record_date", "权益登记日"),
                             ("ex_date", "除息日"), ("pay_date", "现金红利发放日")):
            m = re.search(heading + r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
            require(m is not None and date(*map(int, m.groups())).isoformat() == n[key], f"{pdf.name}: {key}")
        m = re.search(r"本次分红方案[（(]单位[：:]元/10份基金份额[）)]([0-9.]+)", text)
        require(m is not None and D(m[1]) == D(n["cash_per_10_units"]) == 10 * D(n["cash_per_unit"]),
                f"dividend unit error in {pdf.name}")
    rows = table_rows("eastmoney_f10.html", "utf-8", "权益登记日")
    east = [(r[1], r[2], r[4], D(re.search(r"现金([\d.]+)元", r[3])[1]) / 10) for r in rows]
    rows = table_rows("ths_f10.html", "gb18030", "权益登记日")
    ths = [(r[0], r[1], r[2], D(r[3].removesuffix("元")) / 10) for r in rows]
    require(len(east) == 14 and sorted(east) == sorted(ths), "F10 lifetime event mismatch")
    wanted = sorted(x for x in east if START <= x[1] <= END)
    result = [(d["record_date"], d["ex_date"], d["pay_date"], D(d["cash_per_unit"])) for d in divs]
    require(result == wanted, "F10 events differ from delivered inputs")
    splits = table_rows("eastmoney_f10.html", "utf-8", "拆分折算日")
    require(splits == [["2012年", "2012-05-11", "份额折算", "1:0.3709"]], "unexpected split history")
    split_notice = (EVIDENCE / "cninfo_split_notice_2012.html").read_bytes().decode("gb18030")
    require("2012年5月11日" in split_notice and "基金份额折算比例为0.37094933" in split_notice,
            "2012 issuer notice does not confirm the historical conversion")


def check_hashes():
    records = (BASE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    seen = set()
    for record in records:
        digest, name = record.split("  ", 1)
        item = (BASE / name).resolve()
        require(item.is_relative_to(BASE) and name not in seen, f"unsafe or duplicate manifest entry: {name}")
        require(hashlib.sha256(item.read_bytes()).hexdigest() == digest, f"SHA-256 mismatch: {name}")
        seen.add(name)
    expected = {p.relative_to(BASE).as_posix() for p in BASE.rglob("*") if p.is_file()
                and p.name != "SHA256SUMS.txt" and "__pycache__" not in p.parts}
    require(seen == expected, f"manifest set mismatch: {seen ^ expected}")
    return len(records)


def check():
    raw = read_csv(BASE / "raw.csv")
    dates = [r["date"] for r in raw]
    require(tuple(raw[0]) == QUOTE_FIELDS and len(dates) == 1912 and dates == sorted(set(dates)), "quote schema/order")
    require((dates[0], dates[-1]) == (START, END) and SCORE_START in dates, "date coverage")
    calendar = [r["calendar_date"] for r in read_csv(EVIDENCE / "baostock_calendar.csv") if r["is_trading_day"] == "1"]
    require(dates == calendar and sum(d >= SCORE_START for d in dates) == 1412, "missing or extra trading day")
    basic = read_csv(EVIDENCE / "baostock_basic.csv")
    require(len(basic) == 1 and basic[0]["code"] == "sh.510300" and basic[0]["code_name"] == "华泰柏瑞沪深300ETF",
            "identity mismatch")
    sse = read_json("sse_dayk_510300.json")
    require(sse["code"] == "510300", "wrong exchange security")
    bao = read_csv(EVIDENCE / "baostock_510300_kline.csv")
    sina = [r for r in read_json("sina_kline.json") if START <= r["day"] <= END]
    for bar, observed, b, s in zip(raw, sse["kline"], bao, sina, strict=True):
        t = bar["date"]
        require(str(observed[0]) == t.replace("-", "") and b["date"] == s["day"] == t, f"date mismatch: {t}")
        require(b["code"] == "sh.510300" and b["adjustflag"] == "3" and b["tradestatus"] == "1", f"quote status: {t}")
        oq, cq, hq, lq, vol = (D(bar[k]) for k in (*PRICE_FIELDS, "volume"))
        require(0 < lq <= min(oq, cq) <= max(oq, cq) <= hq and vol > 0 and re.fullmatch(r"\d+", bar["volume"]),
                f"invalid OHLCV: {t}")
        require(tuple(map(D, observed[1:])) == (oq, hq, lq, cq, vol, D(b["amount"])), f"exchange mismatch: {t}")
        for k in (*PRICE_FIELDS, "volume"):
            require(D(bar[k]) == D(b[k]) == D(s[k]), f"independent quote mismatch: {t}, {k}")
    by_date = {r["date"]: r for r in raw}
    qq = read_json("tencent_2026_day.json")["data"]["sh510300"]
    require(qq["qt"]["sh510300"][2] == "510300" and [q[0] for q in qq["day"]] == [d for d in dates if d[:4] == "2026"],
            "Tencent coverage/identity mismatch")
    lot_max = D(0)
    for q in qq["day"]:
        row = by_date[q[0]]
        require(tuple(map(D, q[1:5])) == tuple(D(row[k]) for k in PRICE_FIELDS), f"Tencent OHLC mismatch: {q[0]}")
        lot_max = max(lot_max, abs(100 * D(q[5]) - D(row["volume"])))
    require(lot_max <= 50, "Tencent rounded lots do not match exchange volume")
    divs = read_csv(BASE / "dividends.csv")
    require(tuple(divs[0]) == DIV_FIELDS, "dividend schema")
    check_notices(divs)
    factors = read_csv(EVIDENCE / "baostock_adjust_events.csv")
    require(factors[0]["dividOperateDate"] < START and [f["dividOperateDate"] for f in factors[1:]] == [d["ex_date"] for d in divs],
            "unexplained corporate-action date")
    max_err = D(0)
    for prev, f, d in zip(factors[:-1], factors[1:], divs, strict=True):
        rd, ex, pay = (d[k] for k in DIV_FIELDS[:3])
        require(dates[dates.index(rd)+1] == ex < pay and pay in by_date, f"dividend date order: {rd}")
        c, cash = D(by_date[rd]["close"]), D(d["cash_per_unit"])
        require(c > cash > 0, f"invalid dividend: {rd}")
        ratio = D(f["foreAdjustFactor"]) / D(prev["foreAdjustFactor"])
        max_err = max(max_err, abs(ratio - c/(c-cash)))
    require(max_err < D("0.000001"), "adjustment not explained by cash dividend alone")
    return {"status": "pass", "source": "510300.SH exchange-traded, unadjusted", "first": dates[0], "last": dates[-1],
            "warmup_and_scoring_days": len(raw), "scoring_days": sum(x >= SCORE_START for x in dates),
            "calendar_days_missing_or_extra": 0, "rows_by_year": dict(Counter(x[:4] for x in dates)),
            "sse_baostock_ohlc_volume_amount_mismatches": 0, "sina_ohlc_volume_mismatches": 0,
            "tencent_2026_ohlc_mismatches": 0, "tencent_2026_max_lot_rounding_difference_units": str(lot_max),
            "verified_issuer_notices": len(divs), "record_dates_on_or_after_score_start": sum(d['record_date'] >= SCORE_START for d in divs),
            "split_events_in_covered_window": 0, "max_abs_factor_ratio_residual": str(max_err),
            "files_sha256_verified": check_hashes()}


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
