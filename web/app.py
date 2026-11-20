"""本地 Web 看板。

启动：
    python -m uvicorn web.app:app --host 127.0.0.1 --port 8710
或：
    python scripts/03_serve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="A股量化信号终端", docs_url=None, redoc_url=None)


def _json_file(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        raise HTTPException(404, f"{name} 不存在，请先运行 scripts/02_build.py")
    return json.loads(p.read_text(encoding="utf-8"))


def _csv(name: str, **kw) -> pd.DataFrame:
    p = OUT / name
    if not p.exists():
        raise HTTPException(404, f"{name} 不存在，请先运行 scripts/02_build.py")
    return pd.read_csv(p, **kw)


@app.get("/api/health")
def health() -> dict:
    files = ["run_summary.json", "signal.json", "backtest_equity.csv"]
    return {
        "ready": all((OUT / f).exists() for f in files),
        "missing": [f for f in files if not (OUT / f).exists()],
        "output_dir": str(OUT),
    }


@app.get("/api/summary")
def summary() -> dict:
    return _json_file("run_summary.json")


@app.get("/api/signal")
def signal() -> dict:
    return _json_file("signal.json")


@app.get("/api/factors")
def factors() -> dict:
    return _json_file("factors_meta.json")


@app.get("/api/equity")
def equity() -> JSONResponse:
    df = _csv("backtest_equity.csv", index_col=0, parse_dates=True)
    df = df.dropna(how="all")
    # 回撤序列
    eq = df["equity"]
    dd = (eq / eq.cummax() - 1.0) * 100
    bdd = (df["benchmark"] / df["benchmark"].cummax() - 1.0) * 100
    # 超过 3000 点时按周抽样，避免前端卡顿
    if len(df) > 3000:
        df = df.resample("W-FRI").last().dropna(how="all")
        dd = dd.reindex(df.index)
        bdd = bdd.reindex(df.index)
    return JSONResponse({
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "equity": [round(float(v), 5) for v in df["equity"]],
        "benchmark": [round(float(v), 5) if pd.notna(v) else None for v in df["benchmark"]],
        "drawdown": [round(float(v), 3) for v in dd],
        "benchmark_drawdown": [round(float(v), 3) if pd.notna(v) else None for v in bdd],
    })


@app.get("/api/ic")
def ic() -> JSONResponse:
    s = _csv("ic.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
    return JSONResponse({
        "dates": [d.strftime("%Y-%m-%d") for d in s.index],
        "values": [round(float(v), 4) for v in s.values],
        "mean": round(float(s.mean()), 4) if len(s) else None,
    })


@app.get("/api/importance")
def importance() -> JSONResponse:
    df = _csv("importance.csv").head(15)
    return JSONResponse({
        "items": [
            {"factor": str(r["factor"]), "name": str(r.get("name", r["factor"])),
             "group": str(r.get("group", "")), "share": round(float(r["share"]), 4)}
            for _, r in df.iterrows()
        ]
    })


@app.get("/api/layers")
def layers() -> JSONResponse:
    try:
        df = _csv("layers.csv", index_col=0)
    except HTTPException:
        return JSONResponse({"labels": [], "values": []})
    means = (df.mean() * 100).round(2)
    return JSONResponse({
        "labels": [f"第{int(c)+1}层" for c in means.index],
        "values": [float(v) for v in means.values],
    })


@app.get("/api/trades")
def trades() -> JSONResponse:
    try:
        df = _csv("backtest_trades.csv")
    except HTTPException:
        return JSONResponse({"count": 0, "recent": []})
    recent = df.tail(40).iloc[::-1]
    return JSONResponse({
        "count": int(len(df)),
        "recent": [
            {"date": str(r["date"])[:10], "code": str(r["code"]), "side": str(r["side"]),
             "shares": float(r["shares"]), "price": round(float(r["price"]), 2),
             "fee": round(float(r["fee"]), 2)}
            for _, r in recent.iterrows()
        ],
    })


@app.get("/api/holdings")
def holdings() -> JSONResponse:
    try:
        w = _csv("weights.csv", index_col=0, parse_dates=True)
    except HTTPException:
        return JSONResponse({"date": None, "items": []})
    if w.empty:
        return JSONResponse({"date": None, "items": []})
    last = w.iloc[-1].dropna()
    last = last[last > 0].sort_values(ascending=False)
    return JSONResponse({
        "date": str(w.index[-1].date()),
        "items": [
            {"code": str(c), "weight": round(float(v) * 100, 2)}
            for c, v in last.items()
        ],
    })


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index() -> FileResponse:
    p = STATIC / "index.html"
    if not p.exists():
        raise HTTPException(404, "index.html 缺失")
    return FileResponse(p)
