#!/usr/bin/env python3
"""Step 3: DBI on Binance bookDepth. Preregistered in PREREG_STEP3_DBI_BOOKDEPTH_20260930.md.

Q1 (decisive): pooled 74-symbol direction test of sign(z(imb1)) when |z| > 1, 240-min forward return.
Q2 (tradability): the frozen dbi_strategy.backtest on BTCUSDT 1m data, maker-in / taker-out.
Signal and backtest code come from dbi_strategy.py unmodified (sha256 4e40a242...).

Runs on anson-pc:  .venv/bin/python step3.py [--workers 14]
"""
import argparse, glob, math, os, sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
import polars as pl
import analyze, build
import dbi_strategy as S

HERE = os.path.dirname(os.path.abspath(__file__))
BD = os.path.join(HERE, "out", "bookdepth")
OUT = os.path.join(HERE, "out", "step3")
KL = os.path.join(build.B, "monthly", "klines")
START, END = datetime(2023, 1, 1), datetime(2026, 9, 1)
HALF = datetime(2024, 11, 1)
# preregistration deviation 1: Binance bookDepth reference-price anomaly after the 2025-10-10 crash (all symbols)
BAD0, BAD1 = datetime(2025, 10, 10, 15, 0), datetime(2025, 10, 15, 14, 0)
GLITCH = 0.9
MIN = 60_000
THR, STOP, RR, MAXH = 1.0, 60, 2.0, 240
FEES = {"雙邊 maker": (S.MAKER_BPS, S.MAKER_BPS), "進 maker 出 taker": (S.MAKER_BPS, S.TAKER_BPS),
        "雙邊 taker": (S.TAKER_BPS, S.TAKER_BPS)}


def klines_1m(sym):
    d = os.path.join(KL, sym, "1m")
    frames = []
    for f in sorted(os.listdir(d)):
        ym = f[-11:-4]
        if not ("2023-01" <= ym <= "2026-08"):
            continue
        df = build.read_zip_csv(os.path.join(d, f), build.KL_COLS)
        if df is None:
            continue
        df.columns = build.KL_COLS[:len(df.columns)]
        frames.append(df.select(pl.col("open_time").cast(pl.Int64), pl.col("close").cast(pl.Float64)))
    k = pl.concat(frames).unique("open_time").sort("open_time")
    return k.select((pl.from_epoch("open_time", time_unit="ms") + pl.duration(minutes=1)).alias("t"), "close")


def zcol(x):
    return S.zs(np.asarray(x, dtype=np.float64))


def strategy(ts, px, sig, fee):
    """frozen backtest + the preregistered gap / end-truncation filter"""
    tr = S.backtest(ts, px, sig, THR, STOP, RR, MAXH, *fee)
    n = len(ts); kept = []
    for t in tr:
        i, j = t["i"], t["j"]
        if t["how"] == "time" and i + MAXH > n - 1:
            continue
        if np.any(np.diff(ts[i:j + 1]) > 3 * MIN):
            continue
        kept.append((int(ts[i]), t["r"], bool(t["long"])))
    return kept


def work(sym):
    raw = pl.read_parquet(os.path.join(BD, sym + ".parquet")).filter((pl.col("t") > START) & (pl.col("t") <= END))
    k = klines_1m(sym)
    # unfiltered z, kept only for the sensitivity line in the report
    mr = raw.join(k, on="t", how="inner").sort("t")
    zraw = pl.DataFrame({"tm": mr["t"], "z1raw": zcol(mr["imb1"].to_numpy())}).with_columns(pl.col("z1raw").fill_nan(None))
    bd = raw.filter(~((pl.col("t") >= BAD0) & (pl.col("t") < BAD1))).filter(pl.col("imb1").abs() < GLITCH)
    bd = bd.with_columns([pl.when(pl.col(c).abs() < GLITCH).then(pl.col(c)).otherwise(None).alias(c) for c in ("imb02", "band02_1")])
    m = bd.join(k, on="t", how="inner").sort("t")
    ts = m["t"].dt.epoch("ms").to_numpy().astype(np.int64)
    px = m["close"].to_numpy()
    z1 = zcol(m["imb1"].to_numpy())
    # DBI2 on the rows where the 0.2% band exists (2026-01-15 on), z computed on that sub-series
    has2 = m["imb02"].is_not_null().to_numpy() & m["band02_1"].is_not_null().to_numpy()
    z2 = np.full(len(m), np.nan)
    if has2.sum() > 300:
        idx = np.where(has2)[0]
        z2[idx] = (zcol(m["imb02"].to_numpy()[idx]) + zcol(m["band02_1"].to_numpy()[idx])) / 2
    # --- strategy (Q2 for BTC; descriptive for others)
    strat = {name: strategy(ts, px, z1, fee) for name, fee in FEES.items()}
    extra = {}
    if sym == "BTCUSDT":
        for lab, sgn in (("永遠做多", 1.0), ("永遠做空", -1.0)):
            extra[lab] = strategy(ts, px, np.full(len(ts), 2.0 * sgn), FEES["雙邊 maker"])
        if has2.sum() > 300:
            idx = np.where(has2)[0]
            extra["DBI2 進 maker 出 taker"] = strategy(ts[idx], px[idx], z2[idx], FEES["進 maker 出 taker"])
    # --- Q1 frame on the 5m panel grid
    sig = pl.DataFrame({"tm": m["t"], "z1": z1, "z2": z2}).with_columns(pl.col("z1").fill_nan(None), pl.col("z2").fill_nan(None))
    p = (pl.read_parquet(os.path.join(analyze.PANEL, sym + ".parquet"), columns=["t", "close"])
         .filter((pl.col("t") > START) & (pl.col("t") <= END)).sort("t"))
    c = pl.col("close")
    p = p.with_columns([(c.shift(-h) / c - 1).alias("f0_%d" % (5 * h)) for h in (12, 24, 48)] +
                       [(c.shift(-(1 + 48)) / c.shift(-1) - 1).alias("f1_240")])
    q = p.join_asof(sig.sort("tm"), left_on="t", right_on="tm", strategy="backward", tolerance="2m").drop("tm")
    q = q.join_asof(zraw.sort("tm"), left_on="t", right_on="tm", strategy="backward", tolerance="2m").drop("tm")
    q = q.with_columns(pl.lit(sym).alias("sym")).drop("close").with_columns(pl.col(pl.Float64).cast(pl.Float32))
    os.makedirs(OUT, exist_ok=True)
    q.write_parquet(os.path.join(OUT, sym + ".parquet"))
    return sym, len(m), strat, extra


def q1stats(lf, sig, ret, where=None):
    lf2 = lf if where is None else lf.filter(where)
    q = (lf2.select("sym", "t", pl.col(sig).alias("z"), pl.col(ret).alias("r")).drop_nulls()
         .with_columns((pl.col("r") - pl.col("r").mean().over("sym")).alias("x"),
                       pl.when(pl.col("z").abs() > THR).then(pl.col("z").sign()).otherwise(0).alias("g"))
         .collect())
    ic = q.select(pl.corr("z", "r", method="spearman")).item() if q.height > 100 else float("nan")
    q = q.filter(pl.col("g") != 0)
    if q.height < 100:
        return None
    gx = pl.col("g") * pl.col("x")
    day = q.group_by(pl.col("t").dt.date().alias("d")).agg(gx.sum().alias("S"), pl.len().alias("n")).sort("d")
    S_, n_ = day["S"].to_numpy().astype(float), day["n"].to_numpy().astype(float)
    keep = (S_ / n_) <= np.quantile(S_ / n_, 0.95)
    per = q.group_by("sym").agg(gx.mean().alias("e"), pl.len().alias("n")).filter(pl.col("n") >= 1000)
    h1 = q.filter(pl.col("t") < HALF); h2 = q.filter(pl.col("t") >= HALF)
    yr = q.group_by(pl.col("t").dt.year().alias("y")).agg((gx.mean() * 1e4).alias("e")).sort("y")
    return {"n": q.height, "edge": q.select(gx.mean()).item() * 1e4, "t": analyze.nw_t(S_, n_, lags=20),
            "breadth": float((per["e"] > 0).mean()) if per.height else float("nan"), "nsym": per.height,
            "ex5": S_[keep].sum() / n_[keep].sum() * 1e4,
            "h1": h1.select(gx.mean()).item() * 1e4 if h1.height else float("nan"),
            "h2": h2.select(gx.mean()).item() * 1e4 if h2.height else float("nan"),
            "ic": ic, "years": dict(zip(yr["y"].to_list(), yr["e"].to_list()))}


def row(label, r):
    if r is None:
        return "  %-30s 樣本不足" % label
    return "  %-30s n=%9d  edge %+7.2f  t %+5.1f  廣度 %4.0f%%  剔5%%日 %+7.2f  前半 %+7.2f  後半 %+7.2f  IC %+.4f" % (
        label, r["n"], r["edge"], r["t"], 100 * r["breadth"], r["ex5"], r["h1"], r["h2"], r["ic"])


def trow(label, tr):
    if not tr:
        return "  %-26s 無交易" % label
    r = np.array([x[1] for x in tr]); sd = r.std(ddof=1)
    return "  %-26s n=%6d  勝率 %3.0f%%  合計 %+8.1fR  每筆 %+7.4fR  sd %.2f  t %+5.2f" % (
        label, len(r), 100 * (r > 0).mean(), r.sum(), r.mean(), sd, r.mean() / (sd / math.sqrt(len(r))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "step3.txt"))
    a = ap.parse_args()
    syms = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(BD, "*.parquet")))
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for sym, nrows, strat, extra in ex.map(work, syms):
            res[sym] = (nrows, strat, extra)
    L = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    lf = pl.scan_parquet(os.path.join(OUT, "*.parquet"))
    say("# 第 3 步：DBI 在 Binance bookDepth（預先註記 PREREG_STEP3_DBI_BOOKDEPTH_20260930.md）")
    say("訊號 DBI1 = z240(imb±1%%)；啟動 |z| > 1；%d 幣；2023-01 → 2026-08" % len(syms))

    say()
    say("## Q1：方向假說（74 幣合併，240 分鐘，扣漂移）")
    m = q1stats(lf, "z1", "f0_240")
    lag = q1stats(lf, "z1", "f1_240")
    say(row("主規格", m))
    say(row("延後 5 分鐘進場", lag))
    q1_pass = False
    if m:
        gates = [("G1 t ≥ 3", m["t"] >= 3, "t = %+.2f" % m["t"]),
                 ("G2 廣度 ≥ 80%", m["breadth"] >= 0.8, "%.0f%%（%d 幣）" % (100 * m["breadth"], m["nsym"])),
                 ("G3 剔最好 5% 日後 > 0", m["ex5"] > 0, "%+.2f bp" % m["ex5"]),
                 ("G4 兩半期都 > 0", m["h1"] > 0 and m["h2"] > 0, "%+.2f / %+.2f bp" % (m["h1"], m["h2"])),
                 ("G5 延後 5 分鐘同號且 ≥ 一半", lag is not None and np.sign(lag["edge"]) == np.sign(m["edge"]) and abs(lag["edge"]) >= 0.5 * abs(m["edge"]),
                  "%+.2f bp" % (lag["edge"] if lag else float("nan")))]
        for name, ok, val in gates:
            say("  %-28s %-4s %s" % (name, "過" if ok else "不過", val))
        q1_pass = all(ok for _, ok, _ in gates)
        say("  Q1：%s%s" % ("PASS" if q1_pass else "FAIL",
                             "（edge 顯著為負：磁鐵讀法）" if m["edge"] < 0 and m["t"] <= -3 else ""))
        say("  逐年 edge(bp)：" + "  ".join("%d %+.2f" % (y, e) for y, e in m["years"].items()))

    say()
    say("## Q2：可交易性（BTCUSDT，凍結策略，丟跨缺口交易）")
    btc = res.get("BTCUSDT")
    q2_pass = False
    if btc:
        for name in FEES:
            say(trow(name, btc[1][name]))
        r = np.array([x[1] for x in btc[1]["進 maker 出 taker"]])
        if len(r) > 1:
            tq = r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))
            q2_pass = r.mean() > 0 and tq >= 2
            say("  Q2 判準（進 maker 出 taker 每筆 > 0 且 t ≥ 2）：%s  (每筆 %+.4fR, t %+.2f)" % ("PASS" if q2_pass else "FAIL", r.mean(), tq))

    say()
    verdict = ("FAIL —— DBI 假說（此近似下）否證，關閉 DBI 線" if not q1_pass else
               "Q1 PASS / Q2 FAIL —— 深簿失衡有方向，但以此幾何不可交易" if not q2_pass else
               "PASS —— 值得前瞻測試（另立預先註記）")
    say("## 判決：%s" % verdict)

    say()
    say("## 次要報告（描述用，不影響判決）")
    for h in (60, 120):
        say(row("Q1 %d 分鐘" % h, q1stats(lf, "z1", "f0_%d" % h)))
    say(row("Q1 只看 BTC", q1stats(lf, "z1", "f0_240", pl.col("sym") == "BTCUSDT")))
    say(row("DBI2（2026-01-15 起）240 分鐘", q1stats(lf, "z2", "f0_240")))
    say(row("敏感度：不清理 10 月異常與 |imb|≥0.9", q1stats(lf, "z1raw", "f0_240")))
    if btc:
        for lab, tr in btc[2].items():
            say(trow("BTC " + lab, tr))
        say(trow("BTC 做多（進 maker 出 taker）", [x for x in btc[1]["進 maker 出 taker"] if x[2]]))
        say(trow("BTC 做空（進 maker 出 taker）", [x for x in btc[1]["進 maker 出 taker"] if not x[2]]))
    say("### 74 幣都跑凍結策略（固定 60 bps；對非 BTC 不等價）")
    for name in FEES:
        allr = [x for s in res.values() for x in s[1][name]]
        say(trow("合併 " + name, allr))
    pos = sum(1 for s in res.values() if s[1]["進 maker 出 taker"] and np.mean([x[1] for x in s[1]["進 maker 出 taker"]]) > 0)
    say("  進 maker 出 taker 每筆為正的幣：%d / %d" % (pos, sum(1 for s in res.values() if s[1]["進 maker 出 taker"])))
    with open(a.out, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
