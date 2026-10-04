#!/usr/bin/env python3
"""Do the scanner's factors carry directional information? Binance Vision, 74 USDT-M perps, 2020-09 .. 2026-08.

Stage 1 (features): rebuild each scan.py factor score on the 5m panel exactly as scan.py scores it,
  plus forward returns. Writes out/feat/<SYM>.parquet.
Stage 2 (report): per factor x horizon x entry lag, the drift-adjusted directional edge, its
  Newey-West t-stat (days as the unit), the bull-minus-bear spread, hit rate, per-year and
  in/out-of-sample splits, and how many of the 74 symbols agree in sign.

What cannot be rebuilt from Binance dumps: liquidations (no history), smart money (OKX-only),
funding premium (premiumIndexKlines not downloaded), and the live predicted funding rate (we only
have the last SETTLED rate, so the funding factor here lags scan.py by up to one funding interval).

Runs on anson-pc:  .venv/bin/python analyze.py [features|report|all] [--workers 14]
"""
import argparse, glob, math, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import polars as pl
import build

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "out", "panel")
FEAT = os.path.join(HERE, "out", "feat")
H = {"15m": 3, "1h": 12, "4h": 48}
LAGS = (0, 1)                # entry at close of the signal bar, or one 5m bar later
COST_BP = 12.0               # round trip, same as bt.py: 0.05% taker x2 + 0.02% slippage
SPLIT = "2024-01-01"         # in-sample before, out-of-sample from
# scan.py W minus liq/smart. Divergence uses the top-trader POSITION ratio: Binance's top-trader
# ACCOUNT ratio never gets 10 pp away from the crowd (BTC max 8.5 pp in 6 years), so the scan.py rule
# would be permanently silent; the position version fires 41% of bars vs 53% in the OKX scan log.
W = {"oi": 1.0, "funding": 1.0, "crowd": 0.75, "div_pos": 1.0, "volume": 0.75}
FACTORS = ["oi", "funding", "crowd", "div_acc", "div_pos", "div_z", "volume", "total", "verdict"]
LABEL = {"oi": "1 OI 型態", "funding": "2 資金費率 z (反向)", "crowd": "3 散戶多空比 z (反向)",
         "div_acc": "4 大戶-散戶 背離 (大戶帳戶比)", "div_pos": "4' 同上 (大戶持倉比)",
         "div_z": "4'' 背離的 7 日 z 值", "volume": "6 量能 + 主動流",
         "total": "總分 (5 因子加權, 用 4')", "verdict": "總分 ≥ +3 / ≤ -3"}


# ------------------------------------------------------------------ stage 1
def funding_z(sym, n=100):
    """z of each settled rate against the previous n settlements (scan.py: z(), >= 10 points, pstdev)."""
    f = build.funding(sym)
    if f is None or f.height == 0:
        return None
    r = f["funding_last"].to_numpy()
    z = np.zeros(len(r))
    for i in range(len(r)):
        w = r[max(0, i - n):i]
        if len(w) >= 10:
            sd = w.std()
            z[i] = 0.0 if sd == 0 else (r[i] - w.mean()) / sd
    return f.with_columns(pl.Series("fz", z))


def steps(x, lo1, lo2, hi1, hi2, v1=1.0, v2=2.0):
    """scan.py style step scoring: x>=hi2 -> +v2, x>=hi1 -> +v1, x<=lo2 -> -v2, x<=lo1 -> -v1, else 0."""
    return (pl.when(x.is_null()).then(None)
            .when(x >= hi2).then(v2).when(x >= hi1).then(v1)
            .when(x <= lo2).then(-v2).when(x <= lo1).then(-v1).otherwise(0.0))


def zroll(col, n, prior_only=False, min_n=10):
    s = pl.col(col).shift(1) if prior_only else pl.col(col)
    mu = s.rolling_mean(n, min_samples=min_n)
    sd = s.rolling_std(n, min_samples=min_n, ddof=0)
    return pl.when(sd > 0).then((pl.col(col) - mu) / sd).otherwise(None)


def features(sym):
    p = pl.read_parquet(os.path.join(PANEL, sym + ".parquet")).sort("t")
    c = pl.col("close")
    p = p.with_columns(
        (c / c.shift(1) - 1).alias("r1"),
        (c / c.shift(3) - 1).alias("r3"),
        (pl.col("oi_usd") / pl.col("oi_usd").shift(3) - 1).alias("o3"),
        (pl.col("oi_usd") / pl.col("oi_usd").shift(12) - 1).alias("o12"),
        (pl.col("top_ls_acc") / (1 + pl.col("top_ls_acc"))).alias("top_long_acc"),
        (pl.col("top_ls_pos") / (1 + pl.col("top_ls_pos"))).alias("top_long_pos"),
        (pl.col("crowd_ls") / (1 + pl.col("crowd_ls"))).alias("crowd_long"),
        (pl.col("taker_buy_quote").rolling_sum(3) / pl.col("quote_vol").rolling_sum(3)).alias("tb"),
    )
    # 1. OI regime (scan.py: 15m price & OI direction, x1.5 when |1h OI change| > 1%)
    r3, o3, o12 = pl.col("r3"), pl.col("o3"), pl.col("o12")
    base = (pl.when((r3 >= 0) & (o3 >= 0)).then(1.0).when((r3 < 0) & (o3 >= 0)).then(-1.0)
            .when((r3 < 0) & (o3 < 0)).then(-0.5).otherwise(0.5))
    p = p.with_columns(pl.when(r3.is_null() | o3.is_null() | o12.is_null()).then(None)
                       .otherwise((base * pl.when(o12.abs() > 0.01).then(1.5).otherwise(1.0)).clip(-2, 2)).alias("oi"))
    # 2. funding, contrarian: last settled rate's z vs previous 100 settlements
    fz = funding_z(sym)
    if fz is not None:
        p = p.join_asof(fz.select("ft", "fz").sort("ft"), left_on="t", right_on="ft", strategy="backward").drop("ft")
    else:
        p = p.with_columns(pl.lit(None, pl.Float64).alias("fz"))
    p = p.with_columns((-steps(pl.col("fz"), -1.0, -2.0, 1.0, 2.0)).alias("funding"))
    # scan.py tests fz > 1 strictly; steps() uses >=, identical for continuous z
    # 3. crowd L/S z over the last 300 available points (~25h), contrarian at 1.5 / 2.5
    cr = p.select("t", "crowd_ls").drop_nulls().with_columns(zroll("crowd_ls", 300).alias("cz"))
    p = p.join(cr.select("t", "cz"), on="t", how="left")
    p = p.with_columns((-steps(pl.col("cz"), -1.5, -2.5, 1.5, 2.5)).alias("crowd"))
    # 4. top-vs-crowd long share, follow top traders at 10 / 20 pp
    p = p.with_columns((pl.col("top_long_acc") - pl.col("crowd_long")).alias("diff_acc"),
                       (pl.col("top_long_pos") - pl.col("crowd_long")).alias("diff_pos"))
    p = p.with_columns(steps(pl.col("diff_acc"), -0.10, -0.20, 0.10, 0.20).alias("div_acc"),
                       steps(pl.col("diff_pos"), -0.10, -0.20, 0.10, 0.20).alias("div_pos"))
    # 4'' same idea without the exchange-specific level: diff vs its own trailing 7 days
    dz = p.select("t", "diff_acc").drop_nulls().with_columns(zroll("diff_acc", 2016, min_n=500).alias("dz"))
    p = p.join(dz.select("t", "dz"), on="t", how="left")
    p = p.with_columns(steps(pl.col("dz"), -1.0, -2.0, 1.0, 2.0).alias("div_z"))
    # 6. volume z (current bar vs previous 48) signed by last bar, plus 15m taker buy share
    p = p.with_columns(zroll("quote_vol", 48, prior_only=True).alias("vz"))
    vz, tb, r1 = pl.col("vz"), pl.col("tb"), pl.col("r1")
    p = p.with_columns(pl.when(vz.is_null() | tb.is_null() | r1.is_null()).then(None).otherwise(
        pl.when(vz > 1.5).then(pl.when(r1 > 0).then(1.0).otherwise(-1.0)).otherwise(0.0)
        + pl.when(tb > 0.55).then(0.5).when(tb < 0.45).then(-0.5).otherwise(0.0)).clip(-2, 2).alias("volume"))
    # composite (liq and smart money contribute 0, which is what scan.py mostly logs for them)
    p = p.with_columns(pl.sum_horizontal([pl.col(k) * w for k, w in W.items()]).alias("total_raw"),
                       pl.any_horizontal([pl.col(k).is_null() for k in W]).alias("any_null"))
    p = p.with_columns(pl.when(pl.col("any_null")).then(None).otherwise(pl.col("total_raw")).alias("total")).drop("total_raw", "any_null")
    p = p.with_columns(steps(pl.col("total"), -3.0, -1e9, 3.0, 1e9, v1=1.0, v2=1.0).alias("verdict"))
    # forward returns from entry at close of bar t+lag
    fwd = []
    for lag in LAGS:
        for hn, h in H.items():
            fwd.append((c.shift(-(lag + h)) / c.shift(-lag) - 1).alias("f%d_%s" % (lag, hn)))
    p = p.with_columns(fwd)
    keep = ["t"] + FACTORS + [f.meta.output_name() for f in fwd] + ["r3", "o3", "o12", "fz", "cz", "diff_acc", "diff_pos", "dz", "vz", "tb"]
    out = p.select(keep).with_columns(pl.lit(sym).alias("sym"),
                                      pl.col(pl.Float64).cast(pl.Float32))
    os.makedirs(FEAT, exist_ok=True)
    out.write_parquet(os.path.join(FEAT, sym + ".parquet"))
    act = {k: float((out[k].drop_nulls() != 0).mean()) for k in FACTORS}
    return sym, out.height, act


# ------------------------------------------------------------------ stage 2
def lrv(u, lags):
    """Newey-West long-run variance of a mean-zero series (Bartlett kernel)."""
    s = (u * u).mean()
    for l in range(1, min(lags, len(u) - 1) + 1):
        s += 2 * (1 - l / (lags + 1)) * (u[l:] * u[:-l]).mean()
    return s


def nw_t(sums, counts, lags=20):
    """t-stat of the observation-weighted mean sum(sums)/sum(counts), treating days as the unit:
    ratio-estimator residuals u_d = S_d - m*n_d, HAC variance over days. Consistent with `edge`."""
    S, n = np.asarray(sums, float), np.asarray(counts, float)
    D = len(S)
    if D < 30:
        return float("nan")
    m = S.sum() / n.sum()
    v = lrv(S - m * n, lags)
    return m / (math.sqrt(D * v) / n.sum()) if v > 0 else float("nan")


def edge_table(lf, factor, lag, hn):
    """One row of statistics for factor x lag x horizon on a (lazy) frame of feature rows."""
    f = "f%d_%s" % (lag, hn)
    q = (lf.select("sym", "t", pl.col(factor).alias("s"), pl.col(f).alias("r"))
         .drop_nulls()
         .with_columns((pl.col("r") - pl.col("r").mean().over("sym")).alias("x"),   # remove each symbol's drift
                       pl.col("s").sign().alias("g"))
         .collect())
    if q.height == 0:
        return None
    a = q.filter(pl.col("g") != 0)
    if a.height < 100:
        return {"n": a.height}
    strong_th = 3.0 if factor in ("total",) else (1.0 if factor == "verdict" else 1.5)
    st = a.filter(pl.col("s").abs() >= strong_th)
    gx = pl.col("g") * pl.col("x")
    daily = a.group_by(pl.col("t").dt.date()).agg(gx.sum().alias("S"), pl.len().alias("n"), gx.mean().alias("v")).sort("t")
    per_sym = a.group_by("sym").agg(gx.mean().alias("e"), pl.len().alias("n")).filter(pl.col("n") >= 1000)
    yr = a.group_by(pl.col("t").dt.year().alias("y")).agg((gx.mean() * 1e4).alias("e")).sort("y")
    is_ = a.filter(pl.col("t") < pl.lit(SPLIT).str.to_datetime())
    oos = a.filter(pl.col("t") >= pl.lit(SPLIT).str.to_datetime())
    bull = a.filter(pl.col("g") > 0)["r"].mean()
    bear = a.filter(pl.col("g") < 0)["r"].mean()
    return {
        "n": a.height, "active": a.height / q.height,
        "edge_bp": a.select(gx.mean()).item() * 1e4,
        "raw_bp": a.select((pl.col("g") * pl.col("r")).mean()).item() * 1e4,
        "t": nw_t(daily["S"].to_numpy(), daily["n"].to_numpy()),
        "day_bp": daily["v"].mean() * 1e4,          # every day weighted equally
        "spread_bp": ((bull if bull is not None else float("nan")) - (bear if bear is not None else float("nan"))) * 1e4,
        "hit": a.select(((pl.col("g") * pl.col("r")) > 0).mean()).item(),
        "strong_bp": st.select(gx.mean()).item() * 1e4 if st.height >= 100 else float("nan"),
        "strong_n": st.height,
        "is_bp": is_.select(gx.mean()).item() * 1e4 if is_.height >= 100 else float("nan"),
        "oos_bp": oos.select(gx.mean()).item() * 1e4 if oos.height >= 100 else float("nan"),
        "breadth": (per_sym["e"] > 0).mean() if per_sym.height else float("nan"),
        "nsym": per_sym.height,
        "years": dict(zip(yr["y"].to_list(), yr["e"].to_list())),
        "ic": q.select(pl.corr("s", "r")).item(),
    }


def fmt(v, spec="%+6.1f"):
    return ("%6s" % "-") if v is None or (isinstance(v, float) and not math.isfinite(v)) else spec % v


def report(out_path):
    files = sorted(glob.glob(os.path.join(FEAT, "*.parquet")))
    lf_all = pl.scan_parquet(files)
    lf_btc = pl.scan_parquet(os.path.join(FEAT, "BTCUSDT.parquet"))
    L = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    span = lf_all.select(pl.col("t").min().alias("t0"), pl.col("t").max().alias("t1"),
                         pl.col("sym").n_unique(), pl.len()).collect().row(0)
    say("# 掃描器因子方向檢定：Binance Vision metrics + 1m K 線")
    say("樣本 %s ~ %s，%d 個 USDT 永續，%s 根 5 分鐘棒" % (span[0], span[1], span[2], format(span[3], ",")))
    say("edge = 跟著因子方向做、扣掉該標的全期平均漂移後的前瞻報酬 (bp，按啟動棒數加權)；t = 同一加權的日層級 Newey-West(20)；")
    say("日等權 = 每天先平均再跨天平均 (edge 與它差很多，代表結果靠少數長時間啟動的日子撐起)；")
    say("spread = 多方訊號減空方訊號的平均報酬 (本身不受漂移影響)；breadth = 樣本 ≥1000 筆的標的中 edge>0 的比例。")
    say("來回成本參考 %.0f bp (bt.py 同設定)。IS < %s ≤ OOS。" % (COST_BP, SPLIT))
    hdr = "%-26s %4s %9s %6s %7s %6s %7s %7s %6s %7s %7s %7s %6s %7s" % (
        "因子", "持有", "n(啟動)", "啟動%", "edge", "t", "日等權", "spread", "勝率", "強訊號", "IS", "OOS", "廣度", "原始")
    for scope, lf in (("BTCUSDT", lf_btc), ("全 74 標的合併", lf_all)):
        for lag in LAGS:
            say()
            say("## %s，進場延遲 %d 根 5m" % (scope, lag))
            say(hdr)
            for k in FACTORS:
                for hn in H:
                    r = edge_table(lf, k, lag, hn)
                    if r is None or "edge_bp" not in r:
                        say("%-26s %4s  樣本不足 (n=%s)" % (LABEL[k], hn, r and r.get("n")))
                        continue
                    say("%-26s %4s %9d %5.1f%% %s %s %s %s %5.1f%% %s %s %s %6s %s" % (
                        LABEL[k], hn, r["n"], 100 * r["active"], fmt(r["edge_bp"], "%+7.2f"), fmt(r["t"], "%+6.1f"),
                        fmt(r["day_bp"], "%+7.2f"), fmt(r["spread_bp"], "%+7.2f"), 100 * r["hit"], fmt(r["strong_bp"], "%+7.2f"),
                        fmt(r["is_bp"], "%+7.2f"), fmt(r["oos_bp"], "%+7.2f"),
                        "-" if scope == "BTCUSDT" else "%.0f%%" % (100 * r["breadth"]), fmt(r["raw_bp"], "%+7.2f")))
                    if scope == "BTCUSDT" and lag == 0 and hn == "1h":
                        say("%-26s      逐年 edge(bp): %s" % ("", "  ".join("%d %+.1f" % (y, e) for y, e in r["years"].items())))
    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["features", "report", "all"])
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--symbols")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "report.txt"))
    a = ap.parse_args()
    if a.stage in ("features", "all"):
        syms = a.symbols.split(",") if a.symbols else sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(PANEL, "*.parquet")))
        with ProcessPoolExecutor(a.workers) as ex:
            for sym, n, act in ex.map(features, syms):
                print("%-12s %8d  啟動率 " % (sym, n) + " ".join("%s=%.2f" % (k, v) for k, v in act.items()), flush=True)
    if a.stage in ("report", "all"):
        report(a.out)


if __name__ == "__main__":
    sys.exit(main())
