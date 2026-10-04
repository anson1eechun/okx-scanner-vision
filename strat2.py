#!/usr/bin/env python3
"""Re-run bt.py strategy 2 (OI-confirmed breakout) on 6 years of Binance data instead of 15-60 days of OKX.

Uses bt.py's own simulate / breakout / stats unchanged; only the bars come from the 5m panel,
resampled to 15m and 1H, with bt.build()'s derived fields rebuilt the same way:
ATR14, volume z vs previous 48 bars, OI change over 3 bars, 3-bar taker buy share,
funding z vs previous 90 settlements, top-trader ACCOUNT ratio (bt.py uses topacct).

Runs on anson-pc:  .venv/bin/python strat2.py [--symbols BTCUSDT] [--workers 14]
"""
import argparse, glob, math, os, statistics, sys
from concurrent.futures import ProcessPoolExecutor
import polars as pl
import bt
import analyze

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_MS = 1704067200000   # 2024-01-01 UTC
N = 12
VARIANTS = [
    ("純突破，無確認", dict(need_oi=False, need_vol=False, need_taker=False, fund_filter=False)),
    ("突破 + OI 上升", dict(need_vol=False, need_taker=False, fund_filter=False)),
    ("突破 + OI + 量 z>1.5", dict(need_taker=False, fund_filter=False)),
    ("突破 + OI + 量 + 主動流", dict(fund_filter=False)),
    ("完整 (+ 資金費率過濾)", dict()),
]


def bars_for(sym, every):
    p = pl.read_parquet(os.path.join(analyze.PANEL, sym + ".parquet")).sort("t")
    g = (p.group_by_dynamic("t", every=every, closed="right", label="right")
         .agg(pl.col("open").first().alias("o"), pl.col("high").max().alias("h"), pl.col("low").min().alias("l"),
              pl.col("close").last().alias("c"), pl.col("quote_vol").sum().alias("volUsd"),
              pl.col("taker_buy_quote").sum().alias("buy"),
              (pl.col("quote_vol") - pl.col("taker_buy_quote")).sum().alias("sell"),
              pl.col("oi_usd").last().alias("oi"), pl.col("crowd_ls").last().alias("crowd"),
              pl.col("top_ls_acc").last().alias("top"), pl.col("close").count().alias("n")))
    need = {"15m": 3, "1h": 12}[every]
    g = g.filter(pl.col("n") == need).drop("n").drop_nulls(["o", "h", "l", "c", "oi", "crowd", "top"])
    fz = analyze.funding_z(sym, n=90)
    if fz is not None:
        g = g.join_asof(fz.select("ft", "fz", "funding_last").sort("ft"), left_on="t", right_on="ft", strategy="backward").drop("ft")
        g = g.with_columns(pl.col("fz").fill_null(0.0), pl.col("funding_last").fill_null(0.0).alias("frate"))
    else:
        g = g.with_columns(pl.lit(0.0).alias("fz"), pl.lit(0.0).alias("frate"))
    pc = pl.col("c").shift(1)
    tr = pl.max_horizontal(pl.col("h") - pl.col("l"), (pl.col("h") - pc).abs(), (pl.col("l") - pc).abs())
    g = g.with_columns(tr.alias("tr"))
    g = g.with_columns(
        pl.col("tr").rolling_mean(14, min_samples=1).fill_null(pl.col("h") - pl.col("l")).alias("atr"),
        analyze.zroll("volUsd", 48, prior_only=True).fill_null(0.0).alias("volz"),
        analyze.zroll("crowd", 96, prior_only=True).fill_null(0.0).alias("crowdz"),
        (pl.col("oi") / pl.col("oi").shift(3) - 1).fill_null(0.0).alias("oi3"),
        (pl.col("buy").rolling_sum(3, min_samples=1) /
         (pl.col("buy") + pl.col("sell")).rolling_sum(3, min_samples=1)).fill_null(0.5).fill_nan(0.5).alias("tb"),
        (pl.col("top") / (1 + pl.col("top"))).alias("topLong"),
        (pl.col("crowd") / (1 + pl.col("crowd"))).alias("crowdLong"),
        (pl.col("t").dt.epoch("ms")).alias("ts"))
    return g.drop("t", "tr").to_dicts()


def run(args):
    sym, every = args
    bars = bars_for(sym, every)
    out = {}
    for label, kw in VARIANTS:
        sig, ex = bt.breakout(N, **kw)
        out[label] = bt.simulate(bars, sig, ex, 1.5, 48)
    days = (bars[-1]["ts"] - bars[0]["ts"]) / 86400000 if bars else 0
    return sym, every, len(bars), days, out


def pf(p):
    w = sum(x for x in p if x > 0); l = -sum(x for x in p if x <= 0)
    return w / l if l > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "strat2.txt"))
    a = ap.parse_args()
    syms = a.symbols.split(",") if a.symbols else sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(analyze.PANEL, "*.parquet")))
    jobs = [(s, e) for e in ("15m", "1h") for s in syms]
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for sym, every, nb, days, out in ex.map(run, jobs):
            res[(sym, every)] = (nb, days, out)
    L = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    say("# 策略 2 (OI 確認突破) 以 bt.py 原始碼重跑於 Binance 6 年資料  N=%d, 停損 1.5 ATR, 追蹤 2 ATR, 最長 48 根, 成本 %.2f%%/來回" % (N, bt.COST * 100))
    for every in ("15m", "1h"):
        if ("BTCUSDT", every) in res:
            nb, days, out = res[("BTCUSDT", every)]
            say()
            say("==== BTCUSDT %s  %d 根  %.0f 天 ====" % (every, nb, days))
            for label, _ in VARIANTS:
                tr = out[label]
                say(bt.stats(tr, label))
                is_ = [t for t in tr if t["ts"] < SPLIT_MS]; oos = [t for t in tr if t["ts"] >= SPLIT_MS]
                say(bt.stats(is_, "    IS  < 2024"))
                say(bt.stats(oos, "    OOS >= 2024"))
        say()
        say("==== 全部標的 %s：每筆平均淨報酬、合併 PF、PF>1 的標的比例 ====" % every)
        say("%-24s %8s %9s %6s %7s %8s %9s %9s %9s" % ("", "筆數", "均淨(bp)", "t", "合併PF", "PF>1比例", "中位PF", "IS(bp)", "OOS(bp)"))
        base = {}
        for label, _ in VARIANTS:
            allp, allt, pfs, per_sym = [], [], [], {}
            for (sym, e), (nb, days, out) in res.items():
                if e != every:
                    continue
                p = [t["pnl"] for t in out[label]]
                allp += p; allt += out[label]
                if len(p) >= 20:
                    pfs.append(pf(p))
                per_sym[sym] = statistics.mean(p) if p else None
            if label.startswith("純突破"):
                base = per_sym
            # trades on the same day across symbols are not independent: day-clustered HAC t
            by_day = {}
            for t in allt:
                d = t["ts"] // 86400000
                S, n = by_day.get(d, (0.0, 0))
                by_day[d] = (S + t["pnl"], n + 1)
            days_sorted = sorted(by_day)
            tstat = analyze.nw_t([by_day[d][0] for d in days_sorted], [by_day[d][1] for d in days_sorted])
            is_ = [t["pnl"] for t in allt if t["ts"] < SPLIT_MS]; oos = [t["pnl"] for t in allt if t["ts"] >= SPLIT_MS]
            say("%-24s %8d %+9.1f %+6.1f %7.2f %7.0f%% %9.2f %+9.1f %+9.1f" % (
                label, len(allp), 1e4 * statistics.mean(allp) if allp else float("nan"), tstat, pf(allp),
                100 * sum(1 for x in pfs if x > 1) / len(pfs) if pfs else float("nan"),
                statistics.median(pfs) if pfs else float("nan"),
                1e4 * statistics.mean(is_) if is_ else float("nan"), 1e4 * statistics.mean(oos) if oos else float("nan")))
            if not label.startswith("純突破") and base:
                d = [per_sym[s] - base[s] for s in per_sym if per_sym[s] is not None and base.get(s) is not None]
                say("%-24s 相對純突破：每筆均淨改善的標的比例 %.0f%%（%d 標的）" % ("", 100 * sum(1 for x in d if x > 0) / len(d) if d else float("nan"), len(d)))
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
