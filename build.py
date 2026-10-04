#!/usr/bin/env python3
"""Build a 5m panel per symbol from Binance Vision USDT-M futures dumps on the NAS.

Output: out/panel/<SYM>.parquet with one row per 5m bar, labelled by bar END time `t`
(so every field describes information available at `t`):

  open, high, low, close, quote_vol,
  taker_buy_quote                         from 1m klines aggregated to 5m
  oi_usd, top_ls_pos, top_ls_acc,
  crowd_ls, taker_ls                      from daily metrics (re-stamped to completion time, see ALIGN_TOL)
  funding_last                            last settled funding rate known at `t`

Runs on anson-pc:  .venv/bin/python build.py [--symbols BTCUSDT,ETHUSDT] [--workers 12]
"""
import argparse, io, os, sys, zipfile
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import polars as pl

B = "/srv/nas/marketdata/vision/data/futures/um"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "panel")
END_T = datetime(2026, 9, 1)  # last bar = the one closing at 2026-09-01 00:00; monthly klines end with 2026-08
KL_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
# Binance changed the metrics timestamp convention on 2024-03-04 (check_align.py, BTCUSDT):
# before, create_time=T describes the 5m window ENDING at T; from then on, the window STARTING at T.
# Taken at face value, every post-2024-03 row would leak the next 5 minutes. align_metrics() decides
# per symbol-day by matching the metrics taker ratio against the one rebuilt from 1m klines, and only
# uses the no-shift reading on days where that match is confirmed; anything else gets +5m (never early).
ALIGN_TOL = 0.005


def read_zip_csv(path, cols=None):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    if not raw:
        return None
    has_header = not raw[:1].isdigit()
    return pl.read_csv(io.BytesIO(raw), has_header=has_header,
                       new_columns=None if has_header else cols, infer_schema_length=0)


def klines_5m(sym):
    d = os.path.join(B, "monthly", "klines", sym, "1m")
    frames = []
    for f in sorted(os.listdir(d)):
        df = read_zip_csv(os.path.join(d, f), KL_COLS)
        if df is None:
            continue
        df.columns = KL_COLS[:len(df.columns)]
        frames.append(df.select(
            pl.col("open_time").cast(pl.Int64),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("quote_volume").cast(pl.Float64),
            pl.col("taker_buy_quote_volume").cast(pl.Float64)))
    k = pl.concat(frames).unique("open_time").sort("open_time")
    k = k.with_columns(pl.from_epoch("open_time", time_unit="ms").alias("ot"))
    # 5m bucket [T0, T0+5m) labelled by its end T0+5m
    k = k.with_columns((pl.col("ot").dt.truncate("5m") + pl.duration(minutes=5)).alias("t"))
    agg = k.group_by("t").agg(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("quote_volume").sum().alias("quote_vol"),
        pl.col("taker_buy_quote_volume").sum().alias("taker_buy_quote"),
        pl.len().alias("n1m")).sort("t")
    return agg.filter(pl.col("n1m") == 5).drop("n1m")


def metrics(sym):
    d = os.path.join(B, "daily", "metrics", sym)
    frames = []
    for f in sorted(os.listdir(d)):
        df = read_zip_csv(os.path.join(d, f))
        if df is None or df.height == 0:
            continue
        frames.append(df)
    m = pl.concat(frames, how="diagonal_relaxed")
    m = m.select(
        pl.col("create_time").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("ct"),
        pl.col("sum_open_interest_value").cast(pl.Float64, strict=False).alias("oi_usd"),
        pl.col("sum_toptrader_long_short_ratio").cast(pl.Float64, strict=False).alias("top_ls_pos"),
        pl.col("count_toptrader_long_short_ratio").cast(pl.Float64, strict=False).alias("top_ls_acc"),
        pl.col("count_long_short_ratio").cast(pl.Float64, strict=False).alias("crowd_ls"),
        pl.col("sum_taker_long_short_vol_ratio").cast(pl.Float64, strict=False).alias("taker_ls"),
    ).drop_nulls("ct")
    # only rows on the 5m grid; duplicates -> keep last
    m = m.filter(pl.col("ct").dt.minute() % 5 == 0).filter(pl.col("ct").dt.second() == 0)
    m = m.unique("ct", keep="last").sort("ct")
    # 0 / negative ratios are feed glitches
    for c in ["oi_usd", "top_ls_pos", "top_ls_acc", "crowd_ls", "taker_ls"]:
        m = m.with_columns(pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(None).alias(c))
    return m


def align_metrics(m, k):
    """Return m with `t` = time the row's information is complete, plus per-day `align_min`."""
    ks = k.select("t", (pl.col("taker_buy_quote") / pl.col("quote_vol")).alias("k_share"))
    x = m.select("ct", (pl.col("taker_ls") / (1 + pl.col("taker_ls"))).alias("m_share"))
    x = x.join(ks.rename({"t": "ct", "k_share": "k0"}), on="ct", how="left")
    x = x.join(ks.with_columns(pl.col("t") - pl.duration(minutes=5)).rename({"t": "ct", "k_share": "k5"}), on="ct", how="left")
    day = x.group_by(pl.col("ct").dt.date().alias("d")).agg(
        (pl.col("m_share") - pl.col("k0")).abs().median().alias("e0"),
        (pl.col("m_share") - pl.col("k5")).abs().median().alias("e5"))
    day = day.with_columns(
        pl.when((pl.col("e0") < ALIGN_TOL) & (pl.col("e0") < pl.col("e5"))).then(0).otherwise(5).alias("align_min"))
    m = m.with_columns(pl.col("ct").dt.date().alias("d")).join(day.select("d", "align_min"), on="d", how="left")
    m = m.with_columns(pl.col("align_min").fill_null(5))
    m = m.with_columns((pl.col("ct") + pl.duration(minutes=1) * pl.col("align_min")).alias("t")).drop("ct", "d")
    return m.unique("t", keep="last").sort("t")


def funding(sym):
    d = os.path.join(B, "monthly", "fundingRate", sym)
    if not os.path.isdir(d):
        return None
    frames = []
    for f in sorted(os.listdir(d)):
        df = read_zip_csv(os.path.join(d, f), ["calc_time", "funding_interval_hours", "last_funding_rate"])
        if df is not None:
            frames.append(df.select(pl.col("calc_time").cast(pl.Int64), pl.col("last_funding_rate").cast(pl.Float64)))
    if not frames:
        return None
    f = pl.concat(frames).unique("calc_time").sort("calc_time")
    return f.select(pl.from_epoch("calc_time", time_unit="ms").alias("ft"), pl.col("last_funding_rate").alias("funding_last"))


def build(sym):
    try:
        k = klines_5m(sym)
        m = align_metrics(metrics(sym), k)
        start = m["t"].min()
        grid = pl.datetime_range(start, END_T, "5m", eager=True).alias("t").to_frame()
        p = grid.join(k, on="t", how="left").join(m, on="t", how="left")
        fr = funding(sym)
        if fr is not None:
            p = p.sort("t").join_asof(fr.sort("ft"), left_on="t", right_on="ft", strategy="backward").drop("ft")
        else:
            p = p.with_columns(pl.lit(None, pl.Float64).alias("funding_last"))
        os.makedirs(OUT, exist_ok=True)
        p.write_parquet(os.path.join(OUT, sym + ".parquet"))
        return sym, p.height, int(p["close"].null_count()), int(p["oi_usd"].null_count()), str(start)
    except Exception as e:  # keep going; report at the end
        return sym, -1, 0, 0, repr(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    syms = a.symbols.split(",") if a.symbols else sorted(os.listdir(os.path.join(B, "daily", "metrics")))
    with ProcessPoolExecutor(a.workers) as ex:
        for sym, n, nc, no, info in ex.map(build, syms):
            print("%-12s rows=%8d  close_null=%7d  oi_null=%7d  %s" % (sym, n, nc, no, info), flush=True)


if __name__ == "__main__":
    sys.exit(main())
