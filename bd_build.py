#!/usr/bin/env python3
"""Binance Vision um/bookDepth -> one minute series per symbol (step 3, see PREREG_STEP3_DBI_BOOKDEPTH_20260930.md).

For every snapshot keep the cumulative notional at -1%, +1% (and -0.2%, +0.2% where present, from 2026-01-15).
Minute rows take the LAST snapshot inside each minute and are labelled by the minute END, so a row at t only
contains snapshots taken before t. Output: out/bookdepth/<SYM>.parquet with
  t, imb1, imb02, band02_1, snap_s (second-of-minute of the snapshot used)

Runs on anson-pc:  .venv/bin/python bd_build.py [--symbols BTCUSDT] [--workers 14]
"""
import argparse, io, os, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
import polars as pl

B = "/srv/nas/marketdata/vision/data/futures/um/daily/bookDepth"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "bookdepth")
START, END = "2023-01-01", "2026-08-31"
KEEP = {-1.0: "bm1", 1.0: "am1", -0.2: "bm02", 0.2: "am02"}


def read_day(path):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    if not raw:
        return None
    df = pl.read_csv(io.BytesIO(raw), infer_schema_length=0)
    df = df.select(pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("T"),
                   pl.col("percentage").cast(pl.Float64, strict=False).round(2).alias("p"),
                   pl.col("notional").cast(pl.Float64, strict=False).alias("n")).drop_nulls()
    return df.filter(pl.col("p").is_in(list(KEEP)))


def build(sym):
    try:
        d = os.path.join(B, sym)
        files = sorted(f for f in os.listdir(d) if START <= f[-14:-4] <= END)
        frames = [x for x in (read_day(os.path.join(d, f)) for f in files) if x is not None and x.height]
        df = pl.concat(frames)
        df = df.unique(["T", "p"], keep="last")
        w = df.with_columns(pl.col("p").replace_strict(KEEP, return_dtype=pl.Utf8).alias("k")).pivot(
            on="k", index="T", values="n", aggregate_function="last").sort("T")
        for c in KEEP.values():
            if c not in w.columns:
                w = w.with_columns(pl.lit(None, pl.Float64).alias(c))
        w = w.with_columns(pl.col("T").dt.truncate("1m").alias("m"))
        m = w.group_by("m").agg(pl.all().sort_by("T").last()).sort("m")
        m = m.with_columns((pl.col("m") + pl.duration(minutes=1)).alias("t"), pl.col("T").dt.second().alias("snap_s"))
        b1, a1, b02, a02 = pl.col("bm1"), pl.col("am1"), pl.col("bm02"), pl.col("am02")
        m = m.with_columns(
            ((b1 - a1) / (b1 + a1)).alias("imb1"),
            ((b02 - a02) / (b02 + a02)).alias("imb02"),
            (((b1 - b02) - (a1 - a02)) / ((b1 - b02) + (a1 - a02))).alias("band02_1"))
        out = m.select("t", "imb1", "imb02", "band02_1", "snap_s").filter(pl.col("imb1").is_not_null())
        os.makedirs(OUT, exist_ok=True)
        out.write_parquet(os.path.join(OUT, sym + ".parquet"))
        return sym, len(files), out.height, int(out["imb02"].is_not_null().sum()), str(out["t"].min())
    except Exception as e:
        return sym, -1, 0, 0, repr(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    syms = a.symbols.split(",") if a.symbols else sorted(os.listdir(B))
    with ProcessPoolExecutor(a.workers) as ex:
        for sym, nf, nr, n02, info in ex.map(build, syms):
            print("%-12s files=%5d  minute_rows=%8d  with_0.2%%=%7d  %s" % (sym, nf, nr, n02, info), flush=True)


if __name__ == "__main__":
    sys.exit(main())
