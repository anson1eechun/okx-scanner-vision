#!/usr/bin/env python3
"""Verify the built panel: metrics taker share must match the kline bar ending at t (lag 0),
and the chosen per-day alignment should switch on 2024-03-04."""
import sys, polars as pl
sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
p = pl.read_parquet("out/panel/%s.parquet" % sym)
p = p.with_columns((pl.col("taker_ls") / (1 + pl.col("taker_ls"))).alias("m_share"),
                   (pl.col("taker_buy_quote") / pl.col("quote_vol")).alias("k_share"))
for lag in [-1, 0, 1]:
    q = p.select("m_share", pl.col("k_share").shift(-lag).alias("k")).drop_nulls()
    print("%s  metrics(t) vs kline bar ending t%+d min  median|diff|=%.4f" % (sym, 5 * lag, (q["m_share"] - q["k"]).abs().median()))
print(p.group_by(pl.col("t").dt.year().alias("y")).agg(pl.col("align_min").mean().round(3).alias("mean_align_min")).sort("y").rows())
