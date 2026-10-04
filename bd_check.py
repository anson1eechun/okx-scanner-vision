"""Which minute does a bookDepth snapshot stamped T belong to?
Infer mid from the +/-1% (or +/-0.2%) bands (notional/depth = average price), then correlate its
1-minute changes with 1m kline returns at lags -2..+2. Peak at 0 = aligned; +1 = snapshot sees the future."""
import io, os, sys, zipfile
import numpy as np, polars as pl
B = "/srv/nas/marketdata/vision/data/futures/um"
KL = ["open_time","open","high","low","close","volume","close_time","quote_volume","count","taker_buy_volume","taker_buy_quote_volume","ignore"]
def rz(p, cols=None):
    raw = zipfile.ZipFile(p).read(zipfile.ZipFile(p).namelist()[0])
    hh = not raw[:1].isdigit()
    return pl.read_csv(io.BytesIO(raw), has_header=hh, new_columns=None if hh else cols, infer_schema_length=0)
sym, month, band = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = os.path.join(B, "daily/bookDepth", sym)
bd = pl.concat([rz(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f"-{month}-" in f])
bd = bd.select(pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("T"), pl.col("percentage").cast(pl.Float64).alias("p"),
               pl.col("depth").cast(pl.Float64).alias("d"), pl.col("notional").cast(pl.Float64).alias("n"))
bd = bd.filter(pl.col("p").abs() == band).with_columns((pl.col("n")/pl.col("d")).alias("avgpx"))
w = bd.pivot(on="p", index="T", values="avgpx").sort("T")
w.columns = ["T", "bid", "ask"] if w.columns[1].startswith("-") else ["T", "ask", "bid"]
w = w.with_columns(((pl.col("bid")+pl.col("ask"))/2).alias("mhat"))
print("snapshots", w.height, "median spacing s", w["T"].diff().dt.total_seconds().median(), " seconds-of-minute mode", w["T"].dt.second().mode().to_list()[:3])
k = rz(os.path.join(B, "monthly/klines", sym, "1m", f"{sym}-1m-{month}.zip"), KL)
k = k.select(pl.from_epoch(pl.col("open_time").cast(pl.Int64), time_unit="ms").alias("ot"), pl.col("close").cast(pl.Float64))
# minute series: last snapshot in each minute, labelled by the minute it falls in
m = w.with_columns(pl.col("T").dt.truncate("1m").alias("ot")).group_by("ot").agg(pl.col("mhat").last()).sort("ot")
j = k.join(m, on="ot", how="inner").sort("ot")
r_k = np.diff(np.log(j["close"].to_numpy())); r_m = np.diff(np.log(j["mhat"].to_numpy()))
print("rows", len(r_k))
for lag in (-2, -1, 0, 1, 2):
    # lag>0: snapshot change in minute t vs kline return in minute t+lag
    a, b = (r_m[:len(r_m)-lag], r_k[lag:]) if lag >= 0 else (r_m[-lag:], r_k[:len(r_k)+lag])
    print("  snapshot minute t vs kline minute t%+d  corr %.3f" % (lag, np.corrcoef(a, b)[0,1]))
