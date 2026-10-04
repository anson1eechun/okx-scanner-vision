#!/usr/bin/env python3
"""Step 2: sign-corrected scanner composite at 12h / 24h holds, funding included.
Preregistered in PREREG_STEP2_20260930.md; gates G1-G6 are evaluated exactly as written there.

Runs on anson-pc:  .venv/bin/python step2.py [--workers 14]
"""
import argparse, glob, math, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import polars as pl
import analyze, build

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "step2")
COST_BP = 12.0
SPLIT = "2024-01-01"
H = {"12h": 144, "24h": 288}
# sign-corrected components (column name -> expression); weights from scan.py
COMP = {"oi_c": -pl.col("oi"), "funding": pl.col("funding"), "crowd": pl.col("crowd"),
        "div_z": pl.col("div_z"), "volume_c": -pl.col("volume")}
W = {"oi_c": 1.0, "funding": 1.0, "crowd": 0.75, "div_z": 1.0, "volume_c": 0.75}


def build_sym(sym):
    f = pl.read_parquet(os.path.join(analyze.FEAT, sym + ".parquet"), columns=["t", "oi", "funding", "crowd", "div_z", "volume"])
    p = pl.read_parquet(os.path.join(analyze.PANEL, sym + ".parquet"), columns=["t", "close"])
    d = p.join(f, on="t", how="left").sort("t")          # panel grid is complete 5m, so shift(k) = k bars
    fr = build.funding(sym)
    if fr is not None and fr.height:
        fr = fr.sort("ft").with_columns(pl.col("funding_last").cum_sum().alias("fcum"))
        d = d.join_asof(fr.select("ft", "fcum"), left_on="t", right_on="ft", strategy="backward").drop("ft")
        d = d.with_columns(pl.col("fcum").fill_null(0.0))
    else:
        d = d.with_columns(pl.lit(0.0).alias("fcum"))
    d = d.with_columns([e.alias(k) for k, e in COMP.items()])
    d = d.with_columns(pl.when(pl.any_horizontal([pl.col(k).is_null() for k in W])).then(None)
                       .otherwise(pl.sum_horizontal([pl.col(k) * w for k, w in W.items()])).alias("S"))
    c, fc = pl.col("close"), pl.col("fcum")
    rets = []
    for lag, tag in ((0, "0"), (12, "1h"), (288, "24h")):
        for hn, h in H.items():
            if lag == 288 and hn == "12h":
                continue
            price = c.shift(-(lag + h)) / c.shift(-lag) - 1
            fund = fc.shift(-(lag + h)) - fc.shift(-lag)          # funding paid by a long over (t+lag, t+lag+h]
            rets.append((price - fund).alias("R%s_%s" % (tag, hn)))
            if lag == 0:
                rets.append(price.alias("P0_%s" % hn))
    d = d.with_columns(rets).with_columns(pl.lit(sym).alias("sym"))
    keep = ["t", "sym", "S"] + list(W) + [r.meta.output_name() for r in rets]
    out = d.select(keep).with_columns(pl.col(pl.Float64).cast(pl.Float32))
    os.makedirs(OUT, exist_ok=True)
    out.write_parquet(os.path.join(OUT, sym + ".parquet"))
    return sym, out.height


def stats(lf, g_expr, ret):
    """edge statistics for direction g on return column `ret` (drift-adjusted per symbol)."""
    q = (lf.select("sym", "t", g_expr.alias("g"), pl.col(ret).alias("r"))
         .drop_nulls().with_columns((pl.col("r") - pl.col("r").mean().over("sym")).alias("x"))
         .filter(pl.col("g") != 0).collect())
    if q.height < 100:
        return None
    gx = pl.col("g") * pl.col("x")
    day = q.group_by(pl.col("t").dt.date().alias("d")).agg(gx.sum().alias("S"), pl.len().alias("n")).sort("d")
    S_, n_ = day["S"].to_numpy().astype(float), day["n"].to_numpy().astype(float)
    v = S_ / n_
    keep = v <= np.quantile(v, 0.95)
    per = q.group_by("sym").agg(gx.mean().alias("e"), pl.len().alias("n")).filter(pl.col("n") >= 1000)
    split = pl.lit(SPLIT).str.to_datetime()
    recent = q.filter(pl.col("t") >= split); early = q.filter(pl.col("t") < split)
    yr = q.group_by(pl.col("t").dt.year().alias("y")).agg((gx.mean() * 1e4).alias("e")).sort("y")
    return {"n": q.height, "edge": q.select(gx.mean()).item() * 1e4, "t": analyze.nw_t(S_, n_, lags=20),
            "breadth": float((per["e"] > 0).mean()) if per.height else float("nan"), "nsym": per.height,
            "ex5": S_[keep].sum() / n_[keep].sum() * 1e4,
            "early": early.select(gx.mean()).item() * 1e4 if early.height else float("nan"),
            "recent": recent.select(gx.mean()).item() * 1e4 if recent.height else float("nan"),
            "years": dict(zip(yr["y"].to_list(), yr["e"].to_list())), "days": len(day)}


def fmt_row(label, r):
    if r is None:
        return "  %-34s 樣本不足" % label
    return "  %-34s n=%9d  edge %+7.2f  t %+5.1f  廣度 %4.0f%%  剔5%%日 %+7.2f  <2024 %+7.2f  ≥2024 %+7.2f" % (
        label, r["n"], r["edge"], r["t"], 100 * r["breadth"], r["ex5"], r["early"], r["recent"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "step2.txt"))
    a = ap.parse_args()
    syms = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(analyze.FEAT, "*.parquet")))
    if not a.skip_build:
        with ProcessPoolExecutor(a.workers) as ex:
            for sym, n in ex.map(build_sym, syms):
                pass
    L = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    lf = pl.scan_parquet(os.path.join(OUT, "*.parquet"))
    btc = pl.scan_parquet(os.path.join(OUT, "BTCUSDT.parquet"))
    act = pl.when(pl.col("S").abs() >= 3).then(pl.col("S").sign()).otherwise(0)

    say("# 第 2 步：修正方向的掃描器合成分數，持有 12h／24h（預先註記 PREREG_STEP2_20260930.md）")
    say("S = −oi + funding + 0.75·crowd + div_z − 0.75·volume；啟動 |S| ≥ 3；R 含資金費率、扣各幣漂移；成本 %.0f bp" % COST_BP)
    say()
    main_r = stats(lf, act, "R0_24h")
    lag_r = stats(lf, act, "R1h_24h")
    say("## 主規格：74 幣合併，24h，含資金費率")
    say(fmt_row("|S| ≥ 3", main_r))
    say(fmt_row("同上，延後 1 小時進場", lag_r))
    if main_r:
        gates = [
            ("G1 全期 edge > 12 bp", main_r["edge"] > COST_BP, "%+.2f bp" % main_r["edge"]),
            ("G2 t ≥ 3", main_r["t"] >= 3, "t = %+.2f" % main_r["t"]),
            ("G3 廣度 ≥ 80%", main_r["breadth"] >= 0.8, "%.0f%%（%d 幣）" % (100 * main_r["breadth"], main_r["nsym"])),
            ("G4 剔最好 5% 日後 edge > 0", main_r["ex5"] > 0, "%+.2f bp" % main_r["ex5"]),
            ("G5 延後 1h 同號且 ≥ 一半", lag_r is not None and np.sign(lag_r["edge"]) == np.sign(main_r["edge"]) and abs(lag_r["edge"]) >= 0.5 * abs(main_r["edge"]),
             "%+.2f bp" % (lag_r["edge"] if lag_r else float("nan"))),
            ("G6 2024 後 edge > 12 bp", main_r["recent"] > COST_BP, "%+.2f bp" % main_r["recent"]),
        ]
        say()
        say("## 過關條件")
        for name, ok, val in gates:
            say("  %-30s %-4s  %s" % (name, "過" if ok else "不過", val))
        verdict = all(ok for _, ok, _ in gates)
        say()
        say("## 判決：%s" % ("PASS —— 值得前瞻測試（不代表有效）" if verdict else "FAIL —— 掃描器因子不再作為交易訊號研究"))
        say("  逐年 edge(bp)：" + "  ".join("%d %+.1f" % (y, e) for y, e in main_r["years"].items()))

    say()
    say("## 次要報告（描述用，不影響判決）")
    say("### 12h 持有")
    say(fmt_row("|S| ≥ 3, 12h", stats(lf, act, "R0_12h")))
    say(fmt_row("|S| ≥ 3, 12h, 延後 1h", stats(lf, act, "R1h_12h")))
    say("### 其他變化（24h）")
    say(fmt_row("只看符號 sign(S)", stats(lf, pl.col("S").sign(), "R0_24h")))
    say(fmt_row("|S| ≥ 3, 純價格（不含資金費）", stats(lf, act, "P0_24h")))
    say(fmt_row("|S| ≥ 3, 延後 24h 進場", stats(lf, act, "R24h_24h")))
    say(fmt_row("BTC 單獨 |S| ≥ 3", stats(btc, act, "R0_24h")))
    say("### 各分量（已修正方向，sign(score)，含資金費）")
    for k in W:
        for hn in H:
            say(fmt_row("%s %s" % (k, hn), stats(lf, pl.col(k).sign(), "R0_%s" % hn)))
    with open(a.out, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
