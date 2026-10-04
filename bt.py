#!/usr/bin/env python3
"""Backtest strategy 2 (OI-confirmed breakout) and strategy 3 (crowded-trade fade)
on BTC-USDT-SWAP using data/ pulled by fetch_history.py.
Costs: 0.05% taker each side + 0.02% slippage = 0.12% per round trip.
"""
import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
COST = 0.0012

def load(name):
    with open(os.path.join(DATA, name + ".json")) as f:
        return json.load(f)

def zscore(x, window):
    if len(window) < 10: return 0.0
    sd = statistics.pstdev(window)
    return 0.0 if sd == 0 else (x - statistics.mean(window)) / sd

def build(per):
    c = load("candles_" + per)
    oi = {r["ts"]: r["oiUsd"] for r in load("oi_" + per)}
    crowd = {r["ts"]: r["ratio"] for r in load("crowd_" + per)}
    top = {r["ts"]: r["ratio"] for r in load("topacct_" + per)}
    taker = {r["ts"]: r for r in load("taker_" + per)}
    fund = load("funding")
    # funding z per settlement (rolling 90 prior)
    fz = []
    for i, r in enumerate(fund):
        w = [x["rate"] for x in fund[max(0, i - 90):i]]
        fz.append({"ts": r["ts"], "rate": r["rate"], "z": zscore(r["rate"], w)})
    bars = []
    fi = 0
    for i, k in enumerate(c):
        ts = k["ts"]
        if ts not in oi or ts not in crowd or ts not in top or ts not in taker:
            continue
        while fi + 1 < len(fz) and fz[fi + 1]["ts"] <= ts:
            fi += 1
        b = dict(k)
        b["oi"] = oi[ts]; b["crowd"] = crowd[ts]; b["top"] = top[ts]
        b["buy"] = taker[ts]["buy"]; b["sell"] = taker[ts]["sell"]
        b["fz"] = fz[fi]["z"] if fz[fi]["ts"] <= ts else 0.0
        b["frate"] = fz[fi]["rate"]
        bars.append(b)
    # derived: ATR14, vol z (48), crowd z (96), oi chg 3, taker ratio 3
    for i, b in enumerate(bars):
        trs = []
        for j in range(max(1, i - 13), i + 1):
            p = bars[j - 1]["c"]
            trs.append(max(bars[j]["h"] - bars[j]["l"], abs(bars[j]["h"] - p), abs(bars[j]["l"] - p)))
        b["atr"] = statistics.mean(trs) if trs else b["h"] - b["l"]
        b["volz"] = zscore(b["volUsd"], [x["volUsd"] for x in bars[max(0, i - 48):i]])
        b["crowdz"] = zscore(b["crowd"], [x["crowd"] for x in bars[max(0, i - 96):i]])
        b["oi3"] = (b["oi"] / bars[i - 3]["oi"] - 1) if i >= 3 else 0.0
        bb = sum(x["buy"] for x in bars[max(0, i - 2):i + 1]); ss = sum(x["sell"] for x in bars[max(0, i - 2):i + 1])
        b["tb"] = bb / (bb + ss) if bb + ss else 0.5
        b["topLong"] = b["top"] / (1 + b["top"]); b["crowdLong"] = b["crowd"] / (1 + b["crowd"])
    return bars

# ------------------------------------------------------------------ engines
def simulate(bars, signal_fn, exit_fn, stop_atr, max_hold):
    """signal_fn(bars,i)-> +1/-1/0 at close of bar i; enter at open of i+1.
    exit_fn(bars,i,pos)-> True to exit at close of bar i. Returns list of trades."""
    trades, i = [], 0
    while i < len(bars) - 2:
        s = signal_fn(bars, i)
        if s == 0:
            i += 1; continue
        e = bars[i + 1]; entry = e["o"]; atr = bars[i]["atr"]
        stop = entry - s * stop_atr * atr
        best = entry; j = i + 1; exit_px = None; reason = ""
        while j < len(bars):
            b = bars[j]
            if s == 1:
                best = max(best, b["h"])
                if b["l"] <= stop: exit_px, reason = stop, "stop"; break
            else:
                best = min(best, b["l"])
                if b["h"] >= stop: exit_px, reason = stop, "stop"; break
            if exit_fn(bars, j, s, entry, best):
                exit_px, reason = b["c"], "rule"; break
            if j - i >= max_hold:
                exit_px, reason = b["c"], "time"; break
            # trailing stop update (used by breakout strategy through exit_fn state) handled in exit_fn
            j += 1
        if exit_px is None:
            exit_px, reason = bars[-1]["c"], "eod"; j = len(bars) - 1
        pnl = s * (exit_px / entry - 1) - COST
        risk = stop_atr * atr / entry
        trades.append({"ts": e["ts"], "side": s, "entry": entry, "exit": exit_px, "pnl": pnl, "R": pnl / risk, "bars": j - i, "reason": reason})
        i = j + 1
    return trades

def stats(trades, label):
    if not trades:
        return "%-46s  n=0" % label
    p = [t["pnl"] for t in trades]
    wins = [x for x in p if x > 0]; losses = [x for x in p if x <= 0]
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf")
    cum = 0; peak = 0; dd = 0
    for x in p:
        cum += x; peak = max(peak, cum); dd = min(dd, cum - peak)
    return "%-46s  n=%3d  勝率 %4.0f%%  均R %+5.2f  PF %4.2f  淨 %+6.2f%%  最大回撤 %5.2f%%  均持 %4.1f 根" % (
        label, len(p), 100 * len(wins) / len(p), statistics.mean(t["R"] for t in trades), pf, 100 * sum(p), 100 * dd,
        statistics.mean(t["bars"] for t in trades))

# ------------------------------------------------------------------ strategy 2
def breakout(N, volz=1.5, need_oi=True, need_vol=True, need_taker=True, fund_filter=True, top_filter=False, crowd_filter=False):
    def sig(bars, i):
        if i < max(N, 50): return 0
        b = bars[i]
        hi = max(x["h"] for x in bars[i - N:i]); lo = min(x["l"] for x in bars[i - N:i])
        s = 1 if b["c"] > hi else (-1 if b["c"] < lo else 0)
        if s == 0: return 0
        if need_oi and b["oi3"] <= 0: return 0
        if need_vol and b["volz"] < volz: return 0
        if need_taker and ((s == 1 and b["tb"] < 0.55) or (s == -1 and b["tb"] > 0.45)): return 0
        if fund_filter and ((s == 1 and b["fz"] > 2) or (s == -1 and b["fz"] < -2)): return 0
        if top_filter and ((s == 1 and b["top"] < 1.0) or (s == -1 and b["top"] > 1.0)): return 0
        if crowd_filter and ((s == 1 and b["fz"] > 2 and b["crowdz"] > 1.5) or (s == -1 and b["fz"] < -2 and b["crowdz"] < -1.5)): return 0
        return s
    def ex(bars, j, s, entry, best):
        b = bars[j]; atr = b["atr"]
        trail = best - s * 2 * atr
        if (s == 1 and b["c"] < trail) or (s == -1 and b["c"] > trail): return True
        if j >= 3 and bars[j]["oi"] < bars[j-1]["oi"] < bars[j-2]["oi"] < bars[j-3]["oi"]: return True
        return False
    return sig, ex

# ------------------------------------------------------------------ strategy 3
def fade(fz_th=2.0, cz_th=1.5, div_th=0.10):
    def sig(bars, i):
        if i < 100: return 0
        b = bars[i]; div = b["topLong"] - b["crowdLong"]
        if b["fz"] > fz_th and b["crowdz"] > cz_th and div < -div_th: return -1
        if b["fz"] < -fz_th and b["crowdz"] < -cz_th and div > div_th: return 1
        return 0
    def ex(bars, j, s, entry, best):
        return (s == -1 and bars[j]["fz"] < 1.0) or (s == 1 and bars[j]["fz"] > -1.0)
    return sig, ex

def main():
    for per, hold8h in (("15m", 32), ("1H", 8)):
        bars = build(per)
        days = (bars[-1]["ts"] - bars[0]["ts"]) / 86400000
        print("\n==== %s  %d 根可用  %.1f 天 ====" % (per, len(bars), days))
        N = 12
        print("-- 策略 2  OI 確認突破 (N=%d, 停損 1.5 ATR, 追蹤 2 ATR, 最長 48 根)" % N)
        rows = [
            ("純突破，無確認", breakout(N, need_oi=False, need_vol=False, need_taker=False, fund_filter=False)),
            ("突破 + OI 上升", breakout(N, need_vol=False, need_taker=False, fund_filter=False)),
            ("突破 + OI + 量 z>1.5", breakout(N, need_taker=False, fund_filter=False)),
            ("突破 + OI + 量 + 主動流", breakout(N, fund_filter=False)),
            ("完整 (+ 資金費率過濾)", breakout(N)),
            ("完整 + 大戶方向一致", breakout(N, top_filter=True)),
            ("完整 + 策略 3 擁擠過濾", breakout(N, crowd_filter=True)),
        ]
        for label, (sig, ex) in rows:
            print(stats(simulate(bars, sig, ex, 1.5, 48), label))
        print("-- 策略 2 敏感度 (完整版)  N x 量門檻")
        for n in (8, 12, 24):
            for vz in (1.0, 1.5, 2.0):
                sig, ex = breakout(n, volz=vz)
                print(stats(simulate(bars, sig, ex, 1.5, 48), "  N=%2d  volz>%.1f" % (n, vz)))
        print("-- 策略 3  擁擠反向 (停損 2 ATR, 最長 8 小時)")
        for fz_th, cz_th, dv in ((2.0, 1.5, 0.10), (1.5, 1.0, 0.10), (1.0, 1.0, 0.05), (1.0, 0.5, 0.0)):
            sig, ex = fade(fz_th, cz_th, dv)
            print(stats(simulate(bars, sig, ex, 2.0, hold8h), "  費率z>%.1f 多空比z>%.1f 背離>%.2f" % (fz_th, cz_th, dv)))
        # how often is the crowded background even present?
        n_long = sum(1 for b in bars if b["fz"] > 2 and b["crowdz"] > 1.5)
        n_short = sum(1 for b in bars if b["fz"] < -2 and b["crowdz"] < -1.5)
        print("  擁擠背景出現根數：多頭擁擠 %d，空頭擁擠 %d（共 %d 根）" % (n_long, n_short, len(bars)))
        fzs = [b["fz"] for b in bars]
        print("  資金費率 z 分佈：最小 %.2f  中位 %.2f  最大 %.2f；超過 2 的比例 %.1f%%" % (min(fzs), statistics.median(fzs), max(fzs), 100 * sum(1 for x in fzs if x > 2) / len(fzs)))

if __name__ == "__main__":
    main()
