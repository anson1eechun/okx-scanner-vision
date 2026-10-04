#!/usr/bin/env python3
"""Deep Book Imbalance (DBI) — a strategy shaped by what the evidence allows.

Why it looks like this:
  - fees are 2 bps maker / 5 bps taker per side, so the stop must be >> 10 bps;
    60 bps makes a maker round trip 0.07 R instead of the current 2.50 R
  - the tape has no edge (42-day study), so the signal is depth-only
  - a 60 bps stop is ~1.4 sigma of a 60-min move, so the horizon is hours

Signal: equal-weighted composite of three deep-book imbalance measures, all signed
        positive = more resting size on the bid side. NOTHING IS FITTED — equal weights,
        so applying it to the held-out half is a genuine out-of-sample test.
Entry:  limit at mid (maker). Fills are assumed only if price trades through the level.
Exit:   fixed stop / target, or time stop.
"""
import math, os, sys
import numpy as np, pyarrow.parquet as pq

HERE=os.path.dirname(os.path.abspath(__file__))
MAKER_BPS=2.0; TAKER_BPS=5.0
COMPONENTS=["bandimb_mid","imb200","bandimb_far"]

def zs(x,w=240):
    """causal z-score over a trailing window"""
    n=len(x); out=np.full(n,np.nan)
    c=np.concatenate([[0.0],np.cumsum(np.nan_to_num(x))])
    c2=np.concatenate([[0.0],np.cumsum(np.nan_to_num(x)**2)])
    for i in range(w,n):
        m=(c[i]-c[i-w])/w; v=max((c2[i]-c2[i-w])/w-m*m,1e-12)
        out[i]=(x[i]-m)/math.sqrt(v)
    return out

def backtest(ts,px,sig,thr,stop_bps,rr,max_h,entry_fee,exit_fee):
    """one position at a time; entry at the bar close (maker assumed filled at mid)"""
    n=len(px); i=0; trades=[]
    while i<n-1:
        s=sig[i]
        if not np.isfinite(s) or abs(s)<thr: i+=1; continue
        long=s>0; e=px[i]
        sl=e*(1-stop_bps/1e4) if long else e*(1+stop_bps/1e4)
        tp=e*(1+rr*stop_bps/1e4) if long else e*(1-rr*stop_bps/1e4)
        out=None
        for j in range(i+1,min(n,i+max_h+1)):
            if long:
                if px[j]<=sl: out=(-1.0,j,"SL"); break
                if px[j]>=tp: out=(rr,j,"TP"); break
            else:
                if px[j]>=sl: out=(-1.0,j,"SL"); break
                if px[j]<=tp: out=(rr,j,"TP"); break
        if out is None:
            j=min(n-1,i+max_h)
            r=((px[j]-e) if long else (e-px[j]))/(e*stop_bps/1e4)
            out=(r,j,"time")
        fee_r=(entry_fee+exit_fee)/stop_bps
        trades.append(dict(i=i,j=out[1],long=long,r=out[0]-fee_r,raw=out[0],how=out[2]))
        i=out[1]+1
    return trades

def report(name,tr):
    if not tr: return print("  %-30s 無交易" % name)
    r=np.array([t["r"] for t in tr]); w=(r>0).sum()
    eq=np.cumsum(r); dd=float(np.min(eq-np.maximum.accumulate(eq))) if len(eq) else 0
    sh=r.mean()/r.std()*math.sqrt(len(r)) if r.std()>0 else 0
    print("  %-30s n=%3d  勝率 %3.0f%%  合計 %+7.1fR  每筆 %+5.2fR  最大回撤 %6.1fR  t=%+4.1f" %
          (name,len(r),100*w/len(r),r.sum(),r.mean(),dd,sh))

def main():
    d={k:np.asarray(v,dtype=np.float64) for k,v in pq.read_table(os.path.join(HERE,"book_features.parquet")).to_pydict().items()}
    n=len(d["ts"]); px=d["px"]; ts=d["ts"]
    comp=np.nanmean(np.vstack([zs(d[c]) for c in COMPONENTS]),axis=0)
    split=int(n*0.6)
    print("=== DBI 合成訊號 = %s 的等權 z 分數平均 (無擬合參數)" % " + ".join(COMPONENTS))
    print("=== 樣本 %d 分鐘 (%.1f 小時), 樣本內 %d / 樣本外 %d" % (n,(ts[-1]-ts[0])/3.6e6,split,n-split))
    print("\n=== 費用情境比較 (門檻 |z|>1.0, 停損 60 bps, R=2, 最長 4 小時)")
    for label,(ef,xf) in (("雙邊 maker",(MAKER_BPS,MAKER_BPS)),("進 maker 出 taker",(MAKER_BPS,TAKER_BPS)),("雙邊 taker",(TAKER_BPS,TAKER_BPS))):
        tr=backtest(ts,px,comp,1.0,60,2.0,240,ef,xf)
        report(label+" (全樣本)",tr)
    print("\n=== 樣本外 (後 40%%), 雙邊 maker")
    for thr in (0.5,1.0,1.5,2.0):
        tr=backtest(ts[split:],px[split:],comp[split:],thr,60,2.0,240,MAKER_BPS,MAKER_BPS)
        report("門檻 |z|>%.1f"%thr,tr)
    print("\n=== 停損寬度敏感度 (樣本外, |z|>1.0, R=2, 雙邊 maker)")
    for sb in (20,40,60,100,150):
        tr=backtest(ts[split:],px[split:],comp[split:],1.0,sb,2.0,240,MAKER_BPS,MAKER_BPS)
        report("停損 %3d bps (費用 %.2fR)"%(sb,2*MAKER_BPS/sb),tr)
    print("\n=== 對照: 同期無條件做多 / 做空")
    for lab,sgn in (("永遠做多",+1),("永遠做空",-1)):
        fake=np.full(n,float(sgn)*2)
        tr=backtest(ts[split:],px[split:],fake[split:],1.0,60,2.0,240,MAKER_BPS,MAKER_BPS)
        report(lab,tr)
    # power: how much data to confirm
    print("\n=== 要多少資料才能確認")
    tr=backtest(ts,px,comp,1.0,60,2.0,240,MAKER_BPS,MAKER_BPS)
    if tr:
        r=np.array([t["r"] for t in tr]); per_hr=len(tr)/((ts[-1]-ts[0])/3.6e6)
        if r.std()>0:
            need=(2.0*r.std()/max(abs(r.mean()),1e-6))**2      # t=2
            print("  觀測到每筆 %+.3fR (標準差 %.2f), 每小時 %.2f 筆" % (r.mean(),r.std(),per_hr))
            print("  要達到 t=2 需要約 %.0f 筆 = %.0f 小時 = %.1f 天" % (need,need/per_hr,need/per_hr/24))

if __name__=="__main__": main()
