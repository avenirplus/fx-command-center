#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import (
    OHLC, download_zip, read_m1, resample, pivot_arrays, atr_wilder,
    macd_hist, eval_trades_rr_nonoverlap, summarize
)
from backtest_trend_modes import structure_trend

def pivots_l(b, L):
    lo=b["Low"].to_numpy(float); hi=b["High"].to_numpy(float)
    pl=np.ones(len(b),dtype=bool); ph=np.ones(len(b),dtype=bool)
    pl[:L]=False; pl[len(b)-L:]=False
    ph[:L]=False; ph[len(b)-L:]=False
    for k in range(1,L+1):
        pl &= (lo <= np.roll(lo,k)) & (lo <= np.roll(lo,-k))
        ph &= (hi >= np.roll(hi,k)) & (hi >= np.roll(hi,-k))
    pl[:L]=False; pl[len(b)-L:]=False
    ph[:L]=False; ph[len(b)-L:]=False
    return np.flatnonzero(pl),np.flatnonzero(ph)

def build_t2(b, fib_lo, fib_hi, pull_atr, h1_pivot_l):
    tv=structure_trend(b)
    av=atr_wilder(b,14)
    mh=macd_hist(b["Close"])

    h4=b.resample("4h",label="left",closed="left").agg(OHLC).dropna()
    lp4,hp4=pivot_arrays(h4)
    ht=h4.index
    lows4=h4["Low"].to_numpy(float); highs4=h4["High"].to_numpy(float)
    lpt=ht[lp4]; hpt=ht[hp4]

    close=b["Close"].to_numpy(float); high=b["High"].to_numpy(float); low=b["Low"].to_numpy(float)
    avv=av.to_numpy(float); times=b.index
    candidate=np.zeros(len(b),bool); direction=np.zeros(len(b),np.int8)

    for i in range(4,len(b)):
        d=int(tv[i])
        if d==0 or not np.isfinite(avv[i]): continue
        cutoff=times[i]-pd.Timedelta(hours=12)
        if d==1:
            kh=np.searchsorted(hpt,cutoff,side="right")
            if kh<1: continue
            bi=int(hp4[kh-1]); kl=np.searchsorted(lp4,bi,side="left")
            if kl<1: continue
            ai=int(lp4[kl-1]); A=lows4[ai]; B=highs4[bi]
            if B<=A: continue
            depth=(B-close[i])/(B-A)
            counter=close[i-1] < close[i-4]
            pull=(B-low[i]) >= pull_atr*avv[i]
            trigger=close[i] > high[i-1]
        else:
            kl=np.searchsorted(lpt,cutoff,side="right")
            if kl<1: continue
            bi=int(lp4[kl-1]); kh=np.searchsorted(hp4,bi,side="left")
            if kh<1: continue
            ai=int(hp4[kh-1]); A=highs4[ai]; B=lows4[bi]
            if A<=B: continue
            depth=(close[i]-B)/(A-B)
            counter=close[i-1] > close[i-4]
            pull=(high[i]-B) >= pull_atr*avv[i]
            trigger=close[i] < low[i-1]
        if fib_lo <= depth <= fib_hi and counter and pull and trigger:
            candidate[i]=True; direction[i]=d

    lp,hp=pivots_l(b,h1_pivot_l)
    mhv=mh.to_numpy(float)
    hidden=np.zeros(len(b),bool)
    for i in np.flatnonzero(candidate):
        d=int(direction[i]); piv=lp if d==1 else hp
        # Pivot needs L right-hand bars to be confirmed.
        k=np.searchsorted(piv,i-h1_pivot_l,side="right")
        if k<2: continue
        p1,p2=int(piv[k-2]),int(piv[k-1])
        if d==1:
            ps=low[p2] > low[p1]
            hd=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2] < mhv[p1]
        else:
            ps=high[p2] < high[p1]
            hd=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2] > mhv[p1]
        hidden[i]=ps and hd
    return candidate & hidden,direction,av

def run(symbol,b,outdir):
    bands=[("broad",0.45,0.65),("base",0.50,0.618),("narrow",0.52,0.62)]
    pulls=[0.50,0.75,1.00]
    pivots=[1,2,3]
    rows=[]
    for band_name,flo,fhi in bands:
        for pull in pulls:
            for L in pivots:
                mask,direction,av=build_t2(b,flo,fhi,pull,L)
                tr,skip=eval_trades_rr_nonoverlap(b,mask,direction,av,48,2.0)
                s=summarize(tr)
                tc=tr.copy()
                if not tc.empty: tc["R"]=tc["R"]-0.05
                sc=summarize(tc)
                row={
                    "symbol":symbol,"band":band_name,"fib_lo":flo,"fib_hi":fhi,
                    "pull_atr":pull,"h1_pivot_l":L,"trades":s["trades"],"skipped":skip,
                    "pf":s["pf"],"avg_r":s["avg_r"],"max_dd_r":s["max_dd_r"],
                    "pf_cost_005R":sc["pf"],"avg_r_cost_005R":sc["avg_r"]
                }
                rows.append(row)
                print(
                    f"SENS {symbol} band={band_name:6s}[{flo:.3f},{fhi:.3f}] "
                    f"pull={pull:.2f} L={L} n={s['trades']:3d} PF={s['pf']:7.3f} "
                    f"avgR={s['avg_r']:+.4f} DD={s['max_dd_r']:.1f} "
                    f"PF@0.05R={sc['pf']:7.3f}",flush=True
                )
    df=pd.DataFrame(rows)
    df.to_csv(outdir/f"{symbol}_T2_MACD_sensitivity.csv",index=False)
    valid=df[df["trades"]>=15].copy()
    print("\n=== ROBUSTNESS SUMMARY ===")
    print(f"cells={len(df)} valid_n>=15={len(valid)}")
    if len(valid):
        print(f"PF median={valid['pf'].median():.3f} min={valid['pf'].min():.3f} max={valid['pf'].max():.3f}")
        print(f"PF>1 cells={(valid['pf']>1).sum()}/{len(valid)}")
        print(f"PF@0.05R median={valid['pf_cost_005R'].median():.3f} min={valid['pf_cost_005R'].min():.3f}")
        print(f"PF@0.05R>1 cells={(valid['pf_cost_005R']>1).sum()}/{len(valid)}")
        base=valid[(valid.band=="base")&(valid.pull_atr==0.75)&(valid.h1_pivot_l==2)]
        print("BASE:")
        print(base.to_string(index=False))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbol",choices=["USDJPY","XAUUSD"],required=True)
    ap.add_argument("--workdir",default="_research_work")
    ap.add_argument("--outdir",default="research/results")
    args=ap.parse_args()
    work=Path(args.workdir); work.mkdir(parents=True,exist_ok=True)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    z=work/f"{args.symbol}_bid.zip"
    if not z.exists(): z=download_zip(args.symbol,work)
    m1=read_m1(z); h1=resample(m1,"1h"); del m1
    run(args.symbol,h1,out)

if __name__=="__main__":
    main()
