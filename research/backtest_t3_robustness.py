#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import (
    OHLC, download_zip, read_m1, resample, pivot_arrays, atr_wilder,
    macd_hist, h4_trend_from, eval_trades_rr_nonoverlap, summarize
)
from backtest_trend_modes import structure_trend
from backtest_sensitivity import pivots_l

BANDS=[
    ("wide_deep",0.48,0.64),
    ("base",0.50,0.618),
    ("narrow",0.52,0.60),
]

def build(b,flo,fhi,L):
    t1=h4_trend_from(b).to_numpy(np.int8)
    t2=structure_trend(b)
    tv=np.where((t1==t2)&(t1!=0),t1,0).astype(np.int8)
    av=atr_wilder(b,14); mh=macd_hist(b["Close"])
    h4=b.resample("4h",label="left",closed="left").agg(OHLC).dropna()
    lp4,hp4=pivot_arrays(h4)
    ht=h4.index; lows4=h4["Low"].to_numpy(float); highs4=h4["High"].to_numpy(float)
    lpt=ht[lp4]; hpt=ht[hp4]
    close=b["Close"].to_numpy(float); high=b["High"].to_numpy(float); low=b["Low"].to_numpy(float)
    avv=av.to_numpy(float); times=b.index
    cand=np.zeros(len(b),bool); direction=np.zeros(len(b),np.int8)
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
            counter=close[i-1]<close[i-4]
            pull=(B-low[i])>=0.75*avv[i]
            trigger=close[i]>high[i-1]
        else:
            kl=np.searchsorted(lpt,cutoff,side="right")
            if kl<1: continue
            bi=int(lp4[kl-1]); kh=np.searchsorted(hp4,bi,side="left")
            if kh<1: continue
            ai=int(hp4[kh-1]); A=highs4[ai]; B=lows4[bi]
            if A<=B: continue
            depth=(close[i]-B)/(A-B)
            counter=close[i-1]>close[i-4]
            pull=(high[i]-B)>=0.75*avv[i]
            trigger=close[i]<low[i-1]
        if flo<=depth<=fhi and counter and pull and trigger:
            cand[i]=True; direction[i]=d

    lp,hp=pivots_l(b,L); mhv=mh.to_numpy(float)
    hidden=np.zeros(len(b),bool)
    for i in np.flatnonzero(cand):
        d=int(direction[i]); piv=lp if d==1 else hp
        k=np.searchsorted(piv,i-L,side="right")
        if k<2: continue
        p1,p2=int(piv[k-2]),int(piv[k-1])
        if d==1:
            ps=low[p2]>low[p1]
            hd=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2]<mhv[p1]
        else:
            ps=high[p2]<high[p1]
            hd=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2]>mhv[p1]
        hidden[i]=ps and hd
    return cand&hidden,direction,av

def run(symbol,b,outdir):
    rows=[]
    for band,flo,fhi in BANDS:
        for L in [1,2,3]:
            mask,direction,av=build(b,flo,fhi,L)
            tr,skip=eval_trades_rr_nonoverlap(b,mask,direction,av,48,2.0)
            s=summarize(tr)
            tc=tr.copy()
            if not tc.empty: tc["R"]=tc["R"]-0.05
            sc=summarize(tc)
            rows.append({"symbol":symbol,"band":band,"fib_lo":flo,"fib_hi":fhi,"pivot_l":L,"skipped":skip,**s,"pf_cost_005R":sc["pf"],"avg_r_cost_005R":sc["avg_r"]})
            print(f"T3ROB {symbol} band={band:9s}[{flo:.3f},{fhi:.3f}] L={L} n={s['trades']:3d} PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} DD={s['max_dd_r']:.1f} PF@0.05R={sc['pf']:7.3f}")
            if not tr.empty:
                years=pd.to_datetime(tr["signal_time"]).dt.year
                eras=np.select([years<=2018,years<=2022],["2015-2018","2019-2022"],default="2023-2026")
                z=tr.copy(); z["era"]=eras
                for era,g in z.groupby("era",sort=False):
                    es=summarize(g)
                    print(f"T3ERA {symbol} band={band:9s} L={L} {era} n={es['trades']:3d} PF={es['pf']:7.3f} avgR={es['avg_r']:+.4f}")
    df=pd.DataFrame(rows)
    df.to_csv(outdir/f"{symbol}_T3_MACD_pivot_robustness.csv",index=False)
    print("\nT3 SUMMARY")
    print(f"cells={len(df)} PF>1={(df.pf>1).sum()}/{len(df)} median={df.pf.median():.3f} min={df.pf.min():.3f} max={df.pf.max():.3f}")
    print(f"cost0.05 PF>1={(df.pf_cost_005R>1).sum()}/{len(df)} median={df.pf_cost_005R.median():.3f}")

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
