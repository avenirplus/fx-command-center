#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import (
    OHLC, read_m1, resample, download_zip, pivot_arrays, atr_wilder,
    macd_hist, cci, h4_trend_from, eval_trades_rr_nonoverlap, summarize
)
from backtest_trend_modes import structure_trend

BANDS = [
    ("F1_382_618", 0.382, 0.618),
    ("DEEP_500_618", 0.500, 0.618),
    ("SHALLOW_382_500", 0.382, 0.500),
]

def masks_for_band(b, tv, lo_band, hi_band):
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
        if d==0 or not np.isfinite(avv[i]):
            continue
        cutoff=times[i]-pd.Timedelta(hours=12)
        if d==1:
            kh=np.searchsorted(hpt,cutoff,side="right")
            if kh<1: continue
            bi=int(hp4[kh-1])
            kl=np.searchsorted(lp4,bi,side="left")
            if kl<1: continue
            ai=int(lp4[kl-1]); A=lows4[ai]; B=highs4[bi]
            if B<=A: continue
            depth=(B-close[i])/(B-A)
            ok=lo_band<=depth<=hi_band
            counter=close[i-1]<close[i-4]
            pull=(B-low[i])>=0.75*avv[i]
            trigger=close[i]>high[i-1]
        else:
            kl=np.searchsorted(lpt,cutoff,side="right")
            if kl<1: continue
            bi=int(lp4[kl-1])
            kh=np.searchsorted(hp4,bi,side="left")
            if kh<1: continue
            ai=int(hp4[kh-1]); A=highs4[ai]; B=lows4[bi]
            if A<=B: continue
            depth=(close[i]-B)/(A-B)
            ok=lo_band<=depth<=hi_band
            counter=close[i-1]>close[i-4]
            pull=(high[i]-B)>=0.75*avv[i]
            trigger=close[i]<low[i-1]
        if ok and counter and pull and trigger:
            candidate[i]=True; direction[i]=d

    lp,hp=pivot_arrays(b)
    mhv=mh.to_numpy(float)
    ps=np.zeros(len(b),bool); mac=np.zeros(len(b),bool)
    lows=b["Low"].to_numpy(float); highs=b["High"].to_numpy(float)
    for i in np.flatnonzero(candidate):
        d=int(direction[i]); piv=lp if d==1 else hp
        k=np.searchsorted(piv,i-2,side="right")
        if k<2: continue
        p1,p2=int(piv[k-2]),int(piv[k-1])
        if d==1:
            struct=lows[p2]>lows[p1]
            hidden=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2]<mhv[p1]
        else:
            struct=highs[p2]<highs[p1]
            hidden=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2]>mhv[p1]
        ps[i]=struct
        mac[i]=struct and hidden
    return {"candidate":candidate,"price_HL_LH":candidate&ps,"MACD_hidden":candidate&mac},direction,av

def run(symbol,b,outdir):
    t1=h4_trend_from(b).to_numpy(np.int8)
    t2=structure_trend(b)
    t3=np.where((t1==t2)&(t1!=0),t1,0).astype(np.int8)
    rows=[]
    for band,bl,bh in BANDS:
        for mode,tv in [("T1_EMA",t1),("T2_HHHL",t2),("T3_COMPOUND",t3)]:
            masks,direction,av=masks_for_band(b,tv,bl,bh)
            for filt in ["candidate","price_HL_LH","MACD_hidden"]:
                tr,skip=eval_trades_rr_nonoverlap(b,masks[filt],direction,av,48,2.0)
                s=summarize(tr)
                rows.append({"symbol":symbol,"band":band,"trend_mode":mode,"filter":filt,"tp_r":2.0,"skipped":skip,**s})
                print(f"BANDGRID {symbol} {band:15s} {mode:11s} {filt:12s} n={s['trades']:3d} skip={skip:2d} PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} DD={s['max_dd_r']:.1f}")
                if not tr.empty and filt=="MACD_hidden":
                    years=pd.to_datetime(tr["signal_time"]).dt.year
                    eras=np.select([years<=2018,years<=2022],["2015-2018","2019-2022"],default="2023-2026")
                    z=tr.copy(); z["era"]=eras
                    for era,g in z.groupby("era",sort=False):
                        es=summarize(g)
                        print(f"BANDERA  {symbol} {band:15s} {mode:11s} {era} n={es['trades']:3d} PF={es['pf']:7.3f} avgR={es['avg_r']:+.4f}")
                    for d,g in tr.groupby("direction"):
                        ds=summarize(g)
                        side="LONG" if d==1 else "SHORT"
                        print(f"BANDSIDE {symbol} {band:15s} {mode:11s} {side:5s} n={ds['trades']:3d} PF={ds['pf']:7.3f} avgR={ds['avg_r']:+.4f}")
                    for cost in [0.05,0.10]:
                        tc=tr.copy(); tc["R"]=tc["R"]-cost
                        cs=summarize(tc)
                        print(f"BANDCOST {symbol} {band:15s} {mode:11s} cost={cost:.2f}R PF={cs['pf']:7.3f} avgR={cs['avg_r']:+.4f}")
    pd.DataFrame(rows).to_csv(outdir/f"{symbol}_F1_fib_bands.csv",index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbol",choices=["USDJPY","XAUUSD"],required=True)
    ap.add_argument("--workdir",default="_research_work")
    ap.add_argument("--outdir",default="research/results")
    args=ap.parse_args()
    work=Path(args.workdir); work.mkdir(parents=True,exist_ok=True)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    z=download_zip(args.symbol,work)
    m1=read_m1(z)
    h1=resample(m1,"1h")
    run(args.symbol,h1,out)

if __name__=="__main__":
    main()
