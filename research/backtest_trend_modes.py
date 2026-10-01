#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import (
    OHLC, read_m1, resample, download_zip, pivot_arrays, atr_wilder,
    macd_hist, cci, h4_trend_from, eval_trades_rr_nonoverlap, summarize
)

def structure_trend(b):
    h4=b.resample("4h",label="left",closed="left").agg(OHLC).dropna()
    lp,hp=pivot_arrays(h4)
    ht=h4.index
    lows=h4["Low"].to_numpy(float); highs=h4["High"].to_numpy(float)
    lt=ht[lp]; hgt=ht[hp]
    out=np.zeros(len(b),dtype=np.int8)
    for i,t in enumerate(b.index):
        cutoff=t-pd.Timedelta(hours=12)
        kl=np.searchsorted(lt,cutoff,side="right")
        kh=np.searchsorted(hgt,cutoff,side="right")
        if kl<2 or kh<2:
            continue
        l1,l2=lows[int(lp[kl-2])],lows[int(lp[kl-1])]
        h1,h2=highs[int(hp[kh-2])],highs[int(hp[kh-1])]
        if h2>h1 and l2>l1:
            out[i]=1
        elif h2<h1 and l2<l1:
            out[i]=-1
    return out

def masks_for_trend(b,tv):
    av=atr_wilder(b,14)
    mh=macd_hist(b["Close"])
    cv=cci(b,20)
    h4=b.resample("4h",label="left",closed="left").agg(OHLC).dropna()
    lp4,hp4=pivot_arrays(h4)
    ht=h4.index; lows4=h4["Low"].to_numpy(float); highs4=h4["High"].to_numpy(float)
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
            ok=0.50<=depth<=0.618
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
            ok=0.50<=depth<=0.618
            counter=close[i-1]>close[i-4]
            pull=(high[i]-B)>=0.75*avv[i]
            trigger=close[i]<low[i-1]
        if ok and counter and pull and trigger:
            candidate[i]=True; direction[i]=d

    lp,hp=pivot_arrays(b)
    mhv=mh.to_numpy(float); ccv=cv.to_numpy(float)
    ps=np.zeros(len(b),bool); mac=np.zeros(len(b),bool); ccih=np.zeros(len(b),bool)
    for i in np.flatnonzero(candidate):
        d=int(direction[i]); piv=lp if d==1 else hp
        k=np.searchsorted(piv,i-2,side="right")
        if k<2: continue
        p1,p2=int(piv[k-2]),int(piv[k-1])
        if d==1:
            okps=low[p2]>low[p1]
            mhid=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2]<mhv[p1]
            chid=np.isfinite(ccv[p1]) and np.isfinite(ccv[p2]) and ccv[p2]<ccv[p1]
        else:
            okps=high[p2]<high[p1]
            mhid=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2]>mhv[p1]
            chid=np.isfinite(ccv[p1]) and np.isfinite(ccv[p2]) and ccv[p2]>ccv[p1]
        ps[i]=okps; mac[i]=okps and mhid; ccih[i]=okps and chid
    return {
        "candidate":candidate,
        "price_HL_LH":candidate&ps,
        "MACD_hidden":candidate&mac,
        "CCI_hidden":candidate&ccih,
    },direction,av

def run(symbol,b,outdir):
    t1=h4_trend_from(b).to_numpy(np.int8)
    t2=structure_trend(b)
    t3=np.where((t1==t2)&(t1!=0),t1,0).astype(np.int8)
    rows=[]
    for mode,tv in [("T1_EMA",t1),("T2_HHHL",t2),("T3_COMPOUND",t3)]:
        masks,direction,av=masks_for_trend(b,tv)
        print(f"\n=== TRENDMODE {symbol} {mode} active={np.count_nonzero(tv):,} ===")
        for filt in ["candidate","price_HL_LH","MACD_hidden","CCI_hidden"]:
            for tp in [1.25,2.00]:
                tr,skip=eval_trades_rr_nonoverlap(b,masks[filt],direction,av,48,tp)
                s=summarize(tr)
                rows.append({"symbol":symbol,"trend_mode":mode,"filter":filt,"tp_r":tp,"skipped":skip,**s})
                print(f"TRENDGRID {mode:11s} {filt:12s} TP={tp:.2f} n={s['trades']:3d} skip={skip:2d} PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} DD={s['max_dd_r']:.1f}")
                if not tr.empty:
                    years=pd.to_datetime(tr["signal_time"]).dt.year
                    eras=np.select([years<=2018,years<=2022],["2015-2018","2019-2022"],default="2023-2026")
                    z=tr.copy(); z["era"]=eras
                    for era,g in z.groupby("era"):
                        es=summarize(g)
                        print(f"TRENDERA {mode:11s} {filt:12s} TP={tp:.2f} {era} n={es['trades']:3d} PF={es['pf']:7.3f} avgR={es['avg_r']:+.4f}")
                    for d,g in tr.groupby("direction"):
                        es=summarize(g); side="LONG" if d==1 else "SHORT"
                        print(f"TRENDSIDE {mode:11s} {filt:12s} TP={tp:.2f} {side:5s} n={es['trades']:3d} PF={es['pf']:7.3f} avgR={es['avg_r']:+.4f}")
                for cost in [0.05,0.10]:
                    tc=tr.copy()
                    if not tc.empty: tc["R"]=tc["R"]-cost
                    cs=summarize(tc)
                    print(f"TRENDCOST {mode:11s} {filt:12s} TP={tp:.2f} cost={cost:.2f}R PF={cs['pf']:7.3f} avgR={cs['avg_r']:+.4f}")
    pd.DataFrame(rows).to_csv(outdir/f"{symbol}_F1_trend_modes.csv",index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbol",choices=["USDJPY","XAUUSD"],required=True)
    ap.add_argument("--workdir",default="_research_work")
    ap.add_argument("--outdir",default="research/results")
    args=ap.parse_args()
    work=Path(args.workdir); work.mkdir(parents=True,exist_ok=True)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    z=work/f"{args.symbol}_bid.zip"
    if not z.exists():
        z=download_zip(args.symbol,work)
    m1=read_m1(z); h1=resample(m1,"1h"); del m1
    run(args.symbol,h1,out)

if __name__=="__main__":
    main()
