#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import (
    download_zip, read_m1, resample, atr_wilder, h4_trend_from,
    eval_trades_rr_nonoverlap, summarize
)
from backtest_trend_modes import structure_trend

def pullback_then_breakout(b, trend):
    atr=atr_wilder(b,14)
    o=b["Open"].to_numpy(float); h=b["High"].to_numpy(float)
    l=b["Low"].to_numpy(float); c=b["Close"].to_numpy(float)
    av=atr.to_numpy(float)
    tv=np.asarray(trend,dtype=np.int8)
    n=len(b)
    sig=np.zeros(n,dtype=bool)
    direction=np.zeros(n,dtype=np.int8)

    for i in range(6,n):
        d=int(tv[i])
        if d==0 or not np.isfinite(av[i]) or av[i]<=0:
            continue

        # Frozen E3 reconstruction:
        # - pullback/retracement must have occurred within the prior 5 completed bars
        # - countertrend excursion >= 0.5 ATR14
        # - signal bar close breaks the prior 3 completed-bar high/low
        s=i-5; e=i
        wh=h[s:e]; wl=l[s:e]
        if d==1:
            hi_pos=int(np.argmax(wh))
            lo_pos=int(np.argmin(wl))
            # For a long pullback, local high must occur before the local low.
            pull=(hi_pos < lo_pos) and ((wh[hi_pos]-wl[lo_pos]) >= 0.5*av[i])
            level=np.max(h[i-3:i])
            trigger=c[i] > level
        else:
            lo_pos=int(np.argmin(wl))
            hi_pos=int(np.argmax(wh))
            # For a short pullback, local low must occur before the local high.
            pull=(lo_pos < hi_pos) and ((wh[hi_pos]-wl[lo_pos]) >= 0.5*av[i])
            level=np.min(l[i-3:i])
            trigger=c[i] < level

        if pull and trigger:
            sig[i]=True
            direction[i]=d

    # S2 = 1.5 ATR => make 1R equal to 1.5 ATR.
    risk_atr=atr*1.5
    return sig,direction,risk_atr

def run(symbol,b,outdir):
    t1=h4_trend_from(b).to_numpy(np.int8)
    t2=structure_trend(b)
    t3=np.where((t1==t2)&(t1!=0),t1,0).astype(np.int8)

    rows=[]
    for mode,tv in [("T1_EMA",t1),("T2_HHHL",t2),("T3_COMPOUND",t3)]:
        sig,direction,risk=pullback_then_breakout(b,tv)
        tr,skip=eval_trades_rr_nonoverlap(b,sig,direction,risk,48,2.0)
        s=summarize(tr)
        print(f"E3 {symbol} {mode:11s} n={s['trades']:4d} PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} DD={s['max_dd_r']:.1f} skip={skip}")
        row={"symbol":symbol,"mode":mode,"trades":s["trades"],"pf":s["pf"],"avg_r":s["avg_r"],"max_dd_r":s["max_dd_r"],"skipped":skip}
        for cost in [0.05,0.10]:
            tc=tr.copy()
            if not tc.empty: tc["R"]=tc["R"]-cost
            cs=summarize(tc)
            print(f"E3COST {symbol} {mode:11s} cost={cost:.2f}R PF={cs['pf']:7.3f} avgR={cs['avg_r']:+.4f}")
            row[f"pf_cost_{cost:.2f}R"]=cs["pf"]
            row[f"avg_cost_{cost:.2f}R"]=cs["avg_r"]

        if not tr.empty:
            years=pd.to_datetime(tr["signal_time"]).dt.year
            eras=np.select([years<=2018,years<=2022],["2015-2018","2019-2022"],default="2023-2026")
            z=tr.copy(); z["era"]=eras
            for era,g in z.groupby("era",sort=False):
                es=summarize(g)
                print(f"E3ERA {symbol} {mode:11s} {era} n={es['trades']:4d} PF={es['pf']:7.3f} avgR={es['avg_r']:+.4f}")
            for d,g in tr.groupby("direction"):
                ds=summarize(g); side="LONG" if d==1 else "SHORT"
                print(f"E3SIDE {symbol} {mode:11s} {side:5s} n={ds['trades']:4d} PF={ds['pf']:7.3f} avgR={ds['avg_r']:+.4f}")
            tr.to_csv(outdir/f"{symbol}_E3_{mode}_trades.csv",index=False)
        rows.append(row)

    pd.DataFrame(rows).to_csv(outdir/f"{symbol}_E3_summary.csv",index=False)

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
    m1=read_m1(z)
    h1=resample(m1,"1h")
    del m1
    run(args.symbol,h1,out)

if __name__=="__main__":
    main()
