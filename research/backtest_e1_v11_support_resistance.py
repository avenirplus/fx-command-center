#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import (
    download_zip, read_m1, resample, pivot_arrays, atr_wilder,
    eval_trades_rr_nonoverlap, summarize
)
from backtest_e3_all_timeframes import ema_trend_htf, structure_trend_htf

def e1_v11_signal(b,tv):
    atr=atr_wilder(b,14)
    h=b["High"].to_numpy(float)
    l=b["Low"].to_numpy(float)
    c=b["Close"].to_numpy(float)
    av=atr.to_numpy(float)
    trend=np.asarray(tv,dtype=np.int8)

    lp,hp=pivot_arrays(b)
    sig=np.zeros(len(b),dtype=bool)
    direction=np.zeros(len(b),dtype=np.int8)

    for i in range(3,len(b)):
        d=int(trend[i])
        if d==0 or not np.isfinite(av[i]) or av[i]<=0 or not np.isfinite(av[i-1]) or av[i-1]<=0:
            continue

        # L=2 pivot is usable only after its two right-side bars have closed.
        cutoff=i-2

        if d==1:
            k=np.searchsorted(lp,cutoff,side="right")
            if k<1:
                continue
            p=int(lp[k-1])
            support=l[p]
            zone=0.25*av[i-1]
            touched=(l[i-1] >= support-zone) and (l[i-1] <= support+zone)
            reversal=c[i] > h[i-1]
            if touched and reversal:
                sig[i]=True; direction[i]=1
        else:
            k=np.searchsorted(hp,cutoff,side="right")
            if k<1:
                continue
            p=int(hp[k-1])
            resistance=h[p]
            zone=0.25*av[i-1]
            touched=(h[i-1] >= resistance-zone) and (h[i-1] <= resistance+zone)
            reversal=c[i] < l[i-1]
            if touched and reversal:
                sig[i]=True; direction[i]=-1

    return sig,direction,atr*1.5

def one_cell(symbol,label,b,htf_rule,confirm_delay,max_hold):
    t1=ema_trend_htf(b,htf_rule)
    t2=structure_trend_htf(b,htf_rule,confirm_delay)
    t3=np.where((t1==t2)&(t1!=0),t1,0).astype(np.int8)

    rows=[]
    for mode,tv in [("T1_EMA",t1),("T2_HHHL",t2),("T3_COMPOUND",t3)]:
        sig,direction,risk=e1_v11_signal(b,tv)
        tr,skip=eval_trades_rr_nonoverlap(b,sig,direction,risk,max_hold,2.0)
        s=summarize(tr)
        print(f"E1V11 {symbol} {label:8s} {mode:11s} n={s['trades']:5d} PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} DD={s['max_dd_r']:.1f} skip={skip}")
        row={"symbol":symbol,"cell":label,"trend_mode":mode,"trades":s["trades"],"pf":s["pf"],"avg_r":s["avg_r"],"max_dd_r":s["max_dd_r"],"skipped":skip}

        for cost in [0.05,0.10]:
            tc=tr.copy()
            if not tc.empty:
                tc["R"]=tc["R"]-cost
            cs=summarize(tc)
            print(f"E1V11COST {symbol} {label:8s} {mode:11s} cost={cost:.2f}R PF={cs['pf']:7.3f} avgR={cs['avg_r']:+.4f}")
            row[f"pf_cost_{cost:.2f}"]=cs["pf"]

        if not tr.empty:
            years=pd.to_datetime(tr["signal_time"]).dt.year
            eras=np.select([years<=2018,years<=2022],["2015-2018","2019-2022"],default="2023-2026")
            z=tr.copy(); z["era"]=eras
            for era,g in z.groupby("era",sort=False):
                es=summarize(g)
                print(f"E1V11ERA {symbol} {label:8s} {mode:11s} {era} n={es['trades']:4d} PF={es['pf']:7.3f} avgR={es['avg_r']:+.4f}")
            for d,g in tr.groupby("direction"):
                ds=summarize(g); side="LONG" if d==1 else "SHORT"
                print(f"E1V11SIDE {symbol} {label:8s} {mode:11s} {side:5s} n={ds['trades']:4d} PF={ds['pf']:7.3f}")
            tr.to_csv(Path("research/results")/f"{symbol}_E1V11_{label}_{mode}_trades.csv",index=False)
        rows.append(row)
    return rows

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

    m1=read_m1(z)
    h1=resample(m1,"1h")
    m15=resample(m1,"15min")
    del m1

    rows=[]
    rows += one_cell(args.symbol,"H4_H1",h1,"4h",pd.Timedelta(hours=12),48)
    rows += one_cell(args.symbol,"H4_M15",m15,"4h",pd.Timedelta(hours=12),192)
    rows += one_cell(args.symbol,"D1_H1",h1,"1D",pd.Timedelta(days=3),48)
    pd.DataFrame(rows).to_csv(out/f"{args.symbol}_E1V11_all_timeframes.csv",index=False)

if __name__=="__main__":
    main()
