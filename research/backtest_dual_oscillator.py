#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

from backtest_hidden_divergence import download_zip, read_m1, resample, macd_hist, eval_trades_rr_nonoverlap, summarize
from backtest_oscillator_compare import candidate_t3, hidden_mask, rsi_wilder

def run(symbol,b,outdir):
    cand,direction,av=candidate_t3(b)
    mh,_=hidden_mask(b,cand,direction,macd_hist(b["Close"]))
    rh,_=hidden_mask(b,cand,direction,rsi_wilder(b["Close"],14))
    masks={
        "MACD":mh,
        "RSI":rh,
        "BOTH":mh & rh,
        "EITHER":mh | rh,
    }
    rows=[]
    for name,mask in masks.items():
        tr,skip=eval_trades_rr_nonoverlap(b,cand&mask,direction,av,48,2.0)
        s=summarize(tr)
        tc=tr.copy()
        if not tc.empty: tc["R"]=tc["R"]-0.05
        sc=summarize(tc)
        rows.append({"symbol":symbol,"filter":name,"trades":s["trades"],"skipped":skip,"pf":s["pf"],"avg_r":s["avg_r"],"max_dd_r":s["max_dd_r"],"pf_cost_005R":sc["pf"]})
        print(f"DUAL {symbol} {name:6s} n={s['trades']:3d} PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} DD={s['max_dd_r']:.1f} PF@0.05R={sc['pf']:7.3f}")
        if not tr.empty:
            yrs=pd.to_datetime(tr["signal_time"]).dt.year
            eras=pd.Series(index=tr.index,dtype=object)
            eras[yrs<=2018]="2015-2018"; eras[(yrs>=2019)&(yrs<=2022)]="2019-2022"; eras[yrs>=2023]="2023-2026"
            z=tr.copy(); z["era"]=eras.values
            for era,g in z.groupby("era"):
                es=summarize(g)
                print(f"DUALERA {symbol} {name:6s} {era} n={es['trades']:3d} PF={es['pf']:7.3f} avgR={es['avg_r']:+.4f}")
    pd.DataFrame(rows).to_csv(outdir/f"{symbol}_T3_MACD_RSI_dual.csv",index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--symbol",choices=["USDJPY","XAUUSD"],required=True)
    ap.add_argument("--workdir",default="_research_work"); ap.add_argument("--outdir",default="research/results")
    args=ap.parse_args(); work=Path(args.workdir); work.mkdir(parents=True,exist_ok=True); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    z=work/f"{args.symbol}_bid.zip"
    if not z.exists(): z=download_zip(args.symbol,work)
    m1=read_m1(z); h1=resample(m1,"1h"); del m1
    run(args.symbol,h1,out)

if __name__=="__main__": main()
