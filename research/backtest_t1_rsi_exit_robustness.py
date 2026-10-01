#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from backtest_hidden_divergence import download_zip, read_m1, resample, eval_trades_rr_nonoverlap, summarize
from backtest_t1_rsi_robustness import build_t1_rsi

TPS=[1.0,1.5,2.0]

def run(symbol,b,outdir):
    mask,direction,av=build_t1_rsi(b,0.50,0.618,2)
    rows=[]
    for tp in TPS:
        tr,skip=eval_trades_rr_nonoverlap(b,mask,direction,av,48,tp)
        s=summarize(tr)
        tc=tr.copy()
        if not tc.empty: tc["R"]=tc["R"]-0.05
        sc=summarize(tc)
        rows.append({"symbol":symbol,"tp_r":tp,"trades":s["trades"],"pf":s["pf"],"avg_r":s["avg_r"],"max_dd_r":s["max_dd_r"],"pf_cost_005R":sc["pf"]})
        print(f"EXIT {symbol} TP={tp:.2f} n={s['trades']:3d} PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} DD={s['max_dd_r']:.1f} PF@0.05R={sc['pf']:7.3f}")
        if not tr.empty:
            yrs=pd.to_datetime(tr["signal_time"]).dt.year
            eras=np.select([yrs<=2018,yrs<=2022],["2015-2018","2019-2022"],default="2023-2026")
            z=tr.copy(); z["era"]=eras
            for era,g in z.groupby("era",sort=False):
                es=summarize(g)
                print(f"EXITERA {symbol} TP={tp:.2f} {era} n={es['trades']:3d} PF={es['pf']:7.3f} avgR={es['avg_r']:+.4f}")
    pd.DataFrame(rows).to_csv(outdir/f"{symbol}_T1_RSI_exit_robustness.csv",index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--symbol",choices=["USDJPY","XAUUSD"],required=True)
    ap.add_argument("--workdir",default="_research_work"); ap.add_argument("--outdir",default="research/results")
    args=ap.parse_args(); work=Path(args.workdir); work.mkdir(parents=True,exist_ok=True); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    z=work/f"{args.symbol}_bid.zip"
    if not z.exists(): z=download_zip(args.symbol,work)
    m1=read_m1(z); h1=resample(m1,"1h"); del m1
    run(args.symbol,h1,out)

if __name__=="__main__": main()
