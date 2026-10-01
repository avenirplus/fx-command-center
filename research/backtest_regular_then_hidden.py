#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import (
    download_zip, read_m1, resample, pivot_arrays, macd_hist,
    eval_trades_rr_nonoverlap, summarize
)
from backtest_oscillator_compare import candidate_t3, hidden_mask

def regular_before_hidden(b, hidden, direction, osc, window_hours):
    lp,hp=pivot_arrays(b)
    ov=osc.to_numpy(float)
    lows=b["Low"].to_numpy(float); highs=b["High"].to_numpy(float)
    times=b.index
    out=np.zeros(len(b),bool)

    for i in np.flatnonzero(hidden):
        d=int(direction[i])
        # Opposite pivot family:
        # long continuation: prior bearish regular divergence at swing highs
        # short continuation: prior bullish regular divergence at swing lows
        piv=hp if d==1 else lp
        k=np.searchsorted(piv,i-2,side="right")
        if k<2: continue
        p1,p2=int(piv[k-2]),int(piv[k-1])
        if times[i]-times[p2] > pd.Timedelta(hours=window_hours):
            continue
        if d==1:
            regular=(highs[p2]>highs[p1]) and np.isfinite(ov[p1]) and np.isfinite(ov[p2]) and (ov[p2]<ov[p1])
        else:
            regular=(lows[p2]<lows[p1]) and np.isfinite(ov[p1]) and np.isfinite(ov[p2]) and (ov[p2]>ov[p1])
        out[i]=regular
    return out

def run(symbol,b,outdir):
    cand,direction,av=candidate_t3(b)
    osc=macd_hist(b["Close"])
    hidden,price=hidden_mask(b,cand,direction,osc)

    rows=[]
    tr0,sk0=eval_trades_rr_nonoverlap(b,cand&hidden,direction,av,48,2.0)
    s0=summarize(tr0)
    print(f"SEQ {symbol} HIDDEN_ONLY n={s0['trades']:3d} PF={s0['pf']:7.3f} avgR={s0['avg_r']:+.4f} DD={s0['max_dd_r']:.1f}")
    rows.append({"symbol":symbol,"window_h":0,"filter":"hidden_only","skipped":sk0,**s0})

    for wh in [24,48,72]:
        seq=regular_before_hidden(b,cand&hidden,direction,osc,wh)
        tr,skip=eval_trades_rr_nonoverlap(b,cand&hidden&seq,direction,av,48,2.0)
        s=summarize(tr)
        tc=tr.copy()
        if not tc.empty: tc["R"]=tc["R"]-0.05
        sc=summarize(tc)
        rows.append({"symbol":symbol,"window_h":wh,"filter":"regular_then_hidden","skipped":skip,**s,"pf_cost_005R":sc["pf"],"avg_r_cost_005R":sc["avg_r"]})
        print(f"SEQ {symbol} REG->{wh:02d}H->HIDDEN n={s['trades']:3d} PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} DD={s['max_dd_r']:.1f} PF@0.05R={sc['pf']:7.3f}")
        if not tr.empty:
            years=pd.to_datetime(tr["signal_time"]).dt.year
            eras=np.select([years<=2018,years<=2022],["2015-2018","2019-2022"],default="2023-2026")
            z=tr.copy(); z["era"]=eras
            for era,g in z.groupby("era",sort=False):
                es=summarize(g)
                print(f"SEQERA {symbol} {wh:02d}H {era} n={es['trades']:3d} PF={es['pf']:7.3f} avgR={es['avg_r']:+.4f}")
            for d,g in tr.groupby("direction"):
                ds=summarize(g); side="LONG" if d==1 else "SHORT"
                print(f"SEQSIDE {symbol} {wh:02d}H {side:5s} n={ds['trades']:3d} PF={ds['pf']:7.3f} avgR={ds['avg_r']:+.4f}")

    pd.DataFrame(rows).to_csv(outdir/f"{symbol}_regular_then_hidden.csv",index=False)

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
