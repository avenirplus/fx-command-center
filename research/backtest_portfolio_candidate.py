#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import download_zip, read_m1, resample, eval_trades_rr_nonoverlap, summarize
from backtest_sensitivity import build_t2
from backtest_t3_robustness import build as build_t3

def load_symbol(symbol,work):
    z=work/f"{symbol}_bid.zip"
    if not z.exists(): z=download_zip(symbol,work)
    m1=read_m1(z)
    h1=resample(m1,"1h")
    del m1
    return h1

def trades_for(symbol,b,mode):
    if mode=="T2":
        mask,direction,av=build_t2(b,0.50,0.618,0.75,2)
    else:
        mask,direction,av=build_t3(b,0.50,0.618,2)
    tr,skip=eval_trades_rr_nonoverlap(b,mask,direction,av,48,2.0)
    tr["symbol"]=symbol
    tr["mode"]=mode
    return tr,skip

def portfolio_stats(tr):
    z=tr.sort_values("entry_time").reset_index(drop=True)
    r=z["R"].to_numpy(float)
    cum=np.cumsum(r)
    peak=np.maximum.accumulate(np.r_[0.0,cum])
    dd=np.r_[0.0,cum]-peak
    years=pd.to_datetime(z["entry_time"]).dt.year
    annual=z.assign(year=years).groupby("year").agg(trades=("R","size"),R=("R","sum"))
    return z,dict(trades=len(z),sum_r=float(r.sum()),avg_r=float(r.mean()) if len(r) else np.nan,max_dd_r=float(-dd.min()) if len(r) else np.nan,years=int(years.nunique()) if len(r) else 0),annual

def overlap_count(a,b,hours=48):
    ta=pd.to_datetime(a["entry_time"]).sort_values().to_numpy()
    tb=pd.to_datetime(b["entry_time"]).sort_values().to_numpy()
    count=0
    for x in ta:
        if np.any(np.abs((tb-x).astype("timedelta64[h]").astype(int))<=hours):
            count+=1
    return count

def main():
    work=Path("_research_work"); work.mkdir(exist_ok=True)
    out=Path("research/results"); out.mkdir(parents=True,exist_ok=True)
    bars={s:load_symbol(s,work) for s in ["USDJPY","XAUUSD"]}

    all_rows=[]
    for mode in ["T2","T3"]:
        by={}
        print(f"\n=== PORTFOLIO {mode} ===")
        for s in ["USDJPY","XAUUSD"]:
            tr,skip=trades_for(s,bars[s],mode)
            by[s]=tr
            sm=summarize(tr)
            print(f"PORT {mode} {s} n={sm['trades']:3d} PF={sm['pf']:.3f} avgR={sm['avg_r']:+.4f} DD={sm['max_dd_r']:.1f} skip={skip}")
            all_rows.append(tr)
        combo=pd.concat([by["USDJPY"],by["XAUUSD"]],ignore_index=True)
        z,ps,annual=portfolio_stats(combo)
        print(f"PORTCOMBO {mode} n={ps['trades']} sumR={ps['sum_r']:+.2f} avgR={ps['avg_r']:+.4f} maxDD={ps['max_dd_r']:.2f} active_years={ps['years']}")
        print(f"PORTFREQ {mode} trades_per_calendar_year={ps['trades']/12.75:.2f}")
        print(f"PORTOVERLAP {mode} USDJPY entries within 48h of XAUUSD entry={overlap_count(by['USDJPY'],by['XAUUSD'],48)}/{len(by['USDJPY'])}")
        print("PORTANNUAL "+mode)
        for y,row in annual.iterrows():
            print(f"{int(y)} n={int(row['trades']):2d} R={row['R']:+.2f}")
        for risk_pct in [0.5,1.0,2.0,3.0,5.0]:
            f=risk_pct/100.0
            eq=1.0; peak=1.0; maxdd=0.0
            for rr in z["R"].to_numpy(float):
                eq *= (1.0 + f*rr)
                peak=max(peak,eq)
                maxdd=max(maxdd,1.0-eq/peak)
            cagr=eq**(1/11.75)-1 if eq>0 else -1.0
            print(f"PORTRISK {mode} risk={risk_pct:.1f}% final_x={eq:.4f} total_return={(eq-1)*100:+.2f}% maxDD={maxdd*100:.2f}% CAGR={cagr*100:.2f}%")
        z.to_csv(out/f"portfolio_{mode}_trades.csv",index=False)
        annual.to_csv(out/f"portfolio_{mode}_annual.csv")

if __name__=="__main__":
    main()
