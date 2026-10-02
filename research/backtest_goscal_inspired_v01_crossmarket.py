#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import download_zip, read_m1, atr_wilder, ema
from backtest_goscal_components import stochastic_533, ichimoku, cross_up, cross_dn

HORIZONS=[5,10,20,30]

def previous_day_levels(b):
    day=pd.Series(b.index.normalize(),index=b.index)
    daily=b.groupby(day).agg({"High":"max","Low":"min"})
    daily["PDH"]=daily["High"].shift(1)
    daily["PDL"]=daily["Low"].shift(1)
    pdh=day.map(daily["PDH"])
    pdl=day.map(daily["PDL"])
    return pd.Series(pdh.to_numpy(float),index=b.index),pd.Series(pdl.to_numpy(float),index=b.index)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbol",choices=["USDJPY","XAUUSD"],required=True)
    args=ap.parse_args()
    Path("research/results").mkdir(parents=True,exist_ok=True)
    work=Path("_research_work"); work.mkdir(exist_ok=True)
    z=work/f"{args.symbol}_bid.zip"
    if not z.exists(): z=download_zip(args.symbol,work)
    b=read_m1(z)
    atr=atr_wilder(b,14).replace(0,np.nan)

    pdh,pdl=previous_day_levels(b)
    k,d=stochastic_533(b)
    e6=ema(b["Close"],6); e13=ema(b["Close"],13)
    _,_,sa,sb=ichimoku(b)
    cloud_top=pd.concat([sa,sb],axis=1).max(axis=1)
    cloud_bot=pd.concat([sa,sb],axis=1).min(axis=1)
    st_up=cross_up(k,d); st_dn=cross_dn(k,d)
    ema_long=e6>e13; ema_short=e6<e13
    cloud_long=b["Close"]>cloud_top; cloud_short=b["Close"]<cloud_bot

    lo=(b["Close"]>pdh)&(b["Close"].shift(1)<=pdh.shift(1))
    sh=(b["Close"]<pdl)&(b["Close"].shift(1)>=pdl.shift(1))
    variants={
      "BREAKOUT_LEVEL_ONLY":(lo,sh),
      "BREAKOUT_STOCH_CROSS":(lo&st_up,sh&st_dn),
      "BREAKOUT_EMA_ALIGN":(lo&ema_long,sh&ema_short),
      "BREAKOUT_EMA_AND_CLOUD":(lo&ema_long&cloud_long,sh&ema_short&cloud_short),
    }

    rows=[]
    for name,(vl,vs) in variants.items():
        mask=(vl|vs).fillna(False)
        direction=pd.Series(np.where(vl,1,np.where(vs,-1,0)),index=b.index,dtype=float)
        entry=b["Open"].shift(-1)
        for h in HORIZONS:
            rr=direction*(b["Close"].shift(-h)-entry)/atr
            z=rr[mask].replace([np.inf,-np.inf],np.nan).dropna()
            if z.empty: continue
            print(f"GSIXM {args.symbol} {name:30s} h={h:2d} n={len(z):6d} win={100*(z>0).mean():6.2f}% meanATR={z.mean():+.5f} medATR={z.median():+.5f}",flush=True)
            rows.append({"symbol":args.symbol,"signal":name,"h_min":h,"n":len(z),"win_pct":100*(z>0).mean(),"mean_atr":z.mean(),"median_atr":z.median()})
            if h==20:
                zdf=pd.DataFrame({"r":rr,"mask":mask,"year":b.index.year})
                zdf=zdf[zdf["mask"] & zdf["r"].notna()].copy()
                eras=np.select([zdf.year<=2018,zdf.year<=2022],["2015-2018","2019-2022"],default="2023-2026")
                zdf["era"]=eras
                for era,g in zdf.groupby("era",sort=False):
                    print(f"GSIXMERA {args.symbol} {name:27s} {era} n={len(g):6d} win={100*(g.r>0).mean():6.2f}% meanATR={g.r.mean():+.5f} medATR={g.r.median():+.5f}",flush=True)
                hrs=b.index.hour
                sess=np.select([hrs<6,hrs<12,hrs<18],["EET00-05","EET06-11","EET12-17"],default="EET18-23")
                sdf=pd.DataFrame({"r":rr,"mask":mask,"session":sess})
                sdf=sdf[sdf["mask"] & sdf["r"].notna()]
                for sn,g in sdf.groupby("session",sort=False):
                    print(f"GSIXMSES {args.symbol} {name:27s} {sn} n={len(g):6d} win={100*(g.r>0).mean():6.2f}% meanATR={g.r.mean():+.5f} medATR={g.r.median():+.5f}",flush=True)
    pd.DataFrame(rows).to_csv(f"research/results/{args.symbol}_GOSCAL_inspired_v01_crossmarket.csv",index=False)

if __name__=="__main__":
    main()
