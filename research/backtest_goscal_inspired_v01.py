#!/usr/bin/env python3
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

def evaluate(name,mask,direction,b,atr):
    entry=b["Open"].shift(-1)
    av=atr.replace(0,np.nan)
    rows=[]
    print(f"\nGSI SIG {name} raw_n={int(mask.sum())}")
    for h in HORIZONS:
        exitc=b["Close"].shift(-h)
        rr=direction*(exitc-entry)/av
        z=rr[mask].replace([np.inf,-np.inf],np.nan).dropna()
        if z.empty:
            continue
        q=z.quantile([0.01,0.5,0.99])
        print(f"GSI FWD {name:30s} h={h:2d} n={len(z):6d} win={100*(z>0).mean():6.2f}% meanATR={z.mean():+.5f} medATR={z.median():+.5f} q01={q.loc[0.01]:+.3f} q99={q.loc[0.99]:+.3f}")
        rows.append({"signal":name,"h_min":h,"n":len(z),"win_pct":100*(z>0).mean(),"mean_atr":z.mean(),"median_atr":z.median(),"q01":q.loc[0.01],"q99":q.loc[0.99]})

        if h==20:
            zz=pd.DataFrame({"r":rr,"mask":mask,"year":b.index.year})
            zz=zz[zz["mask"] & zz["r"].notna()].copy()
            eras=np.select([zz.year<=2018,zz.year<=2022],["2015-2018","2019-2022"],default="2023-2026")
            zz["era"]=eras
            for era,g in zz.groupby("era",sort=False):
                print(f"GSI ERA20 {name:27s} {era} n={len(g):6d} win={100*(g.r>0).mean():6.2f}% meanATR={g.r.mean():+.5f} medATR={g.r.median():+.5f}")
            hrs=b.index.hour
            sess=np.select([hrs<6,hrs<12,hrs<18],["EET00-05","EET06-11","EET12-17"],default="EET18-23")
            ss=pd.DataFrame({"r":rr,"mask":mask,"session":sess})
            ss=ss[ss["mask"] & ss["r"].notna()]
            for sn,g in ss.groupby("session",sort=False):
                print(f"GSI SES20 {name:27s} {sn} n={len(g):6d} win={100*(g.r>0).mean():6.2f}% meanATR={g.r.mean():+.5f} medATR={g.r.median():+.5f}")
    return rows

def main():
    Path("research/results").mkdir(parents=True,exist_ok=True)
    work=Path("_research_work"); work.mkdir(exist_ok=True)
    z=work/"XAUUSD_bid.zip"
    if not z.exists():
        z=download_zip("XAUUSD",work)
    b=read_m1(z)
    atr=atr_wilder(b,14)

    pdh,pdl=previous_day_levels(b)
    k,d=stochastic_533(b)
    e6=ema(b["Close"],6); e13=ema(b["Close"],13)
    ten,kij,sa,sb=ichimoku(b)
    cloud_top=pd.concat([sa,sb],axis=1).max(axis=1)
    cloud_bot=pd.concat([sa,sb],axis=1).min(axis=1)

    st_up=cross_up(k,d); st_dn=cross_dn(k,d)
    ema_long=e6>e13; ema_short=e6<e13
    cloud_long=b["Close"]>cloud_top; cloud_short=b["Close"]<cloud_bot

    # A: false-break reclaim / rejection
    a_long=(b["Low"]<=pdl)&(b["Close"]>pdl)
    a_short=(b["High"]>=pdh)&(b["Close"]<pdh)

    # B: close breakout
    b_long=(b["Close"]>pdh)&(b["Close"].shift(1)<=pdh.shift(1))
    b_short=(b["Close"]<pdl)&(b["Close"].shift(1)>=pdl.shift(1))

    families={"RECLAIM":(a_long,a_short),"BREAKOUT":(b_long,b_short)}
    out=[]
    for fam,(lo,sh) in families.items():
        variants={
            "LEVEL_ONLY":(lo,sh),
            "STOCH_CROSS":(lo&st_up,sh&st_dn),
            "EMA_ALIGN":(lo&ema_long,sh&ema_short),
            "EMA_AND_CLOUD":(lo&ema_long&cloud_long,sh&ema_short&cloud_short),
        }
        for v,(vl,vs) in variants.items():
            mask=(vl|vs).fillna(False)
            direction=pd.Series(np.where(vl,1,np.where(vs,-1,0)),index=b.index,dtype=float)
            out.extend(evaluate(f"{fam}_{v}",mask,direction,b,atr))

    pd.DataFrame(out).to_csv("research/results/XAUUSD_GOSCAL_inspired_v01_forward.csv",index=False)

if __name__=="__main__":
    main()
