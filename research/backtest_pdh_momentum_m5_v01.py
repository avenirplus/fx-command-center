#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_USD_JPY, INSTRUMENT_FX_METALS_XAU_USD

from backtest_hidden_divergence import download_zip, read_m1, resample, atr_wilder, ema
from backtest_goscal_components import stochastic_533, ichimoku, cross_up, cross_dn

INST={"USDJPY":INSTRUMENT_FX_MAJORS_USD_JPY,"XAUUSD":INSTRUMENT_FX_METALS_XAU_USD}
START=pd.Timestamp("2015-01-01"); END=pd.Timestamp("2026-10-02")
H=[3,6,12,24]  # M5 bars = 15/30/60/120m

def norm_ask(z):
    z=z.copy()
    if not isinstance(z.index,pd.DatetimeIndex):
        if "timestamp" in z.columns:
            z["timestamp"]=pd.to_datetime(z["timestamp"],utc=True); z=z.set_index("timestamp")
        else:
            z.index=pd.to_datetime(z.index,utc=True)
    idx=z.index.tz_localize("UTC") if z.index.tz is None else z.index.tz_convert("UTC")
    z.index=idx.tz_convert("Europe/Helsinki").tz_localize(None)
    z.columns=[str(c).lower() for c in z.columns]
    z=z[[c for c in ["open","high","low","close"] if c in z.columns]]
    return z.loc[~z.index.duplicated()].sort_index()

def load_ask(symbol):
    z=dukascopy_python.fetch(
        instrument=INST[symbol],
        interval=dukascopy_python.INTERVAL_MIN_5,
        offer_side=dukascopy_python.OFFER_SIDE_ASK,
        start=START.to_pydatetime(),end=END.to_pydatetime(),max_retries=5)
    return norm_ask(z)

def prev_levels(b):
    day=pd.Series(b.index.normalize(),index=b.index)
    d=b.groupby(day).agg({"High":"max","Low":"min"})
    return (
      pd.Series(day.map(d["High"].shift(1)).to_numpy(float),index=b.index),
      pd.Series(day.map(d["Low"].shift(1)).to_numpy(float),index=b.index)
    )

def sigs(b):
    pdh,pdl=prev_levels(b)
    k,d=stochastic_533(b)
    e6=ema(b["Close"],6); e13=ema(b["Close"],13)
    _,_,sa,sb=ichimoku(b)
    top=pd.concat([sa,sb],axis=1).max(axis=1); bot=pd.concat([sa,sb],axis=1).min(axis=1)
    up=(b["Close"]>pdh)&(b["Close"].shift(1)<=pdh.shift(1))
    dn=(b["Close"]<pdl)&(b["Close"].shift(1)>=pdl.shift(1))
    return {
      "LEVEL_ONLY":(up,dn),
      "STOCH_CROSS":(up&cross_up(k,d),dn&cross_dn(k,d)),
      "EMA_ALIGN":(up&(e6>e13),dn&(e6<e13)),
      "EMA_AND_CLOUD":(up&(e6>e13)&(b["Close"]>top),dn&(e6<e13)&(b["Close"]<bot)),
    }

def run(symbol,bid,ask):
    atr=atr_wilder(bid,14)
    rows=[]
    for name,(lo,sh) in sigs(bid).items():
        ids=np.flatnonzero((lo|sh).fillna(False).to_numpy())
        bo=bid["Open"].to_numpy(float); bc=bid["Close"].to_numpy(float)
        ao=ask["open"].to_numpy(float); ac=ask["close"].to_numpy(float)
        av=atr.to_numpy(float); t=bid.index
        for h in H:
            rr=[]
            for i in ids:
                ei=i+1; xi=i+h
                if xi>=len(bid) or not np.isfinite(av[i]) or av[i]<=0: continue
                d=1 if bool(lo.iloc[i]) else -1
                vals=[bo[ei],bc[xi],ao[ei],ac[xi]]
                if not all(np.isfinite(x) for x in vals): continue
                entry=ao[ei] if d==1 else bo[ei]
                exitp=bc[xi] if d==1 else ac[xi]
                r=d*(exitp-entry)/av[i]
                spr=(ao[ei]-bo[ei])/av[i]
                rr.append((t[i],d,r,spr))
            z=pd.DataFrame(rr,columns=["signal_time","direction","R","spread_ATR"])
            if z.empty: continue
            mins=h*5
            print(f"PDHM5 {symbol} {name:13s} min={mins:3d} n={len(z):5d} win={100*(z.R>0).mean():6.2f}% meanATR={z.R.mean():+.5f} medATR={z.R.median():+.5f} spreadMed={z.spread_ATR.median():.4f}",flush=True)
            yrs=pd.to_datetime(z.signal_time).dt.year
            z["era"]=np.select([yrs<=2018,yrs<=2022],["2015-2018","2019-2022"],default="2023-2026")
            if mins==30:
                for era,g in z.groupby("era",sort=False):
                    print(f"PDHM5ERA {symbol} {name:13s} {era} n={len(g):5d} meanATR={g.R.mean():+.5f} medATR={g.R.median():+.5f}",flush=True)
            rows.append({"symbol":symbol,"signal":name,"minutes":mins,"n":len(z),"win_pct":100*(z.R>0).mean(),"mean_atr":z.R.mean(),"median_atr":z.R.median(),"spread_med_atr":z.spread_ATR.median()})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--symbol",choices=["USDJPY","XAUUSD"],required=True)
    a=ap.parse_args()
    Path("research/results").mkdir(parents=True,exist_ok=True)
    work=Path("_research_work"); work.mkdir(exist_ok=True)
    z=work/f"{a.symbol}_bid.zip"
    if not z.exists(): z=download_zip(a.symbol,work)
    m1=read_m1(z); bid=resample(m1,"5min"); del m1
    ask=load_ask(a.symbol)

    common=bid.index.intersection(ask.index)
    coverage=len(common)/len(bid)
    print(f"PDHM5ALIGN {a.symbol} bid={len(bid):,} ask={len(ask):,} common={len(common):,} coverage={coverage:.4%}",flush=True)
    if coverage<0.90: raise RuntimeError(f"low alignment {coverage:.2%}")
    out=run(a.symbol,bid.loc[common],ask.loc[common])
    out.to_csv(f"research/results/{a.symbol}_PDH_Momentum_M5_v01.csv",index=False)

if __name__=="__main__":
    main()
