#!/usr/bin/env python3
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_USD_JPY,
    INSTRUMENT_FX_METALS_XAU_USD,
)

from backtest_hidden_divergence import download_zip, read_m1, atr_wilder, ema
from backtest_goscal_components import stochastic_533, ichimoku, cross_up, cross_dn

INST={"USDJPY":INSTRUMENT_FX_MAJORS_USD_JPY,"XAUUSD":INSTRUMENT_FX_METALS_XAU_USD}
START=pd.Timestamp("2015-01-01")
END=pd.Timestamp("2026-10-02")
HORIZONS=[5,10,20,30]

def normalize_ask(df):
    df=df.copy()
    if not isinstance(df.index,pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
            df=df.set_index("timestamp")
        else:
            df.index=pd.to_datetime(df.index,utc=True)
    idx=df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index=idx.tz_convert("Europe/Helsinki").tz_localize(None)
    df.columns=[str(c).lower() for c in df.columns]
    return df[[c for c in ["open","high","low","close"] if c in df.columns]].loc[~df.index.duplicated()].sort_index()

def month_starts():
    x=START
    while x<END:
        y=x+pd.offsets.MonthBegin(1)
        yield x,min(y,END)
        x=y

def load_ask(symbol):
    cache=Path(f"_research_work/{symbol.lower()}_ask_m1"); cache.mkdir(parents=True,exist_ok=True)
    try:
        print(f"{symbol} ASK_FAST_FETCH",flush=True)
        z=dukascopy_python.fetch(
            instrument=INST[symbol],interval=dukascopy_python.INTERVAL_MIN_1,
            offer_side=dukascopy_python.OFFER_SIDE_ASK,
            start=START.to_pydatetime(),end=END.to_pydatetime(),max_retries=5)
        z=normalize_ask(z)
        if len(z)>3_000_000:
            print(f"{symbol} ASK_FAST_OK rows={len(z):,}",flush=True)
            return z
        raise RuntimeError(f"short fetch {len(z)}")
    except Exception as e:
        print(f"{symbol} ASK_FAST_FALLBACK {e}",flush=True)

    chunks=[]
    for s,e in month_starts():
        p=cache/f"{symbol}_ASK_M1_{s:%Y%m}.csv"
        if p.exists() and p.stat().st_size>100:
            z=pd.read_csv(p,index_col=0,parse_dates=True); z.index=pd.DatetimeIndex(z.index)
            chunks.append(z); continue
        last=None
        for a in range(4):
            try:
                z=dukascopy_python.fetch(
                    instrument=INST[symbol],interval=dukascopy_python.INTERVAL_MIN_1,
                    offer_side=dukascopy_python.OFFER_SIDE_ASK,
                    start=s.to_pydatetime(),end=e.to_pydatetime(),max_retries=5)
                z=normalize_ask(z); z.to_csv(p); chunks.append(z)
                print(f"{symbol} ASK_MONTH {s:%Y-%m} n={len(z)}",flush=True)
                last=None; break
            except Exception as exc:
                last=exc; time.sleep(2*(a+1))
        if last is not None: raise RuntimeError(f"{symbol} ask {s:%Y-%m}: {last}")
    return pd.concat(chunks).sort_index().loc[lambda x: ~x.index.duplicated()]

def prev_levels(b):
    day=pd.Series(b.index.normalize(),index=b.index)
    d=b.groupby(day).agg({"High":"max","Low":"min"})
    pdh=day.map(d["High"].shift(1)); pdl=day.map(d["Low"].shift(1))
    return pd.Series(pdh.to_numpy(float),index=b.index),pd.Series(pdl.to_numpy(float),index=b.index)

def signals(b):
    pdh,pdl=prev_levels(b)
    k,d=stochastic_533(b)
    e6=ema(b["Close"],6); e13=ema(b["Close"],13)
    _,_,sa,sb=ichimoku(b)
    ct=pd.concat([sa,sb],axis=1).max(axis=1); cb=pd.concat([sa,sb],axis=1).min(axis=1)
    up=(b["Close"]>pdh)&(b["Close"].shift(1)<=pdh.shift(1))
    dn=(b["Close"]<pdl)&(b["Close"].shift(1)>=pdl.shift(1))
    return {
      "LEVEL_ONLY":(up,dn),
      "STOCH_CROSS":(up&cross_up(k,d),dn&cross_dn(k,d)),
      "EMA_ALIGN":(up&(e6>e13),dn&(e6<e13)),
      "EMA_AND_CLOUD":(up&(e6>e13)&(b["Close"]>ct),dn&(e6<e13)&(b["Close"]<cb)),
    }

def report(symbol,name,lo,sh,bid,ask,atr,h):
    m=(lo|sh).fillna(False).to_numpy()
    ids=np.flatnonzero(m)
    bo=bid["Open"].to_numpy(float); bc=bid["Close"].to_numpy(float)
    ao=ask["open"].to_numpy(float); ac=ask["close"].to_numpy(float)
    av=atr.to_numpy(float)
    times=bid.index
    rows=[]
    for i in ids:
        ei=i+1; xi=i+h
        if xi>=len(bid) or not np.isfinite(av[i]) or av[i]<=0: continue
        d=1 if bool(lo.iloc[i]) else -1
        if not all(np.isfinite(x) for x in [bo[ei],bc[xi],ao[ei],ac[xi]]): continue
        entry=ao[ei] if d==1 else bo[ei]
        exitp=bc[xi] if d==1 else ac[xi]
        r=d*(exitp-entry)/av[i]
        spr=(ao[ei]-bo[ei])/av[i]
        rows.append((times[i],d,r,spr))
    z=pd.DataFrame(rows,columns=["signal_time","direction","R","spread_ATR"])
    if z.empty:return None
    yrs=pd.to_datetime(z.signal_time).dt.year
    z["era"]=np.select([yrs<=2018,yrs<=2022],["2015-2018","2019-2022"],default="2023-2026")
    print(f"PDHEXACT {symbol} {name:13s} h={h:2d} n={len(z):6d} win={100*(z.R>0).mean():6.2f}% meanATR={z.R.mean():+.5f} medATR={z.R.median():+.5f} spreadMed={z.spread_ATR.median():.4f}",flush=True)
    if h==20:
        for era,g in z.groupby("era",sort=False):
            print(f"PDHERA {symbol} {name:13s} {era} n={len(g):6d} win={100*(g.R>0).mean():6.2f}% meanATR={g.R.mean():+.5f} medATR={g.R.median():+.5f}",flush=True)
    return {"symbol":symbol,"signal":name,"h":h,"n":len(z),"win_pct":100*(z.R>0).mean(),"mean_atr":z.R.mean(),"median_atr":z.R.median(),"spread_median_atr":z.spread_ATR.median()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--symbol",choices=["USDJPY","XAUUSD"],required=True)
    a=ap.parse_args()
    Path("research/results").mkdir(parents=True,exist_ok=True)
    work=Path("_research_work"); work.mkdir(exist_ok=True)
    z=work/f"{a.symbol}_bid.zip"
    if not z.exists(): z=download_zip(a.symbol,work)
    bid=read_m1(z); ask=load_ask(a.symbol)

    common=bid.index.intersection(ask.index)
    coverage=len(common)/len(bid)
    print(f"PDHALIGN {a.symbol} bid={len(bid):,} ask={len(ask):,} common={len(common):,} coverage={coverage:.4%}",flush=True)
    if coverage<0.90: raise RuntimeError(f"alignment low {coverage:.2%}")
    bid=bid.loc[common]; ask=ask.loc[common]
    atr=atr_wilder(bid,14)
    out=[]
    for name,(lo,sh) in signals(bid).items():
        for h in HORIZONS:
            row=report(a.symbol,name,lo,sh,bid,ask,atr,h)
            if row: out.append(row)
    pd.DataFrame(out).to_csv(f"research/results/{a.symbol}_PDH_breakout_exact_bidask.csv",index=False)

if __name__=="__main__":
    main()
