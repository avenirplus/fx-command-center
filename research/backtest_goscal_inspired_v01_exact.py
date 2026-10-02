#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

from backtest_hidden_divergence import download_zip, read_m1, atr_wilder, ema
from backtest_goscal_components import stochastic_533, ichimoku, cross_up, cross_dn

START=pd.Timestamp("2015-01-01")
END=pd.Timestamp("2026-10-02")
CACHE=Path("_research_work/xau_ask_m1")
OUT=Path("research/results")
HORIZONS=[5,10,20,30]
CACHE.mkdir(parents=True,exist_ok=True)
OUT.mkdir(parents=True,exist_ok=True)

def normalize_ask(df):
    df=df.copy()
    if not isinstance(df.index,pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
            df=df.set_index("timestamp")
        else:
            df.index=pd.to_datetime(df.index,utc=True)
    if df.index.tz is None:
        idx=df.index.tz_localize("UTC")
    else:
        idx=df.index.tz_convert("UTC")
    # JForex export is labeled EET and follows EET/EEST. Convert the downloaded
    # UTC series to Europe/Helsinki, then drop timezone for alignment to upload.
    df.index=idx.tz_convert("Europe/Helsinki").tz_localize(None)
    df.columns=[str(c).lower() for c in df.columns]
    keep=[c for c in ["open","high","low","close"] if c in df.columns]
    return df[keep].loc[~df.index.duplicated(keep="last")].sort_index()

def month_starts(start,end):
    x=pd.Timestamp(start.year,start.month,1)
    while x<end:
        y=x+pd.offsets.MonthBegin(1)
        yield x,min(y,end)
        x=y

def fetch_ask_month(s,e):
    p=CACHE/f"XAUUSD_ASK_M1_{s:%Y%m}.csv"
    if p.exists() and p.stat().st_size>100:
        z=pd.read_csv(p,index_col=0,parse_dates=True)
        z.index=pd.DatetimeIndex(z.index)
        return z
    last=None
    for attempt in range(4):
        try:
            z=dukascopy_python.fetch(
                instrument=INSTRUMENT_FX_METALS_XAU_USD,
                interval=dukascopy_python.INTERVAL_MIN_1,
                offer_side=dukascopy_python.OFFER_SIDE_ASK,
                start=s.to_pydatetime(),
                end=e.to_pydatetime(),
                max_retries=5,
            )
            z=normalize_ask(z)
            z.to_csv(p)
            print(f"ASK_FETCH {s:%Y-%m} rows={len(z)}",flush=True)
            return z
        except Exception as exc:
            last=exc
            print(f"ASK_RETRY {s:%Y-%m} attempt={attempt+1} err={exc}",flush=True)
            time.sleep(2*(attempt+1))
    raise RuntimeError(f"Ask fetch failed {s:%Y-%m}: {last}")

def load_ask():
    # Try one full request first; monthly fallback keeps the job resumable.
    try:
        print("ASK_FAST_FETCH full range",flush=True)
        z=dukascopy_python.fetch(
            instrument=INSTRUMENT_FX_METALS_XAU_USD,
            interval=dukascopy_python.INTERVAL_MIN_1,
            offer_side=dukascopy_python.OFFER_SIDE_ASK,
            start=START.to_pydatetime(),
            end=END.to_pydatetime(),
            max_retries=5,
        )
        z=normalize_ask(z)
        if len(z)>3_000_000:
            print(f"ASK_FAST_OK rows={len(z):,}",flush=True)
            return z
        raise RuntimeError(f"unexpected short ask fetch {len(z)}")
    except Exception as exc:
        print(f"ASK_FAST_FALLBACK {exc}",flush=True)

    chunks=[]
    for s,e in month_starts(START,END):
        chunks.append(fetch_ask_month(s,e))
    z=pd.concat(chunks).sort_index()
    z=z.loc[~z.index.duplicated(keep="last")]
    return z

def previous_day_levels(b):
    day=pd.Series(b.index.normalize(),index=b.index)
    daily=b.groupby(day).agg({"High":"max","Low":"min"})
    daily["PDH"]=daily["High"].shift(1)
    daily["PDL"]=daily["Low"].shift(1)
    pdh=day.map(daily["PDH"])
    pdl=day.map(daily["PDL"])
    return pd.Series(pdh.to_numpy(float),index=b.index),pd.Series(pdl.to_numpy(float),index=b.index)

def make_signals(b):
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
    return {
        "BREAKOUT_LEVEL_ONLY":(lo,sh),
        "BREAKOUT_STOCH_CROSS":(lo&st_up,sh&st_dn),
        "BREAKOUT_EMA_ALIGN":(lo&ema_long,sh&ema_short),
        "BREAKOUT_EMA_AND_CLOUD":(lo&ema_long&cloud_long,sh&ema_short&cloud_short),
    }

def exact_horizon(name,lo,sh,bid,ask,atr,h):
    mask=(lo|sh).fillna(False)
    idx=np.flatnonzero(mask.to_numpy())
    bo=bid["Open"].to_numpy(float)
    bc=bid["Close"].to_numpy(float)
    ao=ask["open"].to_numpy(float)
    ac=ask["close"].to_numpy(float)
    av=atr.to_numpy(float)
    times=bid.index
    rows=[]
    for i in idx:
        ei=i+1; xi=i+h
        if xi>=len(bid): continue
        d=1 if bool(lo.iloc[i]) else -1
        vals=(av[i],bo[ei],bc[xi],ao[ei],ac[xi])
        if not all(np.isfinite(v) for v in vals) or av[i]<=0:
            continue
        if d==1:
            entry=ao[ei]; exit_px=bc[xi]
        else:
            entry=bo[ei]; exit_px=ac[xi]
        r=d*(exit_px-entry)/av[i]
        spread=ao[ei]-bo[ei]
        rows.append((times[i],d,r,spread/av[i]))
    z=pd.DataFrame(rows,columns=["signal_time","direction","R","entry_spread_ATR"])
    if z.empty:
        return None,z
    yrs=pd.to_datetime(z["signal_time"]).dt.year
    z["era"]=np.select([yrs<=2018,yrs<=2022],["2015-2018","2019-2022"],default="2023-2026")
    print(f"GSIEXACT {name:30s} h={h:2d} n={len(z):6d} win={100*(z.R>0).mean():6.2f}% meanATR={z.R.mean():+.5f} medATR={z.R.median():+.5f} spreadMedATR={z.entry_spread_ATR.median():.4f}",flush=True)
    for era,g in z.groupby("era",sort=False):
        print(f"GSIEXACTERA {name:27s} h={h:2d} {era} n={len(g):6d} win={100*(g.R>0).mean():6.2f}% meanATR={g.R.mean():+.5f} medATR={g.R.median():+.5f}",flush=True)
    return {
        "signal":name,"h_min":h,"n":len(z),"win_pct":100*(z.R>0).mean(),
        "mean_atr":z.R.mean(),"median_atr":z.R.median(),
        "median_entry_spread_atr":z.entry_spread_ATR.median()
    },z

def main():
    work=Path("_research_work"); work.mkdir(exist_ok=True)
    z=work/"XAUUSD_bid.zip"
    if not z.exists():
        z=download_zip("XAUUSD",work)
    bid=read_m1(z)
    ask=load_ask()

    # Align ask to the exact uploaded Bid minute index after UTC->EET/EEST conversion.
    common=bid.index.intersection(ask.index)
    coverage=len(common)/len(bid)
    print(f"ALIGN bid={len(bid):,} ask={len(ask):,} common={len(common):,} coverage={coverage:.4%}",flush=True)
    if coverage<0.90:
        raise RuntimeError(f"Ask/Bid timestamp alignment too low: {coverage:.2%}")

    bid=bid.loc[common].copy()
    ask=ask.loc[common].copy()
    atr=atr_wilder(bid,14)
    signals=make_signals(bid)

    out=[]
    for name,(lo,sh) in signals.items():
        for h in HORIZONS:
            row,ztr=exact_horizon(name,lo,sh,bid,ask,atr,h)
            if row is not None:
                out.append(row)
                if h==20:
                    ztr.to_csv(OUT/f"XAUUSD_{name}_exact_h20.csv",index=False)
    pd.DataFrame(out).to_csv(OUT/"XAUUSD_GOSCAL_inspired_v01_exact_bidask.csv",index=False)

if __name__=="__main__":
    main()
