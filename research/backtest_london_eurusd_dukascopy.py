#!/usr/bin/env python3
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import calendar
import time
import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD

START=pd.Timestamp("2015-01-01")
END=pd.Timestamp("2026-10-01")
OUT=Path("research/results")
CACHE=Path("_research_work/eurusd_m15")
OUT.mkdir(parents=True,exist_ok=True)
CACHE.mkdir(parents=True,exist_ok=True)

def month_starts(start,end):
    x=pd.Timestamp(start.year,start.month,1)
    while x<end:
        y=x+pd.offsets.MonthBegin(1)
        yield x,min(y,end)
        x=y

def fetch_month(side,start,end):
    side_name="bid" if side==dukascopy_python.OFFER_SIDE_BID else "ask"
    p=CACHE/f"EURUSD_M15_{side_name}_{start:%Y%m}.csv"
    if p.exists() and p.stat().st_size>100:
        df=pd.read_csv(p,index_col=0,parse_dates=True)
        df.index=pd.DatetimeIndex(df.index)
        return df
    last=None
    for attempt in range(4):
        try:
            df=dukascopy_python.fetch(
                instrument=INSTRUMENT_FX_MAJORS_EUR_USD,
                interval=dukascopy_python.INTERVAL_MIN_15,
                offer_side=side,
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                max_retries=5,
            )
            if df is None:
                raise RuntimeError("fetch returned None")
            df=df.copy()
            if not isinstance(df.index,pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
                    df=df.set_index("timestamp")
                else:
                    df.index=pd.to_datetime(df.index,utc=True)
            if df.index.tz is not None:
                df.index=df.index.tz_convert("UTC").tz_localize(None)
            else:
                df.index=df.index.tz_localize(None)
            cols={c.lower():c for c in df.columns}
            want=[]
            for k in ["open","high","low","close","volume"]:
                if k in cols: want.append(cols[k])
            df=df[want]
            df.columns=[str(c).lower() for c in df.columns]
            df=df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(p)
            print(f"FETCH {side_name} {start:%Y-%m} rows={len(df)}",flush=True)
            return df
        except Exception as e:
            last=e
            print(f"RETRY {side_name} {start:%Y-%m} attempt={attempt+1} err={e}",flush=True)
            time.sleep(2*(attempt+1))
    raise RuntimeError(f"failed {side_name} {start:%Y-%m}: {last}")

def load_side(side):
    chunks=[]
    for s,e in month_starts(START,END):
        chunks.append(fetch_month(side,s,e))
    z=pd.concat(chunks).sort_index()
    z=z[~z.index.duplicated(keep="last")]
    z=z[(z.index>=START)&(z.index<END)]
    return z

def atr_wilder(b,n=14):
    pc=b["close"].shift(1)
    tr=pd.concat([(b["high"]-b["low"]),(b["high"]-pc).abs(),(b["low"]-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def prepare():
    bid=load_side(dukascopy_python.OFFER_SIDE_BID)
    ask=load_side(dukascopy_python.OFFER_SIDE_ASK)
    common=bid.index.intersection(ask.index)
    bid=bid.loc[common,["open","high","low","close"]].copy()
    ask=ask.loc[common,["open","high","low","close"]].copy()
    # UTC is the native Dukascopy timestamp for downloaded historical bars.
    bid.index=bid.index.tz_localize("UTC").tz_convert("Europe/London")
    ask.index=ask.index.tz_localize("UTC").tz_convert("Europe/London")
    # H1 ATR from BID chart, with last completed H1 only.
    h1=bid.resample("1h",label="left",closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    ah=atr_wilder(h1,14)
    atr15=ah.shift(1).reindex(bid.index,method="ffill")
    return bid,ask,atr15

def sim(bid,ask,atr15):
    T=bid.index
    bo=bid["open"].to_numpy(float); bh=bid["high"].to_numpy(float)
    bl=bid["low"].to_numpy(float); bc=bid["close"].to_numpy(float)
    ao=ask["open"].to_numpy(float); ah=ask["high"].to_numpy(float)
    al=ask["low"].to_numpy(float); ac=ask["close"].to_numpy(float)
    av=atr15.to_numpy(float)
    dates=np.array([x.date() for x in T],dtype=object)
    groups={}
    for i,d in enumerate(dates):
        groups.setdefault(d,[]).append(i)
    rows=[]
    for day,poslist in groups.items():
        pos=np.asarray(poslist,dtype=int)
        rpos=[i for i in pos if 0<=T[i].hour<8]
        wpos=[i for i in pos if 8<=T[i].hour<12]
        # A complete normal London range has 32 M15 bars.
        if len(rpos)!=32 or not wpos:
            continue
        rh=max(bh[i] for i in rpos); rl=min(bl[i] for i in rpos)
        sig=None; d=0
        for i in wpos:
            if bc[i]>rh:
                sig=i;d=1;break
            if bc[i]<rl:
                sig=i;d=-1;break
        if sig is None or sig+1>=len(T) or not np.isfinite(av[sig]) or av[sig]<=0:
            continue
        ei=sig+1
        if T[ei].date()!=day:
            continue
        risk=1.5*av[sig]
        entry=ao[ei] if d==1 else bo[ei]
        sl=entry-d*risk
        tp=entry+d*2.0*risk
        spread_entry=ao[ei]-bo[ei]
        R=None;reason=None;xi=None
        for j in pos:
            if j<ei: continue
            if T[j].hour>=16:
                exit_px=bo[j] if d==1 else ao[j]
                R=d*(exit_px-entry)/risk;reason="TIME16";xi=j;break
            # Conservative same-bar ambiguity: stop first.
            if d==1:
                if bl[j]<=sl:
                    R=-1.0;reason="SL";xi=j;break
                if bh[j]>=tp:
                    R=2.0;reason="TP";xi=j;break
            else:
                if ah[j]>=sl:
                    R=-1.0;reason="SL";xi=j;break
                if al[j]<=tp:
                    R=2.0;reason="TP";xi=j;break
        if R is None:
            j=pos[-1]
            if j<ei: continue
            exit_px=bc[j] if d==1 else ac[j]
            R=d*(exit_px-entry)/risk;reason="DAY_END";xi=j
        rows.append((T[sig],T[ei],T[xi],d,float(R),reason,float(risk),float(spread_entry),float(spread_entry/risk)))
    return pd.DataFrame(rows,columns=["signal_time","entry_time","exit_time","direction","R","reason","risk","entry_spread","entry_spread_R"])

def summarize(tr):
    if tr.empty:
        return dict(trades=0,wins=0,losses=0,win_pct=np.nan,pf=np.nan,avg_r=np.nan,max_dd_r=np.nan,sum_r=0)
    r=tr["R"].to_numpy(float)
    pos=r[r>0].sum();neg=-r[r<0].sum()
    cum=np.cumsum(r);peak=np.maximum.accumulate(np.r_[0.0,cum]);dd=np.r_[0.0,cum]-peak
    wins=(r>0).sum();losses=(r<0).sum()
    return dict(trades=len(r),wins=int(wins),losses=int(losses),win_pct=100*wins/(wins+losses) if wins+losses else np.nan,pf=pos/neg if neg else np.inf,avg_r=float(r.mean()),max_dd_r=float(-dd.min()),sum_r=float(r.sum()))

def report(tr):
    s=summarize(tr)
    print(f"EURUSD_LB_EXACT n={s['trades']} win={s['win_pct']:.2f}% PF={s['pf']:.3f} avgR={s['avg_r']:+.4f} sumR={s['sum_r']:+.2f} DD={s['max_dd_r']:.1f}",flush=True)
    if len(tr):
        print(f"SPREAD entry medianR={tr['entry_spread_R'].median():.4f} meanR={tr['entry_spread_R'].mean():.4f} p90R={tr['entry_spread_R'].quantile(.9):.4f}",flush=True)
        z=tr.copy()
        yrs=pd.to_datetime(z["signal_time"]).dt.year
        z["era"]=np.select([yrs<=2018,yrs<=2022],["2015-2018","2019-2022"],default="2023-2026")
        for era,g in z.groupby("era",sort=False):
            e=summarize(g)
            print(f"EURERA {era} n={e['trades']} PF={e['pf']:.3f} avgR={e['avg_r']:+.4f} DD={e['max_dd_r']:.1f}",flush=True)
        for d,g in z.groupby("direction"):
            e=summarize(g)
            print(f"EURSIDE {'LONG' if d==1 else 'SHORT'} n={e['trades']} PF={e['pf']:.3f} avgR={e['avg_r']:+.4f}",flush=True)
        print(f"EURREASON {z['reason'].value_counts().to_dict()}",flush=True)

def main():
    bid,ask,a15=prepare()
    print(f"ALIGNED_M15 rows={len(bid)} from={bid.index.min()} to={bid.index.max()}",flush=True)
    tr=sim(bid,ask,a15)
    report(tr)
    tr.to_csv(OUT/"EURUSD_LondonBreakout_exact_bidask_trades.csv",index=False)

if __name__=="__main__":
    main()
