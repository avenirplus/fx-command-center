#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import download_zip, read_m1, summarize, atr_wilder

SYMBOL="USDJPY"
TZ_MODES={
    "UTC":"UTC",
    "UTC_PLUS_2":"Etc/GMT-2",
    "UTC_PLUS_3":"Etc/GMT-3",
    "EET_EEST":"Europe/Helsinki",
}

OHLC={"Open":"first","High":"max","Low":"min","Close":"last"}

def attach_timezone(df, tzname):
    x=df.copy()
    idx=x.index
    if tzname=="Europe/Helsinki":
        aware=idx.tz_localize(tzname, ambiguous="NaT", nonexistent="NaT")
    else:
        aware=idx.tz_localize(tzname, ambiguous="NaT", nonexistent="NaT")
    x=x.loc[~aware.isna()].copy()
    aware=aware[~aware.isna()]
    x.index=aware.tz_convert("Europe/London")
    x=x[~x.index.duplicated(keep="last")].sort_index()
    return x

def build_bars(m1_london):
    m15=m1_london.resample("15min",label="left",closed="left").agg(OHLC).dropna()
    h1=m1_london.resample("1h",label="left",closed="left").agg(OHLC).dropna()
    ah=atr_wilder(h1,14)
    # Last completed H1 only.
    atr15=ah.shift(1).reindex(m15.index,method="ffill")
    return m15,atr15

def sim_one_mode(m15,atr15):
    O=m15["Open"].to_numpy(float); H=m15["High"].to_numpy(float)
    L=m15["Low"].to_numpy(float); C=m15["Close"].to_numpy(float)
    A=atr15.to_numpy(float); T=m15.index
    trades=[]

    # Group index positions by London local calendar date.
    dates=np.array([ts.date() for ts in T],dtype=object)
    unique_dates=pd.unique(dates)
    for day in unique_dates:
        pos=np.flatnonzero(dates==day)
        if len(pos)==0: continue
        rpos=[i for i in pos if 0 <= T[i].hour < 8]
        wpos=[i for i in pos if 8 <= T[i].hour < 12]
        if len(rpos)!=32 or not wpos:
            continue
        rh=max(H[i] for i in rpos); rl=min(L[i] for i in rpos)

        sig=None; d=0
        for i in wpos:
            if C[i]>rh:
                sig=i; d=1; break
            if C[i]<rl:
                sig=i; d=-1; break
        if sig is None or sig+1>=len(T) or not np.isfinite(A[sig]) or A[sig]<=0:
            continue

        entry_i=sig+1
        # Entry must still be on the same London day.
        if T[entry_i].date()!=day:
            continue
        entry=O[entry_i]
        risk=1.5*A[sig]
        sl=entry-d*risk
        tp=entry+d*2.0*risk

        R=None; exit_i=None; reason=None
        j=entry_i
        while j<len(T) and T[j].date()==day:
            # Force close at London 16:00 before using the 16:00 bar range.
            if T[j].hour>=16:
                R=d*(O[j]-entry)/risk
                exit_i=j; reason="TIME16"; break
            # Conservative same-bar ambiguity: stop first.
            if d==1:
                if L[j]<=sl:
                    R=-1.0; exit_i=j; reason="SL"; break
                if H[j]>=tp:
                    R=2.0; exit_i=j; reason="TP"; break
            else:
                if H[j]>=sl:
                    R=-1.0; exit_i=j; reason="SL"; break
                if L[j]<=tp:
                    R=2.0; exit_i=j; reason="TP"; break
            j+=1

        if R is None:
            # No 16:00 print: close at final bar close of that London day.
            j=pos[-1]
            if j<entry_i: continue
            R=d*(C[j]-entry)/risk
            exit_i=j; reason="DAY_END"

        trades.append((T[sig],T[entry_i],T[exit_i],d,float(R),reason))

    return pd.DataFrame(trades,columns=["signal_time","entry_time","exit_time","direction","R","reason"])

def stat(tr):
    return summarize(tr)

def era_rows(tr,mode):
    if tr.empty:return []
    years=pd.to_datetime(tr["signal_time"]).dt.year
    era=np.select([years<=2018,years<=2022],["2015-2018","2019-2022"],default="2023-2026")
    z=tr.copy();z["era"]=era
    out=[]
    for e,g in z.groupby("era",sort=False):
        s=stat(g);out.append((e,s))
        print(f"LBERA {mode:10s} {e} n={s['trades']:4d} PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} DD={s['max_dd_r']:.1f}")
    return out

def main():
    work=Path("_research_work");work.mkdir(exist_ok=True)
    out=Path("research/results");out.mkdir(parents=True,exist_ok=True)
    z=work/f"{SYMBOL}_bid.zip"
    if not z.exists(): z=download_zip(SYMBOL,work)
    m1=read_m1(z)

    rows=[]
    for mode,tzname in TZ_MODES.items():
        x=attach_timezone(m1,tzname)
        m15,a15=build_bars(x)
        tr=sim_one_mode(m15,a15)
        s=stat(tr)
        print(f"LB {mode:10s} n={s['trades']:4d} win={s['win_pct']:6.2f}% PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} DD={s['max_dd_r']:.1f}")
        era_rows(tr,mode)
        for cost in [0.02,0.05,0.10]:
            tc=tr.copy()
            if not tc.empty:tc["R"]=tc["R"]-cost
            sc=stat(tc)
            print(f"LBCOST {mode:10s} cost={cost:.2f}R PF={sc['pf']:7.3f} avgR={sc['avg_r']:+.4f}")
        reasons=tr["reason"].value_counts().to_dict() if not tr.empty else {}
        print(f"LBREASON {mode:10s} {reasons}")
        rows.append({"mode":mode,**s,**{f"reason_{k}":v for k,v in reasons.items()}})
        tr.to_csv(out/f"USDJPY_LondonBreakout_{mode}_trades.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"USDJPY_LondonBreakout_timezone_summary.csv",index=False)

if __name__=="__main__":
    main()
