#!/usr/bin/env python3
import argparse, os, sys, zipfile
from pathlib import Path

import gdown
import numpy as np
import pandas as pd

DRIVE_IDS = {
    "USDJPY": "1tuUt8mguCjmzz25zcxjWykKgTcKDwf6V",
    "XAUUSD": "1-OUoJHxWgQtzyVn2bToI6GteVBYnRN5g",
}

OHLC = {"Open":"first","High":"max","Low":"min","Close":"last"}

def download_zip(symbol: str, work: Path) -> Path:
    out = work / f"{symbol}_bid.zip"
    print(f"[download] {symbol} -> {out}", flush=True)
    url = f"https://drive.google.com/uc?id={DRIVE_IDS[symbol]}"
    got = gdown.download(url=url, output=str(out), quiet=False, fuzzy=True, resume=True)
    if not got or not out.exists() or out.stat().st_size < 1_000_000:
        raise RuntimeError(f"download failed for {symbol}")
    print(f"[download] {symbol}: {out.stat().st_size/1024/1024:.1f} MB", flush=True)
    return out

def read_m1(zpath: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV in {zpath}")
        member = members[0]
        print(f"[zip] reading {member}", flush=True)
        with z.open(member) as f:
            df = pd.read_csv(
                f,
                usecols=["Time (EET)","Open","High","Low","Close"],
                parse_dates=["Time (EET)"],
                date_format="%Y.%m.%d %H:%M:%S",
            )
    df = df.rename(columns={"Time (EET)":"Time"}).dropna()
    df = df.drop_duplicates("Time", keep="last").set_index("Time").sort_index()
    for c in ["Open","High","Low","Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    print(f"[m1] rows={len(df):,} from={df.index.min()} to={df.index.max()}", flush=True)
    return df

def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    b = df.resample(rule, label="left", closed="left").agg(OHLC).dropna()
    return b

def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def atr_wilder(b: pd.DataFrame, n=14):
    pc = b["Close"].shift(1)
    tr = pd.concat([
        b["High"]-b["Low"],
        (b["High"]-pc).abs(),
        (b["Low"]-pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def cci(b: pd.DataFrame, n=20):
    tp = (b["High"]+b["Low"]+b["Close"])/3.0
    ma = tp.rolling(n, min_periods=n).mean()
    md = tp.rolling(n, min_periods=n).apply(lambda x: np.mean(np.abs(x-np.mean(x))), raw=True)
    return (tp-ma)/(0.015*md.replace(0, np.nan))

def macd_hist(close: pd.Series):
    m = ema(close,12)-ema(close,26)
    sig = ema(m,9)
    return m-sig

def h4_trend_from(b: pd.DataFrame) -> pd.Series:
    h4 = b.resample("4h", label="left", closed="left").agg(OHLC).dropna()
    e50 = ema(h4["Close"],50)
    e200 = ema(h4["Close"],200)
    t = pd.Series(0, index=h4.index, dtype=np.int8)
    t[(h4["Close"]>e50)&(e50>e200)] = 1
    t[(h4["Close"]<e50)&(e50<e200)] = -1
    # Previous completed H4 only; then forward-fill to entry timeframe.
    return t.shift(1).reindex(b.index, method="ffill").fillna(0).astype(np.int8)

def pivot_arrays(b: pd.DataFrame):
    lo, hi = b["Low"], b["High"]
    pl = (
        (lo <= lo.shift(1)) & (lo < lo.shift(2)) &
        (lo <= lo.shift(-1)) & (lo < lo.shift(-2))
    ).fillna(False).to_numpy()
    ph = (
        (hi >= hi.shift(1)) & (hi > hi.shift(2)) &
        (hi >= hi.shift(-1)) & (hi > hi.shift(-2))
    ).fillna(False).to_numpy()
    return np.flatnonzero(pl), np.flatnonzero(ph)

def summarize(tr: pd.DataFrame):
    if tr.empty:
        return dict(trades=0,wins=0,losses=0,timeouts=0,win_pct=np.nan,pf=np.nan,avg_r=np.nan,max_dd_r=np.nan)
    r = tr["R"].to_numpy(float)
    pos = r[r>0].sum()
    neg = -r[r<0].sum()
    wins = int((r>0).sum())
    losses = int((r<0).sum())
    timeouts = int((r==0).sum())
    pf = pos/neg if neg>0 else np.inf
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0.0,cum])
    dd = np.r_[0.0,cum]-peak
    return dict(
        trades=len(r),wins=wins,losses=losses,timeouts=timeouts,
        win_pct=100*wins/(wins+losses) if wins+losses else np.nan,
        pf=pf,avg_r=float(np.mean(r)),max_dd_r=float(-dd.min())
    )

def eval_trades(b, signal_mask, direction, atr, max_hold):
    idx = np.flatnonzero(signal_mask)
    rows=[]
    o=b["Open"].to_numpy(float); h=b["High"].to_numpy(float); l=b["Low"].to_numpy(float); c=b["Close"].to_numpy(float)
    av=atr.to_numpy(float); times=b.index
    for i in idx:
        if i+1 >= len(b) or not np.isfinite(av[i]) or av[i] <= 0:
            continue
        d=int(direction[i])
        if d not in (-1,1):
            continue
        entry=o[i+1]
        risk=av[i]
        tp=entry+d*risk
        sl=entry-d*risk
        r=0.0
        exit_i=min(i+1+max_hold, len(b)-1)
        for j in range(i+1, exit_i+1):
            # Same-bar ambiguity: stop first (conservative).
            if d==1:
                if l[j] <= sl:
                    r=-1.0; exit_i=j; break
                if h[j] >= tp:
                    r=1.0; exit_i=j; break
            else:
                if h[j] >= sl:
                    r=-1.0; exit_i=j; break
                if l[j] <= tp:
                    r=1.0; exit_i=j; break
        rows.append((times[i], times[i+1], d, r))
    return pd.DataFrame(rows, columns=["signal_time","entry_time","direction","R"])

def build_masks(b: pd.DataFrame, lookback: int, counter_bars: int):
    trend=h4_trend_from(b)
    av=atr_wilder(b,14)
    mh=macd_hist(b["Close"])
    cv=cci(b,20)

    prior_hi=b["High"].shift(1).rolling(lookback,min_periods=lookback).max()
    prior_lo=b["Low"].shift(1).rolling(lookback,min_periods=lookback).min()
    rng=(prior_hi-prior_lo).replace(0,np.nan)

    depth_long=(prior_hi-b["Close"])/rng
    depth_short=(b["Close"]-prior_lo)/rng
    band_long=depth_long.between(0.50,0.618,inclusive="both")
    band_short=depth_short.between(0.50,0.618,inclusive="both")

    # Reconstructed from the recorded GBP research specification:
    # trend -> countertrend/pullback >=0.75 ATR -> 50-61.8% zone -> one-bar resumption break.
    counter_long=b["Close"].shift(1) < b["Close"].shift(1+counter_bars)
    counter_short=b["Close"].shift(1) > b["Close"].shift(1+counter_bars)
    pull_long=(prior_hi-b["Low"]) >= 0.75*av
    pull_short=(b["High"]-prior_lo) >= 0.75*av
    trigger_long=b["Close"] > b["High"].shift(1)
    trigger_short=b["Close"] < b["Low"].shift(1)

    long_c=(trend==1)&band_long&counter_long&pull_long&trigger_long
    short_c=(trend==-1)&band_short&counter_short&pull_short&trigger_short
    candidate=(long_c|short_c).fillna(False).to_numpy()
    direction=np.where(long_c.to_numpy(),1,np.where(short_c.to_numpy(),-1,0)).astype(np.int8)

    low_piv, high_piv=pivot_arrays(b)
    lows=b["Low"].to_numpy(float); highs=b["High"].to_numpy(float)
    mhv=mh.to_numpy(float); ccv=cv.to_numpy(float)

    price_struct=np.zeros(len(b),dtype=bool)
    cci_hidden=np.zeros(len(b),dtype=bool)
    macd_hidden=np.zeros(len(b),dtype=bool)

    cand_idx=np.flatnonzero(candidate)
    for i in cand_idx:
        d=direction[i]
        piv=low_piv if d==1 else high_piv
        # A pivot with L=2 is usable only after its two right-hand bars have closed.
        k=np.searchsorted(piv, i-2, side="right")
        if k < 2:
            continue
        p1,p2=int(piv[k-2]),int(piv[k-1])
        if d==1:
            ps=lows[p2] > lows[p1]       # HL
            ch=np.isfinite(ccv[p1]) and np.isfinite(ccv[p2]) and ccv[p2] < ccv[p1]
            mhid=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2] < mhv[p1]
        else:
            ps=highs[p2] < highs[p1]     # LH
            ch=np.isfinite(ccv[p1]) and np.isfinite(ccv[p2]) and ccv[p2] > ccv[p1]
            mhid=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2] > mhv[p1]
        price_struct[i]=ps
        cci_hidden[i]=ps and ch
        macd_hidden[i]=ps and mhid

    masks={
        "candidate_50_61_8": candidate,
        "price_HL_LH": candidate & price_struct,
        "CCI_hidden": candidate & cci_hidden,
        "MACD_hist_hidden": candidate & macd_hidden,
        "CCI_and_MACD_hidden": candidate & cci_hidden & macd_hidden,
    }
    return masks,direction,av

def run_tf(symbol,b,tf_name,lookback,counter_bars,max_hold,outdir):
    print(f"\n=== {symbol} {tf_name} rows={len(b):,} ===",flush=True)
    masks,direction,av=build_masks(b,lookback,counter_bars)
    summary_rows=[]
    all_trades={}
    era_rows=[]
    side_rows=[]
    for name,mask in masks.items():
        tr=eval_trades(b,mask,direction,av,max_hold)
        all_trades[name]=tr
        s=summarize(tr)
        summary_rows.append({"symbol":symbol,"timeframe":tf_name,"filter":name,**s})
        print(f"{name:24s} n={s['trades']:5d} win={s['win_pct']:6.2f}% PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} maxDD={s['max_dd_r']:.1f} timeouts={s['timeouts']}",flush=True)
        if not tr.empty:
            years=pd.to_datetime(tr["signal_time"]).dt.year
            eras=np.select([years<=2018, years<=2022],["2015-2018","2019-2022"],default="2023-2026")
            tmp=tr.copy()
            tmp["era"]=eras
            for era,g in tmp.groupby("era"):
                era_rows.append({"symbol":symbol,"timeframe":tf_name,"filter":name,"era":era,**summarize(g)})
            for d,g in tr.groupby("direction"):
                side_rows.append({"symbol":symbol,"timeframe":tf_name,"filter":name,"side":"LONG" if d==1 else "SHORT",**summarize(g)})

    summ=pd.DataFrame(summary_rows)
    summ.to_csv(outdir/f"{symbol}_{tf_name}_summary.csv",index=False)
    pd.DataFrame(era_rows).to_csv(outdir/f"{symbol}_{tf_name}_era_stability.csv",index=False)
    pd.DataFrame(side_rows).to_csv(outdir/f"{symbol}_{tf_name}_side_all_filters.csv",index=False)

    print("\nEra stability (all filters):")
    for row in era_rows:
        print(f"ERA {row['filter']:24s} {row['era']}: n={row['trades']:4d} win={row['win_pct']:6.2f}% PF={row['pf']:7.3f} avgR={row['avg_r']:+.4f}")
    print("\nSide stability (all filters):")
    for row in side_rows:
        print(f"SIDE {row['filter']:24s} {row['side']:5s}: n={row['trades']:4d} win={row['win_pct']:6.2f}% PF={row['pf']:7.3f} avgR={row['avg_r']:+.4f}")

    # Detailed yearly breakdown for MACD histogram hidden divergence.
    key="MACD_hist_hidden"
    tr=all_trades[key].copy()
    if not tr.empty:
        tr["year"]=pd.to_datetime(tr["signal_time"]).dt.year
        yr=[]
        for y,g in tr.groupby("year"):
            yr.append({"symbol":symbol,"timeframe":tf_name,"year":int(y),**summarize(g)})
        pd.DataFrame(yr).to_csv(outdir/f"{symbol}_{tf_name}_MACD_hidden_yearly.csv",index=False)
        tr.to_csv(outdir/f"{symbol}_{tf_name}_MACD_hidden_trades.csv",index=False)
        print("\nYearly MACD histogram hidden:")
        for row in yr:
            print(f"{row['year']}: n={row['trades']:4d} win={row['win_pct']:6.2f}% PF={row['pf']:7.3f} avgR={row['avg_r']:+.4f}")
    return summ

def build_masks_h4_swing(b: pd.DataFrame):
    trend=h4_trend_from(b)
    av=atr_wilder(b,14)
    mh=macd_hist(b["Close"])
    cv=cci(b,20)

    h4=b.resample("4h",label="left",closed="left").agg(OHLC).dropna()
    h4_low_piv,h4_high_piv=pivot_arrays(h4)
    h4_times=h4.index
    h4_lows=h4["Low"].to_numpy(float)
    h4_highs=h4["High"].to_numpy(float)

    close=b["Close"].to_numpy(float)
    high=b["High"].to_numpy(float)
    low=b["Low"].to_numpy(float)
    times=b.index
    tv=trend.to_numpy(np.int8)
    avv=av.to_numpy(float)

    candidate=np.zeros(len(b),dtype=bool)
    direction=np.zeros(len(b),dtype=np.int8)

    # H4 pivot at bar t is usable only after two right-side H4 bars have fully closed.
    # With left-labeled 4h bars this is t + 12h, a conservative no-lookahead cutoff.
    h4_low_times=h4_times[h4_low_piv]
    h4_high_times=h4_times[h4_high_piv]

    for i in range(4,len(b)):
        d=int(tv[i])
        if d==0 or not np.isfinite(avv[i]):
            continue
        t=times[i]
        cutoff=t-pd.Timedelta(hours=12)
        if d==1:
            kh=np.searchsorted(h4_high_times,cutoff,side="right")
            if kh<1: continue
            bi=int(h4_high_piv[kh-1])
            kl=np.searchsorted(h4_low_piv,bi,side="left")
            if kl<1: continue
            ai=int(h4_low_piv[kl-1])
            A=h4_lows[ai]; B=h4_highs[bi]
            if not (B>A): continue
            depth=(B-close[i])/(B-A)
            if not (0.50<=depth<=0.618): continue
            counter=close[i-1] < close[i-4]
            pull=(B-low[i]) >= 0.75*avv[i]
            trigger=close[i] > high[i-1]
        else:
            kl=np.searchsorted(h4_low_times,cutoff,side="right")
            if kl<1: continue
            bi=int(h4_low_piv[kl-1])
            kh=np.searchsorted(h4_high_piv,bi,side="left")
            if kh<1: continue
            ai=int(h4_high_piv[kh-1])
            A=h4_highs[ai]; B=h4_lows[bi]
            if not (A>B): continue
            depth=(close[i]-B)/(A-B)
            if not (0.50<=depth<=0.618): continue
            counter=close[i-1] > close[i-4]
            pull=(high[i]-B) >= 0.75*avv[i]
            trigger=close[i] < low[i-1]
        if counter and pull and trigger:
            candidate[i]=True
            direction[i]=d

    low_piv,high_piv=pivot_arrays(b)
    mhv=mh.to_numpy(float); ccv=cv.to_numpy(float)
    price_struct=np.zeros(len(b),dtype=bool)
    cci_hidden=np.zeros(len(b),dtype=bool)
    macd_hidden=np.zeros(len(b),dtype=bool)

    for i in np.flatnonzero(candidate):
        d=int(direction[i])
        piv=low_piv if d==1 else high_piv
        k=np.searchsorted(piv,i-2,side="right")
        if k<2: continue
        p1,p2=int(piv[k-2]),int(piv[k-1])
        if d==1:
            ps=low[p2] > low[p1]
            ch=np.isfinite(ccv[p1]) and np.isfinite(ccv[p2]) and ccv[p2] < ccv[p1]
            mhid=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2] < mhv[p1]
        else:
            ps=high[p2] < high[p1]
            ch=np.isfinite(ccv[p1]) and np.isfinite(ccv[p2]) and ccv[p2] > ccv[p1]
            mhid=np.isfinite(mhv[p1]) and np.isfinite(mhv[p2]) and mhv[p2] > mhv[p1]
        price_struct[i]=ps
        cci_hidden[i]=ps and ch
        macd_hidden[i]=ps and mhid

    masks={
        "candidate_50_61_8":candidate,
        "price_HL_LH":candidate & price_struct,
        "CCI_hidden":candidate & cci_hidden,
        "MACD_hist_hidden":candidate & macd_hidden,
        "CCI_and_MACD_hidden":candidate & cci_hidden & macd_hidden,
    }
    return masks,direction,av


def run_h4_swing(symbol,b,outdir):
    tf_name="H1_H4swing"
    print(f"\n=== {symbol} {tf_name} rows={len(b):,} ===",flush=True)
    masks,direction,av=build_masks_h4_swing(b)
    rows=[]
    for name,mask in masks.items():
        tr=eval_trades(b,mask,direction,av,48)
        s=summarize(tr)
        rows.append({"symbol":symbol,"timeframe":tf_name,"filter":name,**s})
        print(f"H4SW {name:24s} n={s['trades']:5d} win={s['win_pct']:6.2f}% PF={s['pf']:7.3f} avgR={s['avg_r']:+.4f} maxDD={s['max_dd_r']:.1f} timeouts={s['timeouts']}",flush=True)
        if not tr.empty:
            years=pd.to_datetime(tr["signal_time"]).dt.year
            eras=np.select([years<=2018,years<=2022],["2015-2018","2019-2022"],default="2023-2026")
            tmp=tr.copy(); tmp["era"]=eras
            for era,g in tmp.groupby("era"):
                es=summarize(g)
                print(f"H4ERA {name:24s} {era}: n={es['trades']:4d} win={es['win_pct']:6.2f}% PF={es['pf']:7.3f} avgR={es['avg_r']:+.4f}")
    out=pd.DataFrame(rows)
    out.to_csv(outdir/f"{symbol}_{tf_name}_summary.csv",index=False)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbol",choices=["USDJPY","XAUUSD"],required=True)
    ap.add_argument("--workdir",default="_research_work")
    ap.add_argument("--outdir",default="research/results")
    args=ap.parse_args()

    work=Path(args.workdir); work.mkdir(parents=True,exist_ok=True)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)

    z=download_zip(args.symbol,work)
    m1=read_m1(z)
    h1=resample(m1,"1h")
    m15=resample(m1,"15min")
    del m1

    print(f"[bars] H1={len(h1):,}; M15={len(m15):,}",flush=True)
    # H1: original recorded horizon: 30 bars = 30 hours; countertrend lookback ~3 hours; max 48h.
    s1=run_tf(args.symbol,h1,"H1",lookback=30,counter_bars=3,max_hold=48,outdir=out)
    # M15 external variant: preserve the same clock-time horizons: 120 bars=30h, 12 bars=3h, 192 bars=48h.
    s15=run_tf(args.symbol,m15,"M15_time_equiv",lookback=120,counter_bars=12,max_hold=192,outdir=out)
    # Proper Fibonacci validation: confirmed H4 impulse swing A->B, then H1 50-61.8% retracement.
    sh4=run_h4_swing(args.symbol,h1,out)

    combo=pd.concat([s1,s15,sh4],ignore_index=True)
    combo.to_csv(out/f"{args.symbol}_combined_summary.csv",index=False)
    print("\n=== COMBINED SUMMARY ===")
    print(combo.to_string(index=False))
    print("\nNOTE: Bid-only structural screen; spread/Ask execution costs are not included.")
    print("NOTE: This is a reconstructed implementation of the recorded GBP specification, not the lost original code.")

if __name__=="__main__":
    main()
