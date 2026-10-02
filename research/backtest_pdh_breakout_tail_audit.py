#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import download_zip, read_m1, atr_wilder, ema
from backtest_goscal_components import stochastic_533, ichimoku, cross_up, cross_dn

def levels(b):
    day=pd.Series(b.index.normalize(),index=b.index)
    d=b.groupby(day).agg({"High":"max","Low":"min"})
    return (
        pd.Series(day.map(d["High"].shift(1)).to_numpy(float),index=b.index),
        pd.Series(day.map(d["Low"].shift(1)).to_numpy(float),index=b.index)
    )

def signals(b):
    pdh,pdl=levels(b)
    k,d=stochastic_533(b)
    e6=ema(b["Close"],6); e13=ema(b["Close"],13)
    _,_,sa,sb=ichimoku(b)
    top=pd.concat([sa,sb],axis=1).max(axis=1); bot=pd.concat([sa,sb],axis=1).min(axis=1)
    lo=(b["Close"]>pdh)&(b["Close"].shift(1)<=pdh.shift(1))
    sh=(b["Close"]<pdl)&(b["Close"].shift(1)>=pdl.shift(1))
    return {
      "LEVEL_ONLY":(lo,sh),
      "STOCH_CROSS":(lo&cross_up(k,d),sh&cross_dn(k,d)),
      "EMA_ALIGN":(lo&(e6>e13),sh&(e6<e13)),
      "EMA_AND_CLOUD":(lo&(e6>e13)&(b["Close"]>top),sh&(e6<e13)&(b["Close"]<bot)),
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--symbol",choices=["USDJPY","XAUUSD"],required=True)
    a=ap.parse_args()
    work=Path("_research_work"); work.mkdir(exist_ok=True)
    z=work/f"{a.symbol}_bid.zip"
    if not z.exists(): z=download_zip(a.symbol,work)
    b=read_m1(z); atr=atr_wilder(b,14).replace(0,np.nan)
    entry=b["Open"].shift(-1); exitc=b["Close"].shift(-20)

    for name,(lo,sh) in signals(b).items():
        direction=pd.Series(np.where(lo,1,np.where(sh,-1,0)),index=b.index,dtype=float)
        r=(direction*(exitc-entry)/atr)[(lo|sh)].replace([np.inf,-np.inf],np.nan).dropna()
        q01,q99=r.quantile([0.01,0.99])
        winsor=r.clip(lower=q01,upper=q99)
        trimmed=r[(r>=q01)&(r<=q99)]
        total=r.sum()
        top=r[r>q99].sum()
        bottom=r[r<q01].sum()
        print(
          f"TAIL {a.symbol} {name:13s} n={len(r):6d} raw={r.mean():+.5f} "
          f"median={r.median():+.5f} winsor1={winsor.mean():+.5f} trim1={trimmed.mean():+.5f} "
          f"q01={q01:+.3f} q99={q99:+.3f} top1sum={top:+.1f} bottom1sum={bottom:+.1f} total={total:+.1f}",
          flush=True
        )

if __name__=="__main__":
    main()
