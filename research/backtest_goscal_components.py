#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import download_zip, read_m1, atr_wilder, ema

def stochastic_533(b):
    ll=b["Low"].rolling(5,min_periods=5).min()
    hh=b["High"].rolling(5,min_periods=5).max()
    raw=100*(b["Close"]-ll)/(hh-ll).replace(0,np.nan)
    k=raw.rolling(3,min_periods=3).mean()
    d=k.rolling(3,min_periods=3).mean()
    return k,d

def ichimoku(b):
    ten=(b["High"].rolling(9).max()+b["Low"].rolling(9).min())/2
    kij=(b["High"].rolling(26).max()+b["Low"].rolling(26).min())/2
    sa=((ten+kij)/2).shift(26)
    sb=((b["High"].rolling(52).max()+b["Low"].rolling(52).min())/2).shift(26)
    return ten,kij,sa,sb

def cross_up(a,b):
    return (a>b)&(a.shift(1)<=b.shift(1))
def cross_dn(a,b):
    return (a<b)&(a.shift(1)>=b.shift(1))

def summarize_signal(name,mask,direction,b,atr):
    o=b["Open"]; c=b["Close"]
    av=atr.replace(0,np.nan)
    idx=np.flatnonzero(mask.to_numpy())
    print(f"\nSIG {name} raw_n={len(idx)}")
    rows=[]
    for h in [5,10,20,30]:
        entry=o.shift(-1)
        exitc=c.shift(-h)
        rr=direction*(exitc-entry)/av
        z=rr[mask].dropna()
        if len(z)==0: continue
        rows.append((h,len(z),float((z>0).mean()*100),float(z.mean()),float(z.median())))
        print(f"FWD {name:28s} h={h:2d} n={len(z):7d} win={100*(z>0).mean():6.2f}% meanATR={z.mean():+.5f} medATR={z.median():+.5f}")

    # 20-min stability by era and 6-hour EET blocks.
    rr20=direction*(c.shift(-20)-o.shift(-1))/av
    zdf=pd.DataFrame({"r":rr20,"sig":mask,"hour":b.index.hour,"year":b.index.year})
    zdf=zdf[zdf.sig & zdf.r.notna()].copy()
    if len(zdf):
        eras=np.select([zdf.year<=2018,zdf.year<=2022],["2015-2018","2019-2022"],default="2023-2026")
        zdf["era"]=eras
        for era,g in zdf.groupby("era",sort=False):
            print(f"ERA20 {name:28s} {era} n={len(g):7d} win={100*(g.r>0).mean():6.2f}% meanATR={g.r.mean():+.5f}")
        zdf["session"]=pd.cut(zdf.hour,bins=[-1,5,11,17,23],labels=["00-05","06-11","12-17","18-23"])
        for sess,g in zdf.groupby("session",observed=True):
            print(f"SES20 {name:28s} EET{sess} n={len(g):7d} win={100*(g.r>0).mean():6.2f}% meanATR={g.r.mean():+.5f}")
    return rows

def main():
    Path("research/results").mkdir(parents=True,exist_ok=True)
    work=Path("_research_work"); work.mkdir(exist_ok=True)
    z=work/"XAUUSD_bid.zip"
    if not z.exists(): z=download_zip("XAUUSD",work)
    b=read_m1(z)
    print("M1",len(b),b.index.min(),b.index.max())

    atr=atr_wilder(b,14)
    e6=ema(b["Close"],6); e13=ema(b["Close"],13)
    k,d=stochastic_533(b)
    ten,kij,sa,sb=ichimoku(b)
    cloud_top=pd.concat([sa,sb],axis=1).max(axis=1)
    cloud_bot=pd.concat([sa,sb],axis=1).min(axis=1)

    ema_up=cross_up(e6,e13); ema_dn=cross_dn(e6,e13)
    st_up=cross_up(k,d); st_dn=cross_dn(k,d)
    cl_up=(b["Close"]>cloud_top)&(b["Close"].shift(1)<=cloud_top.shift(1))
    cl_dn=(b["Close"]<cloud_bot)&(b["Close"].shift(1)>=cloud_bot.shift(1))

    ema_long=e6>e13; ema_short=e6<e13
    cloud_long=b["Close"]>cloud_top; cloud_short=b["Close"]<cloud_bot
    st_long=k>d; st_short=k<d

    signals={
      "EMA6x13_cross": (ema_up|ema_dn, np.where(ema_up,1,np.where(ema_dn,-1,0))),
      "Stoch533_cross": (st_up|st_dn, np.where(st_up,1,np.where(st_dn,-1,0))),
      "Cloud_break": (cl_up|cl_dn, np.where(cl_up,1,np.where(cl_dn,-1,0))),
      "Stoch_x_EMA_align": ((st_up&ema_long)|(st_dn&ema_short), np.where(st_up&ema_long,1,np.where(st_dn&ema_short,-1,0))),
      "Stoch_x_Cloud_align": ((st_up&cloud_long)|(st_dn&cloud_short), np.where(st_up&cloud_long,1,np.where(st_dn&cloud_short,-1,0))),
      "Stoch_x_EMA_x_Cloud": ((st_up&ema_long&cloud_long)|(st_dn&ema_short&cloud_short), np.where(st_up&ema_long&cloud_long,1,np.where(st_dn&ema_short&cloud_short,-1,0))),
      "EMA_cross_x_Stoch_align": ((ema_up&st_long)|(ema_dn&st_short), np.where(ema_up&st_long,1,np.where(ema_dn&st_short,-1,0))),
      "EMA_cross_x_Cloud_align": ((ema_up&cloud_long)|(ema_dn&cloud_short), np.where(ema_up&cloud_long,1,np.where(ema_dn&cloud_short,-1,0))),
      "Cloud_break_x_EMA_Stoch": ((cl_up&ema_long&st_long)|(cl_dn&ema_short&st_short), np.where(cl_up&ema_long&st_long,1,np.where(cl_dn&ema_short&st_short,-1,0))),
    }

    out=[]
    for name,(mask,direction) in signals.items():
        direction=pd.Series(direction,index=b.index,dtype=float)
        rows=summarize_signal(name,mask.fillna(False),direction,b,atr)
        for h,n,win,mean,med in rows:
            out.append({"signal":name,"h_min":h,"n":n,"win_pct":win,"mean_atr":mean,"median_atr":med})
    pd.DataFrame(out).to_csv("research/results/XAUUSD_GO_SCAL_components_forward.csv",index=False)

if __name__=="__main__":
    main()
