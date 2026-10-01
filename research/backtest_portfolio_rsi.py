#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_hidden_divergence import download_zip, read_m1, resample, eval_trades_rr_nonoverlap, summarize
from backtest_t3_rsi_robustness import build

SYMBOLS=["USDJPY","XAUUSD"]

def load(symbol,work):
    z=work/f"{symbol}_bid.zip"
    if not z.exists():
        z=download_zip(symbol,work)
    m1=read_m1(z)
    h1=resample(m1,"1h")
    del m1
    return h1

def get_trades(symbol,b):
    mask,direction,av=build(b,0.50,0.618,2)
    tr,skip=eval_trades_rr_nonoverlap(b,mask,direction,av,48,2.0)
    tr=tr.copy()
    tr["symbol"]=symbol
    return tr,skip

def max_losing_streak(r):
    best=cur=0
    for x in r:
        if x<0:
            cur+=1; best=max(best,cur)
        else:
            cur=0
    return best

def stats_with_equity(z):
    r=z["R"].to_numpy(float)
    s=summarize(z)
    cum=np.cumsum(r)
    peak=np.maximum.accumulate(np.r_[0.0,cum])
    dd=np.r_[0.0,cum]-peak
    return {
        "trades":len(z),"sumR":float(r.sum()),"avgR":float(r.mean()) if len(r) else np.nan,
        "PF":s["pf"],"maxDD_R":float(-dd.min()),"max_loss_streak":max_losing_streak(r)
    }

def overlap_count(a,b,hours=48):
    ta=pd.to_datetime(a["entry_time"]).sort_values().to_numpy()
    tb=pd.to_datetime(b["entry_time"]).sort_values().to_numpy()
    count=0
    for x in ta:
        if np.any(np.abs((tb-x).astype("timedelta64[h]").astype(int))<=hours):
            count+=1
    return count

def main():
    work=Path("_research_work"); work.mkdir(exist_ok=True)
    out=Path("research/results"); out.mkdir(parents=True,exist_ok=True)
    by={}
    for sym in SYMBOLS:
        b=load(sym,work)
        tr,skip=get_trades(sym,b)
        by[sym]=tr
        s=stats_with_equity(tr)
        print(f"RSIPORT {sym} n={s['trades']} PF={s['PF']:.3f} sumR={s['sumR']:+.2f} avgR={s['avgR']:+.4f} DD={s['maxDD_R']:.2f} lossstreak={s['max_loss_streak']} skip={skip}")
        tr.to_csv(out/f"{sym}_T3_RSI_portfolio_trades.csv",index=False)

    z=pd.concat([by[s] for s in SYMBOLS],ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    s=stats_with_equity(z)
    span=(pd.to_datetime(z["entry_time"]).max()-pd.to_datetime(z["entry_time"]).min()).total_seconds()/(365.2425*24*3600)
    print(f"RSIPORTCOMBO n={s['trades']} PF={s['PF']:.3f} sumR={s['sumR']:+.2f} avgR={s['avgR']:+.4f} DD={s['maxDD_R']:.2f} lossstreak={s['max_loss_streak']} span={span:.3f} trades_per_year={len(z)/span:.2f}")
    print(f"RSIPORTOVERLAP USDJPY_within48h_XAU={overlap_count(by['USDJPY'],by['XAUUSD'],48)}/{len(by['USDJPY'])}")

    z["year"]=pd.to_datetime(z["entry_time"]).dt.year
    for year,g in z.groupby("year"):
        gs=stats_with_equity(g)
        print(f"RSIPORTYEAR {int(year)} n={gs['trades']:2d} PF={gs['PF'] if np.isfinite(gs['PF']) else np.inf:.3f} R={gs['sumR']:+.2f} DD={gs['maxDD_R']:.2f}")

    years=z["year"]
    eras=np.select([years<=2018,years<=2022],["2015-2018","2019-2022"],default="2023-2026")
    z["era"]=eras
    for era,g in z.groupby("era",sort=False):
        gs=stats_with_equity(g)
        print(f"RSIPORTERA {era} n={gs['trades']:2d} PF={gs['PF']:.3f} R={gs['sumR']:+.2f} avgR={gs['avgR']:+.4f} DD={gs['maxDD_R']:.2f}")

    for cost in [0.05,0.10,0.15,0.20]:
        q=z.copy(); q["R"]=q["R"]-cost
        cs=stats_with_equity(q)
        print(f"RSIPORTCOST cost={cost:.2f}R PF={cs['PF']:.3f} sumR={cs['sumR']:+.2f} avgR={cs['avgR']:+.4f} DD={cs['maxDD_R']:.2f}")

    span_years=max(span,1e-9)
    for risk_pct in [0.25,0.50,1.00,1.50,2.00,3.00,5.00]:
        f=risk_pct/100
        eq=1.0; peak=1.0; maxdd=0.0
        for rr in z["R"].to_numpy(float):
            eq*=1+f*rr
            peak=max(peak,eq)
            maxdd=max(maxdd,1-eq/peak)
        cagr=eq**(1/span_years)-1
        print(f"RSIPORTRISK risk={risk_pct:.2f}% final_x={eq:.4f} total={(eq-1)*100:+.2f}% maxDD={maxdd*100:.2f}% CAGR={cagr*100:.2f}%")

    # Simple chronological stress: older two eras as pre-2023, recent era as untouched-style diagnostic.
    pre=z[z["year"]<=2022]
    recent=z[z["year"]>=2023]
    for label,g in [("PRE2023",pre),("RECENT2023_2026",recent)]:
        gs=stats_with_equity(g)
        print(f"RSIPORTHOLD {label} n={gs['trades']} PF={gs['PF']:.3f} R={gs['sumR']:+.2f} avgR={gs['avgR']:+.4f} DD={gs['maxDD_R']:.2f}")

    z.to_csv(out/"portfolio_T3_RSI_USDJPY_XAUUSD.csv",index=False)

if __name__=="__main__":
    main()
