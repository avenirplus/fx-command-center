#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

from backtest_hidden_divergence import download_zip, read_m1, resample, eval_trades_rr_nonoverlap
from backtest_t1_rsi_robustness import build_t1_rsi

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbol",choices=["USDJPY","XAUUSD"],required=True)
    ap.add_argument("--workdir",default="_research_work")
    ap.add_argument("--outdir",default="research/results")
    args=ap.parse_args()
    work=Path(args.workdir); work.mkdir(parents=True,exist_ok=True)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    z=work/f"{args.symbol}_bid.zip"
    if not z.exists(): z=download_zip(args.symbol,work)
    m1=read_m1(z); h1=resample(m1,"1h"); del m1
    mask,direction,av=build_t1_rsi(h1,0.50,0.618,2)
    tr,skip=eval_trades_rr_nonoverlap(h1,mask,direction,av,48,2.0)
    tr["symbol"]=args.symbol
    tr.to_csv(out/f"{args.symbol}_T1_RSI_base_trades.csv",index=False)
    print(f"TRADEEXPORT {args.symbol} n={len(tr)} skipped={skip}")
    for _,r in tr.iterrows():
        print(f"TRADE {args.symbol} {r['signal_time']} {r['entry_time']} {r['exit_time']} dir={int(r['direction'])} R={r['R']:.8f} reason={r['reason']}")

if __name__=="__main__":
    main()
