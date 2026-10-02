# E1-v1.1 preregistered operational specification

Status: NEW OPERATIONAL VARIANT. This is not claimed to be the missing original E1 implementation.

Purpose:
Operationalize the adopted concept "support/resistance reversal" without post-result tuning.

Frozen rules before first backtest:

## Higher-timeframe trend cells
- H4/H1, H4/M15, D1/H1.
- T1: EMA50/EMA200 direction.
- T2: confirmed HH/HL or LL/LH structure.
- T3: T1 and T2 must agree.
- Only completed higher-timeframe information may be used.

## Lower-timeframe support/resistance
- Pivot width L=2.
- Long support = latest confirmed lower-timeframe pivot low.
- Short resistance = latest confirmed lower-timeframe pivot high.
- A pivot is usable only after its two right-side bars have completed.
- No maximum pivot age is imposed in v1.1.

## Zone
- Zone half-width = 0.25 * ATR14 of the lower timeframe.
- Long setup: previous completed bar's low touches/enters [support-0.25ATR, support+0.25ATR].
- Short setup: previous completed bar's high touches/enters [resistance-0.25ATR, resistance+0.25ATR].

## Reversal confirmation
- Long: current signal bar closes above the previous completed bar high.
- Short: current signal bar closes below the previous completed bar low.

## Execution
- Enter at the next lower-timeframe bar open after the signal bar.
- Initial risk distance = 1.5 * ATR14 at the signal bar.
- SL = 1R.
- TP = 2R.
- Maximum hold = 48 hours.
- One position per symbol; overlapping same-symbol trades are blocked.
- Same-bar TP/SL ambiguity is resolved conservatively with SL first.

## Evaluation
- First pass markets: USD/JPY and XAU/USD.
- Report raw PF, Avg R, max DD, trade count.
- Also report 0.05R and 0.10R synthetic friction.
- Report 2015-2018 / 2019-2022 / 2023-2026 and long/short splits.
- No parameter changes are allowed after seeing the first-pass result; any later alternative is a separately named variant.
