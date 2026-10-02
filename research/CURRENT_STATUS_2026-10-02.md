# FX Research — Current Status (2026-10-02)

This file is the authoritative current decision layer.
Older sections in RESULTS_2026-10-01.md are retained for audit history; later realism gates supersede earlier discovery rankings when they conflict.

## Live-trading status

**NO STRATEGY IS APPROVED FOR LIVE TRADING.**

Research capital assumptions do not change this status.

## Current HOLD candidates

### T3 deep pullback + hidden divergence

Core family:
- H4 compound trend
- confirmed H4 impulse
- 50–61.8% retracement
- H1 continuation / HL-LH structure
- hidden divergence
- next-H1 entry
- ATR risk, 2R target

Full 1-minute Bid/Ask realism gate:

USD/JPY:
- T3 + RSI14: 21 trades, PF 1.231, +3R, DD 6R
- T3 + MACD Histogram: 15 trades, PF 1.750, +6R, DD 6R

XAU/USD:
- T3 + RSI14: 14 trades, PF 1.500, +4R, DD 4R
- T3 + MACD Histogram: 16 trades, PF 1.556, +5R, DD 7R

Material weakness:
- both oscillators fail badly in 2019–2022 on one or both exact markets
- frequency is only about 3 trades/year across USDJPY+XAU
- not suitable as a standalone FIRE engine

Classification:
- T3 + RSI: HOLD / WEAK PROVISIONAL
- T3 + MACD Histogram: HOLD / WEAK PROVISIONAL

Do not rescue this family with new filters on the same sample.

## Rejected families

### Formal 27-cell entry family
- E2 EMA20 pullback: REJECTED
- E3 short high/low breakout: REJECTED
- E1 original specification was incomplete
- E1-v1.1 preregistered operational variant: REJECTED

### Method A — H4 20-bar breakout + EMA200
REJECTED.

### Method B — London range breakout
Exploratory USDJPY looked marginal, but intended EURUSD exact validation:
- 2,503 trades
- PF 0.988
- DD about 100.5R
REJECTED.

### T2 + RSI frequency expansion
Exact Gold PF 0.968.
REJECTED.

### M1 previous-day high/low momentum
Bid-only continuation appeared in both XAUUSD and USDJPY and survived tail trimming, but exact execution destroyed it.

XAUUSD EMA+cloud, 20m:
- Bid-only mean +0.1367 M1 ATR
- exact Bid/Ask mean -0.5763
- median entry spread about 0.61 M1 ATR

USDJPY EMA+cloud, 20m:
- Bid-only mean +0.1417 M1 ATR
- exact Bid/Ask mean -0.1398
- median entry spread about 0.19 M1 ATR

REJECT M1 PDH/PDL MOMENTUM.

## GO_SCAL

The real proprietary GO_SCAL method has **not** been faithfully reconstructed.

Confirmed package/template facts:
- EMA 6 / 13
- Ichimoku 9/26/52
- Stochastic 5/3/3
- PrevDayHighLow
- manual Daily/H1 trendlines
- manual Daily/Monthly horizontal lines
- compiled EX5 arrow indicator

Missing:
- EX5 source / arrow logic
- exact time-window rule
- exact manual environment-recognition rule

Therefore:
- negative naive component tests do not reject real GO_SCAL
- GO_SCAL-inspired v0.1 is a separate research variant and must not be presented as the original method

Status: INCOMPLETE SPEC / HOLD.

## PDH Momentum M5 v0.1

Exact Bid/Ask completed. All tested signal variants and 15/30/60/120-minute exits were negative on both USDJPY and XAUUSD.

- USDJPY EMA+cloud: 30m -0.0476 ATR; 60m -0.0457 ATR.
- XAUUSD EMA+cloud: 30m -0.2648 ATR; 120m -0.2348 ATR.

**Status: REJECTED.**

The entire current PDH/PDL momentum family is closed at M1 and M5. Do not tune session/threshold/exit windows on the same sample to rescue it.

## Next independent lane

Hosopi-3 source-fidelity audit and backtest preparation is active. The current public code and the older BB20/2σ configuration are treated as separate specifications.

## Research discipline

1. Exact Bid/Ask execution is mandatory whenever available.
2. H1 spread-field or synthetic-R friction is discovery-only.
3. Any rule chosen after viewing a result is marked post-hoc/development, not independent validation.
4. Failed markets are not silently removed from a portfolio.
5. Small-sample PF is not sufficient for adoption.
6. No live deployment until realism, temporal stability, cross-market/holdout, and capital-DD gates all pass.
