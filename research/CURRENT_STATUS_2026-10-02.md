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

## Active development gate

### PDH Momentum M5 v0.1
Motivated by M1 spread failure; separately preregistered.
Goal: test whether increasing the signal/execution timeframe reduces spread/ATR enough for the underlying positive-skew breakout phenomenon to survive exact Bid/Ask execution.

Markets:
- USDJPY
- XAUUSD

Status: RUNNING / DEVELOPMENT ONLY.

## Research discipline

1. Exact Bid/Ask execution is mandatory whenever available.
2. H1 spread-field or synthetic-R friction is discovery-only.
3. Any rule chosen after viewing a result is marked post-hoc/development, not independent validation.
4. Failed markets are not silently removed from a portfolio.
5. Small-sample PF is not sufficient for adoption.
6. No live deployment until realism, temporal stability, cross-market/holdout, and capital-DD gates all pass.
