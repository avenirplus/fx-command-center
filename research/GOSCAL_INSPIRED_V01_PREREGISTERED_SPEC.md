# GO_SCAL-inspired v0.1 — preregistered level-context screen

Status: EXPLORATORY NEW VARIANT. It is NOT claimed to reproduce the proprietary GO_SCAL signal logic.

Why this exists:
The distributed MT5 template confirms EMA(6), EMA(13), Ichimoku(9,26,52), Stochastic(5,3,3), and a PrevDayHighLow indicator. It also contains manually drawn Daily/H1 trendlines and Daily/Monthly horizontal lines. The EX5 source and exact arrow logic are unavailable.

Frozen first-pass market/timeframe:
- XAU/USD M1, full available Bid history.
- Source clock/calendar as supplied in the dataset.
- No time-of-day filter in v0.1 because the official preferred hours have not been recovered reliably.

Previous-day levels:
- PDH = prior source-calendar-day High.
- PDL = prior source-calendar-day Low.
- Current day may not use its own future high/low.

Indicator settings:
- EMA6 / EMA13.
- Ichimoku 9/26/52 using only information available at the signal time.
- Stochastic 5/3/3.

Two objective level-event families:

A. False-break reclaim / rejection
- Long: current Low <= PDL and current Close > PDL.
- Short: current High >= PDH and current Close < PDH.

B. Close breakout
- Long: current Close > PDH and previous Close <= PDH.
- Short: current Close < PDL and previous Close >= PDL.

For each family, evaluate these nested confirmations without selecting one after the fact:
1. LEVEL_ONLY.
2. + STOCH_CROSS: K crosses D in trade direction on the signal bar.
3. + EMA_ALIGN: EMA6 > EMA13 long / EMA6 < EMA13 short.
4. + EMA_AND_CLOUD: EMA alignment and Close above/below Ichimoku cloud.

Evaluation:
- Entry reference = next M1 open.
- Forward mark-to-market horizons = 5, 10, 20, 30 minutes.
- Normalize by M1 ATR14 on the signal bar.
- Report N, win%, mean ATR return, median ATR return, and eras 2015-2018 / 2019-2022 / 2023-2026.
- No session selection or parameter changes based on this first pass.
- Any later time-window study is descriptive unless official GO_SCAL hours are independently recovered before testing.
