# PDH Momentum M5 v0.1 — preregistered development screen

Status: NEW DEVELOPMENT HYPOTHESIS. Not independent confirmatory evidence because it is motivated by the M1 PDH breakout discovery and M1 spread failure.

Markets:
- XAU/USD
- USD/JPY

Timeframe:
- Signal/execution bars: M5.
- Bid chart for signal construction.
- Ask/Bid exact execution.

Core event:
- Long: M5 close breaks above prior source-calendar-day high, previous M5 close was not above it.
- Short: M5 close breaks below prior source-calendar-day low, previous M5 close was not below it.

Nested signal variants, all evaluated:
1. LEVEL_ONLY
2. + Stochastic 5/3/3 directional cross
3. + EMA6/EMA13 directional alignment
4. + EMA alignment + Ichimoku 9/26/52 cloud direction

Execution:
- next M5 open
- long entry Ask, short entry Bid
- long exit Bid close, short exit Ask close
- time exits at 15, 30, 60, 120 minutes after signal (3/6/12/24 M5 bars)
- normalize P/L by Bid M5 ATR14 at signal
- no stop/target optimization in this screen

Reporting:
- N, win%, mean/median ATR return, median entry spread/ATR
- broad eras 2015-2018 / 2019-2022 / 2023-2026
- no session filter
- no parameter adjustment after first result
