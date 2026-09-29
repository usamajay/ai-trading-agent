# Data notes — market hours, gaps and partial candles

> Measured on 2026-09-29 from Exness `XAUUSDm` / `USOILm` history (Sep 2023 – Sep 2026, M5–D1; M1 last 6 months).
> Binding for Phase 2 (backtesting). Re-measure if the broker or server changes.

## 1. When the market is open

Both symbols follow the US futures (CME) schedule, so rules are written in **New York time**. That makes them follow US daylight saving automatically; in UTC every time shifts by 1 hour twice a year.

| Event | New York time | UTC (US summer / winter) | PKT (summer / winter) |
|---|---|---|---|
| Weekly open | Sun 18:00 (first bar usually 18:00–18:10) | Sun 22:00 / 23:00 | Mon 03:00 / 04:00 |
| Daily break (Mon–Thu) | 17:00 – 18:00 | 21:00–22:00 / 22:00–23:00 | 02:00–03:00 / 03:00–04:00 |
| Weekly close | Fri 17:00 (USOIL often 16:45) | Fri 21:00 / 22:00 | Sat 02:00 / 03:00 |
| US holidays | Early close ~13:00–14:45; some days closed (Good Friday, Christmas, New Year) | | |

Exness server time is UTC (checked), so stored bar times need no conversion.

## 2. Sunday "stub" candles (D1 and H4)

Exness cuts D1 and H4 candles at **00:00 UTC**. The Sunday session (18:00 NY until midnight UTC) is only 1–2 hours long, but it still becomes its own candle:

| | Sunday D1 candle vs a normal weekday (median) |
|---|---|
| XAUUSD | range 11.3 vs ~44 (**26%**), tick volume 10.9k vs ~205k (**5%**) |
| USOIL | range 0.35 vs ~1.9 (**18%**), tick volume 1.1k vs ~45k (**2.5%**) |

It is present in 154 of 157 weeks. The Sunday H4 20:00 candle is the same stub, and the **Friday H4 20:00 candle** is also partial (about 1 hour of trading).

**Why it matters:**
- **ATR is understated.** D1 ATR(14) comes out **10% lower for gold and 13% lower for oil** with stubs included, so stops sized from D1 ATR would be too tight.
- **"Previous day high/low" on Monday is wrong.** In 154 of 157 Mondays, the broker's "previous day" is the Sunday stub, not Friday.
- **Bar-counting indicators become inconsistent.** Bar-counting indicators (EMA(20) on D1, "N-day breakout") see 6 bars per week, one of them fake.
- **Signals can come from a nearly empty candle.** A daily strategy could generate signals on the stub's close, which is based on 1–2 hours of thin trading.

**Rule for backtests (Phase 2):**
1. **Keep the raw broker bars on disk unchanged.** Raw data is the record; anything derived is computed from it.
2. **Build our own "New York close" D1 and H4 bars from H1** for any strategy or indicator that uses D1/H4. The trading day runs 17:00 → 17:00 New York time, which is the standard for FX and metals. Result: exactly **5 daily bars per week**, no stubs. Checked: 155/157/154/154/153 bars on Mon–Fri over 3 years, and D1 ATR matches the ATR of the broker D1 with stubs removed. H4 bars are aligned to 17:00 NY (17, 21, 01, 05, 09, 13 NY) = 6 full bars per day, with no partial Friday bar.
3. **Never generate a signal from a partial (stub) bar** if a strategy must use broker bars.
4. **Add a unit test in Phase 2:** NY-close D1 has no Sunday bars and exactly one bar per trading day.

## 3. Weekend gaps

The market is closed ~49 hours each weekend. Price at Sunday open often differs from Friday's close:

| | Median jump | 90th percentile | Largest | (in H1 ATR units: median / 90% / max) |
|---|---|---|---|---|
| XAUUSD | 3.5 | 21.5 | 98.0 | 0.4× / 2.5× / 11.5× |
| USOIL | 0.29 | 2.10 | 9.06 | 0.8× / 5.7× / 24.5× |

Oil gaps are relatively about twice as large as gold's.

**Rules for backtests:**
1. **Gap-through fills (pessimistic).** If a bar **opens beyond the stop-loss**, the stop fills at that bar's **open**, not at the stop price. This applies after weekends, daily breaks, holidays and data holes. Take-profit fills at the TP price, never at the better open.
2. **Every trade records whether it was held over a weekend.** Reports show weekend-held trades separately, because their loss can exceed 1R (1R = the planned loss if the stop fills exactly).
3. **Scalp and intraday strategies close before the weekend**, by Friday 16:30 NY. This will be a config flag, `flat_before_weekend`. Swing strategies may hold, but their backtest must include the gap risk (rule 1).
4. **Keep the gap in true range / ATR.** The Friday→Sunday price jump is real risk, so it stays in ATR. Never fill in fake bars for the closed hours.
5. **Statistics:** time-based stats use trading bars (252 trading days/year), not calendar time.

## 4. Daily break (1 hour, Mon–Thu)

- **Treat it like a mini-weekend.** The gap-through fill rule applies, and positions stay open across it.
- **Swap** (the overnight fee) is charged at this rollover. MT5 says the triple-swap day is **Wednesday** for XAUUSDm (`swap_rollover3days = 3`). USOILm reports `7`, which is not a normal weekday value; check the Exness contract specs before building the swap cost model.
- **Spreads are wider right after the break and after the weekly open.** Gold's first bar after the break has a median spread of 199 points vs 179 normally. Oil's first bar after the weekend has a median of 26 vs 19 (up to 163). Backtests must use each **bar's own spread**, not the median, and the risk engine's spread filter (≤ 2× median) will block many of these entries anyway.
- **No new entries in the first 15 minutes after an open** (weekly or daily). This will be a config setting, to be confirmed by research in Phase 3.

## 5. Holidays and data holes found by validation

`tradeagent data validate` sorts gaps into **weekend** and **daily break** (normal, counted only), **holiday** (closure longer than normal; logged) and **unexpected** (`gap_intraday`; logged, needs review). Notable unexpected gaps, seen on several timeframes and on both symbols:

| When (New York time) | What | Backtest handling (approved 2026-09-29, see DECISIONS.md) |
|---|---|---|
| Thu 19 Jun 2025 04:50 → Sun 22 Jun | **Data hole**: Friday 20 Jun 2025 is missing (the market traded that day) | **Excluded**: 2025-06-19 08:50 → 2025-06-22 22:05 UTC |
| Fri 6 Dec 2024 → Sun 8 Dec 21:30 | Sunday open ~3.5 h late | **Kept**, treated as a weekend gap |
| Fri 28 Nov 2025 | Real exchange outage (CME, Black Friday); huge spreads (gold 3352 pts = 21× normal) | **No-trade window**: 2025-11-27 23:00 → 2025-11-28 19:45 UTC (no new entries) |
| Several 5–55 min holes (e.g. 3 Jan 2025 06:50, 55 min) | Short broker feed outages; both symbols at the same time | Gap-through rule |

**Rule for data holes:** a backtest must not treat bars on either side of a hole as consecutive. Phase 2 will read `data_quality_log`. Any trade whose holding period overlaps a `gap_intraday` longer than 1 hour is excluded, and signals whose lookback window contains such a hole are skipped, **unless a reviewed decision in `DECISIONS.md` says otherwise** (like the 2024-12-08 late open). New, unreviewed gaps are excluded by default. Exclusions are reported in the backtest results.

## 6. Spikes

Candles larger than 20× the normal ATR are **flagged, not deleted**. The biggest (gold 29 Jan 2026 15:00–15:30 UTC; oil 8–23 Mar 2026) appear at the same moment on M5, M15, H1, H4 and D1, which means they are real market moves, not bad ticks. A single bad tick usually shows on one timeframe only. They stay in the data because they are real risk (**approved 2026-09-29**). If a later check against a second data source (e.g. Dukascopy) shows a spike did not happen, we will record it in `DECISIONS.md` and mask it in the backtest, never by editing the Parquet files.
