# Phase 3 — Strategy engine

**Goal:** the five rule-based strategies of SPEC §4.1 plus the random baseline, each with tests, a look-ahead check and a backtest report, so later phases have real candidates to validate.
**Done means (SPEC §13):** 5 rule-based strategies + baseline, each with tests and a backtest report.

Sources: SPEC §2.3 (features/, strategies/), §4 (interface, ≤ 5 parameters, closed bars, `why`), §4.1 (strategy list), §6 (risk basics applied in backtests), §7 (engine, metrics, robustness). Built on Phase 2 (`docs/PHASE_2_TASKS.md`).

## Research rules for this phase (conservative defaults, see DECISIONS.md)
- **No tuning.** Each strategy runs with fixed, documented defaults taken from common practice. Parameters have ranges (for Phase 6/7 experiments), but Phase 3 does not search them.
- **Train split only.** Validation is left untouched for choosing between variants later; out-of-sample stays locked.
- **Every strategy passes `backtest lookahead-check`** on real data before its results count.
- **Compare with the random baseline on the same timeframe and style** (100-seed distribution): where does the strategy's expectancy fall in the baseline's spread? Informational; the formal p < 0.05 test is Phase 7.
- Results are **reported, not judged** here: most strategies are expected to fail (SPEC §14). Nothing is promoted; promotion is a human action (Phase 7+).
- No news calendar yet, so there is no news filter (known gap, added with the risk engine).

## Tasks

### 3.1 Indicators (`features/indicators.py`)
- EMA, RSI (Wilder), MACD (12/26/9), Bollinger bands (20, 2σ, with bandwidth), ADX (Wilder, 14), session VWAP (anchored at the 17:00 New York trading-day start, tick-volume weighted), rolling average volume. ATR exists (2.1).
- Every indicator uses bars ≤ t only; each is tested against a hand-calculated example and with a truncation test (values do not change when future bars are added).

### 3.2 Market structure (`features/structure.py`)
- Swing highs/lows as fractals of `k` bars on each side, **confirmed only k bars later** (a swing at bar t is known at t + k: no look-ahead).
- Break of structure (close beyond the last confirmed swing), support/resistance levels from confirmed swings, Asian-session range (18:00–03:00 New York) per trading day, previous trading day high/low (New York-close D1).
- Tests: hand-built price paths, confirmation delay, truncation test.

### 3.3 Trend EMA pullback (`strategies/trend_ema_pullback.py`) — SPEC §4.1 #1
- M15 decisions, H1 context, intraday. H1 trend: EMA20 > EMA50 > EMA200 and close above EMA50 (mirror for shorts). M15 pullback: a low touched the M15 EMA20 within the last `pullback_bars`, then RSI(14) crosses back up through `rsi_level` (mirror for shorts).
- Stop `atr_mult` × ATR(14) from the expected entry, target `rr` × stop.
- Parameters: atr_mult 1.5 [1, 3], rr 2.0 [2, 4], rsi_level 45 [35, 55], pullback_bars 5 [2, 10].

### 3.4 Compression breakout (`strategies/breakout_compression.py`) — #2
- M15, intraday. Squeeze: the previous bar's Bollinger bandwidth is in the lowest `squeeze_pct` of the last 100 bars. Break: close above the upper band (long) / below the lower band (short) with tick volume > `vol_mult` × its 20-bar average.
- Stop `atr_mult` × ATR, target `rr` × stop. Parameters: squeeze_pct 0.2 [0.05, 0.4], vol_mult 1.5 [1, 3], atr_mult 1.5 [1, 3], rr 2.0 [2, 4].

### 3.5 Mean reversion (`strategies/mean_reversion_bb.py`) — #3
- M15, intraday. Ranging filter: ADX(14) < `adx_max`. Long when the previous close was below the lower band (`bb_std` σ) and the current close is back inside; mirror for shorts. Target the middle band; stop `atr_mult` × ATR beyond the entry; the strategy skips setups whose reward:risk is below 2 (so the risk check does not have to reject them).
- Parameters: bb_std 2.0 [1.5, 3], adx_max 20 [15, 30], atr_mult 1.0 [0.5, 2].

### 3.6 Session breakout (`strategies/session_breakout.py`) — #4
- M15, intraday. Asian range = high/low of 18:00–03:00 New York. From 03:00 to 11:00 New York, the first M15 close beyond the range (plus `buffer_atr` × ATR) enters in that direction; one trade per trading day. Stop at the range midpoint, kept within 0.5–3 × ATR; target `rr` × stop. Skip days whose range is wider than `max_range_atr` × ATR.
- Parameters: buffer_atr 0.1 [0, 0.5], rr 2.0 [2, 4], max_range_atr 6 [2, 10].

### 3.7 Market-structure retest (`strategies/structure_retest.py`) — #5
- H1 decisions, H4 context, swing (may hold over weekends; gap risk is in the backtest). Trend filter: H4 close above EMA50 (mirror for shorts). Bullish break of structure: H1 close above the last confirmed swing high; then a **buy limit** at the broken level (the retest) valid for `retest_expiry` bars, stop `atr_buffer` × ATR below the level, target `rr` × stop.
- Parameters: swing_k 3 [2, 5], retest_expiry 12 [4, 24], atr_buffer 1.0 [0.5, 2], rr 2.0 [2, 4].

### 3.8 Baselines per timeframe and style
- The random baseline takes a style (intraday / swing) and timeframe. Run 100-seed distributions on train for every combination the strategies use (M15 intraday exists from 2.9; add H1 swing), both symbols.

### 3.9 Strategy report (`docs/reports/PHASE_3_STRATEGIES.md`)
- For each strategy × symbol (train): trades, win rate, expectancy (R) with 95% CI, profit factor, max drawdown, cost stress PASS/FAIL, risk-limit flags, costs % of gross, where the expectancy falls in the matching baseline distribution (percentile), run counts, look-ahead PASS.
- Plain-English verdict per strategy: "worth a Phase 6/7 look" or "no edge on train", without tuning.

### 3.10 Wrap-up
- README, CLAUDE.md commands, SPEC sync, DECISIONS entries; all tests + ruff; commit and push.

## Progress
- ✅ 3.1–3.7 (2026-09-30): indicators, structure, five strategies (all registered), 23 strategy tests (decision rules on hand-set values, look-ahead with real indicators, risk rules never rejecting their signals).
- ✅ Look-ahead check on real data: PASS for all 5 strategies × 2 symbols.
- ✅ 3.8: baselines M15 intraday and H1 swing, 100 seeds × 2 symbols (`data/baselines/`); `RandomBaseline` takes timeframe and style.
- ✅ 3.9: `backtest/compare.py` + `backtest compare`; report `docs/reports/PHASE_3_STRATEGIES.md`. Result: **no strategy has an edge on train with defaults**; all fail the cost stress; none is clearly better than random (best: compression breakout, 82–87th percentile).
- ✅ 3.10: README, CLAUDE.md, DECISIONS.

**Phase 3 is done (2026-09-30)**, except the human review of the strategy report.

## Phase 3 is done when
- 5 strategies + baseline are built, each with tests, a PASS look-ahead check on real data, and a train backtest report for XAUUSD and USOIL.
- The strategy report is written and reviewed with Usama. **(pending: needs Usama)**
