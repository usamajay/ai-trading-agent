# Phase 2 — Backtesting framework

**Goal:** a trustworthy "time machine" that replays past bars one at a time, lets a strategy trade them with realistic costs, and reports honest results.
**Why now:** every strategy in Phase 3 will be judged by this engine. If the engine is wrong (peeks at the future, ignores costs, fills stops too kindly), every later result is wrong too.

Sources: SPEC §4 (strategy interface), §6 (sizing formula, spread filter), §7 (engine rules, splits, metrics), `docs/DATA_NOTES.md` (Sunday stubs, gap fills, holes, spreads), approved data decisions in `docs/DECISIONS.md` (2026-09-29).
Plan approved by Usama on 2026-09-30, with the additions marked **(added)**.

## Scope

**In Phase 2:** market calendar, New York-close D1/H4 bars, data splits + exclusions, strategy interface (market, limit and stop orders), cost model, event engine, look-ahead tests, metrics, saved results + report, random-entry baseline.

**Not in Phase 2 (later phases):**
- Real strategies (Phase 3). Only the random baseline and small test-only strategies here.
- The real risk engine and approval tokens (Phase 4). The backtest has a small, clearly labelled **temporary** copy of the basic rules (task 2.5), read-only from `risk.yaml`, replaced by `risk/` in Phase 4.
- Walk-forward, robustness, Monte Carlo, out-of-sample runs (Phase 7). **The out-of-sample split is locked in Phase 2**: the engine refuses to run on it.
- Results per regime (needs the regime detector, Phase 8). The metrics table has the column ready, empty for now.
- News-window slippage (3× spread) and news blackout: we have no news calendar yet. Recorded as a known gap, added with the risk engine.

## Fixed inputs (answers from Usama, 2026-09-30)
- **Starting balance:** 10,000 USD for every backtest.
- **Swap values** are read from MT5 (`swap_mode`, `swap_long`, `swap_short`, `swap_rollover3days`) and saved in the cost snapshot (task 2.4). Read on 2026-09-30 (demo account): XAUUSDm long −560 pts, short 0, triple swap Wednesday; USOILm long 0, short −186.2 pts, triple-swap day reported as `7` (not a valid weekday) → **provisionally Wednesday** until Usama confirms it in the MT5 Specification window. Dollar conversion re-checked 2026-09-30 three ways (tick_value × point ÷ tick_size, contract size × point, MT5 profit calculator): $0.10 per point per lot for XAUUSD (contract 100 oz), $1.00 for USOIL (contract 1,000 barrels), so −$56 and −$186.20 per lot per night. The USOIL value is ~76%/year of the position's value (gold ~4.8%), which looks unusually high: Usama checks "Swap short" in the MT5 Specification window. Swap rates change over time and we only have today's, so reports show swap as its own cost line.
- **Split dates** are shown to Usama and approved before any backtest uses them (task 2.2).
- **M1 is only in out-of-sample.** M1 starts 2026-03-29, after the out-of-sample start (2026-02-22), so train and validation have no M1 bars. Scalp research trains on **M5**; M1 is used only for fills (2.5), and live/paper M1 keeps accumulating.

## Tasks

### 2.1 Market calendar + New York-close D1/H4 bars
- Move the New York market-hours logic out of `data/validation.py` into `data/market_hours.py` so validation and the backtest share one definition (weekly open/close, daily break, "first N minutes after an open", "Friday 16:30 NY" cut-off).
- `data/resample.py`: build D1 and H4 bars from stored H1 bars on a **17:00 New York** day boundary (H4 at 17, 21, 01, 05, 09, 13 NY). Spread = the median of the H1 spreads inside; tick volume = sum. Raw broker D1/H4 files stay unchanged on disk.
- Bars are never "bridged" across a data hole: a D1/H4 bar missing hours is marked `partial = True`.
- Tests (from DATA_NOTES §2 rule 4): NY-close D1 has **no Sunday bars and exactly one bar per trading day**; H4 has 6 bars per full day and no partial Friday bar; OHLC values match a hand-built example; daylight-saving weeks work.
- ✅ Check: `uv run tradeagent data resample-check --symbol XAUUSD` prints bars per weekday (expected ~153–155 each, 0 on Sunday) and D1 ATR vs broker D1 ATR.

### 2.2 Data splits + exclusions
- **Freeze the split dates.** The 60/20/20 split is computed once from the stored date range, **shown to Usama for approval**, and saved as fixed dates in a new `config/splits.yaml`. Why fixed: new bars arrive every day; if boundaries were recalculated each run, recent out-of-sample data would slowly slide into train.
- New `config/data_exclusions.yaml`, a machine-readable copy of the approved decisions: excluded window (2025-06-19 08:50 → 2025-06-22 22:05 UTC), no-trade window (2025-11-27 23:00 → 2025-11-28 19:45 UTC), "keep" override (2024-12-08 late open).
- `backtest/dataset.py` loads bars for one symbol/timeframe/split and adds per-bar flags:
  - `excluded`: inside an excluded window, or inside a `gap_intraday` > 60 min from `data_quality_log` that has no "keep" override (**unreviewed gaps are excluded by default**);
  - `no_new_entries`: no-trade window, first 15 min after a weekly/daily open, after Friday 16:30 NY for scalp/intraday;
  - `gap_before`: the previous bar is not the normal next bar (weekend, daily break, hole).
- `split = out_of_sample` raises an error in Phase 2.
- Tests: synthetic bars with a hole, a no-trade window and a "keep" gap produce the right flags; loading OOS fails.
- ✅ Check: `uv run tradeagent backtest splits` prints the split dates for both symbols and how many bars are excluded, and why.

### 2.3 Strategy interface (`strategies/base.py`) + ATR
- `Strategy` protocol exactly as SPEC §4 (name, version, style, timeframes, suited_regimes, params ≤ 5 with documented ranges, `generate(ctx)`).
- `Signal`: symbol, direction, stop_loss, take_profit, optional `max_hold_bars`, `why` text, and **(added)** an order type:
  - `market`: fill at the next bar's open;
  - `limit` (buy below / sell above the current price) and `stop` (buy above / sell below) with an `entry_price` and an **expiry** (`expiry_bars`); an unfilled order is cancelled at expiry, at the Friday cut-off for scalp/intraday, and when an excluded window starts.
- `MarketContext`: `now` + read-only bars that **physically contain only closed bars up to t** (numpy views, so no copying on every bar). Asking for a bar after `now` raises an error.
- `features/indicators.py`: only **ATR** for now (the baseline needs it for stop distance); the other indicators come in Phase 3. Tested against a hand-calculated example.
- ✅ Check: tests pass; a strategy missing `why`, with > 5 params, or a limit/stop order without entry price or expiry is rejected.

### 2.4 Cost model (`backtest/costs.py`)
- **(added) Bid/ask.** MT5 bars are **bid** prices; ask = bid + spread.
  - **Long:** enters at the **ask**; its SL/TP trigger and it exits on the **bid** (bar low/high as stored).
  - **Short:** enters at the **bid**; its SL/TP trigger and it exits on the **ask** (bar high/low + spread).
  - Limit/stop entries use the same sides: a buy order is checked against the ask, a sell order against the bid.
- **(added) Spread safety margin.** An MT5 bar's `spread` is usually the **minimum** spread seen in that bar, so it understates the real cost. Spread used in the backtest = `bar spread × spread_margin_multiple + spread_margin_points` (both in `settings.yaml`). A new command `uv run tradeagent backtest spread-check` measures, from the last few weeks of tick data, how the average tick spread inside each bar compares with the bar's recorded spread, and proposes the default; Usama approves the value.
- Slippage: 0.2 × the (margined) spread on market entries, stop entries and stop-loss exits; take-profits and limit entries fill at their price (no slippage, no improvement), except when a bar opens beyond them (see 2.5).
- Commission: per-lot setting, **0 for the Exness Standard account** (its cost is inside the spread).
- Swap (overnight fee): charged when a position is held through the 17:00 NY rollover, in points per lot from the snapshot; triple on the triple-swap day (USOIL provisionally Wednesday, see "Fixed inputs").
- A new command `uv run tradeagent backtest costs --snapshot` saves a **dated cost snapshot** from MT5 (point, tick value, contract size, lot min/step, swap mode/long/short/triple day) to `config/costs.yaml`, so backtests are repeatable and run without MT5 open. This adds read-only fields to `SymbolSpec` (no order functions).
- Tests: long/short entry and exit prices including the margin, a short's SL triggering on ask but not on bid, a 3-night hold charges 3 swaps, a Wednesday-night hold charges triple.

### 2.5 Event engine (`backtest/engine.py`)
- Loop bar by bar. After bar *t* closes: (1) the strategy sees bars ≤ *t*; (2) a new market signal fills at **bar t+1's open, never at bar t's close** **(added: dedicated test)**; limit/stop orders become active from bar *t+1*.
- Fill rules (SPEC §7.1, DATA_NOTES §3–5):
  - **(added) SL and TP both inside one bar:** look at the **lower-timeframe bars inside it** (M1 where stored, i.e. the last 6 months; otherwise M5) and take whichever level was touched first. Use SL-first (pessimistic) only if that is still unclear (both inside the same lower bar, lower bars missing, or no lower timeframe). The same check is used when a limit/stop entry and the SL/TP happen in the same bar. The report counts how many were resolved by M1, by M5, and by the SL-first fallback.
  - Bar **opens beyond the SL** (after a weekend, daily break, holiday or hole) → fill at that **open**, so the loss can be more than 1R. TP never fills at a better open. A stop entry that the bar opens beyond fills at the open (worse); a limit entry that the bar opens beyond fills at the open (better, which is what a real limit order does).
  - No new entries or pending orders on `no_new_entries` bars; no signals when the lookback window touches an `excluded` bar; trades that would be open during an excluded window are dropped and **counted in the report**.
  - `flat_before_weekend`: scalp/intraday trades close, and their pending orders are cancelled, at Friday 16:30 NY.
- **(added) Risk basics, temporary until Phase 4** (`backtest/risk_basics.py`, read-only from `config/risk.yaml`, on by default, clearly labelled): min RR 2.0, max 1 open position (a pending order counts as the slot), SL distance 0.5–3 × ATR, spread ≤ 2 × median spread. Rejected signals are counted by reason. Used by every backtest from Phase 3 on; Phase 4 swaps it for the real risk engine.
- Size: `(equity × 0.5%) / (SL distance × value per point per lot)`, rounded **down** to the lot step; below min lot (0.01) → skipped and counted. Starting balance 10,000 USD.
- Each trade records: order type, signal time, entry/exit time and price, size, SL, TP, costs (spread, slippage, swap, commission), P&L in $ and in **R** (1R = planned loss if the stop fills exactly), exit reason (`tp`/`sl`/`sl_gap`/`weekend_close`/`time`/`end_of_data`), how an intrabar tie was resolved, `held_over_weekend`, and **(added)** the **minimum balance** needed to take it at 0.01 lot and 0.5% risk (= SL distance × value per point for 0.01 lot ÷ 0.005).
- New `backtest:` section in `config/settings.yaml` (starting balance, spread margin, slippage multiple, no-entry minutes after open, Friday cut-off, max hole minutes). `risk.yaml` is **not** touched.
- Tests with tiny hand-built bar sets where the right answer is worked out by hand: signal on bar t fills at t+1 open (not t close), TP hit, SL hit, both in one bar resolved by M1, resolved by M5, unresolved → SL first, gap through SL, limit fill, stop fill, order expiry, weekend close + pending cancel, swap, excluded window, each risk-basics rejection, zero-cost run where the sum of trade P&L equals the equity change.

### 2.6 Look-ahead protection tests
- A "cheating" test strategy that reads the next bar must **fail** (the engine raises an error) (SPEC §7.1).
- **Truncation test:** signals produced on data cut at time T equal the signals on the full data up to T. If a strategy's past signals change when future data is added, it is peeking.
- The lower-timeframe tie check (2.5) is only used for fills, never shown to the strategy; a test checks this.
- These run in every `pytest` run from now on (CLAUDE.md: keep the look-ahead test passing).

### 2.7 Metrics (`backtest/metrics.py`)
- Everything in SPEC §7.3: win rate, profit factor, expectancy (R and $), average trade, Sharpe, Sortino, Calmar, max drawdown (% and duration), recovery factor, number of trades, longest losing streak, exposure time; breakdowns per session (Asia/London/NY), per year, weekend-held vs not. Per-regime left empty until the detector exists.
- **(added) Minimum balance:** for the run's trades, the largest, 95th-percentile and median minimum balance needed at 0.01 lot and 0.5% risk, and how many trades a 10,000 USD account could not take.
- Sharpe/Sortino use daily returns on **252 trading days/year** (DATA_NOTES §3 rule 5).
- Tests: each metric on a small trade list with a hand-calculated answer; edge cases (no trades, no losses).

### 2.8 Saving results + CLI + report
- New SQLite table `backtest_runs`: run_id, strategy, version, params, symbol, timeframe, split, date range, seed, git_commit, config_hash, cost snapshot hash, metrics_json, exclusion/rejection/tie-resolution counts. Backtest trades go to `data/backtests/{run_id}/trades.parquet`, **not** to the `trades` table (that one is for paper/demo/live trades).
- `uv run tradeagent backtest run --strategy random_baseline --symbol XAUUSD --timeframe M15 --split train --seed 1` prints a summary and writes `data/backtests/{run_id}/report.md` + an equity curve picture (adds `matplotlib`). **Every report shows the minimum balance section** (2.7).
- `uv run tradeagent backtest list` shows past runs.
- ✅ Check: running the same command twice gives identical results (same seed, same data, same config_hash).

### 2.9 Random-entry baseline + baseline report
- `strategies/baseline_random.py` (SPEC §4.1 #6): random entry direction and timing (fixed seed), market orders, SL = 1.5 × ATR, TP = 2 × SL distance.
- Run it on the **train split** for XAUUSD and USOIL, 100 seeds each, and write `docs/reports/PHASE_2_BASELINE.md`: average expectancy, spread of results across seeds, cost per trade in R, weekend-held trades, minimum balance.
- **Sanity check:** a random strategy should lose about its costs per trade (expectancy ≈ −cost in R). If it makes clear money, something is wrong (a leak or a cost bug) and we stop and investigate. The 100-seed spread is kept: Phase 7 needs it for "beats the random baseline with p < 0.05".

### 2.10 Wrap-up
- README "How to run" updated with the `backtest` commands; CLAUDE.md useful-commands list updated.
- `docs/DECISIONS.md` entries for every choice above that is not literally in SPEC (fixed split dates, exclusions file, spread margin, lower-timeframe tie rule, temporary risk basics, OOS lock, no news slippage yet, USOIL triple-swap day, backtest_runs table).
- All tests + ruff pass; commit and push.

## Phase 2 is done when
- Engine, costs and metrics built, with hand-checked tests; look-ahead, truncation and t+1-fill tests pass.
- Baseline report for both symbols reviewed with Usama, and the sanity check passes (random ≈ −costs).
- Usama can explain in his own words: what 1R is, why stops can lose more than 1R over a weekend, and why the random baseline should lose money.

## Progress
- ✅ 2.1 (2026-09-30): `data/market_hours.py` (shared with validation), `data/resample.py`, ATR in `features/indicators.py` (moved forward from 2.3), `tradeagent data resample-check`. Real data: NY-close D1 bars per weekday Mon 155 / Tue 157 / Wed 154 / Thu 154 / Fri 153 / Sun 0; median D1 ATR(14) XAUUSD 42.0 (broker with stubs 37.2, without 41.3), USOIL 2.11 (1.81 / 2.07). Partial days are US holidays, the 2024-12-09 late open, the 2025-06-19 hole, the 2025-11-28 outage and the data edges.

- 🟡 2.2 (2026-09-30): built and tested; **waiting for Usama to approve the split dates** before freezing. `config/data_exclusions.yaml` (approved decisions), `backtest/splits.py` (whole-week proposal with 2-week embargo gaps, freeze-once, OOS lock), `backtest/dataset.py` (per-bar flags), `find_gaps()` shared with validation, `tradeagent backtest splits`. Proposed (UTC, [start, end), Sundays): train 2023-10-01 → 2025-06-29 (91 wk), embargo → 2025-07-13, validation → 2026-02-08 (30 wk), embargo → 2026-02-22, out-of-sample → 2026-09-27 (31 wk); later bars are forward data.

## Still open
- USOIL triple-swap day: MT5 reports `7`; confirm on the Exness contract page (Wednesday used until then).
- Spread safety margin value: measured in task 2.4, approved by Usama.
- **Split dates (2.2):** proposed below, waiting for Usama's approval before `--freeze`.
- **Lower-timeframe tie rule vs splits (2.5):** because M1 exists only in out-of-sample, "M1 where available" would resolve SL/TP ties more precisely (usually less pessimistically) in OOS than in train/validation, for the same strategy. Proposal: use the **same** tie source across all splits (M5 for strategies on M15 and above; SL-first for M5 strategies) and report M1 only as a side-by-side check. Usama decides before 2.5.
- **Entries on D1/H4 bars (2.5):** every NY-close D1/H4 bar opens right after the daily break, so the "no entries in the first 15 minutes after a reopen" rule blocks every D1 open and the 17:00 H4 open. Proposal for 2.5: such entries fill 15 minutes later using the M5 open, instead of being skipped.
- **USOIL unreviewed gaps (2.2):** 5 unexpected gaps > 1 h, all in train, excluded by default (listed by `tradeagent backtest splits`). Usama may review them; excluding is the safe default.
