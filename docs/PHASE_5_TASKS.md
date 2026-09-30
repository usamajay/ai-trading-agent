# Phase 5 — Probability / EV engine (lean)

**Goal:** for any signal, estimate `p_win`, its credible interval, and `ev_r` from a history of comparable closed trades, apply the SPEC §5 entry rule, and measure calibration.
**Done means (SPEC §13):** p_win/EV per signal, calibration report. Kept lean on purpose (Usama, 2026-09-30): no strategy has an edge yet, so no model beyond a Beta-binomial, no ML.

Sources: SPEC §5, `config/settings.yaml` `entry_rules`, backtest trade files (`data/backtests/<run>/trades.parquet`).

## Principles
- Deterministic, plain Python (numpy/scipy); no LLM, no ML.
- History is **net of costs** (R multiples after spread, slippage, swap, commission), so `ev_r` needs no separate cost term; an optional `extra_cost_r` covers a signal costing more than history.
- A prediction for a trade uses **only trades closed before that trade's entry** (no look-ahead).
- Out-of-sample runs are refused (split guard; OOS is for Phase 7 gates only).
- Regime: the history is grouped by a key; until the regime detector exists (Phase 8) the key is `all`.

## Tasks

### 5.1 Estimator (`prob/estimator.py`)
- Outcomes: `tp` exit = win, `sl` exit = loss, anything else (weekend/Friday/time exits) = timeout.
- `p_win` = Beta posterior mean with prior Beta(k/2, k/2) (shrinks toward 50%; `prior_strength` k in `settings.yaml`); credible interval at `credible_level` (lower bound used by the entry rule); `confidence` = 1 − interval width.
- `p_timeout` empirical; `p_loss = 1 − p_win − p_timeout`; `avg_win_r`, `avg_loss_r`, `avg_timeout_r` from history (fallbacks when a bucket is empty: planned RR, 1.0, 0.0).
- `ev_r = p_win·avg_win_r − p_loss·avg_loss_r + p_timeout·avg_timeout_r − extra_cost_r`.

### 5.2 Entry gate (`prob/gate.py`)
- SPEC §5 defaults from config: `ev_r ≥ min_ev_r`, `p_win_low ≥ min_p_win_lower`, `n ≥ min_sample_trades`. Every failed rule listed with a reason code (`ev_low`, `p_win_low`, `sample_small`). (RR ≥ 2 is already checked by the risk engine.)

### 5.3 Calibration (`prob/calibration.py`) + CLI
- Walk a run's trades in entry order; predict each from earlier closed trades; record predicted `p_win`, `ev_r`, gate result, and the actual outcome.
- Report: Brier score (vs always-50% and vs the final hit rate), reliability table (10 buckets), predicted vs realised R, and "what the entry gate would have allowed" (trades passed, their realised expectancy).
- `uv run tradeagent prob calibrate --run RUN_ID` (one run) and `--latest` (latest train run of each strategy/symbol) → `docs/reports/PHASE_5_CALIBRATION.md`.

### 5.4 Wrap-up
- Tests for every formula, the gate, the no-look-ahead rule, the OOS refusal. README, CLAUDE.md, SPEC, DECISIONS; pytest + ruff; commit, push.

## Later phases (not now)
- Writing `p_win`/`ev_r` to the `signals` table and the decision journal: Phase 8.
- EV gate inside backtests and regime-conditioned history: Phase 7/8, once a strategy passes train.
