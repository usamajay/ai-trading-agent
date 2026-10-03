# Phase 7 — Walk-forward + out-of-sample framework

**Goal:** the full SPEC §7 validation pipeline and the §8.1 promotion ladder, so that a strategy that passes train can be judged on validation, walk-forward and out-of-sample (OOS) in a fixed, logged order, with every gate decided by code from stored evidence.
**Done means (SPEC §13):** full §7 pipeline + promotion ladder in the registry.

Sources: SPEC §7.2–§7.4, §8.1; CLAUDE.md safety rules 5–6.

## Principles
- **No real strategy touches real OOS in this phase.** Nothing has passed train (Phase 6: 0/18). The pipeline is built and tested on synthetic bars, with a strategy that has a planted, known edge and with the random baseline. The real-data OOS path stays closed until a real candidate reaches `validated`.
- **OOS is opened in exactly one place** (the OOS gate). It refuses unless the candidate is `validated`, the OOS flag was given on purpose, and no earlier OOS touch exists for that candidate. A second touch needs a human override with a written reason. Every touch is logged in `experiments` (SPEC §7.2).
- **Every gate run uses enforced account limits.** Gate code refuses evidence whose `params_json` lacks `"__account_limits": "enforced"` (SPEC §8.1).
- Gate thresholds live in `settings.yaml` → `validation:` (SPEC values). They are promotion rules, not risk limits.
- Promotion is one step at a time. `paper → approved` and `approved → production` are **human only** and recorded in `approvals`. No code path promotes past `paper`.

## Tasks

### 7.1 Controlled OOS unlock (`backtest/splits.py`, runner)
- `split_range(..., allow_oos=False)`. Only the OOS gate passes `True`. Research, `backtest run` and baselines keep refusing OOS. Tests cover every path.

### 7.2 Walk-forward (`backtest/walkforward.py`)
- Rolling windows over train + validation only (default 365 days of history, 91-day test windows, 91-day step; from settings). Strategies have fixed parameters (no per-window re-fit), so the run is made once over the whole span and its trades are cut into test windows by entry time. Trades entering in the embargo gap between train and validation are dropped.
- Per window: trades, expectancy R, profit factor. Pass if ≥ 2/3 of windows with trades have expectancy > 0 after costs. This reads SPEC's "works on at least 2 of 3 walk-forward regimes" as "2 of 3 walk-forward windows"; the windows are reported per regime as well.

### 7.3 Robustness checks (`backtest/robustness.py`)
- Monte Carlo: 1,000 trade-order shuffles of net P&L. The 95th-percentile max drawdown must be ≤ 2× the backtest drawdown and within `max_drawdown_pct`.
- Random baseline: empirical p = (1 + seeds with expectancy ≥ the strategy's) / (1 + seeds), on the same split, timeframe, style and direction. Needs p < 0.05.
- Parameter sensitivity: each parameter ×0.8 and ×1.2 (clipped to its declared range), on train; profit factor must stay ≥ 1.1.
- Multiple testing: Deflated Sharpe Ratio (Bailey & López de Prado) with N = every run of the base strategy (all splits and variants) and the variance of their daily Sharpe ratios. Needs DSR ≥ 0.95.

### 7.4 Registry and promotion ladder (`registry/registry.py`, `registry/gates.py`)
- `strategies` rows: id = name@version#hash(params, variant), plus a code hash of the strategy module.
- Gates (all from stored runs with enforced limits):
  - `research → candidate`: train PF ≥ 1.2.
  - `candidate → validated`: validation expectancy > 0, PF ≥ 1.2, cost stress positive.
  - `validated → oos_passed`: every §7.4 check (≥ 200 trades in train + validation, ≥ 50 OOS, OOS PF ≥ 1.2 and expectancy > 0, OOS cost stress > 0, baseline p < 0.05, sensitivity, Monte Carlo, walk-forward, DSR).
  - `oos_passed → paper`: automatic.
  - `paper → approved` and `approved → production`: human only, with written evidence (the paper-trade checks arrive in Phase 8).
- New table `gate_checks`: every gate evaluation, with each check's required and actual value, pass/fail, and the run ids it used.

### 7.5 CLI (`tradeagent gate ...`)
- `gate register`, `gate status`, `gate check ID` (runs the next automatic gate; the OOS gate also needs `--touch-oos`), `gate approve ID --by NAME --reason "..."`, `gate retire`.

### 7.6 Tests: known-good synthetic strategy and random baseline
- Synthetic M5 bars with a planted, time-of-day momentum edge (direction known at entry; no look-ahead), split into train, validation and OOS. The planted strategy must climb `research → oos_passed → paper` and stop there (human gates refuse it). The random baseline must fail at the first gate. A second OOS touch is refused, and so are runs with flag-mode account limits.

## Status (2026-10-04): framework built
- 7.1–7.6 done, with the `tradeagent gate ...` commands. Tests use a synthetic market: after 02:00, 08:00 and 14:00 UTC the price drifts for 3 hours in the direction of the last closed hour (planted edge; train PF ≈ 2.4).
- The planted strategy passes every gate up to `paper`: OOS PF 2.73 on 65 trades, baseline p 0.048 (it beat all 20 random seeds), walk-forward 8 of 9 windows positive, Monte Carlo 95th-percentile drawdown 5.47% against a 5.84% limit, DSR 1.00. It stops at `paper`; the human steps refuse without a name and a reason.
- The random baseline fails the first gate (train PF 0.76) and cannot retry. A second OOS touch is refused without an override reason. Flag-mode runs fail the enforced-limits check. A test checks that `allow_oos=True` appears only in `registry/gates.py`.
- **No real strategy has been registered, and no real out-of-sample data was read.**
- Paper → approved evidence (≥ 4 weeks, ≥ 50 trades, inside the OOS 90% band) is checked by a human for now; Phase 8 adds the numbers.
