# Phase 6 — logged leads H1–H5 (train only)

## Pre-registration (written before any experiment ran, 2026-09-30)

All experiments are on **train**, seed 1, account limits in flag mode (research; SPEC §8.1 re-runs promotion evidence with limits enforced). Every one counts in the strict research total and under its base strategy's run counter.

**Standard criterion** (all must hold): ≥ 100 trades; expectancy > 0 R after costs; 95% bootstrap CI lower bound > 0; ≥ 95th percentile of the matching random baseline (same symbol, timeframe, style, direction); cost stress (spread and slippage × 1.5) still positive.

| Id | Hypothesis | Variant | Symbol | Criterion |
|---|---|---|---|---|
| E0001 | H1 | trend_ema_pullback on H4/D1, swing | XAUUSD | standard |
| E0002 | H1 | breakout_compression on H4, swing | XAUUSD | standard |
| E0003 | H1 | mean_reversion_bb on H4, swing | XAUUSD | standard |
| E0004 | H1 | structure_retest on H4/D1 (swing) | XAUUSD | standard |
| E0005–E0008 | H1 | same four | USOIL | standard |
| E0009 | H2 | structure_retest H1, **long only** (pays swap) | XAUUSD | standard |
| E0010 | H2 | structure_retest H1, **short only** (pays swap) | USOIL | standard |
| E0011 | H2 | structure_retest H1, **short only** (no swap) | XAUUSD | standard + CI lower bound above E0009's expectancy |
| E0012 | H2 | structure_retest H1, **long only** (no swap) | USOIL | standard + CI lower bound above E0010's expectancy |
| E0013–E0016 | H4 | trend_ema_pullback, breakout_compression, mean_reversion_bb, structure_retest, each only in its declared regimes | XAUUSD | standard |
| E0017–E0020 | H4 | same four | USOIL | standard |

Not run as strategy experiments (no run counted):
- **session_breakout** is left out of H1 (its Asian-range logic is tied to time of day, not to bar size) and of H4 (it declares `any` regime).
- **H3 (oil costs)** is answered from cost-per-trade data already measured (random baselines at M15, H1, H4), not by new strategy runs.
- **H5 (GoldSR EA)** waits for Usama's rules.

## Results

(filled in after the runs)
