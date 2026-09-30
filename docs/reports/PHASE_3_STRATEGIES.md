# Phase 3 strategy report: five rule-based strategies on the train split

> Generated 2026-09-30. Reproduce with `uv run tradeagent backtest run --strategy NAME --symbol SYMBOL --split train --seed 1`
> and `uv run tradeagent backtest compare`. Per-run reports: `data/backtests/<run id>/report.md` (git-ignored).

## How they were tested
- **Train split only** (2023-10-01 → 2025-06-29 UTC). Validation untouched; out-of-sample locked.
- **Fixed defaults, no tuning** (parameters in `docs/PHASE_3_TASKS.md`). One run per strategy and symbol, so the multiple-testing counter stands at 2 train runs per strategy.
- Real per-bar spread × 1.1, slippage, swap; temporary risk basics (min RR 2, stop 0.5–3 × ATR, spread filter, 0.5% risk, one position); starting balance $10,000.
- **Look-ahead check: PASS** for all five strategies on both symbols (last 3,000 train bars, 20 cuts).
- **vs random** = where the strategy's expectancy falls among 100 random-baseline runs on the same timeframe and style (M15 intraday, or H1 swing for the structure retest). 50% = no better than random.

## Results

| Strategy | Symbol | Trades | Win rate | PF | Expectancy (R) [95% CI] | Max DD | Costs (% gross / R per trade) | Stress (R) | vs random | Verdict |
|---|---|---:|---:|---:|---|---:|---|---:|---:|---|
| Trend EMA pullback | XAUUSD | 213 | 33.3% | 0.91 | −0.057 [−0.241, +0.124] | 10.6% | 12.5% / 0.080 | −0.087 | 62% | no edge |
| Trend EMA pullback | USOIL | 275 | 27.3% | 0.67 | −0.242 [−0.390, −0.095] | 30.2% | 29.7% / 0.159 | −0.288 | 6% | no edge |
| Compression breakout | XAUUSD | 410 | 33.7% | 0.95 | −0.017 [−0.157, +0.119] | 21.4% | 10.6% / 0.069 | −0.040 | 87% | no edge |
| Compression breakout | USOIL | 535 | 30.5% | 0.80 | −0.129 [−0.237, −0.013] | 38.7% | 31.5% / 0.190 | −0.208 | 82% | no edge |
| Mean reversion (BB) | XAUUSD | 43 | 25.6% | 0.73 | −0.208 [−0.584, +0.228] | 6.5% | 18.2% / 0.105 | −0.230 | 1% | no edge |
| Mean reversion (BB) | USOIL | 57 | 29.8% | 0.81 | −0.121 [−0.478, +0.259] | 6.6% | 33.8% / 0.222 | −0.228 | 86% | no edge |
| Session breakout | XAUUSD | 328 | 32.9% | 0.89 | −0.073 [−0.216, +0.072] | 13.0% | 7.8% / 0.047 | −0.095 | 51% | no edge |
| Session breakout | USOIL | 405 | 29.1% | 0.74 | −0.176 [−0.302, −0.045] | 30.8% | 19.1% / 0.107 | −0.204 | 43% | no edge |
| Structure retest (H1) | XAUUSD | 330 | 31.8% | 0.90 | −0.074 [−0.227, +0.083] | 21.2% | 8.8% / 0.055 | −0.089 | 34% | no edge |
| Structure retest (H1) | USOIL | 315 | 26.7% | 0.65 | −0.255 [−0.400, −0.102] | 33.3% | 21.1% / 0.112 | −0.298 | 15% | no edge |

Risk-limit flags (informational): every run except mean reversion would have breached the 10% drawdown limit; compression breakout on oil also breached the 4% weekly loss.

## What it means
- **No strategy has an edge on train with default settings.** Every expectancy is negative, every one fails the cost stress, and none is clearly better than random (the best, compression breakout, sits at the 82nd–87th percentile, not the 95th+ needed to stand out). This is the expected outcome for untuned textbook rules (SPEC §14): the pipeline is doing its job by saying no cheaply.
- **Gold is closer to break-even than oil** for every strategy. Oil costs 2–3 times more per trade in R (its spread is large against ATR-sized stops), so an oil strategy needs a much bigger edge.
- **Least bad on gold:** compression breakout (−0.017 R, 95% CI includes +0.12 R) and trend pullback (−0.057 R). These are the most natural starting points for Phase 6 hypotheses, e.g. higher timeframes (lower costs per trade in R), different targets, or regime filters.
- **Mean reversion trades rarely** (43–57 trades in 21 months) because it skips setups under 2R to the mean; its results are too uncertain to read much into (CI roughly ±0.4 R).
- Nothing is promoted: promotion is a human decision and needs the full Phase 7 pipeline.

## Baselines used (100 seeds each, train)

| Baseline | Symbol | Expectancy mean (5th–95th) | Cost per trade | Before costs |
|---|---|---|---:|---:|
| M15 intraday | XAUUSD | −0.074 (−0.157 to +0.012) | 0.078 R | +0.004 R |
| M15 intraday | USOIL | −0.167 (−0.246 to −0.078) | 0.174 R | +0.007 R |
| H1 swing | XAUUSD | −0.042 (−0.248 to +0.139) | 0.048 R | +0.006 R |
| H1 swing | USOIL | −0.151 (−0.337 to +0.009) | 0.117 R | −0.034 R |

The oil H1 swing baseline loses slightly more than its costs. About a quarter of that (−0.008 R per trade) is from stops filled at gapped weekend opens (oil has the largest weekend gaps, and the backtest fills stops at the worse open but never fills targets better); the remaining −0.02 R is about one standard error of noise. A look-ahead leak would show as a profit, not a loss.
