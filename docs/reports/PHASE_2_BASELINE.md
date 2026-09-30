# Phase 2 baseline report: random entries (sanity check of the backtester)

> Generated 2026-09-30 from `uv run tradeagent backtest baseline --symbol <SYMBOL> --seeds 100`
> (files: `data/baselines/<SYMBOL>_M15_intraday_train.parquet`, git-ignored).

## What was run
- **Strategy:** `random_baseline` 1.0, M15, intraday. Each bar: 2% chance of entering, long or short at random (fixed seed per run). Stop 1.5 × ATR(14), target 2 × the stop, both measured from the price actually paid.
- **Data:** train split only, 2023-10-01 → 2025-06-29 (UTC), both symbols. Out-of-sample untouched.
- **Costs:** real per-bar spread × 1.1 safety margin, slippage 0.2 × spread, swap from the MT5 snapshot, no commission. Risk basics on (min RR 2, stop 0.5–3 × ATR, spread ≤ 2 × median, 0.5% risk per trade, one position at a time). Starting balance $10,000.
- **100 seeds** per symbol (seeds 1–100).

## Results

| | XAUUSD | USOIL |
|---|---:|---:|
| Trades per run, median (min–max) | 636 (576–689) | 615 (567–672) |
| **Expectancy per trade, mean over seeds** | **−0.074 R** | **−0.167 R** |
| Expectancy, 5th / 50th / 95th percentile | −0.157 / −0.076 / +0.012 | −0.246 / −0.167 / −0.078 |
| Seeds with positive expectancy | 7 of 100 | 0 of 100 |
| **Cost per trade (spread + slippage + swap), mean** | **0.078 R** | **0.174 R** |
| **Expectancy + cost (the result before costs)** | **+0.004 R** | **+0.007 R** |
| Win rate, mean | 32.7% | 30.7% |
| Profit factor, median | 0.87 | 0.75 |
| Costs as % of gross profit, mean | 13.1% | 30.3% |
| Expectancy with cost stress (spread × 1.5), mean | −0.105 R | −0.225 R |
| Max drawdown (daily mark-to-market), median (worst) | 28.6% (50.2%) | 43.1% (62.7%) |
| Stop-losses filled at a gapped open (`sl_gap`), per run | 1.4 | 2.7 |
| Weekend-held trades (intraday, all seeds) | 0 | 0 |
| Minimum balance for 0.01 lot at 0.5% risk: median / worst trade | $917 / $6,512 | $530 / $2,846 |
| Runs that would breach daily / weekly / drawdown limits | 65 / 48 / 100 | 83 / 87 / 100 |

## Sanity check: PASS
A strategy with no edge should lose about its costs per trade. It does, on both symbols:
- **Before costs the random strategy makes almost exactly zero** (+0.004 R gold, +0.007 R oil, well inside the seed-to-seed spread of ±0.05 R). So the engine neither leaks future information (that would show as a clear profit) nor charges phantom costs (that would show as a loss beyond the listed costs).
- The win rate of ~31–33% with 2R targets matches the break-even rate of 1/3 minus the costs.
- Oil costs about **twice as much per trade in R** as gold (0.174 vs 0.078): its spread is large relative to its ATR-based stops. Any oil strategy needs a larger edge to survive.

## What this means for later phases
- **The baseline to beat** (SPEC §7.4, Phase 7): a strategy's expectancy must beat this seed distribution with p < 0.05. On gold the random 95th percentile is +0.012 R, on oil −0.078 R.
- **The risk limits would stop random trading quickly:** every run breaches the 10% drawdown limit, and most breach the daily or weekly loss limits. The Phase 4 risk engine enforces these.
- **Weekend gaps:** stops filled at a gapped open happen 1–3 times per run; they are why a loss can exceed 1R.

## Found and fixed while running this report
Intraday trades were carried over the weekend on **US holiday early closes** (Black Friday ~13:00) and when **Friday was closed** (Good Friday), because the Friday 16:30 cut-off never came. Intraday trades now also close 30 minutes before any market closure of a day or more (DECISIONS.md, 2026-09-30). After the fix, weekend-held intraday trades went from 62 to 0 (gold, 100 seeds); the averages above barely changed.
