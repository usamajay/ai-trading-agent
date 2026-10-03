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

## H5 pre-registration (written before the runs, 2026-10-01)

| Id | Variant | Symbol | Split | Criterion |
|---|---|---|---|---|
| E0021 | goldsr (EA v2.4 defaults frozen), **re-entry ON** (EA default) | XAUUSD | train | standard, percentile vs random trades with the **same long/short mix** |
| E0022 | goldsr, **re-entry OFF** | XAUUSD | train | same |

Standard = ≥ 100 trades, expectancy > 0 R after costs, 95% CI lower bound > 0, ≥ 95th percentile of the mixed baseline (per seed: long-only and short-only gold M15 swing random baselines weighted by the run's long share), cost stress positive. Our risk engine (0.5% risk, 2% daily loss, news blackout, stop 0.5–3 ATR, RR ≥ 2) overrides the EA's own risk settings. The EA's defaults may be optimizer-fitted, so **validation (one attempt) is the real judge** for a variant that passes train. Reported regardless of verdict: exit distribution (SL / breakeven / TP1-lock / TP3 = first stop / stop after move 1 / stop after move 2 / target), long vs short, and the gold uptrend caveat.

## H5 results: GoldSR EA v2.4 on train — 0 of 2 pass

| | E0021 re-entry ON | E0022 re-entry OFF |
|---|---|---|
| Trades | 188 | 171 |
| Expectancy (R, after costs) | **−0.175** | **−0.166** |
| 95% CI lower bound | −0.327 | −0.326 |
| Win rate / profit factor | 40.4% / 0.70 | 40.4% / 0.72 |
| Cost stress (spread × 1.5) | −0.191 | −0.184 |
| Random trades, same long/short mix (mean) | −0.025 | −0.024 |
| Percentile vs that baseline | **0th** | **0th** |
| Long share | 79% | 80% |
| Longs / shorts expectancy | −0.168 / −0.201 | −0.161 / −0.188 |
| Cost per trade | 0.067 R (15.8% of gross profit) | 0.069 R (15.7%) |

**How trades ended (R distribution), E0022 (E0021 in brackets):**

| Exit | Trades | Mean R |
|---|---|---|
| SL (first stop) | 96 (105) | −1.03 |
| Breakeven (stop after TP1) | 17 (20) | 0.00 |
| TP1-lock (stop after TP2) | 40 (43) | +0.96 |
| TP3 (target) | 17 (18) | +1.95 |
| Friday exit | 1 (2) | −0.48 |

56% of trades lose the full 1R; of the trades that reach TP1, most end at the TP1 lock (+1R) and only about one in four reaches TP3. To break even with this exit mix the first-stop share would need to fall to roughly 45%.

**Checks:**
- **Same-bar stop moves:** 22–24 trades had the trigger and the new stop inside one M5 bar; the verdict uses the pessimistic order. As a bound (analysis only, not a counted run), assuming the favourable order gives **−0.093 R (OFF) / −0.098 R (ON)**: still negative and far below random. M1 data does not cover train, so the true value lies between.
- **Risk engine differences from the EA:** 41 of 260 signals (16%, re-entry OFF) were rejected for a stop outside 0.5–3 × ATR, 4–8 for the news blackout; the EA would have taken them. 460 decision bars were skipped because the 600-bar pivot lookback touched the one excluded gold data hole (June 2025).
- **Re-entry** changes little: 17 more trades, about the same expectancy.
- Look-ahead truncation test: PASS on real gold data (20 cuts). Results reproduce exactly on re-run.
- **Caveat (standing rule):** gold rose +77% over train, and 80% of GoldSR's trades were longs; the comparison is against random trades with the same mix. Any MT5 Strategy Tester results seen earlier had no news filter and used 1% risk.

**Conclusion (Usama, 2026-10-03): H5 falsified** (both variants) for the EA's default settings; neither variant goes to validation. Why it cannot work: the breakeven move at TP1 (1R) caps winners at about +1R (most TP1 trades end at the TP1 lock, only about one in four reaches TP3), while 56% of trades lose the full 1R at the first stop, so expectancy cannot beat costs.

## 6.6 Statistical scans (train, 2026-10-01)

`tradeagent research scan`: the latest train run of each strategy/variant and symbol (27 runs), trades grouped by what is known **at entry**: entry hour (UTC), session, weekday, direction, trend regime and volatility regime of the signal bar. Each group (≥ 30 trades, and ≥ 30 in the rest of the run) is compared with the rest of its run by a Welch z-test; a group is flagged only if p < 0.05 / (groups tested) (Bonferroni).

**Result: 213 groups tested, threshold p < 2.35 × 10⁻⁴, no group flagged.** Closest (none near the threshold):

| p | Run | Group | Mean R (n) | Rest mean R (n) |
|---|---|---|---|---|
| 0.0099 | breakout_compression[reg=suited] XAUUSD | hour 02 UTC | +0.630 (31) | −0.118 (182) |
| 0.016 | breakout_compression XAUUSD | hour 02 UTC | +0.521 (43) | −0.062 (357) |
| 0.020 | session_breakout XAUUSD | high volatility | +0.188 (106) | −0.191 (208) |
| 0.027 | session_breakout USOIL | Tuesday | −0.442 (81) | −0.109 (324) |
| 0.027 | breakout_compression USOIL | ranging | −0.244 (301) | +0.020 (234) |

The two gold 02:00 rows share most of their trades (the filtered run is a subset), so they are one weak signal, not two. **Mistake caught and fixed:** the first version also grouped by holding time and weekend hold; it "found" 12 strong groups (fast exits lose), but holding time is only known after the exit, so these are not usable rules. Those dimensions were removed before any hypothesis was stored. No scan hypotheses were added.



| Hypothesis | Status | Evidence |
|---|---|---|
| H1 higher timeframes | **falsified** | cost per trade in R falls M15 → H1 but not H1 → H4 (swap grows with holding time); H4/D1 variants 0/8 pass with 4–61 train trades |
| H2 swap asymmetry | **supported (cost only)** | measured swap cost; strategy tests withdrawn (drift) |
| H3 oil too costly | **supported** | oil costs 2–3× gold at every timeframe; no oil strategy near the 95th percentile |
| H4 regime filters | **falsified** (4 strategies, labeller `research-1`) | 0/8 pass; no improvement beyond noise |
| H5 GoldSR EA | **falsified** (both variants) | E0021/E0022 0/2, −0.17 R, 0th percentile; breakeven at 1R caps winners at ~+1R while 56% lose 1R |

Strict multiple-testing total: **18 experiment runs** (E0001–E0008, E0013–E0022); E0009–E0012 withdrawn, not counted.

**Standing caveat:** gold rose +77.3% over train (2023-10-01 → 2025-06-27). Any direction-biased idea on gold must beat the same-direction random baseline, and this uptrend may not repeat in validation or out-of-sample.

## Results

### H4 regime filters (E0013–E0020): 0 of 8 pass

Each strategy run only when its decision bar is in its declared regime (research labeller `research-1`), compared with its unfiltered Phase 4 train run.

| Strategy | Symbol | Trades (unfiltered → filtered) | Expectancy R (unfiltered → filtered) | vs random | Verdict |
|---|---|---|---|---|---|
| trend_ema_pullback (trending) | XAUUSD | 203 → 60 | −0.040 → +0.013 (CI low −0.333) | 94th | FAIL (trades, CI, stress) |
| breakout_compression (low_vol_expanding) | XAUUSD | 400 → 213 | 0.000 → −0.009 | 91st | FAIL |
| mean_reversion_bb (ranging, low_vol) | XAUUSD | 43 → 43 | −0.208 → −0.208 | 1st | FAIL |
| structure_retest (trending) | XAUUSD | 324 → 154 | −0.075 → −0.115 | 27th | FAIL |
| trend_ema_pullback (trending) | USOIL | 273 → 62 | −0.245 → −0.345 | 0th | FAIL |
| breakout_compression (low_vol_expanding) | USOIL | 535 → 302 | −0.129 → −0.100 | 91st | FAIL |
| mean_reversion_bb (ranging, low_vol) | USOIL | 57 → 57 | −0.121 → −0.121 | 86th | FAIL |
| structure_retest (trending) | USOIL | 308 → 132 | −0.267 → −0.345 | 6th | FAIL |

- mean_reversion_bb is unchanged because its own rule (ADX(14) < 20) is exactly the `ranging` label: the filter adds nothing (checked: all its trades are labelled ranging).
- The filter moves expectancy by less than its noise everywhere (the best, gold trend_ema_pullback, has only 60 trades and a CI from −0.33 R). **Proposed conclusion: H4 falsified** for these four strategies with this labeller.

### Direction-matched random baselines (H1 swing, train, 100 seeds)

| Baseline | Mean expectancy R | 5–95% | Cost per trade R |
|---|---|---|---|
| XAUUSD long only | **+0.113** | −0.044 to +0.269 | 0.066 |
| XAUUSD short only | −0.185 | −0.349 to −0.030 | 0.031 |
| USOIL long only | −0.062 | −0.222 to +0.108 | 0.060 |
| USOIL short only | −0.223 | −0.388 to −0.048 | 0.176 |

**Random gold longs make money on train (86 of 100 seeds positive).** Cause checked: gold rose **+77.3%** over the train period (1846 → 3275), so random longs ride the trend; random shorts lose (2 of 100 positive). The mean of long and short (−0.036 R) matches the both-direction baseline (−0.044 R), as it should if the engine is right. Oil fell −28.4% yet oil shorts still lose more than longs, because shorts pay swap (cost 0.176 R vs 0.060 R per trade). This is **market drift, not a bug**, but it matters: over this train period any long bias on gold looks good, and the H2 gold comparison (short vs long) is dominated by drift rather than swap. Stopped here to ask Usama (CLAUDE.md: random baseline makes clear money).

### H1 higher timeframes (E0001–E0008): 0 of 8 pass

| Variant | Symbol | Trades | Expectancy R | CI low | vs random (H4 swing) | Cost/trade R |
|---|---|---|---|---|---|---|
| trend_ema_pullback H4/D1 | XAUUSD | 9 | +0.245 | −0.712 | 93rd | 0.110 |
| breakout_compression H4 | XAUUSD | 28 | −0.392 | −0.814 | 5th | 0.048 |
| mean_reversion_bb H4 | XAUUSD | 4 | −0.228 | −1.003 | 19th | 0.019 |
| structure_retest H4/D1 | XAUUSD | 61 | +0.139 | −0.212 | 83rd | 0.059 |
| trend_ema_pullback H4/D1 | USOIL | 14 | −0.546 | −1.164 | 6th | 0.414 |
| breakout_compression H4 | USOIL | 25 | −0.046 | −0.581 | 79th | 0.186 |
| mean_reversion_bb H4 | USOIL | 6 | −0.672 | −1.400 | 0th | 0.255 |
| structure_retest H4/D1 | USOIL | 58 | −0.302 | −0.666 | 31st | 0.142 |

Cost per trade in R from the random baselines (100 seeds, train, same stop rule):

| Symbol | M15 intraday | H1 swing | H4 swing |
|---|---|---|---|
| XAUUSD | 0.078 | 0.048 | 0.052 |
| USOIL | 0.174 | 0.117 | 0.139 |

The cost share falls from M15 to H1 but **not** from H1 to H4: stops are wider, but trades are held longer and pay more swap. Every H4/D1 variant has far fewer than 100 trades in 91 train weeks, so none can show an edge here; the two gold results above zero (9 and 61 trades) have CIs far below zero. By the pre-registered rule H1 is falsified. The Phase 3 ideas themselves were not tuned for H4/D1.

### H2 swap asymmetry: answered by measurement

Strategy experiments E0009–E0012 were withdrawn before running (train gold drift confounds direction tests). From the direction-matched H1 swing random baselines: **gold longs pay about 0.035 R per trade more than shorts** (0.066 vs 0.031 R), **oil shorts about 0.116 R more than longs** (0.176 vs 0.060 R). Swap asymmetry is real and large for oil shorts; it says nothing about an edge. For gold the +77% trend outweighs it on train.

### H3 oil costs: supported

Oil costs 0.117–0.174 R per trade depending on timeframe, 2–3× gold (0.048–0.078 R). A strategy needs a gross edge above that just to break even; every oil strategy so far (Phase 3, H1, H4) is below its random baseline's 95th percentile, and every oil expectancy is negative after costs. **Recommendation:** deprioritise oil M15; oil ideas only on H1+ with a strong reason, and preferably long side (no swap).
