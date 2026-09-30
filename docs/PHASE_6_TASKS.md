# Phase 6 — Hypothesis generation and experiments

**Goal:** turn ideas into pre-registered, logged, reproducible experiments on train (and validation when train passes), starting with the four logged leads in `docs/RESEARCH_HYPOTHESES.md`; then add the statistical scans and the Claude API loop that propose new ones.
**Done means (SPEC §13):** Claude API loop, experiment manager, split guard, lessons table.

Sources: SPEC §2.2, §5, §7.4, §8, `docs/RESEARCH_HYPOTHESES.md`, CLAUDE.md safety rules 5–6.

## Principles
- **Pre-registration:** each experiment's success criterion is written and stored **before** its backtest runs; the verdict is computed by code against that stored criterion, never edited afterwards.
- **Split guard:** experiments run on train; only a train pass may open validation (once per variant); out-of-sample is refused (Phase 7 only). Every run adds to the strategy's run counter (multiple testing, SPEC §7.4).
- **Research runs use flag mode** for account limits (fast, full sample); anything feeding a promotion gate is re-run with `--enforce-account-limits` in Phase 7 (SPEC §8.1).
- Comparisons are against the **random baseline of the same symbol, timeframe, style and direction rule**, with bootstrap CIs; "better" means beyond noise, not a higher point estimate.
- Most hypotheses will fail (SPEC §14). A falsified hypothesis is a result; it gets a lesson row.
- No new strategy parameters are tuned in this phase beyond the variants a hypothesis names up front.

## Tasks

### 6.1 Experiment manager (`research/experiments.py`)
- Tables: add `success_criterion` (JSON, required), `backtest_run_ids`, `status` (`registered/running/done`) to `experiments` (migration). Hypotheses keep `proposed → testing → supported/falsified/inconclusive`.
- `register(hypothesis, variants, criterion)` → rows; `run(experiment_id)` → backtests via the existing runner; `judge()` → verdict from stored metrics vs criterion; writes a lesson.
- Criterion vocabulary (small, typed): expectancy CI lower bound > 0, baseline percentile ≥ 95, stress PASS, min trades, "beats variant X by more than its CI".
- CLI: `tradeagent research load-leads`, `research register`, `research run`, `research list`, `research show ID`.

### 6.2 Split guard (`research/splits_guard.py`)
- One place that decides whether a split may be used: train always; validation only after a recorded train pass of that exact variant (strategy, params, code hash), once; out-of-sample always refused here. Tests for every path, including a second validation attempt.

### 6.3 Direction filter and higher timeframes (for H1–H3)
- A `direction` option (`both`/`long`/`short`) usable by any strategy and by the random baseline, so direction-filtered strategies are compared with direction-filtered baselines.
- Strategies accept H4/D1 as decision timeframe (New York-close bars already exist); random baselines for H4/D1 swing.
- Per-direction and per-timeframe breakdowns in reports (cost per trade in R, swap share).

### 6.4 Regime labels, research-only (for H4)
- `features/regime.py`: deterministic labels on closed bars: trend state (e.g. EMA slope / ADX), volatility state (ATR percentile vs a trailing window), direction. Versioned (`detector_version`), look-ahead truncation test.
- Backtest option `--regimes trending,ranging` to trade only in a strategy's declared `suited_regimes`; reports broken down per regime (fills the empty "per regime" section). The live detector and meta-agent stay in Phase 8.

### 6.5 Run the logged leads (train only)
- **H1 higher timeframes:** Phase 3 ideas on H4/D1 vs H4/D1 baselines; cost per trade in R by timeframe.
- **H2 swap-free directions:** gold short-only and oil long-only swing variants vs direction-matched baselines and vs the opposite direction.
- **H3 oil costs:** break-even edge per timeframe for oil; drop timeframes no strategy could plausibly cover.
- **H4 regime filters:** each strategy only in its declared regimes vs unfiltered and baselines.
- Criteria written into each experiment before running; results and lessons in `docs/reports/PHASE_6_LEADS.md`. Anything that passes train goes to validation (one attempt), then waits for Phase 7.

### 6.6 Statistical scans (`research/scans.py`)
- Deterministic scans over train trades: expectancy by hour, session, weekday, direction, regime, holding time; flags only groups with enough trades and a bootstrap CI clear of the overall mean, with the number of groups tested reported (multiple testing). Each flag becomes a `scan` hypothesis (status `proposed`, not auto-run).

### 6.7 Claude API hypothesis loop (`research/hypothesis.py`)
- Sends a structured summary (experiments, verdicts, lessons, scan flags; **no out-of-sample numbers, no account data**) and asks for ≤ 5 hypotheses in the SPEC §8 JSON schema; validates the JSON; stores them as `llm` hypotheses with status `proposed`. A human (or a later rule) chooses which to register; the LLM never runs backtests or edits strategy code directly in this phase.
- Needs `ANTHROPIC_API_KEY` in `.env`; tests use a fake client (no network). Cost cap per call and a dry-run mode that prints the prompt.

### 6.8 Wrap-up
- README, CLAUDE.md commands, SPEC sync, DECISIONS; pytest + ruff; commit, push; Phase 6 summary.

## Needs Usama's input
- **6.7:** an Anthropic API key in `.env`, and a monthly spend limit (proposal: a few dollars; one call twice a week). Everything before 6.7 needs no key.
- **6.4:** the regime labeller is pulled forward from Phase 8 in research-only form, because H4 cannot be tested without it.
