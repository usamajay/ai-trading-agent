# AI Trading Agent — System Specification

> Version 0.1 · 2026-09-29 · Owner: Usama
> This is the master design document. Every module, rule and phase below is binding.
> If code and this document disagree, fix one of them in the same commit and note why in `docs/DECISIONS.md`.

---

## 0. Plain-English summary

We are building a program that runs on its own, 24/7, and does the work of a careful trading researcher:

1. It **collects** price data for XAUUSD (gold) and USOIL (WTI crude).
2. It **works out what kind of market** it is (trending up, trending down, ranging, very volatile, quiet).
3. It **picks the strategy** that has historically done best in that kind of market.
4. It **estimates the odds** of each possible trade and its expected value (average profit per trade after costs).
5. A separate **risk engine** — which the AI cannot override — decides if the trade is allowed and how big it may be.
6. It **places the trade on paper** (simulated), records everything, and later compares what it expected with what happened.
7. A **research loop** proposes new ideas ("hypotheses"), tests them properly, throws out weak ones, and only suggests a change when the evidence is strong. **A human (Usama) approves every change to live logic.**

Real money is **off** by design until Phase 10, and switching it on needs a manual config change **and** a code-level check that the promotion evidence exists.

---

## 1. Scope

| Item | Initial value |
|---|---|
| Instruments | XAUUSD, USOIL (broker symbols e.g. `XAUUSDm`, `USOILm` on Exness) |
| Broker / platform | Exness via MetaTrader 5 (Python `MetaTrader5` package, **Windows only**) |
| Trading account for testing | **Exness DEMO account** only. Never the real account until Phase 10 |
| Timeframes | M1, M5, M15, H1, H4, D1 stored; strategies declare which they use |
| Styles supported | scalp, intraday, swing |
| Mode flags | `research`, `paper`, `live` (live is locked, see §9) |

Out of scope for v1: crypto exchanges, options, portfolio of >5 symbols, high-frequency (<1 minute) trading.

---

## 2. Architecture

### 2.1 Decision pipeline (runs every new bar)

```
Market Data Service ──► Data Validation ──► Feature Engineering
                                                   │
                                                   ▼
                                        Market Regime Detector
                                                   │
                                                   ▼
                                  Strategy Selector (meta-agent)
                                                   │
                                                   ▼
                                  Signal Generator (active strategies)
                                                   │
                                                   ▼
                          Probability / Expected-Value Engine
                                                   │
                                                   ▼
                  ┌──────── RISK MANAGEMENT ENGINE (hard gate) ────────┐
                  │  deterministic Python, no AI, cannot be bypassed    │
                  └──────────────────────────┬─────────────────────────┘
                                             ▼
                        Execution Engine (paper | live-locked)
                                             │
                                             ▼
                     Trade Monitor ──► Trade Journal ──► Performance Analytics
                                                               │
                                                               ▼
                                                        Learning Engine
```

### 2.2 Research pipeline (runs on a schedule, e.g. nightly / weekends)

```
Observe results ─► Hypothesis Generator (LLM + stats) ─► Experiment Manager
      ▲                                                        │
      │                                                        ▼
      │                                    Backtest (train data only)
      │                                                        │
      │                                                        ▼
      │                                  Validation (validation data)
      │                                                        │
      │                                                        ▼
      │                         Walk-forward + Out-of-sample (touched ONCE)
      │                                                        │
      │                                                        ▼
      │                               Paper trading (min 4 weeks / 50 trades)
      │                                                        │
      │                                                        ▼
      └──────────── Performance review ◄── MANUAL APPROVAL GATE ──► Production registry
```

### 2.3 Modules and folder layout

```
src/tradeagent/
  config.py              # loads config/*.yaml, validates with pydantic
  data/
    mt5_client.py        # connect, fetch bars/ticks, symbol info (Windows)
    historical.py        # bulk download + Dukascopy/CSV import
    validation.py        # gaps, duplicates, spikes, weekend bars, spread sanity
    store.py             # read/write bars to DB/Parquet
  features/
    indicators.py        # EMA, RSI, ATR, VWAP, MACD, Bollinger, ADX
    structure.py         # swing highs/lows, BOS, S/R zones, session H/L
  regime/
    detector.py          # trending/ranging/volatile/quiet/transitional + bull/bear
  strategies/
    base.py              # Strategy interface (see §4)
    trend_ema_pullback.py
    breakout_compression.py
    mean_reversion_bb.py
    session_breakout.py
    ...                  # one file per strategy
  selector/
    meta_agent.py        # picks strategies per regime from registry stats
  probability/
    ev_engine.py         # win prob, P(TP before SL), EV, calibration
  risk/
    engine.py            # THE hard gate (see §6)
    sizing.py            # volatility-adjusted position size
    killswitch.py
  execution/
    paper.py             # simulated fills with spread + slippage model
    live_mt5.py          # locked until Phase 10
  backtest/
    engine.py            # event-driven, bar-by-bar, no look-ahead
    costs.py             # spread, commission, slippage, swap
    walkforward.py
    metrics.py           # see §7
    robustness.py        # sensitivity, Monte Carlo, bootstrap
  research/
    hypothesis.py        # generates hypotheses (Claude API + stats scans)
    experiments.py       # experiment manager, dataset split guard
  registry/
    registry.py          # strategy/model versions + status
  journal/
    journal.py           # trades, rejections, decisions
  monitoring/
    alerts.py            # Telegram / email / n8n webhook
  dashboard/
    app.py               # Streamlit (v1)
  scheduler.py           # APScheduler jobs
  cli.py                 # `tradeagent ...` commands
```

---

## 3. Data layer

### 3.1 Storage
- **v1 (Phases 1–7):** SQLite for journal/registry/experiments + **Parquet files** for price bars (`data/bars/{symbol}/{timeframe}/{year}.parquet`). Zero setup, easy for a beginner.
- **v2 (Phase 8+):** move to PostgreSQL + TimescaleDB when paper trading runs 24/7. Code talks to storage only through `store.py` / repository classes, so the switch is contained.

### 3.2 Core tables (SQLite v1)

| Table | Key columns |
|---|---|
| `bars` (Parquet) | symbol, timeframe, time_utc, open, high, low, close, tick_volume, spread |
| `data_quality_log` | run_id, symbol, timeframe, issue_type, time_utc, details |
| `regimes` | symbol, timeframe, time_utc, trend_state, vol_state, direction, detector_version |
| `strategies` | strategy_id, name, version, params_json, code_hash, status (`research/candidate/validated/paper/approved/production/retired`) |
| `experiments` | experiment_id, hypothesis_id, strategy_id, dataset_split, date_range, git_commit, seed, metrics_json, verdict, created_at |
| `hypotheses` | hypothesis_id, text, rationale, source (`llm`/`scan`/`human`), status, created_at |
| `signals` | signal_id, time_utc, symbol, strategy_id, direction, entry, sl, tp, p_win, ev_r, regime_id |
| `decisions` | decision_id, signal_id, action (`taken`/`rejected`), reason_codes, risk_check_json, explanation_text |
| `trades` | trade_id, decision_id, mode (`paper`/`live`), open_time, close_time, entry, exit, size, sl, tp, fees, slippage, pnl, r_multiple, exit_reason |
| `lessons` | lesson_id, trade_id or experiment_id, text, evidence_json |
| `approvals` | approval_id, strategy_id, from_status, to_status, approved_by, evidence_json, time |
| `risk_events` | time, rule, value, limit, action (block/shutdown/kill) |

Every row that comes from code stores **`git_commit`** and **`config_hash`** so any result can be reproduced.

### 3.3 Data validation rules (must pass before use)
- No duplicate timestamps; monotonic time.
- Gaps flagged (weekend/holiday gaps expected; intraday gaps logged).
- OHLC sanity: `low ≤ open,close ≤ high`; bar range > 20× median ATR flagged as spike.
- Spread recorded and sane (> 0, < 10× median).
- All times stored in **UTC**; display converts to PKT (UTC+5).

---

## 4. Strategy interface

Every strategy is a Python class with the same shape so the backtester, paper trader and selector can treat them the same:

```python
class Strategy(Protocol):
    name: str
    version: str
    style: Literal["scalp", "intraday", "swing"]
    timeframes: list[str]            # e.g. ["M15", "H1"]
    suited_regimes: list[str]        # declared, then verified by stats
    params: dict                     # small, documented, bounded

    def generate(self, ctx: MarketContext) -> list[Signal]:
        """Uses ONLY data up to ctx.now (closed bars). Returns 0..n signals
        each with entry, stop_loss, take_profit, and a short 'why' string."""
```

Rules:
- Max **5 tunable parameters** per strategy; each has a documented range.
- Signals are computed on **closed bars only**; the backtest engine enforces this.
- Every signal carries a human-readable `why` ("EMA20>EMA50>EMA200 on H1, pullback to EMA20 on M15, RSI 45 bounce").

### 4.1 Starting strategy set (Phase 3)

| # | Strategy | Idea | Expected regime |
|---|---|---|---|
| 1 | Trend EMA pullback | Trade with H1 trend on pullback to EMA20/50 | Trending |
| 2 | Compression breakout | Bollinger/ATR squeeze then break with volume > average | Low vol → expanding |
| 3 | Mean reversion | Fade Bollinger 2σ extremes back to mean | Ranging, low vol |
| 4 | Session (London/NY) breakout | Break of Asian range | Any, time-based |
| 5 | Market-structure retest | Break of structure then retest of zone | Trending |
| 6 | Baseline: random entry with same SL/TP | Must be beaten by every strategy | — |

ML-based signals come **after** rule-based strategies work (Phase 6+).

---

## 5. Probability & expected-value engine

For each signal, estimate (from out-of-sample history of that strategy in the same regime, with a calibration model on top):

| Output | How |
|---|---|
| `p_win` = P(TP hit before SL) | Empirical rate in similar conditions, shrunk toward 50% when sample is small (Bayesian Beta prior) |
| `p_loss` | 1 − p_win − p_timeout |
| `avg_win_r`, `avg_loss_r` | From history, in R units (1R = distance to stop) |
| `ev_r` | p_win·avg_win_r − p_loss·avg_loss_r − cost_r |
| `confidence` | Width of the credible interval on p_win (narrow = high) |
| `calibration` | Tracked live: predicted p_win buckets vs actual hit rate (Brier score) |

**Entry rule (defaults, in config):** `ev_r ≥ 0.15`, `p_win lower-bound ≥ 0.40`, `RR ≥ 2.0`, `sample_size ≥ 100 trades` for that strategy/regime. Otherwise → reject and log reason.

A 60% win rate is a **research benchmark, not a target**. Strategies are ranked by out-of-sample expectancy, profit factor, and drawdown — never by win rate alone.

---

## 6. Risk management engine (hard gate)

- Pure deterministic Python in `risk/engine.py`. **No LLM calls, no ML models inside.**
- Every order goes `Execution.submit(order, risk_approval_token)`. Execution **refuses** orders without a valid token issued by the risk engine for that exact order (symbol, side, size, SL).
- The research/AI code has **no import path** to modify risk limits; limits are read from `config/risk.yaml`, which only a human edits. Changes to that file require a note in `docs/DECISIONS.md`.

### 6.1 Default limits (`config/risk.yaml`)

| Rule | Default | Action when hit |
|---|---|---|
| Risk per trade | 0.5% of equity | Size down; reject if min lot exceeds it |
| Max daily loss | 2% | Stop trading until next day (UTC 00:00) |
| Max weekly loss | 4% | Stop until next week |
| Max drawdown from peak | 10% | Full shutdown, needs manual restart |
| Max open positions | 1 (total), 1 per symbol | Reject |
| Consecutive losses | 4 | Pause 24h |
| Correlation | Gold and oil not both open in same direction when 20-day correlation > 0.6 | Reject |
| Stop-loss required | Always; SL distance ≥ 0.5×ATR and ≤ 3×ATR | Reject |
| Min RR | 2.0 (Usama's manual rule; research may test 1.5 in backtests only) | Reject |
| News blackout | No new entries 30 min before/after high-impact USD events | Reject |
| Spread filter | Spread ≤ 2× median | Reject |
| Kill switch | File `KILL` in project root, CLI `tradeagent kill`, or Telegram command | Close all, cancel all, stop |

Position size = `(equity × risk%) / (SL distance × value per point per lot)`, rounded **down** to broker lot step; if < min lot → reject.

---

## 7. Backtesting & validation

### 7.1 Engine requirements
- Event-driven, bar by bar. At bar *t* the strategy sees only bars ≤ *t−1* closed (enforced by the engine, and by a unit test that injects a future-peek and must fail).
- Costs: real historical spread where available, else broker typical spread; commission; slippage model (default 0.2×spread normal, 3×spread in news windows); swap for overnight holds.
- Intrabar ambiguity: if SL and TP both inside one bar, assume **SL hit first** (pessimistic), unless tick data is available.

### 7.2 Data splits (fixed, recorded, never shuffled)

```
|──────── TRAIN 60% ────────|── VALIDATION 20% ──|── OUT-OF-SAMPLE 20% ──|── live paper ──►
```
- Train: build/tune. Validation: choose among variants. **OOS: touched once per candidate**; each touch is logged in `experiments`, and a second touch is blocked by `experiments.py` unless a human overrides with a written reason.
- Walk-forward: rolling windows (e.g. 12 months train / 3 months test, step 3 months).

### 7.3 Metrics reported for every run
Win rate, profit factor, expectancy (R and $), average trade, Sharpe, Sortino, Calmar, max drawdown (% and duration), recovery factor, number of trades, longest losing streak, exposure time, results per regime, per session, per year.

### 7.4 Robustness checks (candidate must pass all)
- ≥ 200 trades in train+validation; ≥ 50 in OOS.
- OOS profit factor ≥ 1.2 and OOS expectancy > 0 after costs.
- Beats the random-entry baseline with p < 0.05 (bootstrap).
- Parameter sensitivity: ±20% change in each parameter keeps PF ≥ 1.1 (no "knife-edge" peaks).
- Monte Carlo (trade order shuffle, 1000 runs): 95th-percentile drawdown ≤ 2× backtest drawdown and within risk limits.
- Works on at least 2 of 3 walk-forward regimes it claims to suit.
- Multiple-testing correction: record how many variants were tried; apply Deflated Sharpe Ratio or Bonferroni-style haircut.

---

## 8. Research & self-improvement loop

`Observe → Hypothesis → Backtest → Validate → Forward test → Evaluate → Learn → Improve → Repeat`

- **Hypothesis generator:** twice a week, sends a structured summary (recent trades, rejected trades, per-regime stats, failing strategies) to the Claude API and asks for ≤ 5 testable hypotheses in a fixed JSON schema: `{statement, rationale, strategy_change, params, expected_effect, how_to_falsify}`. Statistical scans (e.g. "win rate by hour") also produce hypotheses.
- Each hypothesis becomes an **experiment** with a pre-registered success criterion written *before* the test runs.
- The LLM **may** write new strategy code in a sandbox branch; that code must pass tests + the full validation pipeline, and **a human merges it**.
- **Lessons** are written after every closed trade and every experiment (what was expected, what happened, what we learned) and are fed back into the next hypothesis round.
- **Degradation monitor:** rolling 30-trade expectancy and calibration per production strategy; if it falls below the lower 5% band from its OOS distribution → auto-demote to `paper` and alert. (Demotion is automatic; promotion never is.)

### 8.1 Promotion ladder

`research → candidate → validated → oos_passed → paper → approved → production`

| Gate | Requirement |
|---|---|
| research → candidate | Backtest on train looks promising (PF ≥ 1.2) |
| candidate → validated | Validation-set results hold |
| validated → oos_passed | All §7.4 checks pass on OOS |
| oos_passed → paper | Automatic |
| paper → approved | ≥ 4 weeks and ≥ 50 paper trades; paper expectancy within the OOS 90% band; **human review** |
| approved → production | **Human approval** recorded in `approvals` with evidence link |

---

## 9. Execution modes and the live lock

- `mode: paper` (default). Paper engine uses **live MT5 prices from the demo account** but never sends orders, or sends them to the demo account only.
- `mode: live` requires **all** of:
  1. `config/live.yaml` has `enabled: true` and `account_login` matching the connected account,
  2. the strategy is `production` in the registry with an `approvals` row,
  3. environment variable `TRADEAGENT_LIVE_CONFIRM` equals today's date,
  4. startup check passes (risk limits loaded, kill switch absent, account balance matches config).
- If any check fails → the process refuses to start in live mode.

---

## 10. Explainability — every decision answers these

Stored in `decisions.explanation_text` + structured fields:
1. Why did I take / reject this trade? (reason codes + text)
2. What probability and EV did I assign?
3. Which strategy and which regime?
4. What did the backtest/OOS stats predict for this setup?
5. What actually happened? (filled in on close)
6. What did I learn? (lesson row)
7. For strategy changes: which experiments and approvals justify it? (links)

---

## 11. Monitoring, dashboard, alerts

**Dashboard (Streamlit v1):** current regime per symbol, active strategy, open positions, latest signals with p_win/EV, equity curve, daily/weekly/monthly P&L, win rate, PF, Sharpe/Sortino, max DD, per-strategy table, recent trades and rejections with reasons, calibration chart, backtest vs paper comparison, registry status board.

**Alerts (Telegram bot first; n8n webhook optional):** drawdown/daily-loss limit hit, strategy demoted, calibration error rises, regime change, execution error, data gap/spike, MT5 disconnected, process stopped, kill switch used.

---

## 12. Technology stack

| Need | Choice (v1) | Later |
|---|---|---|
| Language | Python 3.11+ | — |
| Env / packages | `uv` (or pip + venv) | — |
| Broker data/execution | `MetaTrader5` Python package (Windows) | Other brokers via adapter |
| Historical data | MT5 history + Dukascopy CSV | Paid tick data |
| Data processing | pandas, numpy, polars (optional) | — |
| Storage | SQLite + Parquet | PostgreSQL + TimescaleDB |
| Indicators | pandas-ta or own implementations (tested) | — |
| Backtesting | Own event-driven engine (+ vectorbt for quick scans) | — |
| Stats / ML | scipy, statsmodels, scikit-learn, LightGBM | — |
| Experiment tracking | MLflow (local) | — |
| LLM (hypotheses, lessons) | Claude API (`anthropic` SDK) | — |
| Scheduling | APScheduler | Prefect |
| Dashboard | Streamlit | FastAPI + React |
| Alerts | Telegram Bot API | n8n workflows |
| Config validation | pydantic + YAML | — |
| Testing / quality | pytest, ruff, mypy | GitHub Actions CI |
| Version control | Git + GitHub | — |
| Runs on | Your Windows PC | Windows VPS (24/7) |

---

## 13. Build phases (each ends with a demo + tests passing)

| Phase | Goal | "Done" means |
|---|---|---|
| 0 | Setup | Python, Git, VS Code, Claude Code, MT5 demo account, repo on GitHub |
| 1 | Market data + database | `tradeagent data fetch` downloads 3+ years M5–D1 for XAUUSD/USOIL; validation report; tests |
| 2 | Backtesting framework | Event engine + costs + metrics; look-ahead test; baseline strategy report |
| 3 | Strategy engine | 5 rule-based strategies + baseline, each with tests and a backtest report |
| 4 | Risk engine | All §6 rules, token gate, kill switch; 100% test coverage on risk module |
| 5 | Probability/EV engine | p_win/EV per signal, calibration report |
| 6 | Hypothesis generation | Claude API loop, experiment manager, split guard, lessons table |
| 7 | Walk-forward + OOS | Full §7 pipeline + promotion ladder in registry |
| 8 | Paper trading | 24/7 on demo prices, journal, regime detector + meta-agent live |
| 9 | Monitoring + self-improvement | Dashboard, alerts, degradation monitor, weekly review report |
| 10 | Controlled live | Only after ≥ 3 months paper success + manual approval; start at minimum size |

Detailed task list for the current phase lives in `docs/PHASE_<n>_TASKS.md`.

---

## 14. Honest expectations

- Most hypotheses will fail. That is the system working, not broken.
- A strategy that looks great in backtest and fails on OOS is the most common outcome; the pipeline exists to catch it cheaply.
- There is no guarantee this system will be profitable. Its main value is to stop bad ideas from reaching real money and to make every decision explainable.
