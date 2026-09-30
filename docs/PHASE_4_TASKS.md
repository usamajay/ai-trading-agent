# Phase 4 — Risk engine (the hard gate)

**Goal:** every SPEC §6 rule in deterministic Python, an approval-token gate that execution cannot bypass, a kill switch, and a news calendar for the blackout rule.
**Done means (SPEC §13):** all §6 rules, token gate, kill switch; **100% test coverage on the risk module**.

Sources: SPEC §6 (rules, limits, token), §9 (execution modes), §10 (explainability), §3.2 (`risk_events`), CLAUDE.md safety rules 1–3, `config/risk.yaml` (read only, never edited).

## Principles
- `src/tradeagent/risk/` is plain deterministic Python: **no LLM calls, no ML, no network**. It reads `risk.yaml` through the frozen config; nothing in the code can change a limit.
- A decision lists **every** failed rule (not just the first) with a reason code and text, for the decision journal (SPEC §10).
- Execution (`execution/paper.py`) **refuses any order without a valid token** issued by the risk engine for that exact order (symbol, side, size, stop-loss), unexpired and unused. Phase 4 execution is simulated only: no MT5 order calls (demo orders arrive in Phase 8).
- Account-level stops use SPEC §6 wording: daily loss resets at **00:00 UTC**; weekly loss resets at the start of the next trading week; the drawdown shutdown needs a **manual restart** (a human command that records who and why).

## Tasks

### 4.1 Risk state (`risk/state.py`)
- Account snapshot and running state: equity and balance, equity peak, equity at the start of the UTC day and of the trading week, consecutive losses, pause-until time, shutdown flag and reason, open positions (symbol, direction, lots, stop).
- Updated on each closed trade and mark-to-market; saved to a new SQLite table `risk_state` for paper/live, kept in memory for backtests. Account-level events go to `risk_events`.

### 4.2 Position sizing (`risk/sizing.py`)
- SPEC §6 formula: `(equity × risk%) / (SL distance × value per point per lot)`, rounded **down** to the lot step, capped at the maximum lot; below the minimum lot → reject. Minimum balance per trade. (Moves the Phase 2 backtest sizing here.)

### 4.3 Rules (`risk/engine.py`)
Every rule of SPEC §6, each with a reason code:

| Rule | Check | Reject code |
|---|---|---|
| Stop-loss required | SL present, on the correct side | `no_stop` |
| Stop distance | 0.5–3 × ATR | `sl_atr` |
| Min reward:risk | ≥ 2.0 from the expected entry | `rr` |
| Spread filter | spread ≤ 2 × median spread | `spread` |
| Risk per trade | 0.5% of equity; size ≥ min lot | `min_lot` |
| Max open positions | 1 total, 1 per symbol | `max_positions` |
| Correlation | no gold and oil in the same direction when 20-day correlation > 0.6 | `correlation` |
| News blackout | no new entries 30 min before/after high-impact USD events | `news` |
| Max daily loss | 2% below the UTC day's starting equity → stop until 00:00 UTC | `daily_loss` |
| Max weekly loss | 4% below the week's starting equity → stop until next week | `weekly_loss` |
| Max drawdown | 10% below the equity peak → full shutdown, manual restart | `max_drawdown` / `shutdown` |
| Consecutive losses | 4 in a row → pause 24 h | `loss_streak` |
| Kill switch | `KILL` file present | `kill_switch` |

### 4.4 Approval tokens (`risk/tokens.py`)
- HMAC-SHA256 over (order id, symbol, side, lots, stop-loss, expiry) with a secret created in memory when the process starts (never stored or logged). Tokens expire (default 60 s) and are **single use**. Verification checks every field against the order being submitted.

### 4.5 Kill switch (`risk/killswitch.py`) + CLI
- Active if the file `KILL` exists in the project root. `uv run tradeagent kill` creates it (with reason and time); `uv run tradeagent kill --status` shows it; clearing needs `--clear --reason "..."` and is logged to `risk_events`. Telegram command comes with alerts (Phase 9).
- When active: the risk engine rejects everything; execution closes all positions and cancels all orders (simulated in Phase 4).

### 4.6 News calendar (`data/news.py`)
- High-impact **USD** events, stored as UTC times. Blackout = 30 min before to 30 min after (`news_blackout_minutes`).
- **Live:** a free weekly feed of this week's events, fetched and cached to `data/news/`; fetch failures are logged and the engine treats "calendar unavailable" conservatively (see DECISIONS).
- **Historical (backtests):** free official release schedules (Federal Reserve FOMC calendar; BLS jobs report and CPI; BEA GDP and PCE) compiled into a committed CSV. If a source cannot be verified, it is recorded as a known limitation.
- CLI: `uv run tradeagent news show` (upcoming events and blackout windows), `uv run tradeagent news fetch`.

### 4.7 Paper execution gate (`execution/paper.py`)
- `PaperBroker.submit(order, token)` refuses missing, forged, expired, reused or mismatched tokens (and logs the refusal); simulated fills only. `close_all()` / `cancel_all()` for the kill switch.

### 4.8 Backtests use the risk engine
- `backtest/risk_basics.py` is replaced by the risk engine. Trade-level rules are enforced in backtests (sizing, stop distance, RR, spread, open positions, **news blackout from the historical calendar**; correlation needs two symbols, so it only applies in multi-symbol runs). Account-level limits stay **reported as flags** by default; `--enforce-account-limits` runs with them enforced.
- Re-run the Phase 3 strategies and the M15/H1 baselines on train with the news blackout, and note the change.

### 4.9 Status, coverage, wrap-up
- `uv run tradeagent risk status`: limits in force, current state (paper), kill switch, shutdown, pauses.
- `uv run pytest --cov=tradeagent.risk --cov-fail-under=100` must pass (adds `pytest-cov`).
- README, CLAUDE.md, SPEC sync, DECISIONS; all tests + ruff; commit and push.

## Phase 4 is done when
- Every §6 rule is implemented and tested, the token gate refuses every kind of bad token, the kill switch stops everything, and risk-module coverage is 100%.
- The news calendar works live, and historically or with a recorded limitation.
