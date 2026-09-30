# AI Trading Agent

An autonomous, probabilistic, risk-controlled trading **research and paper-trading** system for XAUUSD and USOIL on MetaTrader 5.

> ⚠️ Real-money trading is disabled by design until Phase 10 and a recorded manual approval. All testing runs on an Exness MT5 **demo** account; the code refuses real accounts.

## Start here
1. New to this? Follow **[docs/PHASE_0_SETUP.md](docs/PHASE_0_SETUP.md)**.
2. Full design: **[docs/SPEC.md](docs/SPEC.md)**
3. Current work: **[docs/PHASE_1_TASKS.md](docs/PHASE_1_TASKS.md)**
4. Rules for Claude Code: **[CLAUDE.md](CLAUDE.md)**
5. Decision log: **[docs/DECISIONS.md](docs/DECISIONS.md)**
6. Market hours, gaps and candle quirks: **[docs/DATA_NOTES.md](docs/DATA_NOTES.md)**

## Status
| Phase | Name | Status |
|---|---|---|
| 0 | Setup | ✅ done |
| 1 | Market data + database | ✅ done |
| 2 | Backtesting framework | ✅ done |
| 3 | Strategy engine | ✅ done |
| 4 | Risk engine | ✅ done |
| 5 | Probability / EV engine | ✅ done |
| 6 | Hypothesis generation | ⬜ |
| 7 | Walk-forward + OOS | ⬜ |
| 8 | Paper trading | ⬜ |
| 9 | Monitoring + self-improvement | ⬜ |
| 10 | Controlled live deployment | 🔒 |

## How to run

All commands are for **Windows PowerShell**, run from the project folder.

### One-time setup
1. Install the tools from **[docs/PHASE_0_SETUP.md](docs/PHASE_0_SETUP.md)** (Python, uv, Git, MetaTrader 5).
2. In MetaTrader 5, log in to your Exness **demo** account.
3. In MetaTrader 5: **Tools → Options → Charts → Max. bars in chart → Unlimited**, then restart MT5. Without this, MT5 only gives ~100,000 candles per timeframe.
4. Get the code and install it:
   ```powershell
   git clone https://github.com/usamajay/ai-trading-agent.git
   cd ai-trading-agent
   uv sync
   ```
5. Create your secrets file and fill in your **demo** login (it is git-ignored and never uploaded):
   ```powershell
   Copy-Item .env.example .env
   code .env
   ```
6. Check everything:
   ```powershell
   uv run tradeagent check-config   # settings are valid
   uv run tradeagent data ping      # must say "Account type: DEMO"
   ```

### Everyday commands
| Command | What it does |
|---|---|
| `uv run tradeagent data fetch` | Downloads missing candles for all symbols/timeframes (first run: all history; later: only what's new). Options: `--symbol XAUUSD`, `--timeframe M5` |
| `uv run tradeagent data summary` | Shows how many candles are stored and their date range |
| `uv run tradeagent data validate` | Checks the data (gaps, spikes, bad prices, spreads) and saves issues to the database. `--examples 5` shows more detail |
| `uv run tradeagent data watch` | Keeps M1/M5 up to date live, every 10 s. Stop with **Ctrl+C**. `--minutes 60` stops by itself |
| `uv run tradeagent data ping` | Shows the account type, balance and latest prices |
| `uv run tradeagent --help` | Lists all commands |

### Risk and news commands (Phase 4)
| Command | What it does |
|---|---|
| `uv run tradeagent risk status` | Shows the limits from `config/risk.yaml`, the kill switch and any stops in force |
| `uv run tradeagent kill --reason "..."` | **Kill switch**: stops all trading at once (creates the `KILL` file). `--status` shows it; `--clear --reason "..."` removes it |
| `uv run tradeagent news fetch` | Downloads this week's high-impact USD events (run weekly; without it, live entries are blocked) |
| `uv run tradeagent news show` | Upcoming events and whether a news blackout is active now |

### Probability commands (Phase 5)
| Command | What it does |
|---|---|
| `uv run tradeagent prob calibrate --latest` | Checks how well predicted win chances matched real outcomes for the latest backtest of each strategy; writes `docs/reports/PHASE_5_CALIBRATION.md` |

### Backtesting commands (Phase 2)
| Command | What it does |
|---|---|
| `uv run tradeagent backtest splits` | Shows the frozen train / validation / out-of-sample dates and what is excluded |
| `uv run tradeagent backtest costs` | Shows the cost snapshot (contract size, $ per point, swaps); `--snapshot` refreshes it from MT5 |
| `uv run tradeagent backtest spread-check` | Measures real tick spreads vs stored candle spreads (needs MT5 open) |
| `uv run tradeagent backtest run --strategy random_baseline --symbol XAUUSD --split train --seed 1` | Backtests a strategy, saves it and writes a report to `data/backtests/<run id>/report.md` |
| `uv run tradeagent backtest list` | Past runs, with the run counter and cost-stress PASS/FAIL |
| `uv run tradeagent backtest lookahead-check --strategy random_baseline` | Checks a strategy never uses future data |
| `uv run tradeagent backtest baseline --symbol XAUUSD --seeds 100` | Random baseline over many seeds (about 10 min), saved to `data/baselines/` |
| `uv run tradeagent data resample-check` | Compares our New York-close D1/H4 candles with the broker's |
| `uv run tradeagent backtest compare` | Latest run of each strategy vs the random baseline, with a verdict |

Strategies (Phase 3): `trend_ema_pullback`, `breakout_compression`, `mean_reversion_bb`, `session_breakout`, `structure_retest`, and the `random_baseline`. Results: `docs/reports/PHASE_3_STRATEGIES.md`.

Out-of-sample data is locked: backtests run on `train` or `validation` only.

### Where the data lives
All data is on your PC in `data/` (git-ignored, never uploaded):

| Path | Contents |
|---|---|
| `data/bars/<SYMBOL>/<TIMEFRAME>/<YEAR>.parquet` | Price candles, UTC times. e.g. `data/bars/XAUUSD/M5/2025.parquet` |
| `data/tradeagent.db` | SQLite database: data-quality log, backtest runs; trades, experiments, etc. in later phases |
| `data/backtests/<run id>/` | One folder per backtest: report.md, equity.png, trades, daily equity, metrics.json |
| `data/baselines/` | Random-baseline results per symbol/timeframe/split (100 seeds) |
| `data/logs/watch_YYYY-MM-DD.log` | Live updater logs (UTC), kept 30 days |

Symbols use internal names (`XAUUSD`, `USOIL`); the broker names (`XAUUSDm`, `USOILm`) are mapped in `config/settings.yaml`. History depth is set there too (`history_years: 3`, `m1_history_months: 6`).

### Refreshing the data
- **Normal refresh:** `uv run tradeagent data fetch`, then `uv run tradeagent data validate`. Re-runs only download what is missing, so they take seconds.
- **Stay current all day:** leave `uv run tradeagent data watch` running.
- **Start over:** delete the `data/bars` folder, then run `data fetch` again (under a minute).

### Checks before committing
```powershell
uv run pytest            # all tests
uv run ruff check .      # lint
uv run mypy src          # type check
```

### If something goes wrong
| Message | Fix |
|---|---|
| `MT5 initialize failed` | Open MetaTrader 5 and log in; check `MT5_PATH` in `.env` |
| `Refusing REAL account` | Working as designed. Log MT5 in to the **demo** account |
| `Terminal is logged in to …, not MT5_LOGIN` | MT5 is on a different account than `.env`. Switch account or fix `.env` |
| `Config problem: …` | A value in `config/*.yaml` is wrong; the message names it |
| M5 history starts later than 3 years ago | Raise "Max bars in chart" (setup step 3), restart MT5, run `data fetch` |

## Disclaimer
Research software. No guarantee of profit. Trading leveraged products can lose more than you expect.
