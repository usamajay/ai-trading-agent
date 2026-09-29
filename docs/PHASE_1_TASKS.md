# Phase 1 — Market data collection and database

**Goal:** reliable, validated historical and live price data for XAUUSD and USOIL, stored locally, with a command to fetch and check it.
**Why first:** every later part (backtests, strategies, probabilities) is only as good as the data.

## Tasks

### 1.1 Project skeleton
- `pyproject.toml` with dependencies: `MetaTrader5`, `pandas`, `numpy`, `pyarrow`, `pydantic`, `pyyaml`, `python-dotenv`, `typer`, `loguru`; dev: `pytest`, `ruff`, `mypy`.
- `uv sync` works; `uv run pytest` runs.
- `.env.example` with `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_PATH`.
- ✅ Check: `uv run tradeagent --help` prints the CLI help.

### 1.2 Config loader (`config.py`)
- Load `config/settings.yaml`, `config/risk.yaml`, `config/live.yaml` into pydantic models; fail loudly on missing/invalid values.
- Compute `config_hash` (sha256 of the loaded config).
- Tests: valid config loads; bad value (e.g. negative risk) raises an error.

### 1.3 MT5 client (`data/mt5_client.py`)
- `connect()` using `.env`. Log the account type (`account_info().trade_mode`: demo/real) at every start. **The client is read-only**: it exposes only data functions; no order functions exist in this module. If the account is REAL, print a clear banner: "REAL ACCOUNT — READ-ONLY DATA MODE".
- Test: a test fails if `mt5_client.py` (or any module outside `execution/live_mt5.py`) references `order_send`, `order_check`, `positions_*` close/modify or `Buy`/`Sell` helpers.
- `get_bars(symbol, timeframe, start, end)` → pandas DataFrame in UTC.
- `symbol_info(symbol)` → point value, lot step, min lot, typical spread.
- ✅ Check: `uv run tradeagent data ping` shows account type (demo/real), balance, and last XAUUSDm price.

### 1.4 Historical download (`data/historical.py`) + storage (`data/store.py`)
- Download `history_years` of M5, M15, H1, H4, D1 (and M1 for last 6 months) in chunks.
- Save as Parquet: `data/bars/{symbol}/{timeframe}/{year}.parquet`; re-runs only fetch missing periods (incremental).
- SQLite file created with tables from SPEC §3.2 (empty for now except `data_quality_log`).
- ✅ Check: `uv run tradeagent data fetch --symbol XAUUSD` then `data summary` shows bar counts and date ranges.

### 1.5 Validation (`data/validation.py`)
- Rules from SPEC §3.3; write issues to `data_quality_log`.
- `uv run tradeagent data validate` prints a readable report (gaps, spikes, duplicates, spread outliers).
- Tests using small synthetic DataFrames with deliberate errors.

### 1.6 Live bar updater
- `uv run tradeagent data watch` appends each newly closed M1/M5 bar (polling every 10 s), validates it, logs it.
- Stops cleanly with Ctrl+C.

### 1.7 Wrap-up
- Update README "How to run" section.
- Add a short `docs/DECISIONS.md` entry for anything that differed from the spec.
- All tests + ruff pass; commit and push.

## Phase 1 is done when
- 3 years of clean data for both symbols on disk, validation report reviewed with Usama.
- Live updater runs for 1 hour without errors.
- Usama can explain in his own words where the data lives and how to refresh it.
