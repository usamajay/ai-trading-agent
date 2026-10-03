"""Storage: price bars in Parquet, everything else in SQLite (SPEC §3.1, §3.2).

Bars live in `data/bars/{symbol}/{timeframe}/{year}.parquet`, where `symbol`
is our internal name (XAUUSD), not the broker's (XAUUSDm), so files stay the
same if the broker changes. All other code reads/writes bars only through
BarStore, so moving to PostgreSQL later touches only this module.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from tradeagent.data.mt5_client import BAR_COLUMNS, TIME_DTYPE, TIMEFRAMES
from tradeagent.timeutil import ensure_utc


class BarStore:
    """Read/write OHLC bars as one Parquet file per symbol/timeframe/year."""

    def __init__(self, bars_dir: Path) -> None:
        self.bars_dir = bars_dir

    def path(self, symbol: str, timeframe: str, year: int) -> Path:
        return self.bars_dir / symbol / timeframe / f"{year}.parquet"

    def years(self, symbol: str, timeframe: str) -> list[int]:
        folder = self.bars_dir / symbol / timeframe
        if not folder.is_dir():
            return []
        return sorted(int(p.stem) for p in folder.glob("*.parquet") if p.stem.isdigit())

    def write(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> int:
        """Merge bars into the year files. Returns how many bars were new.

        A timestamp already on disk is replaced by the incoming bar, so re-running
        a download never creates duplicates.
        """
        missing = set(BAR_COLUMNS) - set(bars.columns)
        if missing:
            raise ValueError(f"bars are missing columns: {sorted(missing)}")
        if bars.empty:
            return 0

        bars = bars[BAR_COLUMNS].astype({"time_utc": TIME_DTYPE})
        added = 0
        for year, group in bars.groupby(bars["time_utc"].dt.year):
            path = self.path(symbol, timeframe, int(year))
            existing = _read_file(path)
            merged = (
                pd.concat([existing, group], ignore_index=True)
                .drop_duplicates("time_utc", keep="last")
                .sort_values("time_utc")
                .reset_index(drop=True)
            )
            added += len(merged) - len(existing)
            _write_file_atomic(merged, path)
        return added

    def read(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """All stored bars (optionally open time in [start, end]), oldest first."""
        years = self.years(symbol, timeframe)
        if start is not None:
            years = [y for y in years if y >= ensure_utc(start).year]
        if end is not None:
            years = [y for y in years if y <= ensure_utc(end).year]
        frames = [_read_file(self.path(symbol, timeframe, y)) for y in years]
        if not frames:
            return _empty_bars()
        df = pd.concat(frames, ignore_index=True)
        if start is not None:
            df = df[df["time_utc"] >= ensure_utc(start)]
        if end is not None:
            df = df[df["time_utc"] <= ensure_utc(end)]
        return df.reset_index(drop=True)

    def time_range(self, symbol: str, timeframe: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """(first, last) bar open time on disk, or None if nothing is stored."""
        years = self.years(symbol, timeframe)
        if not years:
            return None
        first = _read_file(self.path(symbol, timeframe, years[0]))["time_utc"].min()
        last = _read_file(self.path(symbol, timeframe, years[-1]))["time_utc"].max()
        return first, last

    def summary(self) -> pd.DataFrame:
        """One row per symbol/timeframe: bar count and first/last time."""
        rows = []
        if self.bars_dir.is_dir():
            for symbol_dir in sorted(p for p in self.bars_dir.iterdir() if p.is_dir()):
                for tf_dir in sorted(p for p in symbol_dir.iterdir() if p.is_dir()):
                    df = self.read(symbol_dir.name, tf_dir.name)
                    if df.empty:
                        continue
                    rows.append(
                        {
                            "symbol": symbol_dir.name,
                            "timeframe": tf_dir.name,
                            "bars": len(df),
                            "first_utc": df["time_utc"].iloc[0],
                            "last_utc": df["time_utc"].iloc[-1],
                        }
                    )
        table = pd.DataFrame(rows, columns=["symbol", "timeframe", "bars", "first_utc", "last_utc"])
        order = {tf: i for i, tf in enumerate(TIMEFRAMES)}  # M1, M5, ... D1
        return table.sort_values(
            ["symbol", "timeframe"],
            key=lambda col: col.map(order) if col.name == "timeframe" else col,
        ).reset_index(drop=True)


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}).astype(
        {"time_utc": TIME_DTYPE, "tick_volume": "int64", "spread": "int64"}
    )


def _read_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return _empty_bars()
    return pd.read_parquet(path).astype({"time_utc": TIME_DTYPE})


def _write_file_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# --- SQLite ------------------------------------------------------------------------
# Tables from SPEC §3.2 (bars are Parquet). Rows written by code carry git_commit and
# config_hash so any result can be reproduced. Later phases fill these tables.

SCHEMA = """
CREATE TABLE IF NOT EXISTS data_quality_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    time_utc TEXT,
    details TEXT,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_dql_symbol_tf ON data_quality_log (symbol, timeframe, time_utc);

CREATE TABLE IF NOT EXISTS regimes (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    time_utc TEXT NOT NULL,
    trend_state TEXT,
    vol_state TEXT,
    direction TEXT,
    detector_version TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, time_utc, detector_version)
);

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    params_json TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'research', 'candidate', 'validated', 'oos_passed',
        'paper', 'approved', 'production', 'retired')),
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    rationale TEXT,
    source TEXT NOT NULL CHECK (source IN ('llm', 'scan', 'human')),
    status TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    hypothesis_id TEXT REFERENCES hypotheses (hypothesis_id),
    strategy_id TEXT REFERENCES strategies (strategy_id),
    dataset_split TEXT NOT NULL CHECK (dataset_split IN (
        'train', 'validation', 'out_of_sample', 'walk_forward', 'paper')),
    date_range TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    seed INTEGER,
    metrics_json TEXT,
    verdict TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    time_utc TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy_id TEXT REFERENCES strategies (strategy_id),
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    entry REAL NOT NULL,
    sl REAL NOT NULL,
    tp REAL NOT NULL,
    p_win REAL,
    ev_r REAL,
    regime_id TEXT,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    signal_id TEXT REFERENCES signals (signal_id),
    action TEXT NOT NULL CHECK (action IN ('taken', 'rejected')),
    reason_codes TEXT,
    risk_check_json TEXT,
    explanation_text TEXT,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    decision_id TEXT REFERENCES decisions (decision_id),
    mode TEXT NOT NULL CHECK (mode IN ('paper', 'demo', 'live')),
    open_time TEXT NOT NULL,
    close_time TEXT,
    entry REAL NOT NULL,
    exit REAL,
    size REAL NOT NULL,
    sl REAL NOT NULL,
    tp REAL NOT NULL,
    fees REAL,
    slippage REAL,
    pnl REAL,
    r_multiple REAL,
    exit_reason TEXT,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lessons (
    lesson_id TEXT PRIMARY KEY,
    trade_id TEXT REFERENCES trades (trade_id),
    experiment_id TEXT REFERENCES experiments (experiment_id),
    text TEXT NOT NULL,
    evidence_json TEXT,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (trade_id IS NOT NULL OR experiment_id IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL REFERENCES strategies (strategy_id),
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    rule TEXT NOT NULL,
    value REAL,
    "limit" REAL,
    action TEXT NOT NULL CHECK (action IN ('block', 'shutdown', 'kill')),
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL
);

-- Current risk-engine state for paper/live (Phase 4): one JSON document per key.
CREATE TABLE IF NOT EXISTS risk_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL
);

-- Backtest runs (Phase 2). Trades live in output_dir/trades.parquet, not in `trades`
-- (that table is for paper/demo/live). run_number = how many runs this strategy
-- (by name, any version/params) has had on this split, counting this one: the
-- number of variants tried, for the multiple-testing rule (SPEC §7.4).
CREATE TABLE IF NOT EXISTS gate_checks (
    check_id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL REFERENCES strategies (strategy_id),
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    passed INTEGER NOT NULL,
    checks_json TEXT NOT NULL,
    run_ids_json TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id TEXT PRIMARY KEY,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL NOT NULL,
    worst_case_usd REAL NOT NULL,
    stop_reason TEXT,
    request_id TEXT,
    outcome TEXT NOT NULL,
    response_text TEXT,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    params_json TEXT NOT NULL,
    seed INTEGER,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'out_of_sample')),
    date_start TEXT NOT NULL,
    date_end TEXT NOT NULL,
    run_number INTEGER NOT NULL,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    cost_stress_multiple REAL NOT NULL,
    trades INTEGER NOT NULL,
    insufficient_sample INTEGER NOT NULL,
    expectancy_r REAL,
    stress_expectancy_r REAL,
    stress_pass INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    stress_metrics_json TEXT NOT NULL,
    risk_flags_json TEXT NOT NULL,
    output_dir TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_backtest_runs_strategy ON backtest_runs (strategy, split);
"""

TABLES = (
    "data_quality_log",
    "regimes",
    "strategies",
    "hypotheses",
    "experiments",
    "signals",
    "decisions",
    "trades",
    "lessons",
    "approvals",
    "risk_events",
    "backtest_runs",
    "risk_state",
    "llm_calls",
    "gate_checks",
)


def connect_db(sqlite_path: Path) -> sqlite3.Connection:
    """Open the SQLite file, creating it and any missing tables. Safe to call repeatedly."""
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    return conn


# Columns added after a table was first created (SQLite: ALTER TABLE ... ADD COLUMN).
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "strategies": {  # Phase 7 registry: one row per candidate (variant + symbol)
        "symbol": "TEXT",
        "variant_json": "TEXT",
        "seed": "INTEGER",
        "updated_at": "TEXT",
    },
    "experiments": {  # Phase 6 experiment manager
        "symbol": "TEXT",
        "variant_json": "TEXT",
        "success_criterion": "TEXT",
        "status": "TEXT",
        "run_id": "TEXT",
        "parent_experiment_id": "TEXT",
    },
}


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, columns in ADDED_COLUMNS.items():
        have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    conn.commit()
