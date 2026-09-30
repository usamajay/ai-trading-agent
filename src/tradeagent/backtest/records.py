"""Saving backtest runs to SQLite (`backtest_runs`) and counting runs per strategy."""

import json
import math
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

RUN_SPLITS = ("train", "validation", "out_of_sample")


def clean(value: Any) -> Any:
    """JSON-safe copy: inf -> "inf", nan -> None, numpy/pandas scalars -> Python, dates -> str."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, bool | int | str) or value is None:
        return value
    return str(value)  # dates, timestamps


def to_json(value: Any) -> str:
    return json.dumps(clean(value), sort_keys=True)


def next_run_number(conn: sqlite3.Connection, strategy: str, split: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM backtest_runs WHERE strategy = ? AND split = ?", (strategy, split)
    ).fetchone()
    return int(row[0]) + 1


def run_counts(conn: sqlite3.Connection, strategy: str) -> dict[str, int]:
    """Runs of `strategy` (any version/params) per split: the variants tried so far."""
    rows = dict(
        conn.execute(
            "SELECT split, COUNT(*) FROM backtest_runs WHERE strategy = ? GROUP BY split",
            (strategy,),
        ).fetchall()
    )
    return {split: int(rows.get(split, 0)) for split in RUN_SPLITS}


def save_run(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    columns = list(row)
    with conn:
        conn.execute(
            f"INSERT INTO backtest_runs ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            [row[c] for c in columns],
        )


def list_runs(
    conn: sqlite3.Connection, strategy: str | None = None, limit: int = 20
) -> pd.DataFrame:
    query = (
        "SELECT run_id, created_at, strategy, strategy_version, symbol, timeframe, split, "
        "run_number, trades, insufficient_sample, expectancy_r, stress_expectancy_r, "
        "stress_pass, output_dir FROM backtest_runs"
    )
    params: tuple[Any, ...] = ()
    if strategy:
        query += " WHERE strategy = ?"
        params = (strategy,)
    query += " ORDER BY created_at DESC, run_id DESC LIMIT ?"
    return pd.read_sql_query(query, conn, params=(*params, limit))
