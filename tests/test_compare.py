"""Strategy runs compared with the random-baseline distribution."""

import pandas as pd
import pytest

from tradeagent.backtest.compare import baseline_percentile, verdict


def test_baseline_percentile() -> None:
    base = pd.Series([-0.3, -0.2, -0.1, 0.0, 0.1])
    assert baseline_percentile(-0.15, base) == 40.0  # 2 of 5 below
    assert baseline_percentile(0.0, base) == 70.0  # 3 below + half of 1 tie
    assert baseline_percentile(1.0, base) == 100.0
    assert baseline_percentile(-1.0, base) == 0.0


def row(**changes: object) -> dict[str, object]:
    r: dict[str, object] = {
        "expectancy_r": 0.2,
        "expectancy_ci": [0.05, 0.35],
        "insufficient_sample": False,
        "stress_pass": True,
        "baseline_percentile": 99.0,
    }
    r.update(changes)
    return r


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, "worth a Phase 6/7 look"),
        ({"insufficient_sample": True}, "too few trades to judge"),
        ({"expectancy_r": -0.05}, "no edge on train"),
        ({"stress_pass": False}, "positive but fails the cost stress"),
        ({"expectancy_ci": [-0.02, 0.3]}, "positive but not clearly better than random"),
        ({"baseline_percentile": 80.0}, "positive but not clearly better than random"),
    ],
)
def test_verdict(changes: dict[str, object], expected: str) -> None:
    assert verdict(row(**changes)) == expected


def test_latest_run_per_strategy_and_symbol(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from tradeagent.backtest.compare import latest_runs
    from tradeagent.data.store import connect_db

    conn = connect_db(tmp_path / "t.db")
    base = {
        "strategy_version": "1",
        "params_json": "{}",
        "seed": 1,
        "timeframe": "M15",
        "date_start": "a",
        "date_end": "b",
        "run_number": 1,
        "git_commit": "g",
        "config_hash": "c",
        "data_hash": "d",
        "cost_stress_multiple": 1.5,
        "trades": 1,
        "insufficient_sample": 1,
        "stress_pass": 0,
        "metrics_json": "{}",
        "stress_metrics_json": "{}",
        "risk_flags_json": "{}",
        "output_dir": "o",
    }
    for run_id, created, strategy, symbol, split in [
        ("r1", "2026-01-01", "s1", "XAUUSD", "train"),
        ("r2", "2026-01-02", "s1", "XAUUSD", "train"),  # newer: the one shown
        ("r3", "2026-01-01", "s1", "USOIL", "train"),
        ("r4", "2026-01-03", "random_baseline", "XAUUSD", "train"),  # never shown
        ("r5", "2026-01-04", "s1", "XAUUSD", "validation"),  # other split
    ]:
        row = {
            **base,
            "run_id": run_id,
            "created_at": created,
            "strategy": strategy,
            "symbol": symbol,
            "split": split,
        }
        conn.execute(
            f"INSERT INTO backtest_runs ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})",
            list(row.values()),
        )
    assert sorted(latest_runs(conn, "train")["run_id"]) == ["r2", "r3"]
    conn.close()
