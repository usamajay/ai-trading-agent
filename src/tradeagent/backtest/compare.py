"""Compare recorded strategy runs with the random baseline (Phase 3 report).

For each strategy and symbol, the latest run on a split is placed inside the
100-seed random-baseline distribution for the same timeframe and style: the
percentile says what share of random runs had a lower expectancy. Informational
only; the formal "beats the baseline with p < 0.05" test is Phase 7.
"""

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from tradeagent.backtest.baseline import distribution_path


def baseline_percentile(expectancy: float, baseline: pd.Series) -> float:
    """% of baseline runs with a lower expectancy (ties count half)."""
    values = baseline.dropna().to_numpy(float)
    if len(values) == 0:
        return float("nan")
    below = (values < expectancy).sum()
    equal = (values == expectancy).sum()
    return float((below + 0.5 * equal) / len(values) * 100)


def latest_runs(conn: sqlite3.Connection, split: str) -> pd.DataFrame:
    """The newest run per (strategy, symbol) on `split`, excluding the random baseline."""
    runs = pd.read_sql_query(
        "SELECT * FROM backtest_runs WHERE split = ? AND strategy != 'random_baseline' "
        "ORDER BY created_at, run_id",
        conn,
        params=(split,),
    )
    return runs.groupby(["strategy", "symbol"], as_index=False).tail(1).reset_index(drop=True)


def compare_rows(
    conn: sqlite3.Connection, baselines_dir: Path, split: str, styles: Mapping[str, str]
) -> list[dict[str, Any]]:
    """One row per strategy x symbol; `styles` maps strategy name -> style."""
    rows = []
    for run in latest_runs(conn, split).to_dict("records"):
        m = json.loads(run["metrics_json"])
        flags = json.loads(run["risk_flags_json"])
        t, e, c = m["trades"], m["equity"], m["costs"]
        style = styles.get(run["strategy"], "intraday")
        path = distribution_path(baselines_dir, run["symbol"], run["timeframe"], style, split)
        pct = None
        base_mean = None
        if path.is_file() and t["expectancy_r"] is not None:
            base = pd.read_parquet(path)["expectancy_r"]
            pct = baseline_percentile(float(t["expectancy_r"]), base)
            base_mean = float(base.mean())
        ci = m["confidence_95"]["expectancy_r"]
        rows.append(
            {
                "strategy": run["strategy"],
                "symbol": run["symbol"],
                "timeframe": run["timeframe"],
                "style": style,
                "trades": t["trades"],
                "insufficient_sample": bool(t["insufficient_sample"]),
                "win_rate_pct": t["win_rate_pct"],
                "profit_factor": t["profit_factor"],
                "expectancy_r": t["expectancy_r"],
                "expectancy_ci": ci,
                "net_profit_usd": e["net_profit_usd"],
                "max_dd_pct": e["max_dd_pct"],
                "cost_pct_of_gross": c["total_pct_of_gross_profit"],
                "avg_cost_r": c["avg_cost_r"],
                "stress_expectancy_r": run["stress_expectancy_r"],
                "stress_pass": bool(run["stress_pass"]),
                "breaches": [k for k, v in flags.items() if v["breached"]],
                "baseline_mean_r": base_mean,
                "baseline_percentile": pct,
                "run_number": run["run_number"],
                "run_id": run["run_id"],
            }
        )
    return rows


def verdict(row: dict[str, Any]) -> str:
    """Plain-English verdict on train, without tuning (conservative)."""
    exp, low = row["expectancy_r"], (row["expectancy_ci"] or [None])[0]
    if row["insufficient_sample"]:
        return "too few trades to judge"
    if exp is None or exp <= 0:
        return "no edge on train"
    if not row["stress_pass"]:
        return "positive but fails the cost stress"
    if low is not None and low > 0 and (row["baseline_percentile"] or 0) >= 95:
        return "worth a Phase 6/7 look"
    return "positive but not clearly better than random"
