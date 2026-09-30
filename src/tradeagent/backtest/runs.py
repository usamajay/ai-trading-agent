"""One complete, recorded backtest run (docs/PHASE_2_TASKS.md 2.8).

build the strategy by name -> load inputs (and fingerprint the bars) -> run with
normal costs -> run again with stressed costs (a fresh strategy, same seed) ->
metrics + risk-limit flags -> save to `backtest_runs` + write the report files.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tradeagent.backtest.engine import BacktestResult
from tradeagent.backtest.metrics import compute_metrics
from tradeagent.backtest.records import next_run_number, run_counts, save_run, to_json
from tradeagent.backtest.report import write_outputs
from tradeagent.backtest.risk_flags import risk_limit_flags
from tradeagent.backtest.runner import load_inputs, run_inputs, stressed
from tradeagent.config import AppConfig, SplitName
from tradeagent.data.store import BarStore
from tradeagent.provenance import git_commit, new_run_id
from tradeagent.strategies import registry
from tradeagent.strategies.base import check_strategy
from tradeagent.timeutil import fmt_utc_pkt, utc_now


@dataclass(frozen=True)
class RunRecord:
    run: dict[str, Any]  # the backtest_runs row
    metrics: dict[str, Any]
    stress_metrics: dict[str, Any]
    risk_flags: dict[str, Any]
    counts: dict[str, int]  # runs of this strategy per split, including this one
    result: BacktestResult
    output_dir: Path


def execute_run(
    cfg: AppConfig,
    store: BarStore,
    conn: sqlite3.Connection,
    strategy_name: str,
    symbol: str,
    split: SplitName,
    seed: int,
    params: dict[str, float],
    out_root: Path,
) -> RunRecord:
    strategy = registry.create(strategy_name, seed, params)
    check_strategy(strategy)
    inputs = load_inputs(cfg, store, strategy, symbol, split)
    result = run_inputs(cfg, strategy, inputs)
    multiple = cfg.settings.backtest.cost_stress_multiple
    stress_result = run_inputs(
        cfg, registry.create(strategy_name, seed, params), inputs, stressed(inputs.costs, multiple)
    )
    metrics = compute_metrics(result)
    stress_metrics = compute_metrics(stress_result)
    flags = risk_limit_flags(result.daily, result.start_balance, cfg.risk)

    created = utc_now()
    run_id = new_run_id()
    out_dir = out_root / run_id
    stress_exp = stress_metrics["trades"]["expectancy_r"]
    run: dict[str, Any] = {
        "run_id": run_id,
        "created_at": created.isoformat(),
        "strategy": strategy_name,
        "strategy_version": strategy.version,
        "params_json": json.dumps(strategy.params, sort_keys=True),
        "seed": seed,
        "symbol": symbol,
        "timeframe": inputs.dataset.timeframe,
        "split": split,
        "date_start": inputs.dataset.start_utc.isoformat(),
        "date_end": inputs.dataset.end_utc.isoformat(),
        "run_number": next_run_number(conn, strategy_name, split),
        "git_commit": git_commit(),
        "config_hash": cfg.config_hash,
        "data_hash": inputs.data_hash,
        "cost_stress_multiple": multiple,
        "trades": metrics["trades"]["trades"],
        "insufficient_sample": int(metrics["trades"]["insufficient_sample"]),
        "expectancy_r": metrics["trades"]["expectancy_r"],
        "stress_expectancy_r": stress_exp,
        "stress_pass": int(stress_exp is not None and stress_exp > 0),
        "metrics_json": to_json(metrics),
        "stress_metrics_json": to_json(stress_metrics),
        "risk_flags_json": to_json(flags),
        "output_dir": str(out_dir),
    }
    save_run(conn, run)
    counts = run_counts(conn, strategy_name)
    write_outputs(
        out_dir, run, result, metrics, stress_metrics, flags, counts, fmt_utc_pkt(created)
    )
    return RunRecord(run, metrics, stress_metrics, flags, counts, result, out_dir)
