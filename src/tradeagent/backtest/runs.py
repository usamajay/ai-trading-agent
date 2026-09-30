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
from tradeagent.backtest.metrics import compute_metrics, regime_breakdown
from tradeagent.backtest.records import next_run_number, run_counts, save_run, to_json
from tradeagent.backtest.report import write_outputs
from tradeagent.backtest.risk_flags import risk_limit_flags
from tradeagent.backtest.runner import load_inputs, run_inputs, stressed
from tradeagent.config import AppConfig, SplitName
from tradeagent.data.store import BarStore
from tradeagent.provenance import git_commit, new_run_id
from tradeagent.strategies.base import check_strategy
from tradeagent.strategies.variants import VariantSpec, build
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
    enforce_account_limits: bool = False,
    variant: VariantSpec | None = None,
) -> RunRecord:
    """`variant` (research): run the strategy on other timeframes / one direction / style."""
    spec = variant or VariantSpec(strategy_name, params=params)
    if spec.strategy != strategy_name:
        raise ValueError(f"variant is for {spec.strategy!r}, not {strategy_name!r}")
    params = dict(spec.params)
    strategy = build(spec, seed)
    check_strategy(strategy)
    strategy_name = strategy.name
    inputs = load_inputs(cfg, store, strategy, symbol, split)
    enforce = enforce_account_limits
    result = run_inputs(cfg, strategy, inputs, enforce_account_limits=enforce)
    multiple = cfg.settings.backtest.cost_stress_multiple
    stress_result = run_inputs(
        cfg,
        build(spec, seed),
        inputs,
        stressed(inputs.costs, multiple),
        enforce_account_limits=enforce,
    )
    metrics = compute_metrics(result)
    metrics["by_regime"] = regime_breakdown(
        result.trades, inputs.dataset.bars, inputs.dataset.timeframe
    )
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
        # The account-limit mode is part of how the run was made, so it is stored too.
        "params_json": json.dumps(
            {
                **strategy.params,
                "__account_limits": "enforced" if enforce else "flags",
                **({} if spec.is_plain else {"__variant": spec.to_dict()}),
            },
            sort_keys=True,
        ),
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
