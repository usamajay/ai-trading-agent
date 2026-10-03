"""Random-baseline distribution: the random strategy over many seeds (Task 2.9).

One row per seed with the headline numbers, saved to
data/baselines/{symbol}_{timeframe}_{split}.parquet. Phase 7 uses it for "beats
the random baseline with p < 0.05" (SPEC §7.4). Seeds are not written to
`backtest_runs`: they are one baseline, not strategy variants being tried.
"""

from pathlib import Path

import pandas as pd

from tradeagent.backtest.metrics import compute_metrics
from tradeagent.backtest.risk_flags import risk_limit_flags
from tradeagent.backtest.runner import load_inputs, run_inputs, stressed
from tradeagent.config import AppConfig, SplitName
from tradeagent.data.store import BarStore
from tradeagent.strategies.base import Style
from tradeagent.strategies.baseline_random import RandomBaseline


def baseline_distribution(
    cfg: AppConfig,
    store: BarStore,
    symbol: str,
    seeds: range,
    split: SplitName = "train",
    timeframe: str = "M15",
    params: dict[str, float] | None = None,
    style: Style = "intraday",
    direction: str = "both",
    allow_oos: bool = False,
) -> tuple[pd.DataFrame, str]:
    """(one row per seed, data fingerprint). The bars are loaded once for all seeds."""

    def make(seed: int) -> RandomBaseline:
        return RandomBaseline(seed, params, timeframe, style, direction)

    inputs = load_inputs(cfg, store, make(0), symbol, split, allow_oos)
    hard = stressed(inputs.costs, cfg.settings.backtest.cost_stress_multiple)
    rows = []
    for seed in seeds:
        result = run_inputs(cfg, make(seed), inputs)
        stress = run_inputs(cfg, make(seed), inputs, hard)
        m, sm = compute_metrics(result), compute_metrics(stress)
        t = result.trades
        flags = risk_limit_flags(result.daily, result.start_balance, cfg.risk)
        rows.append(
            {
                "seed": seed,
                "trades": m["trades"]["trades"],
                "win_rate_pct": m["trades"]["win_rate_pct"],
                "profit_factor": m["trades"]["profit_factor"],
                "expectancy_r": m["trades"]["expectancy_r"],
                "net_profit_usd": m["equity"]["net_profit_usd"],
                "max_dd_pct": m["equity"]["max_dd_pct"],
                "sharpe": m["equity"]["sharpe"],
                "avg_cost_r": m["costs"]["avg_cost_r"],
                "total_cost_pct_of_gross": m["costs"]["total_pct_of_gross_profit"],
                "stress_expectancy_r": sm["trades"]["expectancy_r"],
                "weekend_held": int(t["held_over_weekend"].sum()),
                "sl_gap_exits": int((t["exit_reason"] == "sl_gap").sum()),
                "min_balance_max": m["min_balance"]["max"],
                "min_balance_median": m["min_balance"]["median"],
                "daily_loss_breached": flags["daily_loss"]["breached"],
                "weekly_loss_breached": flags["weekly_loss"]["breached"],
                "max_dd_breached": flags["max_drawdown"]["breached"],
            }
        )
    return pd.DataFrame(rows), inputs.data_hash


def distribution_path(
    out_dir: Path, symbol: str, timeframe: str, style: str, split: str, direction: str = "both"
) -> Path:
    """`<SYMBOL>_<TF>_<style>_<split>.parquet`; one-direction baselines add `_long`/`_short`."""
    tail = "" if direction == "both" else f"_{direction}"
    return out_dir / f"{symbol}_{timeframe}_{style}_{split}{tail}.parquet"


def save_distribution(
    df: pd.DataFrame,
    out_dir: Path,
    symbol: str,
    timeframe: str,
    split: str,
    style: str = "intraday",
    direction: str = "both",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = distribution_path(out_dir, symbol, timeframe, style, split, direction)
    df.to_parquet(path, index=False)
    return path
