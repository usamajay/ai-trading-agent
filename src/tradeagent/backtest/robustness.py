"""Robustness checks for the out-of-sample gate (SPEC §7.4; docs/PHASE_7_TASKS.md 7.3).

Pure functions (no I/O). The gate code runs the backtests and passes in their results.
- Monte Carlo: shuffle the order of the trades' net P&L many times; the 95th
  percentile of the shuffled closed-trade max drawdowns is compared with the
  drawdown of the real order (the same closed-trade measure on both sides).
- Random baseline: empirical one-sided p of the strategy's expectancy among random
  runs of the same split: (1 + seeds at or above it) / (1 + seeds).
- Parameter sensitivity: each parameter moved by -pct and +pct, clipped to its range.
- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): the probability that the
  true Sharpe is above what the best of N tries would show by luck alone.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from tradeagent.strategies.base import ParamSpec

EULER_GAMMA = 0.5772156649015329


def closed_max_dd_pct(net_pnl: np.ndarray, start_balance: float) -> float:
    """Largest fall from the running peak of closed-trade equity, in % of that peak."""
    equity = start_balance + np.concatenate([[0.0], np.cumsum(net_pnl)])
    peak = np.maximum.accumulate(equity)
    return float(((peak - equity) / peak).max() * 100)


def monte_carlo_dd(
    net_pnl: np.ndarray, start_balance: float, runs: int, seed: int = 0
) -> tuple[float, float]:
    """(drawdown of the real order, 95th percentile over `runs` shuffles), both in %."""
    rng = np.random.default_rng(seed)
    shuffled = [closed_max_dd_pct(rng.permutation(net_pnl), start_balance) for _ in range(runs)]
    return closed_max_dd_pct(net_pnl, start_balance), float(np.percentile(shuffled, 95))


def baseline_p_value(expectancy: float, baseline: pd.Series) -> float:
    values = baseline.dropna().to_numpy(float)
    return float((1 + (values >= expectancy).sum()) / (1 + len(values)))


def sensitivity_values(
    params: dict[str, float], specs: Sequence[ParamSpec], pct: float
) -> list[tuple[str, float]]:
    """(parameter, moved value) pairs; a move clipped back to the current value is skipped."""
    out = []
    for spec in specs:
        base = params[spec.name]
        for factor in (1 - pct / 100, 1 + pct / 100):
            value = round(min(max(base * factor, spec.low), spec.high), 10)
            if value != base:
                out.append((spec.name, value))
    return out


def deflated_sharpe(
    daily_returns: np.ndarray, trial_sharpes: Sequence[float], n_trials: int
) -> float:
    """DSR from per-day returns. `trial_sharpes`: per-day Sharpe of every try (this one
    included); with fewer than 2 tries the benchmark is 0 (the probabilistic Sharpe)."""
    r = np.asarray(daily_returns, float)
    t = len(r)
    if t < 3 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    benchmark = 0.0
    if n_trials >= 2 and len(trial_sharpes) >= 2:
        sd = float(np.std(trial_sharpes, ddof=1))
        benchmark = sd * (
            (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
            + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e))
        )
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr**2
    if denom <= 0:
        return 0.0
    return float(norm.cdf((sr - benchmark) * math.sqrt(t - 1) / math.sqrt(denom)))
