"""Walk-forward windows and robustness maths (Phase 7.2 / 7.3), no backtests."""

import numpy as np
import pandas as pd
import pytest

from tradeagent.backtest.robustness import (
    baseline_p_value,
    closed_max_dd_pct,
    deflated_sharpe,
    monte_carlo_dd,
    sensitivity_values,
)
from tradeagent.backtest.walkforward import WalkForwardResult, wf_windows, window_results
from tradeagent.strategies.base import ParamSpec

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def _days(n: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(days=n)


def test_windows_start_after_history_and_drop_partial() -> None:
    windows = wf_windows(T0, _days(100), history_days=30, test_days=20, step_days=20)
    assert [(w.start, w.end) for w in windows] == [
        (_days(30), _days(50)),
        (_days(50), _days(70)),
        (_days(70), _days(90)),
    ]  # 90..110 would pass the end: dropped
    assert wf_windows(T0, _days(40), 30, 20, 20) == []


def test_window_results_cut_by_entry_and_drop_the_gap() -> None:
    windows = wf_windows(T0, _days(100), 30, 20, 20)
    trades = pd.DataFrame(
        {
            "entry_time": [_days(31), _days(45), _days(55), _days(75), _days(10)],
            "r_multiple": [1.0, -1.0, 2.0, -1.0, 5.0],
            "net_pnl": [10.0, -10.0, 20.0, -10.0, 50.0],
        }
    )
    res = window_results(trades, windows, gap=(_days(40), _days(50)))
    assert [w.trades for w in res] == [1, 1, 1]  # day 45 is in the gap; day 10 is history
    assert [w.positive for w in res] == [True, True, False]
    wf = WalkForwardResult(res, 0.667)
    assert wf.positive_share == pytest.approx(2 / 3) and not wf.passed  # 0.6667 < 0.667
    assert WalkForwardResult(res, 0.66).passed
    assert not WalkForwardResult(res[:2], 0.5).passed  # fewer than 3 windows


def test_empty_windows_count_as_not_working() -> None:
    windows = wf_windows(T0, _days(100), 30, 20, 20)
    empty = pd.DataFrame({"entry_time": [_days(31)], "r_multiple": [1.0], "net_pnl": [1.0]})
    res = window_results(empty, windows)
    assert [w.trades for w in res] == [1, 0, 0]
    assert WalkForwardResult(res, 0.667).positive_share == pytest.approx(1 / 3)


def test_closed_drawdown_and_monte_carlo() -> None:
    assert closed_max_dd_pct(np.array([100.0, -50.0, -50.0, 200.0]), 1000.0) == pytest.approx(
        100 / 1100 * 100
    )
    assert closed_max_dd_pct(np.array([10.0, 10.0]), 1000.0) == 0.0
    pnl = np.array([50.0] * 30 + [-40.0] * 20)
    dd, p95 = monte_carlo_dd(pnl, 10_000.0, runs=500, seed=3)
    assert monte_carlo_dd(pnl, 10_000.0, runs=500, seed=3) == (dd, p95)  # deterministic
    worst = closed_max_dd_pct(np.sort(pnl), 10_000.0)  # all losses first
    assert 0 < p95 <= worst


def test_baseline_p_value() -> None:
    base = pd.Series(np.linspace(-0.2, 0.0, 20))
    assert baseline_p_value(0.5, base) == pytest.approx(1 / 21)
    assert baseline_p_value(-1.0, base) == pytest.approx(1.0)
    assert baseline_p_value(-0.1, pd.Series([0.0, -0.2, np.nan])) == pytest.approx(2 / 3)


def test_sensitivity_values_clip_and_skip() -> None:
    specs = [ParamSpec("rr", 2.0, 4.0, "target"), ParamSpec("atr_mult", 1.0, 3.0, "stop")]
    values = sensitivity_values({"rr": 2.0, "atr_mult": 1.5}, specs, 20)
    assert values == [("rr", 2.4), ("atr_mult", 1.2), ("atr_mult", 1.8)]  # rr 1.6 -> 2.0 = base


def test_deflated_sharpe_falls_with_more_tries() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.002, 0.01, 250)
    trials = list(rng.normal(0.0, 0.05, 50))
    alone = deflated_sharpe(returns, trials, 1)
    many = deflated_sharpe(returns, trials, 50)
    assert 0 <= many < alone <= 1
    assert deflated_sharpe(-returns, trials, 1) < 0.5
    assert deflated_sharpe(np.zeros(10), trials, 5) == 0.0
