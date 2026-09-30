"""Metrics (SPEC §7.3), each checked against a hand-calculated answer."""

import math

import numpy as np
import pandas as pd
import pytest
from test_engine import LONG, Scripted, frame, run

from tradeagent.backtest.engine import TRADE_COLUMNS, BacktestResult
from tradeagent.backtest.metrics import (
    MIN_TRADES,
    bootstrap_ci,
    cagr,
    compute_metrics,
    cost_breakdown,
    daily_returns,
    drawdown,
    longest_losing_streak,
    profit_factor,
    session_of,
    sharpe_sortino,
    trade_stats,
)

NY = "America/New_York"


def trades(net: list[float], r: list[float] | None = None, **columns: list[object]) -> pd.DataFrame:
    """A trade table with only the columns a test cares about filled in."""
    n = len(net)
    data: dict[str, object] = {c: [0.0] * n for c in TRADE_COLUMNS}
    data.update(
        net_pnl=net,
        r_multiple=r if r is not None else [x / 50 for x in net],
        risk_usd=[50.0] * n,
        entry_time=[pd.Timestamp("2026-01-06 10:00", tz=NY)] * n,
        held_over_weekend=[False] * n,
        exit_reason=["tp" if x > 0 else "sl" for x in net],
        min_balance=[1000.0] * n,
        bars_held=[1] * n,
    )
    data.update(columns)
    return pd.DataFrame(data, columns=TRADE_COLUMNS)


# --- trade statistics ----------------------------------------------------------------


def test_trade_stats() -> None:
    s = trade_stats(trades([100, -50, -50, 100, -50], r=[2, -1, -1, 2, -1]))
    assert s["trades"] == 5 and s["insufficient_sample"]
    assert s["win_rate_pct"] == pytest.approx(40.0)  # 2 of 5
    assert s["profit_factor"] == pytest.approx(200 / 150)
    assert s["expectancy_r"] == pytest.approx(0.2)  # (2 - 1 - 1 + 2 - 1) / 5
    assert s["expectancy_usd"] == pytest.approx(10.0)
    assert (s["avg_win_r"], s["avg_loss_r"]) == (2.0, -1.0)
    assert s["net_pnl_usd"] == 50.0
    assert s["longest_losing_streak"] == 2


def test_sample_size_flag() -> None:
    assert trade_stats(trades([10.0] * (MIN_TRADES - 1)))["insufficient_sample"]
    assert not trade_stats(trades([10.0] * MIN_TRADES))["insufficient_sample"]


def test_profit_factor_edge_cases() -> None:
    assert profit_factor(np.array([10.0, 5.0])) == math.inf  # no losses
    assert profit_factor(np.array([])) is None
    assert profit_factor(np.array([-10.0])) == 0.0
    assert longest_losing_streak(np.array([-1, -1, -1, 1, -1])) == 3


def test_empty_trade_stats() -> None:
    s = trade_stats(trades([]))
    assert s["trades"] == 0 and s["win_rate_pct"] is None and s["expectancy_r"] is None


# --- confidence intervals -------------------------------------------------------------


def test_bootstrap_intervals() -> None:
    mixed = trades([100, -50, -50, 100, -50] * 10, r=[2, -1, -1, 2, -1] * 10)
    ci = bootstrap_ci(mixed)
    low, high = ci["win_rate_pct"]
    assert low < 40.0 < high
    low, high = ci["expectancy_r"]
    assert low < 0.2 < high
    low, high = ci["profit_factor"]
    assert low < 200 / 150 < high
    assert bootstrap_ci(mixed) == ci  # fixed seed: repeatable


def test_bootstrap_edge_cases() -> None:
    wins = bootstrap_ci(trades([10.0] * 5, r=[0.2] * 5))
    assert wins["win_rate_pct"] == [100.0, 100.0]
    assert wins["expectancy_r"] == pytest.approx([0.2, 0.2])
    assert wins["profit_factor"] == [math.inf, math.inf]  # never a loss to divide by
    assert bootstrap_ci(trades([]))["expectancy_r"] is None


# --- equity -----------------------------------------------------------------------------


def test_drawdown() -> None:
    # Start 100 -> 120 (peak) -> 90 -> 95 -> 130 (recovered, 3 days after the peak) -> 117
    dd = drawdown(np.array([120, 90, 95, 130, 117.0]), start=100)
    assert dd["max_dd_pct"] == pytest.approx(25.0)  # 30 / 120
    assert dd["max_dd_usd"] == pytest.approx(30.0)
    assert dd["max_dd_duration"] == 3


def test_drawdown_not_recovered_and_none() -> None:
    dd = drawdown(np.array([110, 100, 105.0]), start=100)
    assert dd["max_dd_pct"] == pytest.approx(10 / 110 * 100)
    assert dd["max_dd_duration"] == 2  # still under water at the end
    assert drawdown(np.array([101, 102.0]), start=100) == {
        "max_dd_pct": 0.0,
        "max_dd_usd": 0.0,
        "max_dd_duration": 0,
    }
    assert drawdown(np.array([95.0]), start=100)["max_dd_pct"] == pytest.approx(5.0)


def test_daily_returns_start_from_the_balance() -> None:
    assert daily_returns(np.array([110, 99.0]), start=100) == pytest.approx([0.1, -0.1])


def test_sharpe_and_sortino() -> None:
    returns = np.array([0.01, -0.01, 0.02, 0.0])
    sharpe, sortino = sharpe_sortino(returns)
    # mean 0.005; sample std = sqrt((0.005^2 + 0.015^2 + 0.015^2 + 0.005^2) / 3)
    assert sharpe == pytest.approx(0.005 / math.sqrt(0.0005 / 3) * math.sqrt(252))
    # downside deviation = sqrt((0 + 0.01^2 + 0 + 0) / 4) = 0.005
    assert sortino == pytest.approx(math.sqrt(252))
    assert sharpe_sortino(np.array([0.01])) == (None, None)
    assert sharpe_sortino(np.array([0.01, 0.01])) == (None, None)  # no variation, no losses


def test_cagr() -> None:
    assert cagr(100, 121, 504) == pytest.approx(0.10)  # +21% over two 252-day years
    assert cagr(100, 121, 0) is None


# --- costs and sessions -------------------------------------------------------------------


def test_costs_as_percent_of_gross_profit() -> None:
    # Trade 1: gross 88 after spread 10 + slippage 2 -> pre-cost +100; paid swap 3.
    # Trade 2: gross -45 after spread 4 + slippage 1 -> pre-cost -40 (not gross profit).
    t = trades(
        [85, -45],
        gross_pnl=[88.0, -45.0],
        spread_cost=[10.0, 4.0],
        slippage_cost=[2.0, 1.0],
        swap=[-3.0, 0.0],
        commission=[0.0, 0.0],
    )
    c = cost_breakdown(t)
    assert c["gross_profit_usd"] == pytest.approx(100.0)
    assert c["spread_pct_of_gross_profit"] == pytest.approx(14.0)
    assert c["slippage_pct_of_gross_profit"] == pytest.approx(3.0)
    assert c["swap_pct_of_gross_profit"] == pytest.approx(3.0)
    assert c["commission_pct_of_gross_profit"] == 0.0
    assert c["total_pct_of_gross_profit"] == pytest.approx(20.0)
    assert c["avg_cost_r"] == pytest.approx(0.2)  # (15/50 + 5/50) / 2


def test_costs_without_gross_profit() -> None:
    t = trades([-45], gross_pnl=[-45.0], spread_cost=[4.0], slippage_cost=[1.0])
    assert cost_breakdown(t)["spread_pct_of_gross_profit"] is None


def test_sessions_in_new_york_time() -> None:
    times = pd.Series(
        pd.DatetimeIndex(
            [
                "2026-01-06 18:30",
                "2026-01-07 02:59",
                "2026-01-07 03:00",
                "2026-01-07 07:59",
                "2026-01-07 08:00",
                "2026-01-07 16:59",
            ]
        ).tz_localize(NY)
    )
    assert session_of(times).tolist() == [
        "asia",
        "asia",
        "london",
        "london",
        "new_york",
        "new_york",
    ]


# --- the whole report ----------------------------------------------------------------------


def result_from(t: pd.DataFrame, daily: list[float], bars_in_split: int = 100) -> BacktestResult:
    final = 10_000 + float(t["net_pnl"].sum())
    closed = [10_000.0, *(10_000 + t["net_pnl"].cumsum()).tolist()]
    return BacktestResult(
        symbol="XAUUSD",
        timeframe="M15",
        start_balance=10_000,
        final_balance=final,
        trades=t,
        equity=pd.DataFrame({"time_utc": [pd.NaT] * len(closed), "equity": closed}),
        counts={"rejected_min_lot": 2},
        daily=pd.DataFrame({"equity": daily}),
        bars_in_split=bars_in_split,
    )


def test_compute_metrics() -> None:
    t = trades([100, -50], r=[2.0, -1.0], bars_held=[10, 15], min_balance=[800.0, 1200.0])
    m = compute_metrics(result_from(t, daily=[10_100, 10_050, 10_050]))
    assert m["trades"]["insufficient_sample"]
    e = m["equity"]
    assert e["net_profit_usd"] == 50 and e["return_pct"] == pytest.approx(0.5)
    assert e["max_dd_usd"] == pytest.approx(50.0) and e["max_dd_duration_days"] == 2
    assert e["closed_trade_max_dd_usd"] == pytest.approx(50.0)
    assert e["recovery_factor"] == pytest.approx(1.0)  # 50 profit / 50 drawdown
    assert e["exposure_pct"] == pytest.approx(25.0)  # 25 of 100 bars
    assert e["trading_days"] == 3
    assert (
        m["min_balance"]["max"] == 1200.0 and m["min_balance"]["signals_too_small_for_min_lot"] == 2
    )
    assert m["by_exit_reason"] == {"tp": 1, "sl": 1}
    assert m["by_regime"] is None
    assert "252" in m["conventions"]["annualisation"]


def test_mark_to_market_drawdown_sees_an_open_trade_dip() -> None:
    # Entry 2001 on Monday 16:55; Monday closes at 1996 (the trade is $45 down: 5 x $9);
    # on Tuesday it reaches its 2011 target. Closed trades never lose; the account did.
    bars = frame(
        [
            ("2026-01-05 16:50", 2000, 2001, 1999, 2000.5),
            ("2026-01-05 16:55", 2001, 2001.5, 1996, 1996),
            ("2026-01-05 18:00", 1996, 2003, 1996, 2002.5),
            ("2026-01-05 18:05", 2002.5, 2011.5, 2002, 2011),
        ]
    )
    result = run(Scripted({0: [LONG]}), bars)
    m = compute_metrics(result)
    assert m["equity"]["closed_trade_max_dd_usd"] == 0.0
    assert m["equity"]["max_dd_usd"] == pytest.approx(45.0)
    assert m["equity"]["max_dd_pct"] == pytest.approx(0.45)


def test_metrics_are_plain_json_numbers() -> None:
    import json

    t = trades([100, -50], r=[2.0, -1.0])
    m = compute_metrics(result_from(t, daily=[10_100, 10_050]))
    for section in ("trades", "equity", "costs", "min_balance"):
        for key, value in m[section].items():
            assert value is None or type(value) in (int, float, bool), (section, key, type(value))
    json.dumps({k: v for k, v in m.items() if k != "confidence_95"})
