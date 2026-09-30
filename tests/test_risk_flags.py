"""Risk-limit flags on daily mark-to-market equity (informational until Phase 4)."""

import datetime as dt

import pandas as pd
import pytest

from tradeagent.backtest.risk_flags import risk_limit_flags
from tradeagent.config import load_config

LIMITS = load_config().risk  # 2% daily, 4% weekly, 10% drawdown


def daily(values: list[float], first: str = "2026-01-05") -> pd.DataFrame:
    """One value per weekday from `first` (a Monday)."""
    days = pd.bdate_range(first, periods=len(values)).date
    return pd.DataFrame({"trading_day": days, "equity": values})


def test_limits_come_from_risk_yaml() -> None:
    assert (LIMITS.max_daily_loss_pct, LIMITS.max_weekly_loss_pct, LIMITS.max_drawdown_pct) == (
        2.0,
        4.0,
        10.0,
    )


def test_daily_loss() -> None:
    # Mon -1% (9,900), Tue 9,700 = -2.02% of 9,900 -> breach; Wed 9,600 = -1.03%.
    flags = risk_limit_flags(daily([9_900, 9_700, 9_600]), 10_000, LIMITS)["daily_loss"]
    assert flags["breached"] and flags["first_breach_day"] == "2026-01-06"
    assert flags["breaches"] == 1
    assert flags["worst_pct"] == pytest.approx(200 / 9_900 * 100)


def test_weekly_loss_counts_from_the_previous_week_close() -> None:
    # Week 1 ends at 10,000. Week 2 falls 1% a day: Thursday is 3.94% down, Friday 4.9%.
    week1 = [10_000] * 5
    week2 = [9_900, 9_801, 9_703, 9_606, 9_510]
    flags = risk_limit_flags(daily(week1 + week2), 10_000, LIMITS)
    assert not flags["daily_loss"]["breached"]  # never 2% in one day
    weekly = flags["weekly_loss"]
    assert weekly["breached"] and weekly["first_breach_day"] == "2026-01-16"
    assert weekly["worst_pct"] == pytest.approx(4.9)


def test_week_resets_after_a_recovery() -> None:
    # -3% in week 1, then week 2 starts from 9,700 and falls 1.5% -> no weekly breach.
    values = [9_900, 9_800, 9_700, 9_700, 9_700, 9_600, 9_555, 9_555, 9_555, 9_555]
    assert not risk_limit_flags(daily(values), 10_000, LIMITS)["weekly_loss"]["breached"]


def test_max_drawdown_from_the_running_peak() -> None:
    # Peak 11,000, then 9,900 = exactly 10% below it -> breach on that day.
    flags = risk_limit_flags(daily([11_000, 10_500, 9_900, 10_200]), 10_000, LIMITS)
    dd = flags["max_drawdown"]
    assert dd["breached"] and dd["first_breach_day"] == "2026-01-07"
    assert dd["worst_pct"] == pytest.approx(10.0)


def test_nothing_breached_and_empty() -> None:
    calm = risk_limit_flags(daily([10_050, 10_020, 10_080]), 10_000, LIMITS)
    assert not any(rule["breached"] for rule in calm.values())
    empty = risk_limit_flags(daily([]), 10_000, LIMITS)
    assert empty["daily_loss"] == {
        "limit_pct": 2.0,
        "breached": False,
        "first_breach_day": None,
        "breaches": 0,
        "worst_pct": 0.0,
    }
    assert isinstance(daily([1.0])["trading_day"].iloc[0], dt.date)
