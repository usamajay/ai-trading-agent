"""Market-hours helpers (New York time, docs/DATA_NOTES.md §1)."""

import datetime as dt

import pandas as pd
import pytest

from tradeagent.data.market_hours import (
    after_weekly_cutoff,
    minutes_since_session_open,
    trading_day,
    trading_day_start,
)

NY = "America/New_York"


def utc(*ny_times: str) -> pd.Series:
    """New York wall times -> a UTC Series, like stored bar times."""
    return pd.Series(pd.DatetimeIndex(ny_times).tz_localize(NY).tz_convert("UTC"))


@pytest.mark.parametrize(
    ("ny_time", "day"),
    [
        ("2026-01-04 18:00", "2026-01-05"),  # Sunday evening session belongs to Monday
        ("2026-01-05 16:59", "2026-01-05"),
        ("2026-01-05 18:00", "2026-01-06"),  # after the daily break: next trading day
        ("2026-01-08 17:30", "2026-01-09"),  # Thursday break belongs to Friday
        ("2026-01-09 16:59", "2026-01-09"),  # last minute of the week
        ("2026-07-05 18:00", "2026-07-06"),  # same rules in US summer time
        ("2026-07-10 16:00", "2026-07-10"),
    ],
)
def test_trading_day(ny_time: str, day: str) -> None:
    assert trading_day(utc(ny_time)).iloc[0] == dt.date.fromisoformat(day)


def test_trading_day_start_follows_daylight_saving() -> None:
    days = pd.Series([dt.date(2026, 1, 5), dt.date(2026, 7, 6)])
    starts = trading_day_start(days)
    # 17:00 New York the day before: 22:00 UTC in winter, 21:00 UTC in summer.
    assert list(starts) == [
        pd.Timestamp("2026-01-04 22:00", tz="UTC"),
        pd.Timestamp("2026-07-05 21:00", tz="UTC"),
    ]


def test_trading_day_start_round_trips() -> None:
    times = utc("2026-03-02 03:00", "2026-03-09 03:00", "2025-11-03 03:00")  # around DST
    starts = trading_day_start(trading_day(times))
    assert trading_day(starts + pd.Timedelta(minutes=1)).tolist() == trading_day(times).tolist()
    assert (starts.dt.tz_convert(NY).dt.hour == 17).all()


def test_minutes_since_session_open() -> None:
    times = utc("2026-01-04 18:10", "2026-01-05 18:00", "2026-01-05 16:59", "2026-07-06 18:14")
    assert minutes_since_session_open(times).tolist() == [10, 0, 1379, 14]


@pytest.mark.parametrize(
    ("ny_time", "expected"),
    [
        ("2026-01-08 16:30", False),  # Thursday: only Friday counts
        ("2026-01-09 16:29", False),
        ("2026-01-09 16:30", True),
        ("2026-01-10 12:00", True),  # Saturday
        ("2026-01-11 17:59", True),
        ("2026-01-11 18:00", False),  # weekly open
    ],
)
def test_after_weekly_cutoff(ny_time: str, expected: bool) -> None:
    assert bool(after_weekly_cutoff(utc(ny_time)).iloc[0]) is expected
