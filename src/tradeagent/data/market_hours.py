"""Market hours for XAUUSD/USOIL, in New York time (docs/DATA_NOTES.md §1).

Measured from Exness XAUUSDm/USOILm history: both trade Sunday 18:00 to Friday
17:00 New York time, with a daily break 17:00-18:00 New York time (Mon-Thu).
Using New York time makes the rules follow US daylight saving automatically.

A "trading day" runs 17:00 -> 17:00 New York time and is named after the day it
ends on, so the Sunday evening session belongs to Monday (the FX/metals standard).
Shared by data validation and the backtester, so both use one definition.
"""

import pandas as pd

NEW_YORK = "America/New_York"

# Closure windows in New York time, widened a little: some sessions close early
# (USOIL often at 16:45) and the first bar after a reopen can come up to ~25 min late.
CLOSE_FROM_MIN = 16 * 60 + 30  # 16:30
OPEN_BY_MIN = 18 * 60 + 30  # 18:30
EARLY_CLOSE_FROM_MIN = 12 * 60 + 45  # US-holiday early closes seen at 13:00-14:45
EARLY_CLOSE_TO_MIN = 14 * 60 + 45
WEEK_CLOSE_FROM = 4 * 1440 + CLOSE_FROM_MIN  # Friday 16:30, in minutes since Monday 00:00
WEEK_OPEN_BY = 6 * 1440 + OPEN_BY_MIN  # Sunday 18:30
# Exact weekend closure (no slack), for spotting bars that should not exist.
WEEKEND_STRICT_FROM = 4 * 1440 + 17 * 60  # Friday 17:00
WEEKEND_STRICT_TO = 6 * 1440 + 18 * 60  # Sunday 18:00

SESSION_OPEN_MIN = 18 * 60  # every session (weekly or daily) opens at 18:00 New York
DAY_END_HOUR = 17  # the trading day ends at 17:00 New York
# Adding this to a New York wall-clock time moves 17:00 to midnight, so the date of
# the result is the trading day.
_DAY_SHIFT = pd.Timedelta(hours=24 - DAY_END_HOUR)


def to_new_york(times_utc: pd.Series) -> pd.Series:
    """UTC times -> New York times (still timezone-aware)."""
    return times_utc.dt.tz_convert(NEW_YORK)


def week_minutes(ny: pd.Series) -> pd.Series:
    """Minutes since Monday 00:00 (New York time)."""
    return ny.dt.dayofweek * 1440 + ny.dt.hour * 60 + ny.dt.minute


def minute_of_day(ny: pd.Series) -> pd.Series:
    return ny.dt.hour * 60 + ny.dt.minute


def in_closure(ny: pd.Series) -> pd.Series:
    """True where a New York time falls in the weekend or a daily break window."""
    week = week_minutes(ny)
    minutes = minute_of_day(ny)
    weekend = (week >= WEEK_CLOSE_FROM) & (week <= WEEK_OPEN_BY)
    daily = (ny.dt.dayofweek <= 3) & (minutes >= CLOSE_FROM_MIN) & (minutes <= OPEN_BY_MIN)
    return weekend | daily


def trading_day(times_utc: pd.Series) -> pd.Series:
    """The trading day (17:00 -> 17:00 New York) each UTC time belongs to, as a date.

    Sunday 18:00 New York -> Monday; Friday 16:59 New York -> Friday.
    """
    wall = to_new_york(times_utc).dt.tz_localize(None)  # New York wall clock
    return (wall + _DAY_SHIFT).dt.date


def trading_day_start(days: pd.Series) -> pd.Series:
    """UTC time when each trading day starts (17:00 New York on the calendar day before).

    Daylight-saving changes happen at 02:00 on a Sunday, while the market is closed,
    so 17:00 New York is never ambiguous here.
    """
    wall = pd.to_datetime(pd.Series(days)) - _DAY_SHIFT
    return wall.dt.tz_localize(NEW_YORK).dt.tz_convert("UTC")


def minutes_since_session_open(times_utc: pd.Series) -> pd.Series:
    """Minutes since the last scheduled 18:00 New York session open (by the clock).

    Times inside the daily break or the weekend are measured from the previous
    open, so they give large values; use together with `in_closure`.
    """
    ny = to_new_york(times_utc)
    return (minute_of_day(ny) - SESSION_OPEN_MIN) % 1440


def after_weekly_cutoff(times_utc: pd.Series, cutoff_min: int = CLOSE_FROM_MIN) -> pd.Series:
    """True from Friday `cutoff_min` (New York, default 16:30) until the Sunday open."""
    week = week_minutes(to_new_york(times_utc))
    return (week >= 4 * 1440 + cutoff_min) & (week < WEEKEND_STRICT_TO)
