"""New York-close D1 and H4 bars built from H1 (SPEC §7.1, docs/DATA_NOTES.md §2).

Exness cuts D1/H4 at 00:00 UTC, which turns the 1-2 hour Sunday session into its
own tiny "stub" candle. Backtests and indicators use these bars instead:
- D1: one bar per trading day, 17:00 -> 17:00 New York time (5 per week).
- H4: six bars per trading day starting 17, 21, 01, 05, 09, 13 New York time.

Raw broker bars on disk are never changed; these are computed on demand.
A bar is only built from H1 bars that exist, so a data hole is never "bridged":
a D1/H4 bar with fewer H1 bars than a normal day is marked `partial`.
"""

from typing import Literal

import numpy as np
import pandas as pd

from tradeagent.data.market_hours import DAY_END_HOUR, NEW_YORK, to_new_york
from tradeagent.data.mt5_client import BAR_COLUMNS

NyTimeframe = Literal["D1", "H4"]

H4_HOURS = 4
# H1 bars in a full trading day: 18:00 -> 17:00 New York (17:00-18:00 is the break).
FULL_DAY_H1 = 23
# H1 bars in each full H4 block; the first (17-21 New York) loses the break hour.
FULL_H4_H1 = (3, 4, 4, 4, 4, 4)

RESAMPLED_COLUMNS = [*BAR_COLUMNS, "end_utc", "trading_day", "h1_bars", "partial"]

_DAY_SHIFT = pd.Timedelta(hours=24 - DAY_END_HOUR)


def resample_ny_close(h1: pd.DataFrame, timeframe: NyTimeframe) -> pd.DataFrame:
    """Build New York-close D1 or H4 bars from H1 bars.

    Returns BAR_COLUMNS plus:
      end_utc      when the bar's time window ends (the bar is closed after this)
      trading_day  the 17:00 -> 17:00 New York day the bar belongs to (a date)
      h1_bars      how many H1 bars went into it
      partial      fewer H1 bars than a full day/block (hole, holiday, early close,
                   or the edge of the data); never generate signals from these
    """
    if timeframe not in ("D1", "H4"):
        raise ValueError(f"timeframe must be D1 or H4, got {timeframe!r}")
    missing = set(BAR_COLUMNS) - set(h1.columns)
    if missing:
        raise ValueError(f"H1 bars are missing columns: {sorted(missing)}")
    if h1.empty:
        return _empty()

    df = h1.drop_duplicates("time_utc", keep="last").sort_values("time_utc")
    # New York wall clock moved so 17:00 becomes midnight: its date is the trading day.
    shifted = to_new_york(df["time_utc"]).dt.tz_localize(None) + _DAY_SHIFT
    day = shifted.dt.normalize()
    block = shifted.dt.hour // H4_HOURS if timeframe == "H4" else pd.Series(0, index=df.index)

    grouped = df.assign(_day=day, _block=block).groupby(["_day", "_block"], sort=True)
    out = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        tick_volume=("tick_volume", "sum"),
        spread=("spread", "median"),
        h1_bars=("time_utc", "size"),
    ).reset_index()

    hours = 24 if timeframe == "D1" else H4_HOURS
    start_wall = out["_day"] - _DAY_SHIFT + out["_block"] * pd.Timedelta(hours=H4_HOURS)
    # Daylight saving changes at 02:00 Sunday while the market is closed, so these
    # 17/21/01/05/09/13 New York wall times always exist and are never ambiguous.
    out["time_utc"] = start_wall.dt.tz_localize(NEW_YORK).dt.tz_convert("UTC")
    out["end_utc"] = out["time_utc"] + pd.Timedelta(hours=hours)
    out["trading_day"] = out["_day"].dt.date
    # A bar's spread is usually its minimum, so round the median up, never down.
    out["spread"] = np.ceil(out["spread"]).astype("int64")
    out["tick_volume"] = out["tick_volume"].astype("int64")
    full = FULL_DAY_H1 if timeframe == "D1" else out["_block"].map(dict(enumerate(FULL_H4_H1)))
    out["partial"] = out["h1_bars"] < full
    out["h1_bars"] = out["h1_bars"].astype("int64")
    return out[RESAMPLED_COLUMNS].reset_index(drop=True)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "tick_volume": pd.Series(dtype="int64"),
            "spread": pd.Series(dtype="int64"),
            "end_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "trading_day": pd.Series(dtype="object"),
            "h1_bars": pd.Series(dtype="int64"),
            "partial": pd.Series(dtype="bool"),
        }
    )
