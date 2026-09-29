"""Synthetic bars that follow Exness market hours, shared by tests.

Market open: Sunday 18:00 -> Friday 17:00 New York time, daily break 17:00-18:00.
"""

import numpy as np
import pandas as pd

NY = "America/New_York"


def market_open(ny: pd.DatetimeIndex) -> np.ndarray:
    day, minute = ny.dayofweek, ny.hour * 60 + ny.minute
    weekend = (day == 5) | ((day == 4) & (minute >= 17 * 60)) | ((day == 6) & (minute < 18 * 60))
    daily_break = (day <= 3) & (minute >= 17 * 60) & (minute < 18 * 60)
    return ~(weekend | daily_break)


def trading_bars(first_sunday: str, weeks: int = 2, freq: str = "5min") -> pd.DataFrame:
    """Clean bars from Sunday 18:00 NY for `weeks` weeks, only while the market is open."""
    start = pd.Timestamp(f"{first_sunday} 18:00", tz=NY)
    ny = pd.date_range(start, start + pd.Timedelta(weeks=weeks), freq=freq, inclusive="left")
    ny = ny[market_open(ny)]
    rng = np.random.default_rng(7)
    close = 2000 + np.cumsum(rng.normal(0, 0.5, len(ny)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "time_utc": ny.tz_convert("UTC").astype("datetime64[ns, UTC]"),
            "open": open_,
            "high": np.maximum(open_, close) + 0.3,
            "low": np.minimum(open_, close) - 0.3,
            "close": close,
            "tick_volume": 100,
            "spread": 20,
        }
    )
