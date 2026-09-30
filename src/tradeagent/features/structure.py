"""Market structure (SPEC §2.3): swings, breaks of structure, session levels.

Everything is known only when a live trader would know it:
- A swing high at bar p is a high strictly above the `k` highs before and after
  it. It is **confirmed at bar p + k** (the k later bars must close first), so the
  columns below only show it from bar p + k on. Swing lows mirror this.
- Break of structure (BOS) up at bar t: close above the last swing high confirmed
  by bar t - 1, after the previous close was not above it. Down mirrors this.
- Asian range: high/low of the 18:00-03:00 New York part of each trading day,
  shown on bars that start from 03:00 until the 17:00 close of that day.
- Previous trading day high/low (17:00 New York days), shown on every bar of the
  next trading day.
"""

import numpy as np
import pandas as pd

from tradeagent.data.market_hours import to_new_york, trading_day

ASIA_START_HOUR, ASIA_END_HOUR = 18, 3  # New York


def swing_points(bars: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """is_swing_high / is_swing_low at the pivot bar itself (for inspection only:
    using these at the pivot bar would be look-ahead; use `swings`)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    high, low = bars["high"], bars["low"]
    is_high = pd.Series(True, index=bars.index)
    is_low = pd.Series(True, index=bars.index)
    for j in range(1, k + 1):
        is_high &= (high > high.shift(j)) & (high > high.shift(-j))
        is_low &= (low < low.shift(j)) & (low < low.shift(-j))
    return pd.DataFrame({"is_swing_high": is_high, "is_swing_low": is_low})


def swings(bars: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Latest confirmed swing levels known at each bar's close.

    Columns: swing_high, swing_low (price of the latest confirmed swing),
    swing_high_bar, swing_low_bar (index position of that pivot bar).
    """
    points = swing_points(bars, k)
    position = pd.Series(np.arange(len(bars)), index=bars.index, dtype="float64")
    # A pivot at p becomes known at p + k: shift everything k bars forward.
    high_level = bars["high"].where(points["is_swing_high"]).shift(k)
    low_level = bars["low"].where(points["is_swing_low"]).shift(k)
    high_bar = position.where(points["is_swing_high"]).shift(k)
    low_bar = position.where(points["is_swing_low"]).shift(k)
    return pd.DataFrame(
        {
            "swing_high": high_level.ffill(),
            "swing_low": low_level.ffill(),
            "swing_high_bar": high_bar.ffill(),
            "swing_low_bar": low_bar.ffill(),
        }
    )


def break_of_structure(bars: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """bos_up / bos_down flags and the level that was broken (bos_level)."""
    s = swings(bars, k)
    close, prev_close = bars["close"], bars["close"].shift()
    high_before, low_before = s["swing_high"].shift(), s["swing_low"].shift()
    up = (close > high_before) & ~(prev_close > high_before)
    down = (close < low_before) & ~(prev_close < low_before)
    level = pd.Series(np.nan, index=bars.index)
    level[up] = high_before[up]
    level[down] = low_before[down]
    return pd.DataFrame(
        {"bos_up": up.fillna(False), "bos_down": down.fillna(False), "bos_level": level}
    )


def asian_range(bars: pd.DataFrame) -> pd.DataFrame:
    """asia_high / asia_low of the trading day, on bars starting 03:00-17:00 New York."""
    hour = to_new_york(bars["time_utc"]).dt.hour
    day = pd.Series(trading_day(bars["time_utc"]), index=bars.index)
    in_asia = (hour >= ASIA_START_HOUR) | (hour < ASIA_END_HOUR)
    highs = bars["high"].where(in_asia).groupby(day).transform("max")
    lows = bars["low"].where(in_asia).groupby(day).transform("min")
    after = (hour >= ASIA_END_HOUR) & (hour < 17)  # every Asian bar has closed by then
    return pd.DataFrame({"asia_high": highs.where(after), "asia_low": lows.where(after)})


def previous_day_levels(bars: pd.DataFrame) -> pd.DataFrame:
    """prev_day_high / prev_day_low: the previous trading day with bars (17:00 New York)."""
    day = pd.Series(trading_day(bars["time_utc"]), index=bars.index)
    per_day = bars.groupby(day).agg(high=("high", "max"), low=("low", "min"))
    previous = per_day.shift()
    return pd.DataFrame(
        {"prev_day_high": day.map(previous["high"]), "prev_day_low": day.map(previous["low"])},
        index=bars.index,
    )
