"""Technical indicators (SPEC §2.3). Each value at bar t uses bars <= t only.

Conventions (standard definitions; tests check each against hand-worked values):
- EMA(n): alpha 2/(n+1), seeded with the simple average of the first n values.
- Wilder smoothing (ATR, RSI, ADX): seeded with the simple average of the first n
  values, then avg_t = (avg_(t-1) * (n - 1) + x_t) / n.
- Bollinger bands: simple average of n closes +/- k population standard deviations.
- VWAP: typical price (high + low + close) / 3 weighted by tick volume, restarting at
  each trading day's start (17:00 New York).
Values before an indicator has enough history are NaN.
"""

import numpy as np
import pandas as pd

from tradeagent.data.market_hours import trading_day


def true_range(bars: pd.DataFrame) -> pd.Series:
    """max(high - low, |high - previous close|, |low - previous close|).

    The first bar has no previous close, so it uses high - low. A weekend or data
    gap stays inside the true range on purpose: the jump is real risk
    (docs/DATA_NOTES.md §3 rule 4).
    """
    prev_close = bars["close"].shift()
    ranges = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range with Wilder's smoothing.

    The first value (bar period-1) is the plain average of the first `period` true
    ranges; after that ATR_t = (ATR_{t-1} * (period - 1) + TR_t) / period.
    Earlier bars are NaN.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    tr = true_range(bars).to_numpy(dtype="float64")
    out = np.full(len(tr), np.nan)
    if len(tr) >= period:
        out[period - 1] = tr[:period].mean()
        for i in range(period, len(tr)):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(out, index=bars.index, name=f"atr_{period}")


def wilder(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing of `values`, starting at the first run of `period` valid values."""
    x = np.asarray(values, dtype="float64")
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < period:
        return out
    start = valid[0]
    seed_end = start + period - 1
    if np.isnan(x[start : seed_end + 1]).any():
        return out
    out[seed_end] = x[start : seed_end + 1].mean()
    for i in range(seed_end + 1, len(x)):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


def ema_values(values: np.ndarray, period: int) -> np.ndarray:
    """EMA seeded with the simple average of the first `period` valid values."""
    x = np.asarray(values, dtype="float64")
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < period:
        return out
    start = valid[0]
    seed_end = start + period - 1
    out[seed_end] = x[start : seed_end + 1].mean()
    alpha = 2 / (period + 1)
    for i in range(seed_end + 1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def ema(series: pd.Series, period: int) -> pd.Series:
    _check(period)
    return pd.Series(ema_values(series.to_numpy(), period), index=series.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI: 100 - 100 / (1 + average gain / average loss); 100 with no losses."""
    _check(period)
    delta = close.diff().to_numpy(dtype="float64")
    gain = wilder(np.where(np.isnan(delta), np.nan, np.maximum(delta, 0)), period)
    loss = wilder(np.where(np.isnan(delta), np.nan, np.maximum(-delta, 0)), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(loss == 0, 100.0, 100 - 100 / (1 + gain / loss))
    value[np.isnan(gain)] = np.nan
    return pd.Series(value, index=close.index)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line (EMA fast - EMA slow), its signal line (EMA of MACD) and histogram."""
    line = ema_values(close.to_numpy(), fast) - ema_values(close.to_numpy(), slow)
    sig = ema_values(line, signal)
    return pd.DataFrame(
        {"macd": line, "macd_signal": sig, "macd_hist": line - sig}, index=close.index
    )


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Middle (simple average), upper/lower bands and bandwidth = (upper - lower) / middle."""
    _check(period)
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper, lower = mid + num_std * std, mid - num_std * std
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": (upper - lower) / mid},
        index=close.index,
    )


def adx(bars: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder's ADX with +DI and -DI."""
    _check(period)
    high, low = bars["high"].to_numpy(float), bars["low"].to_numpy(float)
    up = np.r_[np.nan, high[1:] - high[:-1]]
    down = np.r_[np.nan, low[:-1] - low[1:]]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[0] = minus_dm[0] = np.nan
    tr = true_range(bars).to_numpy(float, copy=True)
    tr[0] = np.nan  # the first bar has no previous bar to move from
    tr_avg = wilder(tr, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100 * wilder(plus_dm, period) / tr_avg
        minus_di = 100 * wilder(minus_dm, period) / tr_avg
        total = plus_di + minus_di
        dx = np.where(total > 0, 100 * np.abs(plus_di - minus_di) / total, 0.0)
    dx[np.isnan(total)] = np.nan
    return pd.DataFrame(
        {"adx": wilder(dx, period), "plus_di": plus_di, "minus_di": minus_di}, index=bars.index
    )


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """VWAP restarting at each trading day (17:00 New York), weighted by tick volume."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    volume = bars["tick_volume"].astype(float)
    day = pd.Series(trading_day(bars["time_utc"]), index=bars.index)
    weighted = (typical * volume).groupby(day).cumsum()
    total = volume.groupby(day).cumsum()
    return (weighted / total.where(total > 0)).rename("vwap")


def average_volume(bars: pd.DataFrame, period: int = 20) -> pd.Series:
    """Simple average of tick volume over the last `period` bars (this bar included)."""
    _check(period)
    return bars["tick_volume"].astype(float).rolling(period).mean()


def _check(period: int) -> None:
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
