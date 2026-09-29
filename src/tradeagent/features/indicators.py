"""Technical indicators (SPEC §2.3). Each value at bar t uses bars <= t only.

Only ATR for now (Phase 2 needs it for stop distances); the rest arrive in Phase 3.
"""

import numpy as np
import pandas as pd


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
