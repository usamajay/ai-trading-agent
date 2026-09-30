"""Market regime labels, research-only version (docs/PHASE_6_TASKS.md 6.4).

Deterministic, from closed bars only (a label at bar t uses bars <= t):
- trend: ADX(14) >= 25 -> `trending`; ADX < 20 -> `ranging`; in between: neither.
- volatility: rank of ATR(14) among the last 250 bars (including this one):
  bottom third `low_vol`, top third `high_vol`, else `normal_vol`.
- `low_vol_expanding`: `low_vol` on at least one of the previous 10 bars and ATR now
  >= 1.1 x ATR 5 bars ago (a quiet market starting to move).
Labels are 0/1 float columns `regime_<name>` (0 during warm-up), because strategy
frames carry numeric columns only. The live detector and meta-agent are Phase 8.
"""

import numpy as np
import pandas as pd

from tradeagent.data.mt5_client import TIMEFRAMES
from tradeagent.features.indicators import adx, atr

DETECTOR_VERSION = "research-1"
REGIMES = ("trending", "ranging", "low_vol", "normal_vol", "high_vol", "low_vol_expanding")
TREND_ADX, RANGE_ADX = 25.0, 20.0
VOL_WINDOW, EXPAND_LOOKBACK, EXPAND_SHIFT, EXPAND_RATIO = 250, 10, 5, 1.1


def _rank_in_window(values: np.ndarray) -> float:
    last = values[-1]
    return float(np.mean(values <= last)) if np.isfinite(last) else np.nan


def regime_labels(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per bar: `regime_<name>` = 1.0 when the bar is in that regime, else 0.0."""
    a = adx(bars)["adx"]
    vol = atr(bars, 14)
    rank = vol.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).apply(_rank_in_window, raw=True)
    low = rank <= 1 / 3
    out = pd.DataFrame(index=bars.index)
    out["regime_trending"] = a >= TREND_ADX
    out["regime_ranging"] = a < RANGE_ADX
    out["regime_low_vol"] = low
    out["regime_high_vol"] = rank > 2 / 3
    out["regime_normal_vol"] = rank.notna() & ~low & ~(rank > 2 / 3)
    was_low = low.shift(1, fill_value=False).rolling(EXPAND_LOOKBACK, min_periods=1).max() > 0
    out["regime_low_vol_expanding"] = was_low & (vol >= EXPAND_RATIO * vol.shift(EXPAND_SHIFT))
    return out.astype(float)  # NaN comparisons are False -> 0.0 during warm-up


def bar_end(bars: pd.DataFrame, timeframe: str) -> pd.Series:
    if "end_utc" in bars.columns:
        return bars["end_utc"]
    return bars["time_utc"] + pd.Timedelta(TIMEFRAMES[timeframe][1])


def trade_regimes(trades: pd.DataFrame, bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Trend and volatility label of the decision bar each trade's signal came from."""
    labels = regime_labels(bars)
    trend = np.select(
        [labels["regime_trending"] == 1, labels["regime_ranging"] == 1],
        ["trending", "ranging"],
        "neither",
    )
    vol = np.select(
        [
            labels["regime_low_vol"] == 1,
            labels["regime_high_vol"] == 1,
            labels["regime_normal_vol"] == 1,
        ],
        ["low_vol", "high_vol", "normal_vol"],
        "warm_up",
    )
    table = pd.DataFrame({"end": bar_end(bars, timeframe).to_numpy(), "trend": trend, "vol": vol})
    by_end = table.drop_duplicates("end", keep="last").set_index("end")
    found = by_end.reindex(pd.DatetimeIndex(trades["signal_time"]))
    return pd.DataFrame(
        {
            "trend": found["trend"].fillna("unknown").to_numpy(),
            "vol": found["vol"].fillna("unknown").to_numpy(),
        },
        index=trades.index,
    )
