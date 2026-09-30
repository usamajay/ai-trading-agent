"""How much wider is the real spread than the spread MT5 stores per bar?

MT5 stores the *minimum* spread seen inside each bar. From real bid/ask ticks we
measure, per bar, the time-weighted average spread and the spread at the bar's
open, and compare both with the stored value. The result suggests the
`spread_margin_multiple` used by backtests (docs/PHASE_2_TASKS.md 2.4).
"""

import math

import numpy as np
import pandas as pd

SPREAD_COLUMNS = ["bar_spread", "tick_min", "tick_twavg", "tick_open", "ticks"]


def bar_tick_spreads(
    ticks: pd.DataFrame, bars: pd.DataFrame, point: float, bar_minutes: int
) -> pd.DataFrame:
    """Per bar (index time_utc): stored spread and the spreads seen in its ticks, in points.

    Each tick's spread counts for as long as it lasted, until the next tick or the
    bar's end. Bars without ticks or with a stored spread of 0 are left out.
    """
    t = ticks.sort_values("time_utc").reset_index(drop=True)
    spread = ((t["ask"] - t["bid"]) / point).round()
    bar = t["time_utc"].dt.floor(f"{bar_minutes}min")
    bar_end = bar + pd.Timedelta(minutes=bar_minutes)
    next_tick = t["time_utc"].shift(-1).fillna(bar_end)
    lasted = (np.minimum(next_tick, bar_end) - t["time_utc"]).dt.total_seconds().clip(lower=0)
    per_bar = (
        pd.DataFrame({"bar": bar, "spread": spread, "weighted": spread * lasted, "lasted": lasted})
        .groupby("bar")
        .agg(
            tick_min=("spread", "min"),
            tick_open=("spread", "first"),
            weighted=("weighted", "sum"),
            lasted=("lasted", "sum"),
            ticks=("spread", "size"),
        )
    )
    per_bar["tick_twavg"] = per_bar["weighted"] / per_bar["lasted"].replace(0, np.nan)
    per_bar["tick_twavg"] = per_bar["tick_twavg"].fillna(per_bar["tick_open"])
    stored = bars.set_index("time_utc")["spread"].rename("bar_spread")
    joined = per_bar.join(stored, how="inner")
    joined = joined[joined["bar_spread"] > 0]
    return joined[SPREAD_COLUMNS]


def summarize(spreads: pd.DataFrame) -> dict[str, float]:
    """Headline numbers: how often the stored spread is the minimum, and the ratios."""
    if spreads.empty:
        return {"bars": 0}
    twavg = spreads["tick_twavg"] / spreads["bar_spread"]
    at_open = spreads["tick_open"] / spreads["bar_spread"]
    return {
        "bars": float(len(spreads)),
        "stored_is_min": float((spreads["bar_spread"] == spreads["tick_min"]).mean()),
        "twavg_median": float(twavg.median()),
        "twavg_mean": float(twavg.mean()),
        "twavg_p90": float(twavg.quantile(0.9)),
        "twavg_p99": float(twavg.quantile(0.99)),
        "twavg_max": float(twavg.max()),
        "open_p99": float(at_open.quantile(0.99)),
        "open_max": float(at_open.max()),
    }


def propose_multiple(summaries: list[dict[str, float]], step: float = 0.05) -> float:
    """Smallest multiple (rounded up to `step`) covering the 99th-percentile ratio of
    time-weighted to stored spread across all symbols, never below 1."""
    p99 = [s["twavg_p99"] for s in summaries if s.get("bars", 0) > 0]
    if not p99:
        return 1.0
    return round(max(1.0, math.ceil(round(max(p99) / step, 6)) * step), 4)
