"""Session breakout (SPEC §4.1 #4): break of the Asian range in London/New York hours.

Asian range = high/low of 18:00-03:00 New York. From 03:00 to 11:00 New York, the
first M15 close beyond the range plus `buffer_atr` x ATR(14) enters in that
direction; at most one signal per trading day. Days whose range is wider than
`max_range_atr` x ATR are skipped. Stop at the range midpoint, kept within
0.5-3 x ATR; target `rr` x the stop. Intraday.
"""

from collections.abc import Sequence

import pandas as pd

from tradeagent.data.market_hours import to_new_york, trading_day
from tradeagent.features.indicators import atr
from tradeagent.features.structure import asian_range
from tradeagent.strategies import registry
from tradeagent.strategies.base import (
    BaseStrategy,
    Direction,
    MarketContext,
    ParamSpec,
    Signal,
    Style,
)
from tradeagent.strategies.common import finite, market_signal

NAME = "session_breakout"
DEFAULTS = {"buffer_atr": 0.1, "rr": 2.0, "max_range_atr": 6.0}
WINDOW_START, WINDOW_END = 3, 11  # New York hours in which a breakout may trigger
STOP_MIN_ATR, STOP_MAX_ATR = 0.55, 2.95  # just inside the 0.5-3 x ATR risk rule


class SessionBreakout(BaseStrategy):
    name = NAME
    version = "1.0"
    style: Style = "intraday"
    timeframes: Sequence[str] = ("M15",)
    suited_regimes: Sequence[str] = ("any",)
    lookback_bars = 40  # the Asian session is 36 M15 bars
    param_specs: Sequence[ParamSpec] = (
        ParamSpec("buffer_atr", 0.0, 0.5, "distance beyond the range, in ATR(14), to count"),
        ParamSpec("rr", 2.0, 4.0, "target as a multiple of the stop distance"),
        ParamSpec("max_range_atr", 2.0, 10.0, "skip days whose Asian range is wider (in ATR)"),
    )

    def __init__(self, seed: int = 0, params: dict[str, float] | None = None) -> None:
        super().__init__(**{**DEFAULTS, **(params or {})})
        self.last_signal_day = -1.0

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out = dict(frames)
        m = out["M15"]
        day = pd.to_datetime(pd.Series(trading_day(m["time_utc"]), index=m.index))
        out["M15"] = pd.concat([m, asian_range(m)], axis=1).assign(
            atr14=atr(m, 14),
            ny_hour=to_new_york(m["time_utc"]).dt.hour.astype(float),
            day_id=(day - pd.Timestamp("1970-01-01")).dt.days.astype(float),
        )
        return out

    def generate(self, ctx: MarketContext) -> list[Signal]:
        m = ctx.bars("M15")
        hour, day = m.last("ny_hour"), m.last("day_id")
        if not (WINDOW_START <= hour < WINDOW_END) or day == self.last_signal_day:
            return []
        high, low, a, close = (
            m.last("asia_high"),
            m.last("asia_low"),
            m.last("atr14"),
            m.last("close"),
        )
        if not finite(high, low, a) or a <= 0:
            return []
        if high - low > self.params["max_range_atr"] * a:
            return []
        buffer = self.params["buffer_atr"] * a
        direction: Direction
        if close > high + buffer:
            direction, side = "long", "above the high"
        elif close < low - buffer:
            direction, side = "short", "below the low"
        else:
            return []
        entry = ctx.expected_entry(direction)
        mid = (high + low) / 2
        stop = min(max(abs(entry - mid), STOP_MIN_ATR * a), STOP_MAX_ATR * a)
        self.last_signal_day = day
        why = f"M15 close {side} of the Asian range {low:.2f}-{high:.2f}; stop at the midpoint"
        return [market_signal(ctx, direction, stop, self.params["rr"], why)]


registry.register(NAME, lambda seed, params: SessionBreakout(seed, params))
