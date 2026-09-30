"""Compression breakout (SPEC §4.1 #2): a Bollinger squeeze, then a break with volume.

On closed M15 bars:
- squeeze: the previous bar's Bollinger bandwidth (20, 2) ranks in the lowest
  `squeeze_pct` of the last 100 bars;
- break: the close is above the upper band (long) or below the lower band (short),
  and this bar's tick volume is above `vol_mult` x the average of the 20 bars before.
Stop `atr_mult` x ATR(14) from the expected entry, target `rr` x the stop. Intraday.
"""

from collections.abc import Sequence

import pandas as pd

from tradeagent.features.indicators import atr, average_volume, bollinger
from tradeagent.strategies import registry
from tradeagent.strategies.base import BaseStrategy, MarketContext, ParamSpec, Signal, Style
from tradeagent.strategies.common import finite, market_signal

NAME = "breakout_compression"
DEFAULTS = {"squeeze_pct": 0.2, "vol_mult": 1.5, "atr_mult": 1.5, "rr": 2.0}
RANK_WINDOW = 100


class BreakoutCompression(BaseStrategy):
    name = NAME
    version = "1.0"
    style: Style = "intraday"
    timeframes: Sequence[str] = ("M15",)
    suited_regimes: Sequence[str] = ("low_vol_expanding",)
    lookback_bars = RANK_WINDOW + 20
    param_specs: Sequence[ParamSpec] = (
        ParamSpec("squeeze_pct", 0.05, 0.4, "bandwidth percentile (last 100 bars) for a squeeze"),
        ParamSpec("vol_mult", 1.0, 3.0, "breakout volume vs the 20-bar average"),
        ParamSpec("atr_mult", 1.0, 3.0, "stop distance in ATR(14)"),
        ParamSpec("rr", 2.0, 4.0, "target as a multiple of the stop distance"),
    )

    def __init__(self, seed: int = 0, params: dict[str, float] | None = None) -> None:
        super().__init__(**{**DEFAULTS, **(params or {})})

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out = dict(frames)
        m = out["M15"]
        bb = bollinger(m["close"], 20, 2.0)
        out["M15"] = pd.concat([m, bb], axis=1).assign(
            width_rank=bb["bb_width"].rolling(RANK_WINDOW).rank(pct=True),
            avg_vol_before=average_volume(m, 20).shift(),  # the 20 bars before this one
            atr14=atr(m, 14),
            volume=m["tick_volume"].astype(float),
        )
        return out

    def generate(self, ctx: MarketContext) -> list[Signal]:
        m = ctx.bars("M15")
        if len(m) < 2:
            return []
        rank_before = m.last("width_rank", 1)
        close, upper, lower = m.last("close"), m.last("bb_upper"), m.last("bb_lower")
        volume, avg_vol, a = m.last("volume"), m.last("avg_vol_before"), m.last("atr14")
        if not finite(rank_before, upper, lower, avg_vol, a) or a <= 0 or avg_vol <= 0:
            return []
        if rank_before > self.params["squeeze_pct"]:
            return []
        if volume <= self.params["vol_mult"] * avg_vol:
            return []
        stop, rr = self.params["atr_mult"] * a, self.params["rr"]
        why = (
            f"squeeze (bandwidth rank {rank_before:.0%} of 100 bars), "
            f"volume {volume / avg_vol:.1f}x average"
        )
        if close > upper:
            return [market_signal(ctx, "long", stop, rr, why + ", close above upper band")]
        if close < lower:
            return [market_signal(ctx, "short", stop, rr, why + ", close below lower band")]
        return []


registry.register(NAME, lambda seed, params: BreakoutCompression(seed, params))
