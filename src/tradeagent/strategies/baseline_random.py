"""Baseline: random entries with the same stop/target rules (SPEC §4.1 #6).

Every real strategy must beat this. On each decision bar it enters with
probability `p_entry`, long or short at random (fixed seed), with the stop
`atr_mult` x ATR(14) away and the target `rr` x the stop distance, both measured
from the price it will actually pay (ask for longs, bid for shorts, with the spread
margin). With no edge, its expectancy should be about minus its costs.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from tradeagent.features.indicators import atr
from tradeagent.strategies import registry
from tradeagent.strategies.base import BaseStrategy, MarketContext, ParamSpec, Signal, Style

NAME = "random_baseline"
DEFAULTS = {"atr_mult": 1.5, "rr": 2.0, "p_entry": 0.02}


class RandomBaseline(BaseStrategy):
    name = NAME
    version = "1.0"
    style: Style = "intraday"
    timeframes: Sequence[str] = ("M15",)
    suited_regimes: Sequence[str] = ("any",)
    lookback_bars = 15  # ATR(14) needs 15 bars
    param_specs: Sequence[ParamSpec] = (
        ParamSpec("atr_mult", 0.5, 3.0, "stop distance in ATR(14) of the decision timeframe"),
        ParamSpec("rr", 1.5, 4.0, "target distance as a multiple of the stop distance"),
        ParamSpec("p_entry", 0.001, 0.2, "chance of entering on any decision bar"),
    )

    def __init__(
        self,
        seed: int = 0,
        params: dict[str, float] | None = None,
        timeframe: str = "M15",
        style: Style = "intraday",
    ) -> None:
        """`timeframe` and `style` let the baseline match the strategy it is compared with."""
        super().__init__(**{**DEFAULTS, **(params or {})})
        self.timeframes = (timeframe,)
        self.style = style
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        tf = self.timeframes[0]
        out = dict(frames)
        out[tf] = out[tf].assign(atr14=atr(out[tf], 14))
        return out

    def generate(self, ctx: MarketContext) -> list[Signal]:
        # Two draws on every decision, so a run is repeatable from its seed alone.
        enter, side = self.rng.random(), self.rng.random()
        view = ctx.bars(self.timeframes[0])
        a = view.last("atr14")
        if enter >= self.params["p_entry"] or not np.isfinite(a) or a <= 0:
            return []
        long = side < 0.5
        entry = ctx.expected_entry("long" if long else "short")
        stop = self.params["atr_mult"] * a
        target = self.params["rr"] * stop
        if long:
            sl, tp = entry - stop, entry + target
        else:
            sl, tp = entry + stop, entry - target
        return [
            Signal(
                symbol=ctx.symbol,
                direction="long" if long else "short",
                stop_loss=sl,
                take_profit=tp,
                why=f"random entry (seed {self.seed}), stop {self.params['atr_mult']} x ATR",
            )
        ]


registry.register(NAME, lambda seed, params: RandomBaseline(seed, params))
