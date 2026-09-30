"""Mean reversion (SPEC §4.1 #3): fade Bollinger extremes back to the mean in a range.

On closed M15 bars, when ADX(14) < `adx_max` (no strong trend):
- long: the previous close was below the lower band (`bb_std` sigma) and this close
  is back inside it; target the middle band;
- short mirrors this at the upper band.
Stop `atr_mult` x ATR(14) from the expected entry. Setups whose reward:risk to the
middle band is below 2 are skipped (the risk rules would reject them). Intraday.
"""

from collections.abc import Sequence

import pandas as pd

from tradeagent.features.indicators import adx, atr, bollinger
from tradeagent.strategies import registry
from tradeagent.strategies.base import BaseStrategy, MarketContext, ParamSpec, Signal, Style
from tradeagent.strategies.common import finite

NAME = "mean_reversion_bb"
DEFAULTS = {"bb_std": 2.0, "adx_max": 20.0, "atr_mult": 1.0}
MIN_RR = 2.0  # config/risk.yaml min_reward_risk (read-only copy for the setup filter)


class MeanReversionBB(BaseStrategy):
    name = NAME
    version = "1.0"
    style: Style = "intraday"
    timeframes: Sequence[str] = ("M15",)
    suited_regimes: Sequence[str] = ("ranging", "low_vol")
    lookback_bars = 30
    param_specs: Sequence[ParamSpec] = (
        ParamSpec("bb_std", 1.5, 3.0, "Bollinger band width in standard deviations"),
        ParamSpec("adx_max", 15.0, 30.0, "ADX(14) must be below this (ranging market)"),
        ParamSpec("atr_mult", 0.5, 2.0, "stop distance in ATR(14)"),
    )

    def __init__(self, seed: int = 0, params: dict[str, float] | None = None) -> None:
        super().__init__(**{**DEFAULTS, **(params or {})})

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out = dict(frames)
        m = out["M15"]
        bands = bollinger(m["close"], 20, self.params["bb_std"])
        out["M15"] = pd.concat([m, bands, adx(m, 14)], axis=1).assign(atr14=atr(m, 14))
        return out

    def generate(self, ctx: MarketContext) -> list[Signal]:
        m = ctx.bars("M15")
        if len(m) < 2:
            return []
        close, close_before = m.last("close"), m.last("close", 1)
        lower, upper, mid = m.last("bb_lower"), m.last("bb_upper"), m.last("bb_mid")
        lower_before, upper_before = m.last("bb_lower", 1), m.last("bb_upper", 1)
        trend, a = m.last("adx"), m.last("atr14")
        if not finite(lower, upper, mid, lower_before, upper_before, trend, a) or a <= 0:
            return []
        if trend >= self.params["adx_max"]:
            return []
        stop = self.params["atr_mult"] * a
        why = f"ADX {trend:.0f} < {self.params['adx_max']:.0f}; "
        if close_before < lower_before and close > lower:
            entry = ctx.expected_entry("long")
            if mid - entry >= MIN_RR * stop:
                why += "close back inside the lower band; target the mean"
                return [Signal(ctx.symbol, "long", entry - stop, mid, why=why)]
        if close_before > upper_before and close < upper:
            entry = ctx.expected_entry("short")
            if entry - mid >= MIN_RR * stop:
                why += "close back inside the upper band; target the mean"
                return [Signal(ctx.symbol, "short", entry + stop, mid, why=why)]
        return []


registry.register(NAME, lambda seed, params: MeanReversionBB(seed, params))
