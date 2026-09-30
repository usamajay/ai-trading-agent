"""Market-structure retest (SPEC §4.1 #5): break of structure, then trade the retest.

On closed H1 bars, with the H4 trend agreeing (H4 close above EMA50 for longs):
- a bullish break of structure (close above the last confirmed swing high, swings
  of `swing_k` bars) places a **buy limit at the broken level** (the retest), valid
  for `retest_expiry` H1 bars;
- stop `atr_buffer` x ATR(14) H1 below the level; target `rr` x the stop.
Short mirrors this with a sell limit. Swing: may hold over weekends.
"""

from collections.abc import Sequence

import pandas as pd

from tradeagent.features.indicators import atr, ema
from tradeagent.features.structure import break_of_structure
from tradeagent.strategies import registry
from tradeagent.strategies.base import BaseStrategy, MarketContext, ParamSpec, Signal, Style
from tradeagent.strategies.common import finite

NAME = "structure_retest"
DEFAULTS = {"swing_k": 3.0, "retest_expiry": 12.0, "atr_buffer": 1.0, "rr": 2.0}


class StructureRetest(BaseStrategy):
    name = NAME
    version = "1.0"
    style: Style = "swing"
    timeframes: Sequence[str] = ("H1", "H4")
    suited_regimes: Sequence[str] = ("trending",)
    lookback_bars = 50
    param_specs: Sequence[ParamSpec] = (
        ParamSpec("swing_k", 2.0, 5.0, "bars on each side that make a swing high/low"),
        ParamSpec("retest_expiry", 4.0, 24.0, "H1 bars the retest order stays valid"),
        ParamSpec("atr_buffer", 0.5, 2.0, "stop distance beyond the level, in ATR(14) H1"),
        ParamSpec("rr", 2.0, 4.0, "target as a multiple of the stop distance"),
    )

    def __init__(self, seed: int = 0, params: dict[str, float] | None = None) -> None:
        super().__init__(**{**DEFAULTS, **(params or {})})

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out = dict(frames)
        h1, h4 = out["H1"], out["H4"]
        bos = break_of_structure(h1, int(self.params["swing_k"]))
        out["H1"] = h1.assign(
            bos_up=bos["bos_up"].astype(float),
            bos_down=bos["bos_down"].astype(float),
            bos_level=bos["bos_level"],
            atr14=atr(h1, 14),
        )
        out["H4"] = h4.assign(ema50=ema(h4["close"], 50))
        return out

    def generate(self, ctx: MarketContext) -> list[Signal]:
        h, h4 = ctx.bars("H1"), ctx.bars("H4")
        if len(h4) < 1:
            return []
        level, a = h.last("bos_level"), h.last("atr14")
        trend_close, trend_ema = h4.last("close"), h4.last("ema50")
        if not finite(level, a, trend_close, trend_ema) or a <= 0:
            return []
        stop = self.params["atr_buffer"] * a
        rr, expiry = self.params["rr"], int(self.params["retest_expiry"])
        if h.last("bos_up") and trend_close > trend_ema:
            why = f"H1 break above swing high {level:.2f}, H4 above EMA50; buy the retest"
            return [
                Signal(
                    ctx.symbol,
                    "long",
                    level - stop,
                    level + rr * stop,
                    why=why,
                    order_type="limit",
                    entry_price=level,
                    expiry_bars=expiry,
                )
            ]
        if h.last("bos_down") and trend_close < trend_ema:
            why = f"H1 break below swing low {level:.2f}, H4 below EMA50; sell the retest"
            return [
                Signal(
                    ctx.symbol,
                    "short",
                    level + stop,
                    level - rr * stop,
                    why=why,
                    order_type="limit",
                    entry_price=level,
                    expiry_bars=expiry,
                )
            ]
        return []


registry.register(NAME, lambda seed, params: StructureRetest(seed, params))
