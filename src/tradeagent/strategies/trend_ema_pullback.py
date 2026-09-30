"""Trend EMA pullback (SPEC §4.1 #1): trade with the H1 trend after a pullback on M15.

Long when, on closed bars:
- H1 trend is up: EMA20 > EMA50 > EMA200 and the H1 close is above EMA50;
- M15 pulled back: a low touched the M15 EMA20 within the last `pullback_bars` bars;
- M15 momentum returns: RSI(14) crosses up through `rsi_level` and the close is
  back above EMA20.
Short mirrors this (RSI crossing down through 100 - rsi_level). Stop `atr_mult` x
ATR(14) M15 from the expected entry; target `rr` x the stop. Intraday.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from tradeagent.features.indicators import atr, ema, rsi
from tradeagent.strategies import registry
from tradeagent.strategies.base import BaseStrategy, MarketContext, ParamSpec, Signal, Style
from tradeagent.strategies.common import finite, market_signal

NAME = "trend_ema_pullback"
DEFAULTS = {"atr_mult": 1.5, "rr": 2.0, "rsi_level": 45.0, "pullback_bars": 5.0}


class TrendEmaPullback(BaseStrategy):
    name = NAME
    version = "1.0"
    style: Style = "intraday"
    timeframes: Sequence[str] = ("M15", "H1")
    suited_regimes: Sequence[str] = ("trending",)
    lookback_bars = 30  # M15 bars behind a signal (RSI 14 + pullback window + margin)
    param_specs: Sequence[ParamSpec] = (
        ParamSpec("atr_mult", 1.0, 3.0, "stop distance in ATR(14) M15"),
        ParamSpec("rr", 2.0, 4.0, "target as a multiple of the stop distance"),
        ParamSpec("rsi_level", 35.0, 55.0, "RSI(14) level the pullback must cross back through"),
        ParamSpec("pullback_bars", 2.0, 10.0, "M15 bars in which the low must touch EMA20"),
    )

    def __init__(self, seed: int = 0, params: dict[str, float] | None = None) -> None:
        super().__init__(**{**DEFAULTS, **(params or {})})

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out = dict(frames)
        m15, h1 = out["M15"], out["H1"]
        out["M15"] = m15.assign(
            ema20=ema(m15["close"], 20), rsi14=rsi(m15["close"], 14), atr14=atr(m15, 14)
        )
        out["H1"] = h1.assign(
            ema20=ema(h1["close"], 20), ema50=ema(h1["close"], 50), ema200=ema(h1["close"], 200)
        )
        return out

    def generate(self, ctx: MarketContext) -> list[Signal]:
        m, h = ctx.bars("M15"), ctx.bars("H1")
        n = int(self.params["pullback_bars"])
        if len(m) < n + 2 or len(h) < 1:
            return []
        e20, e50, e200 = h.last("ema20"), h.last("ema50"), h.last("ema200")
        h_close = h.last("close")
        rsi_now, rsi_before = m.last("rsi14"), m.last("rsi14", 1)
        close, m_ema, a = m.last("close"), m.last("ema20"), m.last("atr14")
        if not finite(e20, e50, e200, rsi_now, rsi_before, m_ema, a) or a <= 0:
            return []
        lows, highs, emas = m["low"][-n:], m["high"][-n:], m["ema20"][-n:]
        level = self.params["rsi_level"]
        stop = self.params["atr_mult"] * a
        if (
            e20 > e50 > e200
            and h_close > e50
            and bool(np.any(lows <= emas))
            and rsi_before < level <= rsi_now
            and close > m_ema
        ):
            why = (
                f"H1 EMA20>50>200 uptrend; M15 pullback to EMA20 in {n} bars; "
                f"RSI {rsi_before:.0f}->{rsi_now:.0f} crossed {level:.0f}"
            )
            return [market_signal(ctx, "long", stop, self.params["rr"], why)]
        mirror = 100 - level
        if (
            e20 < e50 < e200
            and h_close < e50
            and bool(np.any(highs >= emas))
            and rsi_before > mirror >= rsi_now
            and close < m_ema
        ):
            why = (
                f"H1 EMA20<50<200 downtrend; M15 pullback to EMA20 in {n} bars; "
                f"RSI {rsi_before:.0f}->{rsi_now:.0f} crossed {mirror:.0f}"
            )
            return [market_signal(ctx, "short", stop, self.params["rr"], why)]
        return []


registry.register(NAME, lambda seed, params: TrendEmaPullback(seed, params))
