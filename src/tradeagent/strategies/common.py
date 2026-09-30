"""Small helpers shared by strategies."""

import math

from tradeagent.strategies.base import Direction, MarketContext, Signal


def finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def market_signal(
    ctx: MarketContext,
    direction: Direction,
    stop_distance: float,
    rr: float,
    why: str,
    max_hold_bars: int | None = None,
) -> Signal:
    """Market order with the stop `stop_distance` away from the expected entry (ask for
    longs, bid for shorts) and the target `rr` x that distance, so the reward:risk
    check sees exactly `rr`."""
    entry = ctx.expected_entry(direction)
    target = rr * stop_distance
    if direction == "long":
        sl, tp = entry - stop_distance, entry + target
    else:
        sl, tp = entry + stop_distance, entry - target
    return Signal(
        symbol=ctx.symbol,
        direction=direction,
        stop_loss=sl,
        take_profit=tp,
        why=why,
        max_hold_bars=max_hold_bars,
    )
