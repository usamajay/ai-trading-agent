"""Position sizing (SPEC §6): risk a fixed share of equity, rounded down to the lot step.

size = (equity x risk%) / (stop distance x value per point per lot), rounded DOWN to
the broker's lot step and capped at the maximum lot. Below the minimum lot the
trade is rejected: taking the minimum would risk more than the limit.
"""

import math

from tradeagent.config import SymbolCosts

_EPS = 1e-9


def loss_per_lot(stop_distance: float, spec: SymbolCosts) -> float:
    """Account currency lost per 1 lot if the stop fills exactly."""
    return stop_distance / spec.point * spec.value_per_point_per_lot


def position_size(equity: float, risk_pct: float, stop_distance: float, spec: SymbolCosts) -> float:
    """Lots risking `risk_pct`% of `equity`; 0 when nothing fits."""
    if not all(math.isfinite(v) and v > 0 for v in (stop_distance, equity, risk_pct)):
        return 0.0
    raw = equity * risk_pct / 100 / loss_per_lot(stop_distance, spec)
    lots = math.floor(raw / spec.volume_step + _EPS) * spec.volume_step
    return round(min(lots, spec.volume_max), 8)


def min_balance(stop_distance: float, spec: SymbolCosts, risk_pct: float) -> float:
    """Smallest balance for which the minimum lot risks at most `risk_pct`%."""
    return loss_per_lot(stop_distance, spec) * spec.volume_min / (risk_pct / 100)
