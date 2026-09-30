"""TEMPORARY risk basics for backtests, until the Phase 4 risk engine replaces them.

Deterministic Python, no AI. Limits are read (never written) from config/risk.yaml:
minimum reward:risk, stop distance 0.5-3 x ATR, spread <= 2 x median spread, and
the SPEC §6 position size (risk % of equity, rounded down to the lot step). The
"one open position" rule is enforced by the engine, which holds one slot.
"""

import math

from tradeagent.config import RiskLimits, SymbolCosts

# Reasons a signal is rejected, as counted in backtest reports.
REJECT_REASONS = (
    "invalid_signal",  # broke the Signal rules (raised inside the strategy)
    "wrong_symbol",
    "entry_side",  # buy limit above / buy stop below the price (or the reverse for sells)
    "position_open",  # the one slot is taken by a position or pending order
    "rr",  # reward:risk below the minimum
    "no_atr",  # ATR not available yet (too little history)
    "sl_atr",  # stop distance outside sl_atr_min..sl_atr_max x ATR
    "spread",  # bar spread above max_spread_multiple x median spread
    "min_lot",  # the size for the risk budget is below the broker's minimum lot
)

_EPS = 1e-9


def position_size(equity: float, risk_pct: float, sl_distance: float, spec: SymbolCosts) -> float:
    """Lots so that a stop filled exactly loses `risk_pct`% of equity, rounded DOWN
    to the lot step and capped at the maximum lot (SPEC §6). May return 0."""
    if sl_distance <= 0 or equity <= 0:
        return 0.0
    loss_per_lot = sl_distance / spec.point * spec.value_per_point_per_lot
    raw = equity * risk_pct / 100 / loss_per_lot
    lots = math.floor(raw / spec.volume_step + _EPS) * spec.volume_step
    return round(min(lots, spec.volume_max), 8)


def min_balance(sl_distance: float, spec: SymbolCosts, risk_pct: float) -> float:
    """Smallest account balance for which the minimum lot risks at most `risk_pct`%."""
    loss_at_min_lot = sl_distance / spec.point * spec.value_per_point_per_lot * spec.volume_min
    return loss_at_min_lot / (risk_pct / 100)


def check_signal(
    *,
    entry_ref: float,
    stop_loss: float,
    take_profit: float,
    atr: float,
    bar_spread: float,
    median_spread: float,
    equity: float,
    limits: RiskLimits,
    spec: SymbolCosts,
) -> tuple[str | None, float]:
    """(reject reason or None, lots). `entry_ref` is the expected entry price."""
    risk = abs(entry_ref - stop_loss)
    reward = abs(take_profit - entry_ref)
    if risk <= 0 or reward / risk < limits.min_reward_risk - _EPS:
        return "rr", 0.0
    if not math.isfinite(atr) or atr <= 0:
        return "no_atr", 0.0
    in_atr = risk / atr
    if in_atr < limits.sl_atr_min - _EPS or in_atr > limits.sl_atr_max + _EPS:
        return "sl_atr", 0.0
    if median_spread > 0 and bar_spread > limits.max_spread_multiple * median_spread + _EPS:
        return "spread", 0.0
    lots = position_size(equity, limits.risk_per_trade_pct, risk, spec)
    if lots < spec.volume_min - _EPS:
        return "min_lot", 0.0
    return None, lots
