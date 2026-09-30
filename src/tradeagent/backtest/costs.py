"""Trading costs for backtests: spread, slippage, commission, swap (SPEC §7.1).

Prices in stored bars are **bid** prices. The ask is bid + spread, where the
spread used is the bar's stored spread (usually the bar's minimum, see
`spread_check.py`) times `spread_margin_multiple`, plus `spread_margin_points`:

- long:  enters at the ask, exits on the bid (its SL/TP are checked against bid prices)
- short: enters at the bid, exits on the ask (its SL/TP are checked against ask prices)

Slippage (`slippage_spread_multiple` x spread) makes market entries, stop entries
and stop-loss exits worse; take-profits and limit entries fill at their price.

Swap is charged once for each 17:00 New York rollover a position is held through
(Monday-Friday), and x3 on the symbol's triple-swap day, if it has one
(`swap_rollover3days = 7` means none; confirmed for USOIL on 2026-09-30).
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from tradeagent.config import AppConfig, BacktestSettings, CostSnapshot, SymbolCosts
from tradeagent.data.market_hours import NEW_YORK
from tradeagent.data.mt5_client import SymbolSpec
from tradeagent.strategies.base import Direction

ROLLOVER_HOUR_NY = 17


class CostError(Exception):
    """Costs are missing or unusable for a backtest."""


@dataclass(frozen=True)
class CostModel:
    symbol: str
    spec: SymbolCosts
    spread_margin_multiple: float
    spread_margin_points: float
    slippage_spread_multiple: float
    commission_per_lot_usd: float

    @classmethod
    def from_config(cls, cfg: AppConfig, symbol: str) -> "CostModel":
        if cfg.costs is None:
            raise CostError("no cost snapshot: run `tradeagent backtest costs --snapshot`")
        if symbol not in cfg.costs.symbols:
            raise CostError(f"cost snapshot has no {symbol}")
        return cls.from_settings(symbol, cfg.costs.symbols[symbol], cfg.settings.backtest)

    @classmethod
    def from_settings(cls, symbol: str, spec: SymbolCosts, bt: BacktestSettings) -> "CostModel":
        return cls(
            symbol=symbol,
            spec=spec,
            spread_margin_multiple=bt.spread_margin_multiple,
            spread_margin_points=bt.spread_margin_points,
            slippage_spread_multiple=bt.slippage_spread_multiple,
            commission_per_lot_usd=bt.commission_per_lot_usd,
        )

    # --- prices --------------------------------------------------------------------

    def spread(self, bar_spread_points: float) -> float:
        """Spread used for a bar, in price units (stored spread + safety margin)."""
        points = bar_spread_points * self.spread_margin_multiple + self.spread_margin_points
        return points * self.spec.point

    def slippage(self, bar_spread_points: float) -> float:
        """Slippage for one market/stop fill, in price units."""
        return self.slippage_spread_multiple * self.spread(bar_spread_points)

    def entry_side(self, direction: Direction, bid: float, bar_spread_points: float) -> float:
        """The price a new position trades at: ask for a buy, bid for a sell."""
        return bid + self.spread(bar_spread_points) if direction == "long" else bid

    def exit_side(self, direction: Direction, bid: float, bar_spread_points: float) -> float:
        """The price an open position closes at: bid for a long, ask for a short."""
        return bid if direction == "long" else bid + self.spread(bar_spread_points)

    def entry_fill(
        self, direction: Direction, price: float, bar_spread_points: float, slip: bool
    ) -> float:
        """Fill for an entry at `price` (already on the entry side), with slippage if `slip`."""
        if not slip:
            return price
        s = self.slippage(bar_spread_points)
        return price + s if direction == "long" else price - s

    def exit_fill(
        self, direction: Direction, price: float, bar_spread_points: float, slip: bool
    ) -> float:
        """Fill for an exit at `price` (already on the exit side), with slippage if `slip`."""
        if not slip:
            return price
        s = self.slippage(bar_spread_points)
        return price - s if direction == "long" else price + s

    def exit_range(
        self, direction: Direction, high: float, low: float, bar_spread_points: float
    ) -> tuple[float, float]:
        """(high, low) a position's SL/TP are checked against in a bar: bid for longs,
        ask (bid + spread) for shorts."""
        if direction == "long":
            return high, low
        s = self.spread(bar_spread_points)
        return high + s, low + s

    def entry_range(
        self, direction: Direction, high: float, low: float, bar_spread_points: float
    ) -> tuple[float, float]:
        """(high, low) a pending order is checked against: ask for buys, bid for sells."""
        if direction == "short":
            return high, low
        s = self.spread(bar_spread_points)
        return high + s, low + s

    # --- money ---------------------------------------------------------------------

    def pnl(self, direction: Direction, entry: float, exit: float, lots: float) -> float:
        """Price P&L in account currency (before commission and swap)."""
        move = exit - entry if direction == "long" else entry - exit
        return move / self.spec.point * self.spec.value_per_point_per_lot * lots

    def commission(self, lots: float) -> float:
        """Round-turn commission in account currency (a cost, so >= 0)."""
        return self.commission_per_lot_usd * lots

    def swap(
        self, direction: Direction, lots: float, open_utc: pd.Timestamp, close_utc: pd.Timestamp
    ) -> float:
        """Swap in account currency for holding from open to close (negative = paid)."""
        nights = rollover_nights(open_utc, close_utc, self.spec.triple_swap_weekday)
        points = self.spec.swap_long if direction == "long" else self.spec.swap_short
        return nights * points * self.spec.value_per_point_per_lot * lots


def rollover_nights(
    open_utc: pd.Timestamp, close_utc: pd.Timestamp, triple_weekday: int | None
) -> int:
    """Swap nights charged between open and close.

    One per 17:00 New York rollover (Monday-Friday) strictly between the two times;
    the rollover on `triple_weekday` (Python weekday, Monday=0) counts 3.
    """
    if close_utc <= open_utc:
        return 0
    first = open_utc.tz_convert(NEW_YORK).date()
    last = close_utc.tz_convert(NEW_YORK).date()
    nights = 0
    day: date = first
    while day <= last:
        if day.weekday() <= 4:
            rollover = pd.Timestamp(
                year=day.year, month=day.month, day=day.day, hour=ROLLOVER_HOUR_NY, tz=NEW_YORK
            )
            if open_utc < rollover < close_utc:
                nights += 3 if day.weekday() == triple_weekday else 1
        day += timedelta(days=1)
    return nights


# --- the cost snapshot ---------------------------------------------------------------


def symbol_costs(spec: SymbolSpec) -> SymbolCosts:
    """Copy the cost-relevant fields of MT5 symbol_info (refuses unsupported swap modes)."""
    return SymbolCosts(
        broker_symbol=spec.symbol,
        digits=spec.digits,
        point=spec.point,
        tick_size=spec.tick_size,
        tick_value=spec.tick_value,
        contract_size=spec.contract_size,
        volume_min=spec.volume_min,
        volume_step=spec.volume_step,
        volume_max=spec.volume_max,
        swap_mode=spec.swap_mode,  # type: ignore[arg-type]  # validated: must be 1
        swap_long=spec.swap_long,
        swap_short=spec.swap_short,
        swap_rollover3days=spec.swap_rollover3days,
    )


def write_snapshot(snapshot: CostSnapshot, path: Path) -> None:
    header = (
        "# Broker cost snapshot for backtests (docs/PHASE_2_TASKS.md 2.4). Written by\n"
        "# `tradeagent backtest costs --snapshot`; refresh it the same way, and note\n"
        "# changed swap values in docs/DECISIONS.md. Swaps are points per lot per night;\n"
        "# swap_rollover3days: MT5 day number (0=Sunday..6=Saturday), 7 = no triple day.\n"
    )
    body = yaml.safe_dump(snapshot.model_dump(mode="json"), sort_keys=False)
    path.write_text(header + body, encoding="utf-8")
