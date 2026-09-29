"""Strategy interface (SPEC §4): every strategy has the same shape.

- A strategy declares name, version, style, timeframes, suited regimes and at most
  5 tunable parameters, each with a documented range (`ParamSpec`).
- `timeframes[0]` is the decision timeframe: the backtest steps through its bars,
  and `generate()` is called after each one closes. Other timeframes are context.
- `generate(ctx)` sees only closed bars through `MarketContext`, and returns 0..n
  `Signal`s, each with stop-loss, take-profit, an order type and a `why` text.
- Optional `prepare(frames)` adds indicator columns once per run (e.g. ATR), so a
  strategy doesn't recompute them on every bar. Every value at bar t must use bars
  <= t only; the truncation test (docs/PHASE_2_TASKS.md 2.6) checks this.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from tradeagent.data.mt5_client import TIMEFRAMES

Style = Literal["scalp", "intraday", "swing"]
Direction = Literal["long", "short"]
OrderType = Literal["market", "limit", "stop"]

MAX_PARAMS = 5
STYLES: tuple[Style, ...] = ("scalp", "intraday", "swing")


class InvalidSignal(ValueError):
    """A signal that breaks the interface rules (engine counts and skips it)."""


class InvalidStrategy(ValueError):
    """A strategy that breaks the interface rules (refused before any backtest)."""


# --- parameters ----------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter: its allowed range [low, high] and what it does."""

    name: str
    low: float
    high: float
    description: str

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise InvalidStrategy(f"parameter {self.name!r} needs a description")
        if self.low > self.high:
            raise InvalidStrategy(f"parameter {self.name!r}: low {self.low} > high {self.high}")


# --- signals -------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """A trade idea from closed bars.

    market: fills at the next bar's open.
    limit:  buy below / sell above the current price, at `entry_price` or better.
    stop:   buy above / sell below the current price, at `entry_price` or worse.
    limit/stop orders are cancelled if unfilled after `expiry_bars` decision bars.
    `max_hold_bars` (optional) closes the trade after that many bars.
    """

    symbol: str
    direction: Direction
    stop_loss: float
    take_profit: float
    why: str
    order_type: OrderType = "market"
    entry_price: float | None = None
    expiry_bars: int | None = None
    max_hold_bars: int | None = None

    def __post_init__(self) -> None:
        if not self.why.strip():
            raise InvalidSignal("every signal needs a human-readable `why`")
        if self.direction not in ("long", "short"):
            raise InvalidSignal(f"direction must be long or short, got {self.direction!r}")
        if self.order_type not in ("market", "limit", "stop"):
            raise InvalidSignal(f"unknown order type {self.order_type!r}")
        prices = [self.stop_loss, self.take_profit]
        if self.entry_price is not None:
            prices.append(self.entry_price)
        if not all(np.isfinite(p) and p > 0 for p in prices):
            raise InvalidSignal(f"prices must be positive numbers: {prices}")
        if self.max_hold_bars is not None and self.max_hold_bars < 1:
            raise InvalidSignal("max_hold_bars must be at least 1")

        if self.order_type == "market":
            if self.entry_price is not None or self.expiry_bars is not None:
                raise InvalidSignal("market orders fill at the next open: no entry_price/expiry")
        else:
            if self.entry_price is None:
                raise InvalidSignal(f"{self.order_type} orders need an entry_price")
            if self.expiry_bars is None or self.expiry_bars < 1:
                raise InvalidSignal(f"{self.order_type} orders need expiry_bars >= 1")

        # Stop below and target above for longs; the reverse for shorts. For pending
        # orders the entry must lie between them.
        low, high = (
            (self.stop_loss, self.take_profit)
            if self.direction == "long"
            else (self.take_profit, self.stop_loss)
        )
        if not low < high:
            side = "below" if self.direction == "long" else "above"
            raise InvalidSignal(f"{self.direction}: stop-loss must be {side} the take-profit")
        if self.entry_price is not None and not low < self.entry_price < high:
            raise InvalidSignal("entry_price must lie between stop-loss and take-profit")

    def entry_side_error(self, price: float) -> str | None:
        """Why this pending order is on the wrong side of the current price, or None.

        A buy limit must sit below the price and a buy stop above it (sells: the
        reverse); otherwise it would fill at once and really be a market order.
        """
        if self.order_type == "market" or self.entry_price is None:
            return None
        buy = self.direction == "long"
        below = self.entry_price < price
        above = self.entry_price > price
        ok = {
            ("limit", True): below,
            ("stop", True): above,
            ("limit", False): above,
            ("stop", False): below,
        }[(self.order_type, buy)]
        if ok:
            return None
        where = "below" if (self.order_type == "limit") == buy else "above"
        side = "buy" if buy else "sell"
        return f"{side} {self.order_type} at {self.entry_price} must be {where} price {price}"


# --- what a strategy may see ---------------------------------------------------------


@dataclass(frozen=True)
class BarView:
    """Read-only arrays of closed bars for one timeframe, oldest first.

    Index -1 is the latest closed bar. Asking past the end raises IndexError.
    """

    timeframe: str
    columns: Mapping[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.columns["close"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    @property
    def time_utc(self) -> np.ndarray:
        return self.columns["time_utc"]

    @property
    def open(self) -> np.ndarray:
        return self.columns["open"]

    @property
    def high(self) -> np.ndarray:
        return self.columns["high"]

    @property
    def low(self) -> np.ndarray:
        return self.columns["low"]

    @property
    def close(self) -> np.ndarray:
        return self.columns["close"]

    def last(self, name: str, back: int = 0) -> float:
        """Value `back` bars before the latest closed bar (back=0: the latest)."""
        values = self.columns[name]
        if back < 0 or back >= len(values):
            raise IndexError(f"{self.timeframe}: no closed bar {back} back (have {len(values)})")
        return float(values[-1 - back])


class Frames:
    """All bars of one run, prepared once. Hands out MarketContexts, one per decision.

    A bar is visible at time `now` only when it has closed: its end time <= now. The
    end is `end_utc` when present (New York-close D1/H4), else open time + length.
    """

    def __init__(self, symbol: str, frames: Mapping[str, pd.DataFrame]) -> None:
        if not frames:
            raise ValueError("need at least one timeframe of bars")
        self.symbol = symbol
        self._columns: dict[str, dict[str, np.ndarray]] = {}
        self._end_ns: dict[str, np.ndarray] = {}
        for tf, df in frames.items():
            if tf not in TIMEFRAMES:
                raise ValueError(f"unknown timeframe {tf!r}")
            if str(df["time_utc"].dtype) != "datetime64[ns, UTC]":
                raise ValueError(f"{tf}: time_utc must be datetime64[ns, UTC]")
            df = df.sort_values("time_utc").reset_index(drop=True)
            if "end_utc" in df.columns:
                end = df["end_utc"]
            else:
                end = df["time_utc"] + pd.Timedelta(TIMEFRAMES[tf][1])
            self._end_ns[tf] = _read_only(end.dt.tz_convert("UTC").to_numpy("datetime64[ns]"))
            cols: dict[str, np.ndarray] = {
                "time_utc": _read_only(
                    df["time_utc"].dt.tz_convert("UTC").to_numpy("datetime64[ns]")
                )
            }
            for name in df.columns:
                if name in ("time_utc", "end_utc"):
                    continue
                if pd.api.types.is_numeric_dtype(df[name]) or pd.api.types.is_bool_dtype(df[name]):
                    cols[name] = _read_only(df[name].to_numpy())
            self._columns[tf] = cols

    @property
    def timeframes(self) -> list[str]:
        return list(self._columns)

    def context(self, now: pd.Timestamp) -> "MarketContext":
        return MarketContext(self, pd.Timestamp(now).tz_convert("UTC"))

    def _view(self, timeframe: str, now: pd.Timestamp) -> BarView:
        if timeframe not in self._columns:
            raise KeyError(f"timeframe {timeframe!r} was not loaded for this run")
        visible = int(np.searchsorted(self._end_ns[timeframe], now.to_datetime64(), side="right"))
        return BarView(
            timeframe, {name: arr[:visible] for name, arr in self._columns[timeframe].items()}
        )


@dataclass
class MarketContext:
    """What a strategy may see at decision time `now`: only bars closed by then."""

    _frames: Frames
    now: pd.Timestamp
    _cache: dict[str, BarView] = field(default_factory=dict, repr=False)

    @property
    def symbol(self) -> str:
        return self._frames.symbol

    def bars(self, timeframe: str) -> BarView:
        if timeframe not in self._cache:
            self._cache[timeframe] = self._frames._view(timeframe, self.now)
        return self._cache[timeframe]


def _read_only(values: np.ndarray) -> np.ndarray:
    values = np.array(values, copy=True)
    values.setflags(write=False)
    return values


# --- the strategy itself -------------------------------------------------------------


@runtime_checkable
class Strategy(Protocol):
    name: str
    version: str
    style: Style
    timeframes: Sequence[str]  # timeframes[0] = decision timeframe
    suited_regimes: Sequence[str]  # declared, then verified by stats
    params: dict[str, float]  # small, documented, bounded (see param_specs)
    param_specs: Sequence[ParamSpec]

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Add indicator columns once per run. Values at bar t use bars <= t only."""
        ...

    def generate(self, ctx: MarketContext) -> list[Signal]:
        """Use ONLY ctx (closed bars up to ctx.now); return 0..n signals."""
        ...


def check_strategy(strategy: Strategy) -> None:
    """Refuse a strategy that breaks the SPEC §4 rules, before it can be backtested."""
    if not strategy.name.strip() or not strategy.version.strip():
        raise InvalidStrategy("strategy needs a name and a version")
    if strategy.style not in STYLES:
        raise InvalidStrategy(f"style must be one of {STYLES}, got {strategy.style!r}")
    if not strategy.timeframes:
        raise InvalidStrategy("strategy must declare at least one timeframe")
    unknown = [tf for tf in strategy.timeframes if tf not in TIMEFRAMES]
    if unknown:
        raise InvalidStrategy(f"unknown timeframes: {unknown}")
    if len(set(strategy.timeframes)) != len(strategy.timeframes):
        raise InvalidStrategy("timeframes are listed twice")

    specs = {spec.name: spec for spec in strategy.param_specs}
    if len(specs) > MAX_PARAMS:
        raise InvalidStrategy(f"at most {MAX_PARAMS} tunable parameters, got {len(specs)}")
    if set(specs) != set(strategy.params):
        raise InvalidStrategy(
            f"params {sorted(strategy.params)} must match param_specs {sorted(specs)}"
        )
    for name, value in strategy.params.items():
        spec = specs[name]
        if not spec.low <= value <= spec.high:
            raise InvalidStrategy(
                f"parameter {name}={value} is outside its range [{spec.low}, {spec.high}]"
            )


class BaseStrategy:
    """Convenience base: default `prepare` (no extra columns) and param defaults."""

    name: str = ""
    version: str = ""
    style: Style = "intraday"
    timeframes: Sequence[str] = ()  # tuples, e.g. ("M15", "H1")
    suited_regimes: Sequence[str] = ()
    param_specs: Sequence[ParamSpec] = ()

    def __init__(self, **params: float) -> None:
        self.params: dict[str, float] = dict(params)

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return frames

    def generate(self, ctx: MarketContext) -> list[Signal]:
        raise NotImplementedError
