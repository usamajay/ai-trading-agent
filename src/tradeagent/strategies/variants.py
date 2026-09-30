"""Strategy variants for research (docs/PHASE_6_TASKS.md 6.3/6.4).

A variant wraps a registered strategy without changing its code:
- **timeframes**: run the same idea on other bars, e.g. trend_ema_pullback's
  (M15, H1) as (H4, D1). The inner strategy still asks for "M15"/"H1"; the wrapper
  hands it the H4/D1 bars under those names.
- **direction**: keep only long or only short signals.
- **style**: e.g. `swing` for H4/D1 (no forced Friday close).
Variants are named `base[key=value,...]` and counted under their base strategy, so
every variant tried adds to the base strategy's run counter (SPEC §7.4).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from tradeagent.strategies import registry
from tradeagent.strategies.base import (
    BarView,
    Direction,
    MarketContext,
    Signal,
    Strategy,
    Style,
)

DirectionRule = Literal["both", "long", "short"]


@dataclass(frozen=True)
class VariantSpec:
    strategy: str
    timeframes: tuple[str, ...] = ()  # empty = the strategy's own
    direction: DirectionRule = "both"
    style: Style | None = None  # None = the strategy's own
    params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in ("both", "long", "short"):
            raise ValueError(f"direction must be both, long or short, got {self.direction!r}")

    @property
    def is_plain(self) -> bool:
        return not self.timeframes and self.direction == "both" and self.style is None

    def name(self) -> str:
        tags = []
        if self.timeframes:
            tags.append("tf=" + "/".join(self.timeframes))
        if self.direction != "both":
            tags.append(f"dir={self.direction}")
        if self.style is not None:
            tags.append(f"style={self.style}")
        return f"{self.strategy}[{','.join(tags)}]" if tags else self.strategy

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "timeframes": list(self.timeframes),
            "direction": self.direction,
            "style": self.style,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VariantSpec":
        return cls(
            strategy=d["strategy"],
            timeframes=tuple(d.get("timeframes") or ()),
            direction=d.get("direction", "both"),
            style=d.get("style"),
            params=dict(d.get("params") or {}),
        )


def base_name(name: str) -> str:
    """`trend_ema_pullback[tf=H4/D1]` -> `trend_ema_pullback`."""
    return name.split("[", 1)[0]


class _RenamedContext:
    """Shows the inner strategy the variant's bars under the inner timeframe names."""

    def __init__(self, ctx: MarketContext, inner_to_outer: Mapping[str, str]) -> None:
        self._ctx = ctx
        self._map = inner_to_outer
        self.symbol = ctx.symbol
        self.now = ctx.now

    def bars(self, timeframe: str) -> BarView:
        return self._ctx.bars(self._map.get(timeframe, timeframe))

    def expected_entry(self, direction: Direction) -> float:
        return self._ctx.expected_entry(direction)


class Variant:
    """A registered strategy run with other timeframes, one direction, or another style."""

    def __init__(self, spec: VariantSpec, inner: Strategy) -> None:
        own = tuple(inner.timeframes)
        if spec.timeframes and len(spec.timeframes) != len(own):
            raise ValueError(
                f"{inner.name} uses {len(own)} timeframes {own}; got {spec.timeframes}"
            )
        self.spec = spec
        self.inner = inner
        self.name = spec.name()
        self.version = inner.version
        self.style: Style = spec.style or inner.style
        self.timeframes: Sequence[str] = spec.timeframes or own
        self.suited_regimes = inner.suited_regimes
        self.lookback_bars = inner.lookback_bars
        self.params = inner.params
        self.param_specs = inner.param_specs
        self._to_outer = dict(zip(own, self.timeframes, strict=True))
        self._to_inner = {v: k for k, v in self._to_outer.items()}

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        as_inner = {self._to_inner.get(tf, tf): df for tf, df in frames.items()}
        done = self.inner.prepare(as_inner)
        return {self._to_outer.get(tf, tf): df for tf, df in done.items()}

    def generate(self, ctx: MarketContext) -> list[Signal]:
        inner_ctx = _RenamedContext(ctx, self._to_outer) if self.spec.timeframes else ctx
        signals = self.inner.generate(inner_ctx)  # type: ignore[arg-type]
        if self.spec.direction == "both":
            return signals
        return [s for s in signals if s.direction == self.spec.direction]


def build(spec: VariantSpec, seed: int = 0) -> Strategy:
    """The strategy for a spec: the registered one itself when nothing is changed."""
    inner = registry.create(spec.strategy, seed, dict(spec.params))
    return inner if spec.is_plain else Variant(spec, inner)
