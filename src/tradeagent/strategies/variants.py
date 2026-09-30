"""Strategy variants for research (docs/PHASE_6_TASKS.md 6.3/6.4).

A variant wraps a registered strategy without changing its code:
- **timeframes**: run the same idea on other bars, e.g. trend_ema_pullback's
  (M15, H1) as (H4, D1). The inner strategy still asks for "M15"/"H1"; the wrapper
  hands it the H4/D1 bars under those names.
- **direction**: keep only long or only short signals.
- **style**: e.g. `swing` for H4/D1 (no forced Friday close).
- **regimes**: trade only when the decision bar is in one of these regimes
  (`features/regime.py`); `suited` = the strategy's declared `suited_regimes`.
Variants are named `base[key=value,...]` and counted under their base strategy, so
every variant tried adds to the base strategy's run counter (SPEC §7.4).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from tradeagent.features.regime import REGIMES, regime_labels
from tradeagent.strategies import registry
from tradeagent.strategies.base import (
    BarView,
    Direction,
    MarketContext,
    Signal,
    Strategy,
    Style,
    TradeEvent,
)

DirectionRule = Literal["both", "long", "short"]


@dataclass(frozen=True)
class VariantSpec:
    strategy: str
    timeframes: tuple[str, ...] = ()  # empty = the strategy's own
    direction: DirectionRule = "both"
    style: Style | None = None  # None = the strategy's own
    params: Mapping[str, float] = field(default_factory=dict)
    regimes: tuple[str, ...] = ()  # empty = no filter; "suited" = the declared ones

    def __post_init__(self) -> None:
        if self.direction not in ("both", "long", "short"):
            raise ValueError(f"direction must be both, long or short, got {self.direction!r}")
        unknown = [r for r in self.regimes if r not in (*REGIMES, "suited")]
        if unknown:
            raise ValueError(f"unknown regimes {unknown}; known: {REGIMES} or 'suited'")

    @property
    def is_plain(self) -> bool:
        return (
            not self.timeframes
            and self.direction == "both"
            and self.style is None
            and not self.regimes
        )

    def name(self) -> str:
        tags = []
        if self.timeframes:
            tags.append("tf=" + "/".join(self.timeframes))
        if self.direction != "both":
            tags.append(f"dir={self.direction}")
        if self.style is not None:
            tags.append(f"style={self.style}")
        if self.regimes:
            tags.append("reg=" + "+".join(self.regimes))
        return f"{self.strategy}[{','.join(tags)}]" if tags else self.strategy

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "timeframes": list(self.timeframes),
            "direction": self.direction,
            "style": self.style,
            "params": dict(self.params),
            "regimes": list(self.regimes),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VariantSpec":
        return cls(
            strategy=d["strategy"],
            timeframes=tuple(d.get("timeframes") or ()),
            direction=d.get("direction", "both"),
            style=d.get("style"),
            params=dict(d.get("params") or {}),
            regimes=tuple(d.get("regimes") or ()),
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
        allowed: list[str] = []
        for r in spec.regimes:
            allowed += list(inner.suited_regimes) if r == "suited" else [r]
        if spec.regimes and not allowed:
            raise ValueError(f"{inner.name} declares no regimes to filter on")
        # "any" (e.g. session breakout) means no filter; unknown declared names are refused
        self.regimes: tuple[str, ...] = () if "any" in allowed else tuple(dict.fromkeys(allowed))
        unknown = [r for r in self.regimes if r not in REGIMES]
        if unknown:
            raise ValueError(f"{inner.name} declares regimes the labeller lacks: {unknown}")

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        as_inner = {self._to_inner.get(tf, tf): df for tf, df in frames.items()}
        done = self.inner.prepare(as_inner)
        out = {self._to_outer.get(tf, tf): df for tf, df in done.items()}
        if self.regimes:
            decision = self.timeframes[0]
            labels = regime_labels(out[decision])
            out[decision] = out[decision].assign(**{c: labels[c] for c in labels.columns})
        return out

    def on_trade_opened(self, event: TradeEvent) -> None:
        hook = getattr(self.inner, "on_trade_opened", None)
        if hook is not None:
            hook(event)

    def on_trade_closed(self, event: TradeEvent) -> None:
        hook = getattr(self.inner, "on_trade_closed", None)
        if hook is not None:
            hook(event)

    def in_regime(self, ctx: MarketContext) -> bool:
        view = ctx.bars(self.timeframes[0])
        return any(view.last(f"regime_{r}") == 1.0 for r in self.regimes)

    def generate(self, ctx: MarketContext) -> list[Signal]:
        # The inner strategy always sees every bar (it may keep state); filters apply after.
        inner_ctx = _RenamedContext(ctx, self._to_outer) if self.spec.timeframes else ctx
        signals = self.inner.generate(inner_ctx)  # type: ignore[arg-type]
        if self.regimes and signals and not self.in_regime(ctx):
            return []
        if self.spec.direction == "both":
            return signals
        return [s for s in signals if s.direction == self.spec.direction]


def build(spec: VariantSpec, seed: int = 0) -> Strategy:
    """The strategy for a spec: the registered one itself when nothing is changed."""
    inner = registry.create(spec.strategy, seed, dict(spec.params))
    return inner if spec.is_plain else Variant(spec, inner)
