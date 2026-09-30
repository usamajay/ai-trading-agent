"""Strategies by name, so the CLI can build them: `--strategy random_baseline`.

A factory takes (seed, params) and returns a fresh strategy; building a fresh one
for every run keeps runs independent and repeatable (same seed -> same result).
"""

import importlib
from collections.abc import Callable

from tradeagent.strategies.base import Strategy

Factory = Callable[[int, dict[str, float]], Strategy]

_FACTORIES: dict[str, Factory] = {}

# Modules that register built-in strategies when imported (added phase by phase).
BUILTIN_MODULES: tuple[str, ...] = (
    "tradeagent.strategies.baseline_random",
    "tradeagent.strategies.trend_ema_pullback",
    "tradeagent.strategies.breakout_compression",
    "tradeagent.strategies.mean_reversion_bb",
    "tradeagent.strategies.session_breakout",
    "tradeagent.strategies.structure_retest",
)


def load_builtins() -> None:
    for module in BUILTIN_MODULES:
        importlib.import_module(module)


def register(name: str, factory: Factory) -> None:
    if name in _FACTORIES:
        raise ValueError(f"strategy {name!r} is already registered")
    _FACTORIES[name] = factory


def unregister(name: str) -> None:
    _FACTORIES.pop(name, None)


def names() -> list[str]:
    return sorted(_FACTORIES)


def create(name: str, seed: int = 0, params: dict[str, float] | None = None) -> Strategy:
    if name not in _FACTORIES:
        raise KeyError(f"unknown strategy {name!r}; known: {', '.join(names()) or 'none yet'}")
    return _FACTORIES[name](seed, dict(params or {}))
