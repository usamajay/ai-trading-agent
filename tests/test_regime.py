"""Research regime labels (Phase 6.4): closed bars only, trend/vol states, variant filter."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from tradeagent.features.regime import (
    REGIMES,
    VOL_WINDOW,
    bar_end,
    regime_labels,
    trade_regimes,
)
from tradeagent.strategies import registry
from tradeagent.strategies.base import BaseStrategy, Frames, MarketContext, Signal
from tradeagent.strategies.variants import VariantSpec, build


def _bars(close: np.ndarray, spread: np.ndarray | None = None) -> pd.DataFrame:
    n = len(close)
    width = np.full(n, 0.5) if spread is None else spread
    t = pd.date_range("2025-01-06", periods=n, freq="1h", tz="UTC").as_unit("ns")
    return pd.DataFrame(
        {
            "time_utc": t,
            "open": close,
            "high": close + width,
            "low": close - width,
            "close": close,
            "tick_volume": 1,
            "spread": 10,
        }
    )


def _noise(n: int, seed: int = 3) -> np.ndarray:
    return 100 + np.random.default_rng(seed).normal(0, 0.3, n).cumsum() * 0.1


def test_columns_warm_up_and_values() -> None:
    lab = regime_labels(_bars(_noise(400)))
    assert list(lab.columns) == [
        f"regime_{r}"
        for r in ("trending", "ranging", "low_vol", "high_vol", "normal_vol", "low_vol_expanding")
    ]
    assert set(REGIMES) == {c.removeprefix("regime_") for c in lab.columns}
    assert set(np.unique(lab.to_numpy())) <= {0.0, 1.0}
    vol = lab[["regime_low_vol", "regime_normal_vol", "regime_high_vol"]]
    assert (vol.iloc[: VOL_WINDOW - 1].sum(axis=1) == 0).all()  # warm-up: no vol label
    assert (vol.iloc[VOL_WINDOW + 20 :].sum(axis=1) == 1).all()  # then exactly one


def test_labels_use_closed_bars_only() -> None:
    bars = _bars(_noise(600))
    full = regime_labels(bars)
    for n in (260, 400, 599):
        part = regime_labels(bars.iloc[:n])
        pd.testing.assert_frame_equal(part, full.iloc[:n])


def test_trend_and_range() -> None:
    trend = regime_labels(_bars(np.linspace(100, 200, 300)))
    assert trend["regime_trending"].iloc[-50:].eq(1).all()
    zigzag = 100 + np.tile([0.0, 1.0], 150)
    flat = regime_labels(_bars(zigzag))
    assert flat["regime_ranging"].iloc[-50:].eq(1).all()
    assert flat["regime_trending"].iloc[-50:].eq(0).all()


def test_volatility_states_and_expansion() -> None:
    width = np.r_[np.full(300, 1.0), np.full(60, 0.1), np.linspace(0.1, 3.0, 40)]
    lab = regime_labels(_bars(np.full(400, 100.0), width))
    assert lab["regime_low_vol"].iloc[330:360].eq(1).all()
    assert lab["regime_low_vol_expanding"].iloc[360:].eq(1).any()
    assert lab["regime_high_vol"].iloc[-1] == 1
    assert lab["regime_low_vol_expanding"].iloc[:300].eq(0).all()


def test_trade_regimes_and_bar_end() -> None:
    bars = _bars(np.linspace(100, 200, 300))
    ends = bar_end(bars, "H1")
    assert ends.iloc[0] == bars["time_utc"].iloc[0] + pd.Timedelta("1h")
    assert bar_end(bars.assign(end_utc=bars["time_utc"]), "H1").equals(bars["time_utc"])
    trades = pd.DataFrame(
        {"signal_time": [ends.iloc[-1], ends.iloc[5], pd.Timestamp("2030-01-01", tz="UTC")]}
    )
    got = trade_regimes(trades, bars, "H1")
    assert got["trend"].tolist() == ["trending", "neither", "unknown"]
    assert got["vol"].tolist()[1:] == ["warm_up", "unknown"]


# --- variant filter ------------------------------------------------------------------

NAME = "test_regime_strategy"


class AlwaysLong(BaseStrategy):
    name, version, style = NAME, "1", "swing"
    timeframes = ("H1",)
    suited_regimes = ("trending",)
    lookback_bars = 1

    def __init__(self, seed: int = 0, params: dict[str, float] | None = None) -> None:
        super().__init__()
        self.calls = 0

    def generate(self, ctx: MarketContext) -> list[Signal]:
        self.calls += 1
        c = ctx.bars("H1").last("close")
        return [Signal(ctx.symbol, "long", c - 1, c + 2, why="always")]


@pytest.fixture(autouse=True)
def registered() -> Iterator[None]:
    registry.register(NAME, AlwaysLong)
    yield
    registry.unregister(NAME)


def test_variant_regime_filter() -> None:
    v = build(VariantSpec(NAME, regimes=("suited",)))
    assert v.name == f"{NAME}[reg=suited]" and v.regimes == ("trending",)  # type: ignore[attr-defined]
    close = np.r_[100 + np.tile([0.0, 1.0], 150), np.linspace(101, 200, 300)]
    frames = Frames("XAUUSD", v.prepare({"H1": _bars(close)}))
    t0 = pd.Timestamp("2025-01-06", tz="UTC")
    ranging = v.generate(frames.context(t0 + pd.Timedelta(hours=290)))
    trending = v.generate(frames.context(t0 + pd.Timedelta(hours=590)))
    assert ranging == [] and len(trending) == 1
    assert v.inner.calls == 2  # type: ignore[attr-defined]  # still called outside the regime


def test_variant_regime_validation() -> None:
    with pytest.raises(ValueError, match="unknown regimes"):
        VariantSpec(NAME, regimes=("bullish",))
    spec = VariantSpec(NAME, regimes=("ranging", "high_vol"))
    assert VariantSpec.from_dict(spec.to_dict()) == spec and not spec.is_plain
    assert build(spec).regimes == ("ranging", "high_vol")  # type: ignore[attr-defined]
    AlwaysLong.suited_regimes = ("any",)
    try:
        assert build(VariantSpec(NAME, regimes=("suited",))).regimes == ()  # type: ignore[attr-defined]
        AlwaysLong.suited_regimes = ("moonphase",)
        with pytest.raises(ValueError, match="labeller lacks"):
            build(VariantSpec(NAME, regimes=("suited",)))
        AlwaysLong.suited_regimes = ()
        with pytest.raises(ValueError, match="declares no regimes"):
            build(VariantSpec(NAME, regimes=("suited",)))
    finally:
        AlwaysLong.suited_regimes = ("trending",)
