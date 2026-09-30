"""Strategy variants (Phase 6.3): names, timeframe renaming, direction filter."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from tradeagent.strategies import registry
from tradeagent.strategies.base import BaseStrategy, Frames, MarketContext, Signal
from tradeagent.strategies.baseline_random import RandomBaseline
from tradeagent.strategies.variants import Variant, VariantSpec, base_name, build

NAME = "test_two_tf"


class TwoTimeframes(BaseStrategy):
    """Reads M15 and H1 by name; one long and one short signal on every bar."""

    name, version, style = NAME, "1", "intraday"
    timeframes = ("M15", "H1")
    lookback_bars = 1

    def __init__(self, seed: int = 0, params: dict[str, float] | None = None) -> None:
        super().__init__()

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return {tf: df.assign(tag=float(len(tf))) for tf, df in frames.items()}

    def generate(self, ctx: MarketContext) -> list[Signal]:
        m, h = ctx.bars("M15"), ctx.bars("H1")
        c = m.last("close")
        why = f"{m.timeframe}/{h.timeframe}"
        return [
            Signal(ctx.symbol, "long", c - 1, c + 2, why=why),
            Signal(ctx.symbol, "short", c + 1, c - 2, why=why),
        ]


@pytest.fixture(autouse=True)
def registered() -> Iterator[None]:
    registry.register(NAME, TwoTimeframes)
    yield
    registry.unregister(NAME)


def _bars(freq: str, n: int) -> pd.DataFrame:
    t = pd.date_range("2026-01-05", periods=n, freq=freq, tz="UTC").as_unit("ns")
    close = np.linspace(100, 110, n)
    return pd.DataFrame(
        {
            "time_utc": t,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "tick_volume": 1,
            "spread": 10,
        }
    )


def test_spec_name_and_round_trip() -> None:
    spec = VariantSpec(
        "s", timeframes=("H4", "D1"), direction="short", style="swing", params={"rr": 3.0}
    )
    assert spec.name() == "s[tf=H4/D1,dir=short,style=swing]"
    assert VariantSpec.from_dict(spec.to_dict()) == spec
    assert VariantSpec("s").name() == "s" and VariantSpec("s").is_plain
    assert base_name(spec.name()) == "s" == base_name("s")
    with pytest.raises(ValueError):
        VariantSpec("s", direction="up")  # type: ignore[arg-type]


def test_build_plain_is_the_strategy_itself() -> None:
    assert isinstance(build(VariantSpec(NAME)), TwoTimeframes)
    assert isinstance(build(VariantSpec(NAME, direction="long")), Variant)


def test_timeframes_are_renamed_both_ways() -> None:
    v = build(VariantSpec(NAME, timeframes=("H4", "D1"), style="swing"))
    assert list(v.timeframes) == ["H4", "D1"] and v.style == "swing"
    assert v.lookback_bars == 1 and v.params == {}
    prepared = v.prepare({"H4": _bars("4h", 30), "D1": _bars("1D", 10)})
    assert list(prepared) == ["H4", "D1"]  # decision timeframe stays first
    assert prepared["H4"]["tag"].iloc[0] == 3.0  # prepared as the inner "M15"
    frames = Frames("XAUUSD", prepared)
    signals = v.generate(frames.context(pd.Timestamp("2026-01-10", tz="UTC")))
    assert [s.why for s in signals] == ["H4/D1"] * 2  # inner asked for M15/H1, got H4/D1


def test_direction_filter() -> None:
    frames = Frames("XAUUSD", {"M15": _bars("15min", 20), "H1": _bars("1h", 10)})
    ctx = frames.context(pd.Timestamp("2026-01-05 06:00", tz="UTC"))
    assert [s.direction for s in build(VariantSpec(NAME, direction="short")).generate(ctx)] == [
        "short"
    ]
    assert len(build(VariantSpec(NAME, direction="both", style="swing")).generate(ctx)) == 2


def test_wrong_number_of_timeframes() -> None:
    with pytest.raises(ValueError, match="uses 2 timeframes"):
        build(VariantSpec(NAME, timeframes=("H4",)))


def test_random_baseline_direction() -> None:
    frames = Frames("XAUUSD", {"M15": _bars("15min", 60)})
    b = RandomBaseline(1, {"p_entry": 0.2}, direction="short")
    prepared = Frames("XAUUSD", b.prepare({"M15": _bars("15min", 60)}))
    sides = set()
    for i in range(20, 60):
        ctx = prepared.context(pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(minutes=15 * i))
        sides |= {s.direction for s in b.generate(ctx)}
    assert sides == {"short"} and frames.timeframes == ["M15"]
    with pytest.raises(ValueError):
        RandomBaseline(1, direction="up")


def test_leads_parse() -> None:
    from tradeagent.research.leads import parse_leads

    md = (
        "| # | Hypothesis | Why | How | Source | Status |\n|---|---|---|---|---|---|\n"
        "| H1 | **Big idea** here | evidence | re-run it | me | proposed |\n"
        "| H2 | short |\nnot a row\n"
    )
    leads = parse_leads(md)
    assert [(x.hypothesis_id, x.text) for x in leads] == [("H1", "Big idea here")]
    assert leads[0].rationale == "evidence Test: re-run it"
