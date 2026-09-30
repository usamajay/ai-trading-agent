"""Strategy interface (SPEC §4): signals, parameters, and the closed-bars-only context."""

import numpy as np
import pandas as pd
import pytest
from synthetic_bars import NY, trading_bars

from tradeagent.data.resample import resample_ny_close
from tradeagent.strategies.base import (
    BaseStrategy,
    Frames,
    InvalidSignal,
    InvalidStrategy,
    MarketContext,
    ParamSpec,
    Signal,
    Strategy,
    check_strategy,
)

# --- signals ---------------------------------------------------------------------------


def long_signal(**changes: object) -> Signal:
    fields: dict[str, object] = {
        "symbol": "XAUUSD",
        "direction": "long",
        "stop_loss": 1990.0,
        "take_profit": 2020.0,
        "why": "test",
    }
    fields.update(changes)
    return Signal(**fields)  # type: ignore[arg-type]


def test_valid_market_signals() -> None:
    assert long_signal().order_type == "market"
    short = long_signal(direction="short", stop_loss=2020.0, take_profit=1990.0)
    assert short.direction == "short"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"why": "  "}, "why"),
        ({"stop_loss": 2030.0}, "stop-loss must be below"),
        ({"direction": "short"}, "stop-loss must be above"),
        ({"direction": "up"}, "long or short"),
        ({"stop_loss": -1.0}, "positive"),
        ({"take_profit": float("nan")}, "positive"),
        ({"entry_price": 2000.0}, "market orders"),
        ({"expiry_bars": 3}, "market orders"),
        ({"order_type": "limit", "expiry_bars": 3}, "need an entry_price"),
        ({"order_type": "stop", "entry_price": 2000.0}, "expiry_bars"),
        ({"order_type": "limit", "entry_price": 2000.0, "expiry_bars": 0}, "expiry_bars"),
        ({"order_type": "limit", "entry_price": 2025.0, "expiry_bars": 3}, "between"),
        ({"order_type": "fok"}, "unknown order type"),
        ({"max_hold_bars": 0}, "max_hold_bars"),
    ],
)
def test_invalid_signals(changes: dict[str, object], message: str) -> None:
    with pytest.raises(InvalidSignal, match=message):
        long_signal(**changes)


@pytest.mark.parametrize(
    ("direction", "order_type", "entry", "ok"),
    [
        ("long", "limit", 1995.0, True),  # buy limit below the price
        ("long", "limit", 2005.0, False),
        ("long", "stop", 2005.0, True),  # buy stop above the price
        ("long", "stop", 1995.0, False),
        ("short", "limit", 2005.0, True),  # sell limit above the price
        ("short", "limit", 1995.0, False),
        ("short", "stop", 1995.0, True),  # sell stop below the price
        ("short", "stop", 2005.0, False),
    ],
)
def test_pending_order_side(direction: str, order_type: str, entry: float, ok: bool) -> None:
    sl, tp = (1980.0, 2030.0) if direction == "long" else (2030.0, 1980.0)
    signal = long_signal(
        direction=direction,
        order_type=order_type,
        entry_price=entry,
        expiry_bars=4,
        stop_loss=sl,
        take_profit=tp,
    )
    error = signal.entry_side_error(price=2000.0)
    assert (error is None) is ok
    assert long_signal().entry_side_error(2000.0) is None  # market orders: always fine


# --- strategy rules --------------------------------------------------------------------


class Demo(BaseStrategy):
    name = "demo"
    version = "1.0"
    style = "intraday"
    timeframes = ("M15", "H1")
    suited_regimes = ("trending",)
    lookback_bars = 50
    param_specs = (
        ParamSpec("atr_mult", 0.5, 3.0, "stop distance in ATRs"),
        ParamSpec("rr", 1.5, 4.0, "take-profit distance as a multiple of the stop"),
    )

    def generate(self, ctx: MarketContext) -> list[Signal]:
        return []


def test_valid_strategy() -> None:
    demo = Demo(atr_mult=1.5, rr=2.0)
    assert isinstance(demo, Strategy)
    check_strategy(demo)


def test_param_spec_needs_description_and_range() -> None:
    with pytest.raises(InvalidStrategy, match="description"):
        ParamSpec("x", 0, 1, " ")
    with pytest.raises(InvalidStrategy, match="low"):
        ParamSpec("x", 2, 1, "x")


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda s: setattr(s, "params", {"atr_mult": 5.0, "rr": 2.0}), "outside its range"),
        (lambda s: setattr(s, "params", {"atr_mult": 1.0}), "must match"),
        (lambda s: setattr(s, "timeframes", ("M15", "M7")), "unknown timeframes"),
        (lambda s: setattr(s, "timeframes", ()), "at least one timeframe"),
        (lambda s: setattr(s, "timeframes", ("H1", "H1")), "twice"),
        (lambda s: setattr(s, "style", "hft"), "style"),
        (lambda s: setattr(s, "version", ""), "name and a version"),
        (lambda s: setattr(s, "lookback_bars", 0), "lookback_bars"),
        (
            lambda s: setattr(
                s, "param_specs", tuple(ParamSpec(f"p{i}", 0, 1, "x") for i in range(6))
            ),
            "at most 5",
        ),
    ],
)
def test_invalid_strategies(setup, message: str) -> None:  # type: ignore[no-untyped-def]
    demo = Demo(atr_mult=1.5, rr=2.0)
    setup(demo)
    with pytest.raises(InvalidStrategy, match=message):
        check_strategy(demo)


# --- what a strategy can see ----------------------------------------------------------


@pytest.fixture
def frames() -> Frames:
    m15 = trading_bars("2026-01-04", weeks=1, freq="15min")
    h1 = trading_bars("2026-01-04", weeks=1, freq="1h")
    d1 = resample_ny_close(h1, "D1")
    return Frames("XAUUSD", {"M15": m15, "H1": h1, "D1": d1})


def ny(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=NY)


def utc_ns(text: str) -> np.datetime64:
    return ny(text).tz_convert("UTC").tz_localize(None).to_datetime64()


def test_only_closed_bars_are_visible(frames: Frames) -> None:
    ctx = frames.context(ny("2026-01-06 10:15"))
    m15, h1 = ctx.bars("M15"), ctx.bars("H1")
    assert m15.time_utc[-1] == utc_ns("2026-01-06 10:00")  # 10:00-10:15 just closed
    assert h1.time_utc[-1] == utc_ns("2026-01-06 09:00")  # 10:00-11:00 is still forming
    # D1 for Tuesday (Mon 17:00 -> Tue 17:00 New York) is not closed yet; Monday's is.
    d1 = ctx.bars("D1")
    assert len(d1) == 1 and d1.time_utc[-1] == utc_ns("2026-01-04 17:00")


def test_bar_becomes_visible_exactly_at_its_close(frames: Frames) -> None:
    before = frames.context(ny("2026-01-06 10:59")).bars("H1")
    at_close = frames.context(ny("2026-01-06 11:00")).bars("H1")
    assert len(at_close) == len(before) + 1
    assert at_close.time_utc[-1] == utc_ns("2026-01-06 10:00")


def test_views_are_read_only_and_end_at_now(frames: Frames) -> None:
    view = frames.context(ny("2026-01-06 10:15")).bars("M15")
    with pytest.raises(ValueError):
        view.close[-1] = 0.0
    with pytest.raises(IndexError):
        view.last("close", back=len(view))
    assert view.last("close") == view.close[-1]
    assert view.last("close", back=1) == view.close[-2]
    with pytest.raises(KeyError, match="not loaded"):
        frames.context(ny("2026-01-06 10:15")).bars("M5")


def test_prepared_columns_are_passed_through(frames: Frames) -> None:
    m15 = trading_bars("2026-01-04", weeks=1, freq="15min")
    m15["atr_14"] = np.arange(len(m15), dtype=float)
    ctx = Frames("XAUUSD", {"M15": m15}).context(ny("2026-01-05 10:00"))
    view = ctx.bars("M15")
    assert view.last("atr_14") == len(view) - 1


def test_frames_reject_bad_input() -> None:
    bars = trading_bars("2026-01-04", weeks=1, freq="1h")
    with pytest.raises(ValueError, match="unknown timeframe"):
        Frames("XAUUSD", {"H2": bars})
    naive = bars.assign(time_utc=bars["time_utc"].dt.tz_localize(None))
    with pytest.raises(ValueError, match="UTC"):
        Frames("XAUUSD", {"H1": naive})
