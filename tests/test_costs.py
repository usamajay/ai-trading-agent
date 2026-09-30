"""Cost model: bid/ask sides, spread margin, slippage, P&L, commission and swap.

Every expected number below is worked out by hand in the comment next to it.
"""

import pandas as pd
import pydantic
import pytest

from tradeagent.backtest.costs import CostError, CostModel, rollover_nights, symbol_costs
from tradeagent.config import BacktestSettings, SymbolCosts, load_config
from tradeagent.data.mt5_client import SymbolSpec

NY = "America/New_York"

GOLD = SymbolCosts(
    broker_symbol="XAUUSDm",
    digits=3,
    point=0.001,
    tick_size=0.001,
    tick_value=0.1,
    contract_size=100.0,
    volume_min=0.01,
    volume_step=0.01,
    volume_max=200.0,
    swap_mode=1,
    swap_long=-560.0,
    swap_short=0.0,
    swap_rollover3days=3,  # Wednesday
)
OIL = GOLD.model_copy(
    update={
        "broker_symbol": "USOILm",
        "tick_value": 1.0,
        "contract_size": 1000.0,
        "swap_long": 0.0,
        "swap_short": -186.2,
        "swap_rollover3days": 7,  # no triple day
    }
)
BT = BacktestSettings(
    embargo_weeks=2,
    max_hole_minutes=60,
    no_entry_minutes_after_open=15,
    friday_cutoff_ny="16:30",
    exclusions_source_timeframe="M5",
    spread_margin_multiple=1.1,
    spread_margin_points=0,
    slippage_spread_multiple=0.2,
    commission_per_lot_usd=0.0,
    starting_balance=10_000,
)


@pytest.fixture
def gold() -> CostModel:
    return CostModel.from_settings("XAUUSD", GOLD, BT)


def ny(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


# --- prices ----------------------------------------------------------------------------


def test_spread_margin_and_slippage(gold: CostModel) -> None:
    # stored 200 points x 1.1 = 220 points = 0.220; slippage 0.2 x 0.220 = 0.044
    assert gold.spread(200) == pytest.approx(0.220)
    assert gold.slippage(200) == pytest.approx(0.044)
    extra = CostModel.from_settings(
        "XAUUSD", GOLD, BT.model_copy(update={"spread_margin_points": 5})
    )
    assert extra.spread(200) == pytest.approx(0.225)  # 220 + 5 points


def test_long_enters_at_ask_and_exits_on_bid(gold: CostModel) -> None:
    ask = gold.entry_side("long", bid=2000.0, bar_spread_points=200)
    assert ask == pytest.approx(2000.22)
    assert gold.entry_fill("long", ask, 200, slip=True) == pytest.approx(2000.264)  # + 0.044
    assert gold.exit_side("long", bid=2010.0, bar_spread_points=200) == 2010.0
    assert gold.exit_fill("long", 1990.0, 200, slip=True) == pytest.approx(1989.956)  # SL
    assert gold.exit_fill("long", 2020.0, 200, slip=False) == 2020.0  # TP: exact price


def test_short_enters_at_bid_and_exits_on_ask(gold: CostModel) -> None:
    bid = gold.entry_side("short", bid=2000.0, bar_spread_points=200)
    assert bid == 2000.0
    assert gold.entry_fill("short", bid, 200, slip=True) == pytest.approx(1999.956)
    assert gold.exit_side("short", bid=1990.0, bar_spread_points=200) == pytest.approx(1990.22)
    assert gold.exit_fill("short", 2010.0, 200, slip=True) == pytest.approx(2010.044)  # SL


def test_short_stop_triggers_on_ask_not_bid(gold: CostModel) -> None:
    # Bid bar 1999.0-2001.0, short SL at 2001.10: the bid high never reaches it, but
    # the ask high (2001.0 + 0.22) does, so the stop is hit.
    high, low = gold.exit_range("short", high=2001.0, low=1999.0, bar_spread_points=200)
    assert (high, low) == pytest.approx((2001.22, 1999.22))
    stop_loss = 2001.10
    assert 2001.0 < stop_loss <= high
    # A long's exits use the bid bar unchanged.
    assert gold.exit_range("long", 2001.0, 1999.0, 200) == (2001.0, 1999.0)


def test_pending_orders_are_checked_on_their_side(gold: CostModel) -> None:
    # Buy limit at 1999.10: the bid low 1999.00 would touch it, the ask low 1999.22 doesn't.
    _, low = gold.entry_range("long", high=2001.0, low=1999.0, bar_spread_points=200)
    assert low == pytest.approx(1999.22) and not low <= 1999.10
    # Sell orders use the bid bar.
    assert gold.entry_range("short", 2001.0, 1999.0, 200) == (2001.0, 1999.0)


# --- money -----------------------------------------------------------------------------


def test_pnl_in_dollars() -> None:
    gold = CostModel.from_settings("XAUUSD", GOLD, BT)
    oil = CostModel.from_settings("USOIL", OIL, BT)
    # Gold 0.01 lot = 1 oz: a $10 move is $10.  (10 / 0.001 points x $0.1 x 0.01 lot)
    assert gold.pnl("long", 2000.0, 2010.0, 0.01) == pytest.approx(10.0)
    assert gold.pnl("short", 2000.0, 2010.0, 0.01) == pytest.approx(-10.0)
    # Oil 0.01 lot = 10 barrels: a $1 move is $10.  (1000 points x $1 x 0.01 lot)
    assert oil.pnl("long", 80.0, 81.0, 0.01) == pytest.approx(10.0)
    assert oil.spec.value_per_point_per_lot == 1.0


def test_commission(gold: CostModel) -> None:
    assert gold.commission(1.0) == 0.0  # Exness Standard
    paid = CostModel.from_settings(
        "XAUUSD", GOLD, BT.model_copy(update={"commission_per_lot_usd": 7})
    )
    assert paid.commission(0.5) == pytest.approx(3.5)


# --- swap ------------------------------------------------------------------------------


def test_triple_swap_day_mapping() -> None:
    assert GOLD.triple_swap_weekday == 2  # MT5 3 = Wednesday
    assert OIL.triple_swap_weekday is None  # MT5 7 = no triple day
    assert GOLD.model_copy(update={"swap_rollover3days": 0}).triple_swap_weekday == 6  # Sunday
    assert GOLD.model_copy(update={"swap_rollover3days": 1}).triple_swap_weekday == 0  # Monday


@pytest.mark.parametrize(
    ("open_ny", "close_ny", "triple", "nights"),
    [
        ("2026-01-05 10:00", "2026-01-05 16:55", 2, 0),  # closed before the rollover
        ("2026-01-05 10:00", "2026-01-05 18:05", 2, 1),  # held through Monday 17:00
        ("2026-01-05 10:00", "2026-01-08 10:00", None, 3),  # Mon, Tue, Wed: 3 nights
        ("2026-01-05 10:00", "2026-01-08 10:00", 2, 5),  # gold: Wed counts 3 -> 1+1+3
        ("2026-01-06 10:00", "2026-01-08 10:00", 2, 4),  # Tue 1 + Wed 3
        ("2026-01-09 10:00", "2026-01-12 10:00", None, 1),  # oil over the weekend: Friday only
        ("2026-01-09 10:00", "2026-01-12 10:00", 2, 1),  # gold over the weekend: Friday x1
        ("2026-07-06 10:00", "2026-07-09 10:00", 2, 5),  # same in US summer time
        ("2026-01-05 10:00", "2026-01-05 09:00", 2, 0),  # close before open
    ],
)
def test_rollover_nights(open_ny: str, close_ny: str, triple: int | None, nights: int) -> None:
    assert rollover_nights(ny(open_ny), ny(close_ny), triple) == nights


def test_swap_in_dollars() -> None:
    gold = CostModel.from_settings("XAUUSD", GOLD, BT)
    oil = CostModel.from_settings("USOIL", OIL, BT)
    tue, thu = ny("2026-01-06 10:00"), ny("2026-01-08 10:00")
    # Gold long Tue->Thu: 4 nights x -560 pts x $0.1 x 0.01 lot = -$2.24; shorts pay nothing.
    assert gold.swap("long", 0.01, tue, thu) == pytest.approx(-2.24)
    assert gold.swap("short", 0.01, tue, thu) == 0.0
    # Oil short Fri->Mon: 1 night x -186.2 pts x $1 x 0.01 lot = -$1.862.
    assert oil.swap("short", 0.01, ny("2026-01-09 10:00"), ny("2026-01-12 10:00")) == pytest.approx(
        -1.862
    )
    # Oil short Mon->Thu: 3 nights, Wednesday is not x3 for oil = -$5.586.
    assert oil.swap("short", 0.01, ny("2026-01-05 10:00"), thu) == pytest.approx(-5.586)
    assert oil.swap("long", 0.01, ny("2026-01-05 10:00"), thu) == 0.0


# --- snapshot --------------------------------------------------------------------------


def test_symbol_costs_from_mt5_spec() -> None:
    spec = SymbolSpec(
        symbol="USOILm",
        digits=3,
        point=0.001,
        tick_size=0.001,
        tick_value=1.0,
        contract_size=1000.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread_points=20,
        typical_spread_points=20.0,
        swap_mode=1,
        swap_long=0.0,
        swap_short=-186.2,
        swap_rollover3days=7,
    )
    assert symbol_costs(spec) == OIL.model_copy(update={"volume_max": 100.0})
    with pytest.raises(pydantic.ValidationError):
        symbol_costs(SymbolSpec(**{**spec.__dict__, "swap_mode": 2}))  # only points supported


def test_committed_snapshot_matches_confirmed_values() -> None:
    """Values Usama confirmed in the MT5 Specification window (DECISIONS.md 2026-09-30)."""
    costs = load_config().costs
    assert costs is not None
    gold, oil = costs.symbols["XAUUSD"], costs.symbols["USOIL"]
    assert (gold.swap_long, gold.swap_short, gold.triple_swap_weekday) == (-560.0, 0.0, 2)
    assert (oil.swap_long, oil.swap_short, oil.triple_swap_weekday) == (0.0, -186.2, None)
    assert (gold.contract_size, oil.contract_size) == (100.0, 1000.0)
    assert gold.value_per_point_per_lot == pytest.approx(0.1)
    assert oil.value_per_point_per_lot == pytest.approx(1.0)


def test_cost_model_needs_a_snapshot() -> None:
    cfg = load_config()
    assert CostModel.from_config(cfg, "USOIL").spec.swap_short == -186.2
    with pytest.raises(CostError, match="no cost snapshot"):
        CostModel.from_config(cfg.model_copy(update={"costs": None}), "USOIL")
    with pytest.raises(CostError, match="no BTCUSD"):
        CostModel.from_config(cfg, "BTCUSD")
