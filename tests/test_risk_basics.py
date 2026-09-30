"""Temporary backtest risk basics: sizing, minimum balance and signal checks."""

import pytest

from tradeagent.backtest.risk_basics import check_signal, min_balance, position_size
from tradeagent.config import RiskLimits, SymbolCosts, load_config

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
    swap_rollover3days=3,
)
OIL = GOLD.model_copy(update={"tick_value": 1.0, "contract_size": 1000.0})


@pytest.fixture
def limits() -> RiskLimits:
    return load_config().risk  # the real risk.yaml, read-only


@pytest.mark.parametrize(
    ("spec", "sl_distance", "lots"),
    [
        (GOLD, 5.0, 0.10),  # $50 risk / ($5 x 100 oz = $500 per lot) = 0.1
        (GOLD, 7.3, 0.06),  # 50 / 730 = 0.0684 -> rounded DOWN to 0.06
        (GOLD, 60.0, 0.0),  # 50 / 6000 = 0.0083 -> below 0.01: nothing
        (OIL, 0.5, 0.10),  # $50 / ($0.5 x 1000 barrels = $500 per lot) = 0.1
    ],
)
def test_position_size(spec: SymbolCosts, sl_distance: float, lots: float) -> None:
    assert position_size(10_000, 0.5, sl_distance, spec) == pytest.approx(lots)


def test_position_size_edge_cases() -> None:
    assert position_size(10_000, 0.5, 0.0, GOLD) == 0.0
    assert position_size(-5, 0.5, 5.0, GOLD) == 0.0
    assert position_size(1e12, 0.5, 5.0, GOLD) == GOLD.volume_max


def test_min_balance() -> None:
    # Gold, $60 stop: 0.01 lot loses $60; at 0.5% risk that needs $12,000.
    assert min_balance(60.0, GOLD, 0.5) == pytest.approx(12_000)
    # Gold, $5 stop: 0.01 lot loses $5 -> $1,000.
    assert min_balance(5.0, GOLD, 0.5) == pytest.approx(1_000)


def check(limits: RiskLimits, **changes: float) -> tuple[str | None, float]:
    args = {
        "entry_ref": 2000.0,
        "stop_loss": 1995.0,
        "take_profit": 2010.0,  # RR exactly 2.0
        "atr": 5.0,  # stop = 1.0 x ATR
        "bar_spread": 190.0,
        "median_spread": 190.0,
        "equity": 10_000.0,
    }
    args.update(changes)
    return check_signal(limits=limits, spec=GOLD, **args)


def test_accepted_signal_gets_a_size(limits: RiskLimits) -> None:
    assert check(limits) == (None, pytest.approx(0.10))


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"take_profit": 2009.9}, "rr"),  # RR 1.98 < 2.0
        ({"stop_loss": 2000.0}, "rr"),  # no risk distance
        ({"atr": float("nan")}, "no_atr"),
        ({"atr": 12.0}, "sl_atr"),  # stop 0.42 x ATR < 0.5
        ({"atr": 1.5, "take_profit": 2020.0}, "sl_atr"),  # stop 3.33 x ATR > 3
        ({"bar_spread": 381.0}, "spread"),  # > 2 x 190
        ({"equity": 900.0}, "min_lot"),  # $4.50 risk budget < $5 at 0.01 lot
    ],
)
def test_rejections(limits: RiskLimits, changes: dict[str, float], reason: str) -> None:
    assert check(limits, **changes)[0] == reason


def test_boundaries_are_allowed(limits: RiskLimits) -> None:
    assert check(limits, bar_spread=380.0)[0] is None  # exactly 2 x median
    assert check(limits, atr=10.0)[0] is None  # exactly 0.5 x ATR
    assert check(limits, stop_loss=1985.0, take_profit=2030.0)[0] is None  # 3 x ATR, RR 2
