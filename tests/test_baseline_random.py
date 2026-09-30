"""Random-entry baseline (SPEC §4.1 #6)."""

import pandas as pd
from synthetic_bars import trading_bars

from tradeagent.backtest.costs import CostModel
from tradeagent.backtest.dataset import WINDOW_COLUMNS, Dataset, flag_bars
from tradeagent.backtest.engine import run_backtest
from tradeagent.backtest.lookahead import check_lookahead
from tradeagent.config import DataExclusions, load_config
from tradeagent.strategies import registry
from tradeagent.strategies.base import check_strategy
from tradeagent.strategies.baseline_random import NAME, RandomBaseline

NONE = DataExclusions(excluded_windows=[], no_trade_windows=[], keep_gaps=[])
WINDOWS = pd.DataFrame(columns=WINDOW_COLUMNS).astype(
    {"start_utc": "datetime64[ns, UTC]", "end_utc": "datetime64[ns, UTC]"}
)


def m15() -> pd.DataFrame:
    return trading_bars("2026-01-04", weeks=3, freq="15min").assign(spread=250)


def backtest(seed: int):  # type: ignore[no-untyped-def]
    cfg = load_config()
    bars = flag_bars(m15(), "M15", "XAUUSD", WINDOWS, NONE, cfg.settings.backtest, True)
    dataset = Dataset(
        "XAUUSD",
        "M15",
        "train",
        "intraday",
        bars["time_utc"].iloc[0],
        bars["time_utc"].iloc[-1],
        bars,
        WINDOWS,
    )
    return run_backtest(
        RandomBaseline(seed),
        dataset,
        CostModel.from_config(cfg, "XAUUSD"),
        cfg.risk,
        cfg.settings.backtest,
    )


def test_registered_and_valid() -> None:
    registry.load_builtins()
    strategy = registry.create(NAME, seed=3, params={"p_entry": 0.05})
    check_strategy(strategy)
    assert strategy.params == {"atr_mult": 1.5, "rr": 2.0, "p_entry": 0.05}


def test_no_lookahead() -> None:
    frames = {"M15": m15()}
    assert check_lookahead(lambda: RandomBaseline(7, {"p_entry": 0.1}), "XAUUSD", frames) == []


def test_signals_pass_reward_risk_from_the_real_entry() -> None:
    result = backtest(seed=1)
    assert result.counts.get("signals", 0) > 20
    assert result.counts.get("rejected_rr", 0) == 0  # stop/target measured from the ask
    assert len(result.trades) > 10


def test_same_seed_same_trades_and_different_seed_different() -> None:
    a, b, c = backtest(1), backtest(1), backtest(2)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    assert not a.trades["entry_time"].equals(c.trades["entry_time"])


def test_both_directions() -> None:
    directions = set(backtest(1).trades["direction"])
    assert directions == {"long", "short"}
