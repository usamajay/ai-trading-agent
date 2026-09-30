"""Event engine on tiny hand-built bar sets; every expected number is worked out by hand.

Gold contract: $0.1 per point per lot (point 0.001). Unless a test says otherwise:
spread 0, no slippage, no reopen wait, equity 10,000, risk 0.5% = $50 per trade.
A long signal at bar 0's close 2000.5 with SL 1995 is sized on a 5.5 stop:
$50 / (5,500 points x $0.1) = 0.0909 -> 0.09 lots.
"""

from collections.abc import Sequence

import pandas as pd
import pytest

from tradeagent.backtest.costs import CostModel
from tradeagent.backtest.dataset import WINDOW_COLUMNS, Dataset, flag_bars
from tradeagent.backtest.engine import BacktestResult, held_over_weekend, run_backtest
from tradeagent.config import BacktestSettings, DataExclusions, SymbolCosts, load_config
from tradeagent.strategies.base import BaseStrategy, MarketContext, Signal, Style

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
    swap_rollover3days=3,
)
BT = BacktestSettings(
    embargo_weeks=2,
    max_hole_minutes=60,
    no_entry_minutes_after_open=0,
    friday_cutoff_ny="16:30",
    exclusions_source_timeframe="M5",
    spread_margin_multiple=1.0,
    spread_margin_points=0,
    slippage_spread_multiple=0.0,
    commission_per_lot_usd=0.0,
    starting_balance=10_000,
    cost_stress_multiple=1.5,
)
LIMITS = load_config().risk
NO_RULES = DataExclusions(excluded_windows=[], no_trade_windows=[], keep_gaps=[])
NO_WINDOWS = pd.DataFrame(
    {
        "start_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "end_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "source": pd.Series(dtype="object"),
        "reason": pd.Series(dtype="object"),
    }
)[WINDOW_COLUMNS]

LONG = {"direction": "long", "stop_loss": 1995.0, "take_profit": 2011.0}

Row = tuple[float, float, float, float] | tuple[str, float, float, float, float]


def ny(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=NY)


def frame(
    rows: Sequence[Row], start: str = "2026-01-06 10:00", minutes: int = 5, spread: int = 0
) -> pd.DataFrame:
    """Bars from (open, high, low, close) rows, one every `minutes` from `start`
    (New York); a row starting with a time string sets that bar's time instead."""
    t = ny(start)
    times, prices = [], []
    for row in rows:
        if len(row) == 5:
            t = ny(row[0])  # type: ignore[arg-type]
            prices.append(row[1:])
        else:
            prices.append(row)
        times.append(t)
        t += pd.Timedelta(minutes=minutes)
    o, h, lo, c = zip(*prices, strict=True)
    return pd.DataFrame(
        {
            "time_utc": pd.DatetimeIndex(times).tz_convert("UTC").astype("datetime64[ns, UTC]"),
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "tick_volume": 1,
            "spread": spread,
        }
    )


def aggregate(m5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Higher-timeframe bars built from M5 bars (like the broker does)."""
    key = m5["time_utc"].dt.floor(f"{minutes}min")
    out = m5.groupby(key).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        tick_volume=("tick_volume", "sum"),
        spread=("spread", "max"),
    )
    return out.rename_axis("time_utc").reset_index()


def flagged(
    bars: pd.DataFrame,
    tf: str,
    style: Style,
    settings: BacktestSettings = BT,
    windows: pd.DataFrame = NO_WINDOWS,
) -> pd.DataFrame:
    flat = style in ("scalp", "intraday")
    return flag_bars(bars, tf, "XAUUSD", windows, NO_RULES, settings, flat)


class Scripted(BaseStrategy):
    """Emits the signals planned for each decision bar index (by the bar that closed)."""

    name = "scripted"
    version = "1"

    def __init__(
        self,
        plan: dict[int, list[dict[str, object]]],
        timeframes: tuple[str, ...] = ("M5",),
        style: Style = "swing",
        lookback_bars: int = 1,
    ) -> None:
        super().__init__()
        self.plan = plan
        self.timeframes = timeframes
        self.style = style
        self.lookback_bars = lookback_bars
        self.seen: list[int] = []
        self.last_close_seen: list[float] = []

    def generate(self, ctx: MarketContext) -> list[Signal]:
        view = ctx.bars(self.timeframes[0])
        index = len(view) - 1
        self.seen.append(index)
        self.last_close_seen.append(view.last("close"))
        return [
            Signal(symbol="XAUUSD", why="test", **fields)  # type: ignore[arg-type]
            for fields in self.plan.get(index, [])
        ]


def run(
    strategy: Scripted,
    bars: pd.DataFrame,
    exec_m5: pd.DataFrame | None = None,
    m1: pd.DataFrame | None = None,
    risk: bool = False,
    settings: BacktestSettings = BT,
    windows: pd.DataFrame = NO_WINDOWS,
) -> BacktestResult:
    tf, style = strategy.timeframes[0], strategy.style
    data = flagged(bars, tf, style, settings, windows)
    dataset = Dataset(
        symbol="XAUUSD",
        timeframe=tf,
        split="train",
        style=style,
        start_utc=data["time_utc"].iloc[0],
        end_utc=data["time_utc"].iloc[-1] + pd.Timedelta(days=1),
        bars=data,
        exclusions=windows,
    )
    exec_bars = None if exec_m5 is None else flagged(exec_m5, "M5", style, settings, windows)
    costs = CostModel.from_settings("XAUUSD", GOLD, settings)
    return run_backtest(
        strategy,
        dataset,
        costs,
        LIMITS,
        settings,
        exec_bars=exec_bars,
        m1_bars=m1,
        risk_rules=risk,
    )


def only_trade(result: BacktestResult) -> pd.Series:
    assert len(result.trades) == 1, result.counts
    return result.trades.iloc[0]


# --- fills and exits --------------------------------------------------------------------


def test_signal_on_bar_t_fills_at_bar_t_plus_1_open_not_bar_t_close() -> None:
    strategy = Scripted({0: [LONG]})
    bars = frame(
        [(2000, 2001, 1999, 2000.5), (2001, 2002, 2000.5, 2001.5), (2001.5, 2002, 2001, 2001.5)]
    )
    t = only_trade(run(strategy, bars))
    assert t["entry_price"] == 2001.0  # bar 1's open, not bar 0's close (2000.5)
    assert t["entry_time"] == ny("2026-01-06 10:05")
    assert t["signal_time"] == ny("2026-01-06 10:05")  # bar 0 closed at 10:05
    assert t["lots"] == pytest.approx(0.09)
    assert t["exit_reason"] == "end_of_data"
    assert t["exit_price"] == 2001.5 and t["exit_time"] == ny("2026-01-06 10:15")
    assert strategy.last_close_seen[0] == 2000.5  # at the decision it saw only bar 0


def test_take_profit() -> None:
    bars = frame(
        [(2000, 2001, 1999, 2000.5), (2001, 2002, 2000.5, 2001.5), (2001.5, 2011.5, 2001, 2011)]
    )
    t = only_trade(run(Scripted({0: [LONG]}), bars))
    assert (t["exit_reason"], t["exit_price"]) == ("tp", 2011.0)
    # +$10 x 10,000 points/$... : 10 / 0.001 x $0.1 x 0.09 lots = $90; 1R = 6 -> $54
    assert t["net_pnl"] == pytest.approx(90.0)
    assert t["risk_usd"] == pytest.approx(54.0)
    assert t["r_multiple"] == pytest.approx(90 / 54)
    assert t["min_balance"] == pytest.approx(1200.0)  # $6 stop x 1 oz = $6 at 0.01 lot / 0.5%
    assert t["tie"] == "none"


def test_stop_loss() -> None:
    bars = frame(
        [(2000, 2001, 1999, 2000.5), (2001, 2002, 2000.5, 2001.5), (2001.5, 2002, 1994, 1996)]
    )
    t = only_trade(run(Scripted({0: [LONG]}), bars))
    assert (t["exit_reason"], t["exit_price"]) == ("sl", 1995.0)
    assert t["net_pnl"] == pytest.approx(-54.0) and t["r_multiple"] == pytest.approx(-1.0)


def test_both_in_one_m5_bar_is_stop_loss_first() -> None:
    bars = frame(
        [(2000, 2001, 1999, 2000.5), (2001, 2002, 2000.5, 2001.5), (2001.5, 2012, 1994, 2000)]
    )
    result = run(Scripted({0: [LONG]}), bars)
    t = only_trade(result)
    assert (t["exit_reason"], t["tie"]) == ("sl", "sl_first")
    assert result.counts["tie_sl_first"] == 1


def test_gap_through_stop_fills_at_the_open() -> None:
    bars = frame(
        [
            (2000, 2001, 1999, 2000.5),
            (2001, 2002, 2000.5, 2001.5),
            ("2026-01-06 18:00", 1990, 1991, 1989, 1990.5),  # after the daily break
        ]
    )
    t = only_trade(run(Scripted({0: [LONG]}), bars))
    assert (t["exit_reason"], t["exit_price"]) == ("sl_gap", 1990.0)
    # -$11 x 0.09 lots = -$99, plus the 17:00 rollover swap -$5.04 (held 10:05 -> 18:00);
    # 1R = $54, so more than 1R is lost.
    assert t["swap"] == pytest.approx(-5.04)
    assert t["r_multiple"] == pytest.approx((-99 - 5.04) / 54)


def test_gap_beyond_take_profit_never_fills_better() -> None:
    bars = frame(
        [
            (2000, 2001, 1999, 2000.5),
            (2001, 2002, 2000.5, 2001.5),
            ("2026-01-06 18:00", 2020, 2021, 2019, 2020),
        ]
    )
    t = only_trade(run(Scripted({0: [LONG]}), bars))
    assert (t["exit_reason"], t["exit_price"]) == ("tp", 2011.0)


def test_short_stop_triggers_on_the_ask() -> None:
    # Spread 200 points = 0.2. Short entry 2000 (bid open); SL 2001.1. Bar 2's bid high
    # is 2001.0 (below the stop) but its ask high is 2001.2, so the stop is hit.
    short = {"direction": "short", "stop_loss": 2001.1, "take_profit": 1999.3}
    bars = frame(
        [
            (2000.4, 2000.6, 2000.3, 2000.5),
            (2000, 2000.5, 1999.9, 2000.2),
            (2000, 2001.0, 1999.8, 2000.5),
        ],
        spread=200,
    )
    t = only_trade(run(Scripted({0: [short]}), bars))
    assert t["entry_price"] == 2000.0
    assert (t["exit_reason"], t["exit_price"]) == ("sl", 2001.1)
    # 0.6 stop -> $60 per lot -> 0.83 lots; shorts pay the spread on exit: 0.2 -> $16.60
    assert t["lots"] == pytest.approx(0.83)
    assert t["spread_cost"] == pytest.approx(16.6)


def test_slippage_and_spread_on_a_long() -> None:
    settings = BT.model_copy(update={"slippage_spread_multiple": 0.2})
    bars = frame(
        [(2000, 2001, 1999, 2000.5), (2001, 2002, 2000.5, 2001.5), (2001.5, 2002, 1994, 1996)],
        spread=100,
    )
    t = only_trade(run(Scripted({0: [LONG]}), bars, settings=settings))
    # Entry: ask 2001.1 + slippage 0.02; stop exit 1995 - 0.02.
    assert t["entry_price"] == pytest.approx(2001.12)
    assert t["exit_price"] == pytest.approx(1994.98)
    assert t["slippage_cost"] == pytest.approx(0.04 / 0.001 * 0.1 * t["lots"])


def test_time_exit() -> None:
    signal = {**LONG, "max_hold_bars": 2}
    bars = frame([(2000, 2001, 1999, 2000.5)] + [(2001, 2002, 2000.5, 2001.5)] * 4)
    t = only_trade(run(Scripted({0: [signal]}), bars))
    assert t["exit_reason"] == "time" and t["bars_held"] == 2
    assert t["exit_time"] == ny("2026-01-06 10:15")  # close of bar 2


# --- pending orders -----------------------------------------------------------------------


def test_limit_order_fills_at_its_price() -> None:
    limit = {
        "direction": "long",
        "order_type": "limit",
        "entry_price": 1998.0,
        "expiry_bars": 3,
        "stop_loss": 1990.0,
        "take_profit": 2014.0,
    }
    bars = frame(
        [(2000, 2001, 1999, 2000.5), (2000, 2001, 1997.5, 2000), (2000, 2014.5, 1999, 2014)]
    )
    t = only_trade(run(Scripted({0: [limit]}), bars))
    assert t["entry_price"] == 1998.0 and t["order_type"] == "limit"
    assert (t["exit_reason"], t["exit_price"]) == ("tp", 2014.0)


def test_limit_order_opened_beyond_fills_at_the_better_open() -> None:
    limit = {
        "direction": "long",
        "order_type": "limit",
        "entry_price": 1998.0,
        "expiry_bars": 3,
        "stop_loss": 1990.0,
        "take_profit": 2014.0,
    }
    bars = frame([(2000, 2001, 1999, 2000.5), ("2026-01-06 18:00", 1997, 1998, 1996.5, 1997.5)])
    assert only_trade(run(Scripted({0: [limit]}), bars))["entry_price"] == 1997.0


def test_target_in_the_fill_bar_does_not_count() -> None:
    limit = {
        "direction": "long",
        "order_type": "limit",
        "entry_price": 1998.0,
        "expiry_bars": 3,
        "stop_loss": 1990.0,
        "take_profit": 2014.0,
    }
    bars = frame(
        [(2000, 2001, 1999, 2000.5), (2000, 2015, 1997.5, 2010), (2010, 2010, 2009, 2009.5)]
    )
    t = only_trade(run(Scripted({0: [limit]}), bars))
    assert t["exit_reason"] == "end_of_data"  # the 2015 high came in the fill bar: unknown order


def test_stop_order_fills_at_level_or_worse_open() -> None:
    stop = {
        "direction": "long",
        "order_type": "stop",
        "entry_price": 2003.0,
        "expiry_bars": 3,
        "stop_loss": 1995.0,
        "take_profit": 2020.0,
    }
    touched = frame(
        [(2000, 2001, 1999, 2000.5), (2001, 2004, 2000.5, 2003.5), (2003.5, 2004, 2003, 2003.5)]
    )
    assert only_trade(run(Scripted({0: [stop]}), touched))["entry_price"] == 2003.0
    gapped = frame([(2000, 2001, 1999, 2000.5), ("2026-01-06 18:00", 2005, 2006, 2004.5, 2005.5)])
    assert only_trade(run(Scripted({0: [stop]}), gapped))["entry_price"] == 2005.0


def test_pending_order_expires() -> None:
    limit = {
        "direction": "long",
        "order_type": "limit",
        "entry_price": 1990.0,
        "expiry_bars": 2,
        "stop_loss": 1985.0,
        "take_profit": 2010.0,
    }
    bars = frame([(2000, 2001, 1999, 2000.5)] * 5)
    result = run(Scripted({0: [limit]}), bars)
    assert result.trades.empty
    assert result.counts["orders_expired"] == 1


def test_pending_order_on_the_wrong_side_is_rejected() -> None:
    wrong = {
        "direction": "long",
        "order_type": "limit",
        "entry_price": 2005.0,
        "expiry_bars": 2,
        "stop_loss": 1995.0,
        "take_profit": 2025.0,
    }
    result = run(Scripted({0: [wrong]}), frame([(2000, 2001, 1999, 2000.5)] * 3))
    assert result.counts["rejected_entry_side"] == 1 and result.trades.empty


def test_one_slot_only() -> None:
    other = {**LONG, "take_profit": 2012.0}
    bars = frame([(2000, 2001, 1999, 2000.5)] + [(2001, 2002, 2000.5, 2001.5)] * 3)
    result = run(Scripted({0: [LONG, other], 1: [LONG]}), bars)
    assert result.counts["rejected_position_open"] == 2  # same bar + while the trade is open
    assert len(result.trades) == 1


# --- weekends, swap, exclusions ----------------------------------------------------------


def test_intraday_trade_closes_at_friday_cutoff_and_pending_is_cancelled() -> None:
    bars = frame(
        [(2000, 2001, 1999, 2000.5)] + [(2001, 2002, 2000.5, 2001.5)] * 5, start="2026-01-09 16:10"
    )
    t = only_trade(run(Scripted({0: [LONG]}, style="intraday"), bars))
    assert t["exit_reason"] == "weekend_close"
    assert t["exit_time"] == ny("2026-01-09 16:30")
    assert t["exit_price"] == 2001.0  # 16:30 bar's open (bid)

    limit = {
        "direction": "long",
        "order_type": "limit",
        "entry_price": 1990.0,
        "expiry_bars": 10,
        "stop_loss": 1985.0,
        "take_profit": 2010.0,
    }
    result = run(Scripted({1: [limit]}, style="intraday"), bars)
    assert result.counts["cancelled_weekend"] == 1 and result.trades.empty


def test_swing_trade_holds_over_the_weekend() -> None:
    bars = frame(
        [
            ("2026-01-09 16:50", 2000, 2001, 1999, 2000.5),
            ("2026-01-09 16:55", 2001, 2002, 2000.5, 2001.5),
            ("2026-01-11 18:00", 2002, 2003, 2001.5, 2002.5),
        ]
    )
    t = only_trade(run(Scripted({0: [LONG]}), bars))
    assert t["held_over_weekend"]
    assert t["swap"] == pytest.approx(-560 * 0.1 * 0.09)  # Friday rollover x1 = -$5.04


def test_swap_charged_through_the_daily_rollover() -> None:
    bars = frame(
        [
            ("2026-01-05 16:50", 2000, 2001, 1999, 2000.5),
            ("2026-01-05 16:55", 2001, 2002, 2000.5, 2001.5),
            ("2026-01-05 18:00", 2002, 2003, 2001.5, 2002.5),
        ]
    )
    t = only_trade(run(Scripted({0: [LONG]}), bars))
    assert t["swap"] == pytest.approx(-5.04)  # Monday 17:00 rollover, -560 pts x $0.1 x 0.09
    assert not t["held_over_weekend"]
    assert t["net_pnl"] == pytest.approx(t["gross_pnl"] - 5.04)


def window(start: str, end: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_utc": [ny(start).tz_convert("UTC")],
            "end_utc": [ny(end).tz_convert("UTC")],
            "source": ["decision"],
            "reason": ["test"],
        }
    ).astype({"start_utc": "datetime64[ns, UTC]", "end_utc": "datetime64[ns, UTC]"})


def hole_bars(spread: int = 0) -> pd.DataFrame:
    """Bars 10:00 and 10:05, then nothing until 12:00: a 1h50 data hole."""
    return frame(
        [
            (2000, 2001, 1999, 2000.5),
            (2001, 2002, 2000.5, 2001.5),
            ("2026-01-06 12:00", 1980, 1981, 1979, 1980.5),
            (1980.5, 1981, 1980, 1980.5),
        ],
        spread=spread,
    )


def test_trade_open_at_a_data_hole_closes_before_it() -> None:
    # Spread 100 points (0.1), slippage 0.2 x spread = 0.02.
    settings = BT.model_copy(update={"slippage_spread_multiple": 0.2})
    hole = window("2026-01-06 10:10", "2026-01-06 12:00")
    result = run(Scripted({0: [LONG]}), hole_bars(spread=100), settings=settings, windows=hole)
    t = only_trade(result)  # kept in the results, not dropped
    assert t["exit_reason"] == "data_gap"
    assert t["exit_time"] == ny("2026-01-06 10:10")  # close of the last bar before the hole
    assert t["entry_price"] == pytest.approx(2001.12)  # ask 2001.1 + slippage
    assert t["exit_price"] == pytest.approx(2001.48)  # bid close 2001.5 - slippage
    assert result.counts["closed_data_gap"] == 1
    assert result.final_balance == pytest.approx(10_000 + t["net_pnl"])
    assert "dropped_excluded" not in result.counts


def test_short_at_a_data_hole_closes_on_the_ask() -> None:
    settings = BT.model_copy(update={"slippage_spread_multiple": 0.2})
    short = {"direction": "short", "stop_loss": 2006.0, "take_profit": 1990.0}
    hole = window("2026-01-06 10:10", "2026-01-06 12:00")
    t = only_trade(
        run(Scripted({0: [short]}), hole_bars(spread=100), settings=settings, windows=hole)
    )
    assert t["exit_reason"] == "data_gap"
    assert t["exit_price"] == pytest.approx(2001.62)  # ask 2001.5 + 0.1, + slippage 0.02


def test_data_gap_close_uses_the_last_m5_bar_for_higher_timeframes() -> None:
    m5 = frame(
        [(2000, 2001, 1999, 2000.5)] * 3
        + [(2001, 2002, 2000.5, 2001.5), (2001.5, 2002, 2001, 2001.8)]
    )
    m5 = pd.concat(
        [m5, frame([(1980, 1981, 1979, 1980.5)] * 3, start="2026-01-06 12:00")], ignore_index=True
    )
    hole = window("2026-01-06 10:25", "2026-01-06 12:00")
    result = run(
        Scripted({0: [LONG]}, timeframes=("M15",)), aggregate(m5, 15), exec_m5=m5, windows=hole
    )
    t = only_trade(result)
    assert t["exit_reason"] == "data_gap"
    assert (t["exit_time"], t["exit_price"]) == (ny("2026-01-06 10:25"), 2001.8)  # M5 10:20 close


def test_signals_whose_lookback_touches_an_excluded_window_are_skipped() -> None:
    strategy = Scripted({}, lookback_bars=3)
    bars = frame([(2000, 2001, 1999, 2000.5)] * 6)
    result = run(strategy, bars, windows=window("2026-01-06 10:01", "2026-01-06 10:02"))
    assert result.counts["skipped_lookback_excluded"] == 3  # bars 0, 1, 2 see bar 0
    assert strategy.seen == [3, 4]


# --- M5 execution bars for higher timeframes ---------------------------------------------


def m15_case(
    sub_rows: list[tuple[float, float, float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """M5 bars 10:00-10:40 (bar 0 of M15 is quiet); `sub_rows` are M15 bar 1's M5 bars."""
    quiet = [(2000, 2001, 1999, 2000.5)] * 3
    tail = [(2001, 2002, 2000.5, 2001.5)] * 3
    m5 = frame(quiet + sub_rows + tail)
    return m5, aggregate(m5, 15)


def test_m5_bars_settle_a_tie_inside_an_m15_bar() -> None:
    tp_first = [
        (2001, 2002, 2000.5, 2001.5),
        (2001.5, 2011.5, 2001, 2011),
        (2011, 2011.2, 1994, 1995),
    ]
    m5, m15 = m15_case(tp_first)
    result = run(Scripted({0: [LONG]}, timeframes=("M15",)), m15, exec_m5=m5)
    t = only_trade(result)
    assert (t["exit_reason"], t["tie"]) == ("tp", "m5")
    assert t["exit_time"] == ny("2026-01-06 10:20")
    assert result.counts["tie_m5"] == 1

    sl_first = [
        (2001, 2002, 2000.5, 2001.5),
        (2001.5, 2002, 1994, 1995),  # stop first
        (1995, 2011.5, 1995, 2011),  # target later in the same M15 bar
    ]
    m5, m15 = m15_case(sl_first)
    t = only_trade(run(Scripted({0: [LONG]}, timeframes=("M15",)), m15, exec_m5=m5))
    assert (t["exit_reason"], t["tie"]) == ("sl", "m5")


def test_tie_inside_one_m5_bar_is_stop_first() -> None:
    both = [(2001, 2002, 2000.5, 2001.5), (2001.5, 2012, 1994, 2000), (2000, 2001, 1999.5, 2000)]
    m5, m15 = m15_case(both)
    t = only_trade(run(Scripted({0: [LONG]}, timeframes=("M15",)), m15, exec_m5=m5))
    assert (t["exit_reason"], t["tie"]) == ("sl", "sl_first")


def test_mismatched_m5_bars_fall_back_to_the_decision_bar() -> None:
    tp_first = [
        (2001, 2002, 2000.5, 2001.5),
        (2001.5, 2011.5, 2001, 2011),
        (2011, 2011.2, 1994, 1995),
    ]
    m5, m15 = m15_case(tp_first)
    m5.loc[4, "high"] = 2010.0  # M5 no longer reaches the M15 high
    result = run(Scripted({0: [LONG]}, timeframes=("M15",)), m15, exec_m5=m5)
    t = only_trade(result)
    assert (t["exit_reason"], t["tie"]) == ("sl", "sl_first")
    assert result.counts["exec_fallback_bars"] >= 1


def test_entry_waits_for_the_end_of_the_reopen_window() -> None:
    settings = BT.model_copy(update={"no_entry_minutes_after_open": 15})
    before = frame([(2000, 2001, 1999, 2000.5)] * 24, start="2026-01-06 15:00")  # 15:00-16:55
    after = frame(
        [(2000 + k, 2000 + k + 0.5, 2000 + k - 0.5, 2000 + k) for k in range(12)],
        start="2026-01-06 18:00",
    )  # opens 2000, 2001, ... 2011 at 18:00..18:55
    m5 = pd.concat([before, after], ignore_index=True)
    h1 = aggregate(m5, 60)
    signal = {"direction": "long", "stop_loss": 1990.0, "take_profit": 2030.0}
    t = only_trade(
        run(Scripted({1: [signal]}, timeframes=("H1",)), h1, exec_m5=m5, settings=settings)
    )
    assert t["entry_time"] == ny("2026-01-06 18:15")  # 18:00-18:10 are inside the window
    assert t["entry_price"] == 2003.0

    # An M5 strategy has no later bar inside the same bar: the entry is skipped.
    m5_only = Scripted({23: [signal]})
    result = run(m5_only, m5, settings=settings)
    assert result.trades.empty and result.counts["entry_skipped_blocked"] == 1


def test_m1_check_is_reported_but_never_changes_the_result() -> None:
    bars = frame(
        [(2000, 2001, 1999, 2000.5), (2001, 2002, 2000.5, 2001.5), (2001.5, 2012, 1994, 2000)]
    )
    m1 = frame(
        [(2001.5, 2011.5, 2001, 2011), (2011, 2012, 1994, 1995)],
        start="2026-01-06 10:10",
        minutes=1,
    )
    result = run(Scripted({0: [LONG]}), bars, m1=m1)
    t = only_trade(result)
    assert (t["exit_reason"], t["tie"], t["m1_check"]) == ("sl", "sl_first", "tp")
    assert result.counts["m1_check_differs"] == 1


# --- risk basics and accounting ------------------------------------------------------------


def test_risk_rules_reject_and_accept() -> None:
    # 20 bars with range 2 and no gaps: ATR(14) = 2. Stop 2 (1 x ATR).
    bars = frame([(2000, 2001, 1999, 2000)] * 20)
    low_rr = {"direction": "long", "stop_loss": 1998.0, "take_profit": 2003.0}  # RR 1.5
    good = {"direction": "long", "stop_loss": 1998.0, "take_profit": 2004.0}  # RR 2.0
    result = run(Scripted({3: [good], 15: [low_rr], 16: [good]}), bars, risk=True)
    assert result.counts["rejected_sl_atr"] == 1  # bar 3: ATR not ready
    assert result.counts["rejected_rr"] == 1
    assert result.counts["orders"] == 1 and len(result.trades) == 1


def test_zero_cost_trades_add_up_to_the_equity_change() -> None:
    rows = []
    for k in range(6):
        rows += [(2000, 2001, 1999, 2000.5), (2001, 2002, 2000.5, 2001.5)]
        rows.append((2001.5, 2011.5, 2001, 2011) if k % 2 else (2001.5, 2002, 1994, 1996))
    plan = {3 * k: [LONG] for k in range(6)}
    result = run(Scripted(plan), frame(rows))
    assert len(result.trades) == 6
    assert result.trades["net_pnl"].sum() == pytest.approx(result.final_balance - 10_000)
    assert result.equity["equity"].iloc[-1] == pytest.approx(result.final_balance)
    assert result.equity["equity"].iloc[0] == 10_000


def test_held_over_weekend_helper() -> None:
    assert held_over_weekend(ny("2026-01-09 16:00"), ny("2026-01-11 18:30"))
    assert not held_over_weekend(ny("2026-01-05 10:00"), ny("2026-01-09 16:30"))


def test_daily_mark_to_market_values_the_open_trade() -> None:
    bars = frame(
        [
            ("2026-01-05 16:50", 2000, 2001, 1999, 2000.5),
            ("2026-01-05 16:55", 2001, 2002, 2000.5, 2001.5),  # entry 2001; Monday closes
            ("2026-01-05 18:00", 2002, 2003, 2001.5, 2002.5),  # Tuesday's trading day
            ("2026-01-05 18:05", 2002.5, 2011.5, 2002, 2011),  # take-profit 2011
            ("2026-01-05 18:10", 2011, 2012, 2010, 2011),
        ]
    )
    result = run(Scripted({0: [LONG]}), bars)
    t = only_trade(result)
    daily = result.daily
    assert [str(d) for d in daily["trading_day"]] == ["2026-01-05", "2026-01-06"]
    monday = daily.iloc[0]
    # Open at Monday's 17:00 close: +0.5 x 10,000 points/$... = 0.5/0.001 x $0.1 x 0.09 = $4.50;
    # no swap yet (the 17:00 rollover is not strictly before the 17:00 mark).
    assert monday["time_utc"] == ny("2026-01-05 17:00")
    assert monday["equity"] == pytest.approx(10_004.5)
    assert monday["closed_equity"] == 10_000
    # Tuesday (last day): the trade has closed, so both equal the final balance.
    tuesday = daily.iloc[1]
    assert tuesday["equity"] == pytest.approx(result.final_balance)
    assert tuesday["closed_equity"] == pytest.approx(10_000 + t["net_pnl"])
    assert result.bars_in_split == 5


def test_stress_costs_change_fills_not_which_trades_are_taken() -> None:
    # 20 bars of range 2 (ATR 2), spread 100 points (0.1). A long sized from the ask
    # 2000.1 with SL 1998.1 and TP 2004.1 has RR exactly 2.0 at normal costs.
    from tradeagent.backtest.runner import stressed

    bars = frame([(2000, 2001, 1999, 2000)] * 20, spread=100)
    signal = {"direction": "long", "stop_loss": 1998.1, "take_profit": 2004.1}
    data = flagged(bars, "M5", "swing")
    dataset = Dataset(
        "XAUUSD",
        "M5",
        "train",
        "swing",
        data["time_utc"].iloc[0],
        data["time_utc"].iloc[-1] + pd.Timedelta(days=1),
        data,
        NO_WINDOWS,
    )
    normal = CostModel.from_settings("XAUUSD", GOLD, BT)
    hard = stressed(normal, 1.5)

    def go(costs: CostModel, decision: CostModel | None) -> BacktestResult:
        return run_backtest(
            Scripted({15: [signal]}), dataset, costs, LIMITS, BT, decision_costs=decision
        )

    base = go(normal, None)
    assert len(base.trades) == 1
    # Judged with stressed costs, the ask is 2000.15 and RR falls below 2: rejected.
    assert go(hard, None).counts.get("rejected_rr") == 1
    # The stress run judges with normal costs: same trade, filled 0.05 worse.
    stress = go(hard, normal)
    assert len(stress.trades) == 1
    assert stress.trades["entry_price"].iloc[0] == pytest.approx(
        base.trades["entry_price"].iloc[0] + 0.05
    )


def test_news_blackout_rejects_signals_in_backtests() -> None:
    from tradeagent.data.news import NewsCalendar, NewsEvent

    bars = frame([(2000, 2001, 1999, 2000)] * 20)
    good = {"direction": "long", "stop_loss": 1998.0, "take_profit": 2004.0}
    event = NewsEvent(ny("2026-01-06 11:30"), "CPI", "test")  # bar 15 closes at 11:20
    covered = ((ny("2026-01-01"), ny("2026-02-01")),)
    data = flagged(bars, "M5", "swing")
    dataset = Dataset(
        "XAUUSD",
        "M5",
        "train",
        "swing",
        data["time_utc"].iloc[0],
        data["time_utc"].iloc[-1] + pd.Timedelta(days=1),
        data,
        NO_WINDOWS,
    )
    costs = CostModel.from_settings("XAUUSD", GOLD, BT)

    def go(news: NewsCalendar) -> BacktestResult:
        return run_backtest(Scripted({15: [good]}), dataset, costs, LIMITS, BT, news=news)

    blocked = go(NewsCalendar((event,), covered, 30))
    assert blocked.counts.get("rejected_news") == 1 and blocked.trades.empty
    assert len(go(NewsCalendar((), covered, 30)).trades) == 1
    unknown = go(NewsCalendar((), (), 30))  # no calendar for these dates: not traded
    assert unknown.counts.get("rejected_news_unknown") == 1


def test_account_limits_enforced_only_when_asked() -> None:
    # Four stop-outs in a row (bars of range 2, ATR 2, stop 2): the fifth signal falls
    # inside the 24-hour pause when account limits are enforced.
    rows = [(2000, 2001, 1999, 2000)] * 16
    for _ in range(5):
        rows += [(2000, 2001, 1999, 2000), (2000, 2000.5, 1997.5, 1998)]
    bars = frame(rows)
    signal = {"direction": "long", "stop_loss": 1998.0, "take_profit": 2004.0}
    plan = {16 + 2 * k: [signal] for k in range(5)}  # on the flat bars (close 2000)
    data = flagged(bars, "M5", "swing")
    dataset = Dataset(
        "XAUUSD",
        "M5",
        "train",
        "swing",
        data["time_utc"].iloc[0],
        data["time_utc"].iloc[-1] + pd.Timedelta(days=1),
        data,
        NO_WINDOWS,
    )
    costs = CostModel.from_settings("XAUUSD", GOLD, BT)

    free = run_backtest(Scripted(plan), dataset, costs, LIMITS, BT)
    assert len(free.trades) == 5 and "rejected_loss_streak" not in free.counts
    enforced = run_backtest(Scripted(plan), dataset, costs, LIMITS, BT, enforce_account_limits=True)
    assert len(enforced.trades) == 4
    assert enforced.counts["account_loss_streak"] == 1
    assert enforced.counts["rejected_loss_streak"] == 1
