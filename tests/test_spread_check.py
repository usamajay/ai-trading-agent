"""Spread measurement: real tick spreads vs the spread MT5 stores per bar."""

import pandas as pd
import pytest

from tradeagent.backtest.spread_check import bar_tick_spreads, propose_multiple, summarize


def ticks(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """(time, spread in points) -> bid/ask ticks with bid 2000.000 and point 0.001."""
    return pd.DataFrame(
        {
            "time_utc": pd.to_datetime([t for t, _ in rows], utc=True),
            "bid": 2000.0,
            "ask": [2000.0 + s * 0.001 for _, s in rows],
        }
    )


def bars(rows: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"time_utc": pd.to_datetime([t for t, _ in rows], utc=True), "spread": [s for _, s in rows]}
    )


def test_time_weighted_spread_per_bar() -> None:
    t = ticks(
        [
            ("2026-01-05 10:00:00", 20),  # lasts 150 s
            ("2026-01-05 10:02:30", 30),  # lasts 150 s, until the bar ends at 10:05
            ("2026-01-05 10:05:00", 40),  # the only tick of the next bar
            ("2026-01-05 10:10:00", 25),  # bar with stored spread 0: left out
        ]
    )
    b = bars(
        [
            ("2026-01-05 10:00", 20),
            ("2026-01-05 10:05", 40),
            ("2026-01-05 10:10", 0),
            ("2026-01-05 10:15", 22),  # no ticks: left out
        ]
    )
    s = bar_tick_spreads(t, b, point=0.001, bar_minutes=5)
    assert len(s) == 2
    first = s.iloc[0]
    assert (first["bar_spread"], first["tick_min"], first["tick_open"]) == (20, 20, 20)
    assert first["tick_twavg"] == pytest.approx(25.0)  # (150 x 20 + 150 x 30) / 300
    assert first["ticks"] == 2
    assert s.iloc[1]["tick_twavg"] == pytest.approx(40.0)

    summary = summarize(s)
    assert summary["bars"] == 2
    assert summary["stored_is_min"] == 1.0
    assert summary["twavg_median"] == pytest.approx(1.125)  # ratios 1.25 and 1.0
    assert summary["twavg_max"] == pytest.approx(1.25)


def test_summary_of_nothing() -> None:
    empty = bar_tick_spreads(ticks([("2026-01-05 10:00", 20)]), bars([]), 0.001, 5)
    assert summarize(empty) == {"bars": 0}


@pytest.mark.parametrize(
    ("p99s", "proposed"),
    [
        ([1.083, 1.0], 1.1),  # the measured 4 weeks (gold, oil)
        ([1.0], 1.0),
        ([1.1], 1.1),  # exactly on a step: not rounded up again
        ([1.661], 1.7),
        ([0.9], 1.0),  # never below 1
        ([], 1.0),
    ],
)
def test_propose_multiple(p99s: list[float], proposed: float) -> None:
    summaries = [{"bars": 100.0, "twavg_p99": p} for p in p99s]
    assert propose_multiple(summaries) == proposed
