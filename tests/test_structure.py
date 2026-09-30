"""Market structure: swings confirmed late, breaks of structure, session levels."""

import numpy as np
import pandas as pd
import pytest
from synthetic_bars import NY, trading_bars

from tradeagent.features.structure import (
    asian_range,
    break_of_structure,
    previous_day_levels,
    swing_points,
    swings,
)


def path(highs: list[float], lows: list[float] | None = None, closes: list[float] | None = None):  # type: ignore[no-untyped-def]
    lows = lows if lows is not None else [h - 1 for h in highs]
    closes = closes if closes is not None else [h - 0.5 for h in highs]
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


def test_swing_high_is_confirmed_k_bars_later() -> None:
    bars = path([1, 2, 5, 2, 1, 2, 3])
    assert swing_points(bars, 2)["is_swing_high"].tolist() == [False, False, True] + [False] * 4
    s = swings(bars, 2)
    assert np.isnan(s["swing_high"].iloc[3])  # not yet: bars 3 and 4 must close first
    assert s["swing_high"].iloc[4:].tolist() == [5, 5, 5]
    assert s["swing_high_bar"].iloc[4] == 2


def test_swing_low_and_ties() -> None:
    bars = path([9, 8, 7, 8, 9, 8, 8], lows=[8, 7, 3, 7, 8, 7, 7])
    s = swings(bars, 2)
    assert s["swing_low"].iloc[4] == 3 and np.isnan(s["swing_low"].iloc[3])
    flat = path([1, 5, 5, 1, 1])  # equal highs: neither is strictly above the other
    assert not swing_points(flat, 1)["is_swing_high"].any()


def test_break_of_structure() -> None:
    # Swing high 5 at bar 2 (k=1, known at bar 3); bar 5 closes 6 > 5: BOS up once.
    bars = path([1, 2, 5, 3, 4, 6.5, 7], closes=[1, 2, 4, 2.5, 3.5, 6, 6.8])
    b = break_of_structure(bars, 1)
    assert b["bos_up"].tolist() == [False] * 5 + [True, False]
    assert b["bos_level"].iloc[5] == 5
    assert not b["bos_down"].any()


def test_structure_has_no_lookahead() -> None:
    bars = trading_bars("2026-01-04", weeks=1, freq="15min")
    cut = 250
    full = pd.concat([swings(bars, 3), break_of_structure(bars, 3)], axis=1).iloc[:cut]
    part = pd.concat([swings(bars.iloc[:cut], 3), break_of_structure(bars.iloc[:cut], 3)], axis=1)
    pd.testing.assert_frame_equal(full, part)


def ny_bars(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    times = pd.DatetimeIndex([t for t, _, _ in rows]).tz_localize(NY).tz_convert("UTC")
    return pd.DataFrame(
        {"time_utc": times, "high": [h for _, h, _ in rows], "low": [lo for _, _, lo in rows]}
    )


def test_asian_range_known_after_0300() -> None:
    bars = ny_bars(
        [
            ("2026-01-05 18:00", 101, 99),  # Tuesday's trading day starts
            ("2026-01-05 23:00", 103, 100),
            ("2026-01-06 02:45", 102, 98),
            ("2026-01-06 03:00", 110, 97),  # London: not part of the range
            ("2026-01-06 16:45", 111, 96),
        ]
    )
    r = asian_range(bars)
    assert r["asia_high"].isna().iloc[:3].all()  # unknown during the Asian session
    assert r["asia_high"].iloc[3:].tolist() == [103, 103]
    assert r["asia_low"].iloc[3:].tolist() == [98, 98]


def test_previous_day_levels() -> None:
    bars = ny_bars(
        [
            ("2026-01-05 10:00", 101, 99),  # Monday
            ("2026-01-05 16:00", 105, 100),
            ("2026-01-05 18:00", 103, 102),  # Tuesday's trading day
            ("2026-01-06 10:00", 104, 90),
        ]
    )
    p = previous_day_levels(bars)
    assert p["prev_day_high"].isna().iloc[:2].all()
    assert p["prev_day_high"].iloc[2:].tolist() == [105, 105]
    assert p["prev_day_low"].iloc[2:].tolist() == [99, 99]


def test_bad_k() -> None:
    with pytest.raises(ValueError):
        swings(path([1, 2, 3]), 0)
