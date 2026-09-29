"""Indicators checked against hand-calculated values."""

import numpy as np
import pandas as pd
import pytest

from tradeagent.features.indicators import atr, true_range


@pytest.fixture
def bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 11.0, 14.0],
            "low": [8.0, 9.0, 9.0, 10.0, 10.0],
            "close": [9.0, 10.0, 11.0, 10.5, 13.0],
        }
    )


def test_true_range(bars: pd.DataFrame) -> None:
    # bar 0: 10-8; bar 2: max(3, |12-10|, |9-10|); bar 4: max(4, |14-10.5|, |10-10.5|)
    assert true_range(bars).tolist() == [2.0, 2.0, 3.0, 1.0, 4.0]


def test_true_range_keeps_a_gap() -> None:
    # Friday close 100, Monday bar 110-111: the 10-point weekend jump counts.
    gap = pd.DataFrame({"high": [101.0, 111.0], "low": [99.0, 110.0], "close": [100.0, 110.5]})
    assert true_range(gap).tolist() == [2.0, 11.0]


def test_atr_wilder(bars: pd.DataFrame) -> None:
    values = atr(bars, period=3).tolist()
    assert np.isnan(values[0]) and np.isnan(values[1])
    # (2+2+3)/3, then (7/3*2 + 1)/3 = 17/9, then (17/9*2 + 4)/3 = 70/27
    assert values[2:] == pytest.approx([7 / 3, 17 / 9, 70 / 27])


def test_atr_uses_past_bars_only(bars: pd.DataFrame) -> None:
    changed = bars.copy()
    changed.loc[4, ["high", "close"]] = [50.0, 49.0]
    assert atr(changed, 3)[:4].equals(atr(bars, 3)[:4])


def test_atr_short_input_and_bad_period(bars: pd.DataFrame) -> None:
    assert atr(bars, period=10).isna().all()
    with pytest.raises(ValueError):
        atr(bars, period=0)
