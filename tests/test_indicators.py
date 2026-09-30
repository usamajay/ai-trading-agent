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


# --- Phase 3 indicators -------------------------------------------------------------

from synthetic_bars import NY, trading_bars

from tradeagent.features.indicators import (
    adx,
    average_volume,
    bollinger,
    ema,
    macd,
    rsi,
    session_vwap,
)


def test_ema_seeded_with_simple_average() -> None:
    values = ema(pd.Series([1.0, 2, 3, 4, 5]), 3).tolist()
    assert np.isnan(values[0]) and np.isnan(values[1])
    # seed (1+2+3)/3 = 2; alpha 0.5: 0.5*4 + 0.5*2 = 3; 0.5*5 + 0.5*3 = 4
    assert values[2:] == pytest.approx([2.0, 3.0, 4.0])


def test_rsi_wilder() -> None:
    # deltas +1 +1 -1 +1 +1. Seed: gain 2/3, loss 1/3 -> RS 2 -> 66.67;
    # then gain 7/9, loss 2/9 -> 77.78; then gain 23/27, loss 4/27 -> 85.19.
    values = rsi(pd.Series([1.0, 2, 3, 2, 3, 4]), 3).tolist()
    assert all(np.isnan(v) for v in values[:3])
    assert values[3:] == pytest.approx([100 - 100 / 3, 100 - 100 / 4.5, 100 - 100 / 6.75])
    assert rsi(pd.Series([1.0, 2, 3, 4]), 2).iloc[-1] == 100.0  # no losses


def test_macd_is_ema_difference_with_signal() -> None:
    close = pd.Series(np.linspace(100, 130, 60) + np.sin(np.arange(60)))
    m = macd(close, 12, 26, 9)
    expected = ema(close, 12) - ema(close, 26)
    assert np.allclose(m["macd"], expected, equal_nan=True)
    assert np.allclose(m["macd_signal"], ema(m["macd"], 9), equal_nan=True)
    assert m["macd_signal"].first_valid_index() == 25 + 8  # 26 bars, then 9 MACD values


def test_bollinger() -> None:
    b = bollinger(pd.Series([1.0, 2, 3, 4]), 3, 2.0).iloc[2]
    std = np.sqrt(2 / 3)  # population std of 1, 2, 3
    assert b["bb_mid"] == pytest.approx(2.0)
    assert (b["bb_upper"], b["bb_lower"]) == pytest.approx((2 + 2 * std, 2 - 2 * std))
    assert b["bb_width"] == pytest.approx(4 * std / 2)


def test_adx_wilder_by_hand() -> None:
    bars = pd.DataFrame(
        {
            "high": [10.0, 11, 12, 11.5, 12.5],
            "low": [8.0, 9, 10, 9.5, 10],
            "close": [9.0, 10.5, 11.5, 10, 12],
        }
    )
    a = adx(bars, 2)
    # bar 2: +DM avg 1, -DM avg 0, TR avg 2 -> +DI 50, -DI 0, DX 100
    # bar 3: +DM 0.5, -DM 0.25, TR 2 -> +DI 25, -DI 12.5, DX 33.33; ADX = (100 + 33.33) / 2
    # bar 4: +DM 0.75, -DM 0.125, TR 2.25 -> +DI 33.33, -DI 5.56, DX 71.43; ADX = mean
    assert a["plus_di"].iloc[2] == pytest.approx(50.0) and a["minus_di"].iloc[2] == 0.0
    assert a["adx"].iloc[3] == pytest.approx((100 + 100 / 3) / 2)
    dx4 = 100 * (100 / 3 - 50 / 9) / (100 / 3 + 50 / 9)
    assert a["adx"].iloc[4] == pytest.approx(((100 + 100 / 3) / 2 + dx4) / 2)
    assert np.isnan(a["adx"].iloc[2])


def test_adx_high_in_a_trend_low_in_a_range() -> None:
    n = 120
    trend = pd.DataFrame({"close": np.arange(n, dtype=float)})
    trend = trend.assign(high=trend["close"] + 0.5, low=trend["close"] - 0.5)
    chop = pd.DataFrame({"close": 100 + np.tile([0.0, 1.0], n // 2)})
    chop = chop.assign(high=chop["close"] + 0.5, low=chop["close"] - 0.5)
    assert adx(trend)["adx"].iloc[-1] > 50
    assert adx(chop)["adx"].iloc[-1] < 20


def test_session_vwap_restarts_each_trading_day() -> None:
    times = (
        pd.DatetimeIndex(
            ["2026-01-05 16:50", "2026-01-05 16:55", "2026-01-05 18:00", "2026-01-05 18:05"]
        )
        .tz_localize(NY)
        .tz_convert("UTC")
    )
    bars = pd.DataFrame(
        {
            "time_utc": times,
            "high": [11.0, 13, 21, 23],
            "low": [9.0, 11, 19, 21],
            "close": [10.0, 12, 20, 22],
            "tick_volume": [1, 3, 2, 2],
        }
    )
    v = session_vwap(bars).tolist()
    # Monday: (10*1 + 12*3) / 4 = 11.5; Tuesday restarts at 18:00: 20, then (20*2 + 22*2)/4 = 21
    assert v == pytest.approx([10.0, 11.5, 20.0, 21.0])


def test_average_volume() -> None:
    bars = pd.DataFrame({"tick_volume": [1, 2, 3, 4]})
    assert average_volume(bars, 2).tolist()[1:] == [1.5, 2.5, 3.5]


def test_indicators_do_not_change_when_future_bars_are_added() -> None:
    bars = trading_bars("2026-01-04", weeks=1, freq="15min")
    cut = 300
    head = bars.iloc[:cut]

    def compute(b: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                ema(b["close"], 20).rename("ema"),
                rsi(b["close"]).rename("rsi"),
                macd(b["close"]),
                bollinger(b["close"]),
                adx(b),
                atr(b).rename("atr"),
                session_vwap(b),
                average_volume(b).rename("vol"),
            ],
            axis=1,
        )

    full, part = compute(bars).iloc[:cut], compute(head)
    pd.testing.assert_frame_equal(full, part)
