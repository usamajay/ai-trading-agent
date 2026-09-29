"""New York-close D1/H4 bars built from H1 (docs/DATA_NOTES.md §2 rule 4)."""

import pandas as pd
import pytest
from synthetic_bars import NY, trading_bars

from tradeagent.data.mt5_client import BAR_COLUMNS
from tradeagent.data.resample import RESAMPLED_COLUMNS, resample_ny_close


@pytest.fixture
def h1() -> pd.DataFrame:
    return trading_bars("2026-01-04", weeks=2, freq="1h")  # US winter time


def ny_hours(times: pd.Series) -> list[int]:
    return times.dt.tz_convert(NY).dt.hour.tolist()


def weekdays(d1: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(d1["trading_day"]).dt.dayofweek


def test_d1_has_no_sunday_bars_and_one_bar_per_trading_day(h1: pd.DataFrame) -> None:
    d1 = resample_ny_close(h1, "D1")
    assert list(d1.columns) == RESAMPLED_COLUMNS
    assert len(d1) == 10  # two weeks, Monday-Friday
    assert d1["trading_day"].is_unique
    assert sorted(weekdays(d1).value_counts().to_dict().items()) == [(d, 2) for d in range(5)]
    assert not d1["partial"].any()
    assert (d1["h1_bars"] == 23).all()
    # Each day starts 17:00 New York the day before (Sunday for Monday) and lasts 24 h.
    assert ny_hours(d1["time_utc"]) == [17] * 10
    assert (d1["end_utc"] - d1["time_utc"] == pd.Timedelta(hours=24)).all()


def test_d1_values_match_the_h1_bars_inside(h1: pd.DataFrame) -> None:
    d1 = resample_ny_close(h1, "D1")
    monday = h1[(h1["time_utc"] >= d1["time_utc"][0]) & (h1["time_utc"] < d1["end_utc"][0])]
    row = d1.iloc[0]
    assert len(monday) == 23
    assert row["open"] == monday["open"].iloc[0]
    assert row["high"] == monday["high"].max()
    assert row["low"] == monday["low"].min()
    assert row["close"] == monday["close"].iloc[-1]
    assert row["tick_volume"] == monday["tick_volume"].sum()


def test_h4_has_six_full_blocks_per_day_and_no_partial_friday_bar(h1: pd.DataFrame) -> None:
    h4 = resample_ny_close(h1, "H4")
    assert len(h4) == 60
    assert not h4["partial"].any()
    assert h4.groupby("trading_day").size().eq(6).all()
    assert ny_hours(h4["time_utc"])[:6] == [17, 21, 1, 5, 9, 13]
    assert h4["h1_bars"].tolist()[:6] == [3, 4, 4, 4, 4, 4]  # 17:00-18:00 is the break
    last = h4.iloc[-1]
    assert last["end_utc"] == pd.Timestamp("2026-01-16 17:00", tz=NY)  # Friday close


def test_h4_hand_built_block() -> None:
    times = pd.DatetimeIndex(
        ["2026-01-06 01:00", "2026-01-06 02:00", "2026-01-06 03:00", "2026-01-06 04:00"]
    ).tz_localize(NY)
    h1 = pd.DataFrame(
        {
            "time_utc": times.tz_convert("UTC"),
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.5, 14.0, 12.5, 13.5],
            "low": [9.5, 10.5, 8.0, 12.0],
            "close": [11.0, 12.0, 13.0, 12.5],
            "tick_volume": [5, 6, 7, 8],
            "spread": [20, 21, 25, 30],
        }
    )
    [row] = resample_ny_close(h1, "H4").to_dict("records")
    assert row["time_utc"] == pd.Timestamp("2026-01-06 01:00", tz=NY)
    assert row["end_utc"] == pd.Timestamp("2026-01-06 05:00", tz=NY)
    assert (row["open"], row["high"], row["low"], row["close"]) == (10.0, 14.0, 8.0, 12.5)
    assert row["tick_volume"] == 26
    assert row["spread"] == 23  # median 23
    assert row["h1_bars"] == 4
    assert not row["partial"]

    # Median 20.5 is rounded up: a bar's recorded spread is usually its minimum.
    h1.loc[2:, "spread"] = [21, 20]
    assert resample_ny_close(h1, "H4")["spread"].iloc[0] == 21


@pytest.mark.parametrize(
    ("first_sunday", "utc_hours"),
    [
        ("2026-03-01", [22] * 5 + [21] * 5),  # US summer time starts Sunday 8 March
        ("2025-10-26", [21] * 5 + [22] * 5),  # US summer time ends Sunday 2 November
    ],
)
def test_daylight_saving_weeks(first_sunday: str, utc_hours: list[int]) -> None:
    h1 = trading_bars(first_sunday, weeks=2, freq="1h")
    # "2 weeks" of clock time crosses the DST hour, so cut at the second Friday close
    # (otherwise one H1 bar of the next Sunday slips in and forms a partial day).
    friday_close = (pd.Timestamp(first_sunday) + pd.Timedelta(days=12, hours=17)).tz_localize(NY)
    h1 = h1[h1["time_utc"] < friday_close]
    d1 = resample_ny_close(h1, "D1")
    assert len(d1) == 10
    assert not d1["partial"].any()
    assert ny_hours(d1["time_utc"]) == [17] * 10
    assert d1["time_utc"].dt.hour.tolist() == utc_hours
    h4 = resample_ny_close(h1, "H4")
    assert len(h4) == 60 and not h4["partial"].any()


def test_data_hole_is_marked_partial_not_bridged(h1: pd.DataFrame) -> None:
    ny = h1["time_utc"].dt.tz_convert(NY)
    hole = (ny >= pd.Timestamp("2026-01-07 10:00", tz=NY)) & (
        ny < pd.Timestamp("2026-01-07 12:00", tz=NY)
    )
    d1 = resample_ny_close(h1[~hole], "D1")
    assert d1["partial"].sum() == 1
    wednesday = d1[d1["partial"]].iloc[0]
    assert str(wednesday["trading_day"]) == "2026-01-07"
    assert wednesday["h1_bars"] == 21

    h4 = resample_ny_close(h1[~hole], "H4")
    [block] = h4[h4["partial"]].to_dict("records")
    assert block["time_utc"] == pd.Timestamp("2026-01-07 09:00", tz=NY)
    assert block["h1_bars"] == 2


def test_missing_day_gets_no_bar(h1: pd.DataFrame) -> None:
    ny = h1["time_utc"].dt.tz_convert(NY)
    wednesday = (ny >= pd.Timestamp("2026-01-06 18:00", tz=NY)) & (
        ny < pd.Timestamp("2026-01-07 17:00", tz=NY)
    )
    d1 = resample_ny_close(h1[~wednesday], "D1")
    assert len(d1) == 9
    assert "2026-01-07" not in {str(d) for d in d1["trading_day"]}
    assert not d1["partial"].any()


def test_input_order_and_duplicates_do_not_matter(h1: pd.DataFrame) -> None:
    messy = pd.concat([h1.iloc[::-1], h1.iloc[:5]], ignore_index=True)
    before = messy.copy()
    pd.testing.assert_frame_equal(resample_ny_close(messy, "D1"), resample_ny_close(h1, "D1"))
    pd.testing.assert_frame_equal(messy, before)  # input not modified


def test_empty_and_bad_input() -> None:
    empty = resample_ny_close(pd.DataFrame(columns=BAR_COLUMNS), "D1")
    assert empty.empty and list(empty.columns) == RESAMPLED_COLUMNS
    with pytest.raises(ValueError, match="D1 or H4"):
        resample_ny_close(pd.DataFrame(columns=BAR_COLUMNS), "M15")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing columns"):
        resample_ny_close(pd.DataFrame({"time_utc": []}), "D1")
