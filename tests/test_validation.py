"""Validator tests on synthetic weeks that follow Exness market hours.

Market open: Sunday 18:00 -> Friday 17:00 New York time, daily break 17:00-18:00.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradeagent.data.store import connect_db
from tradeagent.data.validation import ISSUE_TYPES, log_issues, validate_bars
from tradeagent.provenance import git_commit, new_run_id

NY = "America/New_York"


def market_open(ny: pd.DatetimeIndex) -> np.ndarray:
    day, minute = ny.dayofweek, ny.hour * 60 + ny.minute
    weekend = (day == 5) | ((day == 4) & (minute >= 17 * 60)) | ((day == 6) & (minute < 18 * 60))
    daily_break = (day <= 3) & (minute >= 17 * 60) & (minute < 18 * 60)
    return ~(weekend | daily_break)


def trading_bars(first_sunday: str, weeks: int = 2, freq: str = "5min") -> pd.DataFrame:
    """Clean bars from Sunday 18:00 NY for `weeks` weeks, only while the market is open."""
    start = pd.Timestamp(f"{first_sunday} 18:00", tz=NY)
    ny = pd.date_range(start, start + pd.Timedelta(weeks=weeks), freq=freq, inclusive="left")
    ny = ny[market_open(ny)]
    rng = np.random.default_rng(7)
    close = 2000 + np.cumsum(rng.normal(0, 0.5, len(ny)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "time_utc": ny.tz_convert("UTC").astype("datetime64[ns, UTC]"),
            "open": open_,
            "high": np.maximum(open_, close) + 0.3,
            "low": np.minimum(open_, close) - 0.3,
            "close": close,
            "tick_volume": 100,
            "spread": 20,
        }
    )


@pytest.fixture
def week() -> pd.DataFrame:
    return trading_bars("2026-01-04")  # US winter time


def only_issue(bars: pd.DataFrame, timeframe: str = "M5") -> dict[str, int]:
    counts = validate_bars(bars, "XAUUSD", timeframe).counts()
    return {k: v for k, v in counts.items() if v}


@pytest.mark.parametrize("sunday", ["2026-01-04", "2026-07-05"])  # winter and summer time
def test_clean_weeks_have_no_issues(sunday: str) -> None:
    bars = trading_bars(sunday)
    result = validate_bars(bars, "XAUUSD", "M5")
    assert result.counts() == dict.fromkeys(ISSUE_TYPES, 0)
    assert result.expected["gap_weekend"] == 1
    assert result.expected["gap_daily_break"] == 8  # Mon-Thu breaks, two weeks


def test_clean_h1_week(week: pd.DataFrame) -> None:
    bars = trading_bars("2026-01-04", freq="1h")
    assert only_issue(bars, "H1") == {}


def test_input_is_not_modified(week: pd.DataFrame) -> None:
    before = week.copy()
    validate_bars(week, "XAUUSD", "M5")
    pd.testing.assert_frame_equal(week, before)


def test_duplicate_and_out_of_order(week: pd.DataFrame) -> None:
    dup = pd.concat([week, week.iloc[[100]]], ignore_index=True)
    assert only_issue(dup) == {"duplicate": 1, "not_monotonic": 1}
    swapped = week.copy()
    swapped.iloc[[50, 51]] = swapped.iloc[[51, 50]].to_numpy()
    assert only_issue(swapped) == {"not_monotonic": 1}


def test_bad_ohlc(week: pd.DataFrame) -> None:
    bad = week.copy()
    bad.loc[10, "low"] = bad.loc[10, "open"] + 5  # low above open
    bad.loc[20, "close"] = np.nan
    assert only_issue(bad) == {"ohlc_invalid": 2}


def test_spike(week: pd.DataFrame) -> None:
    bad = week.copy()
    bad.loc[1500, "high"] += 200  # ~400x the normal bar size
    assert only_issue(bad) == {"spike": 1}


def test_spreads(week: pd.DataFrame) -> None:
    bad = week.copy()
    bad.loc[1200, "spread"] = 0
    bad.loc[1300, "spread"] = 500  # 25x the median of 20
    bad.loc[1400, "spread"] = 150  # 7.5x: wide but allowed
    assert only_issue(bad) == {"spread_zero": 1, "spread_outlier": 1}


def test_misaligned_bar(week: pd.DataFrame) -> None:
    bad = week.copy()
    bad.loc[300, "time_utc"] += pd.Timedelta(minutes=2)
    assert only_issue(bad)["misaligned"] == 1


def test_saturday_bar_is_flagged(week: pd.DataFrame) -> None:
    saturday = week.iloc[[0]].copy()
    saturday["time_utc"] = pd.Timestamp("2026-01-10 15:00", tz="UTC")
    bars = pd.concat([week, saturday]).sort_values("time_utc", ignore_index=True)
    counts = only_issue(bars)
    assert counts["weekend_bar"] == 1


def test_missing_hour_midweek_is_intraday_gap(week: pd.DataFrame) -> None:
    ny = week["time_utc"].dt.tz_convert(NY)
    hole = (ny >= "2026-01-07 10:00-05:00") & (ny < "2026-01-07 11:00-05:00")  # Wednesday
    result = validate_bars(week[~hole], "XAUUSD", "M5")
    gaps = result.issues[result.issues["issue_type"] == "gap_intraday"]
    assert len(gaps) == 1 and "1h00m" in gaps["details"].iloc[0]
    assert gaps["severity"].iloc[0] == 60  # minutes without a bar


def test_early_close_before_holiday_is_expected(week: pd.DataFrame) -> None:
    # Close at 13:30 on Monday 2026-01-05 and reopen at the normal 18:00.
    for timeframe, freq in (("M5", "5min"), ("H1", "1h")):
        bars = week if timeframe == "M5" else trading_bars("2026-01-04", freq=freq)
        ny_tf = bars["time_utc"].dt.tz_convert(NY)
        closed = (ny_tf >= "2026-01-05 13:30-05:00") & (ny_tf < "2026-01-05 18:00-05:00")
        assert only_issue(bars[~closed], timeframe) == {"gap_holiday": 1}, timeframe


def test_d1_sunday_stubs_are_counted_not_flagged() -> None:
    days = pd.date_range("2026-01-04", "2026-01-17", freq="D", tz="UTC")
    days = days[days.dayofweek != 5]  # no Saturday bars; Sunday stubs present
    bars = pd.DataFrame(
        {
            "time_utc": days.astype("datetime64[ns, UTC]"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "tick_volume": 10,
            "spread": 20,
        }
    )
    result = validate_bars(bars, "USOIL", "D1")
    assert result.expected["sunday_stub"] == 2  # Jan 4 and Jan 11
    assert result.expected["gap_weekend"] == 1  # Fri Jan 9 -> Sun Jan 11
    assert not any(result.counts().values())


def test_rejects_non_utc_times(week: pd.DataFrame) -> None:
    naive = week.assign(time_utc=week["time_utc"].dt.tz_localize(None))
    with pytest.raises(ValueError, match="UTC"):
        validate_bars(naive, "XAUUSD", "M5")


def test_log_issues_writes_provenance(tmp_path: Path, week: pd.DataFrame) -> None:
    bad = week.copy()
    bad.loc[10, "spread"] = 0
    result = validate_bars(bad, "XAUUSD", "M5")
    conn = connect_db(tmp_path / "t.db")
    run_id = new_run_id()
    assert log_issues(conn, result, run_id, git_commit(), "abc123") == 1
    row = conn.execute(
        "SELECT run_id, symbol, timeframe, issue_type, time_utc, git_commit, config_hash "
        "FROM data_quality_log"
    ).fetchone()
    conn.close()
    assert row[:4] == (run_id, "XAUUSD", "M5", "spread_zero")
    assert row[4].endswith("+00:00")  # stored as UTC
    assert row[5] and row[6] == "abc123"
