"""Bar validation (SPEC §3.3). Flags problems; never changes or deletes data.

Market hours used to tell expected gaps from real ones live in
`data/market_hours.py` (New York time, measured from Exness data; see
docs/DATA_NOTES.md).
"""

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradeagent.data.market_hours import (
    EARLY_CLOSE_FROM_MIN,
    EARLY_CLOSE_TO_MIN,
    NEW_YORK,
    WEEK_CLOSE_FROM,
    WEEK_OPEN_BY,
    WEEKEND_STRICT_FROM,
    WEEKEND_STRICT_TO,
    in_closure,
    week_minutes,
)
from tradeagent.data.mt5_client import BAR_COLUMNS, TIMEFRAMES
from tradeagent.timeutil import utc_now

# Logged to data_quality_log.
ISSUE_TYPES = (
    "duplicate",
    "not_monotonic",
    "ohlc_invalid",
    "misaligned",
    "weekend_bar",
    "gap_intraday",  # missing bars while the market should be open
    "gap_holiday",  # a closure longer than the normal weekend/daily break
    "spike",
    "spread_zero",
    "spread_outlier",
)
# Normal market behaviour: counted in the report, not logged.
EXPECTED_TYPES = ("gap_weekend", "gap_daily_break", "sunday_stub")

# severity: how bad, for sorting (gap minutes, or multiple of normal for spikes/spreads)
ISSUE_COLUMNS = ["issue_type", "time_utc", "details", "severity"]


@dataclass(frozen=True)
class ValidationResult:
    symbol: str
    timeframe: str
    bars: int
    issues: pd.DataFrame  # columns: ISSUE_COLUMNS
    expected: dict[str, int] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        found = self.issues["issue_type"].value_counts()
        return {t: int(found.get(t, 0)) for t in ISSUE_TYPES}


def validate_bars(
    bars: pd.DataFrame,
    symbol: str,
    timeframe: str,
    spike_atr_multiple: float = 20.0,
    spread_multiple: float = 10.0,
    median_window: int = 2000,
) -> ValidationResult:
    """Run every SPEC §3.3 rule on one symbol/timeframe of bars."""
    missing = set(BAR_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f"bars are missing columns: {sorted(missing)}")
    if str(bars["time_utc"].dtype) != "datetime64[ns, UTC]":
        raise ValueError(f"time_utc must be datetime64[ns, UTC], got {bars['time_utc'].dtype}")
    bar_length = pd.Timedelta(TIMEFRAMES[timeframe][1])

    found: list[pd.DataFrame] = []
    t = bars["time_utc"]
    found.append(_rows("duplicate", t[t.duplicated(keep="first")], "repeated timestamp"))
    back = t.diff() < pd.Timedelta(0)
    found.append(_rows("not_monotonic", t[back], "earlier than the bar before it"))

    # Every later rule needs one bar per time, oldest first.
    df = (
        bars.drop_duplicates("time_utc", keep="last").sort_values("time_utc").reset_index(drop=True)
    )
    found += [
        _check_ohlc(df),
        _check_alignment(df, timeframe, bar_length),
        _check_weekend_bars(df, bar_length),
        _check_spikes(df, spike_atr_multiple, median_window),
        _check_spreads(df, spread_multiple, median_window),
    ]
    gaps, expected = _check_gaps(df, bar_length)
    found.append(gaps)
    if bar_length >= pd.Timedelta(hours=4):
        # Broker H4/D1 bars are cut at 00:00 UTC, so Sunday's 1-2 hours of trading
        # form their own tiny bar (see docs/DATA_NOTES.md).
        expected["sunday_stub"] = int((df["time_utc"].dt.dayofweek == 6).sum())
    else:
        expected["sunday_stub"] = 0

    non_empty = [f for f in found if not f.empty]
    issues = (
        pd.concat(non_empty, ignore_index=True).sort_values(["time_utc", "issue_type"])
        if non_empty
        else pd.DataFrame(columns=ISSUE_COLUMNS)
    )
    return ValidationResult(
        symbol=symbol,
        timeframe=timeframe,
        bars=len(bars),
        issues=issues.reset_index(drop=True),
        expected=expected,
    )


def log_issues(
    conn: sqlite3.Connection,
    result: ValidationResult,
    run_id: str,
    git_commit: str,
    config_hash: str,
) -> int:
    """Write the issues to data_quality_log. Returns rows written."""
    created_at = utc_now().isoformat()
    issues = result.issues
    rows = [
        (
            run_id,
            result.symbol,
            result.timeframe,
            issue_type,
            time_utc.isoformat(),
            details,
            git_commit,
            config_hash,
            created_at,
        )
        for issue_type, time_utc, details in zip(
            issues["issue_type"], issues["time_utc"], issues["details"], strict=True
        )
    ]
    with conn:
        conn.executemany(
            "INSERT INTO data_quality_log (run_id, symbol, timeframe, issue_type, time_utc, "
            "details, git_commit, config_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


# --- individual rules --------------------------------------------------------------


def _rows(
    issue_type: str,
    times: pd.Series,
    details: pd.Series | str,
    severity: pd.Series | float = 0.0,
) -> pd.DataFrame:
    # Plain arrays (not Series) so rows line up by position, never by index label.
    n = len(times)

    def column(value: pd.Series | str | float) -> list[object] | np.ndarray:
        return value.to_numpy() if isinstance(value, pd.Series) else [value] * n

    return pd.DataFrame(
        {
            "issue_type": [issue_type] * n,
            "time_utc": times.to_numpy(),
            "details": column(details),
            "severity": column(severity),
        },
        columns=ISSUE_COLUMNS,
    ).astype({"time_utc": "datetime64[ns, UTC]", "severity": "float64"})


def _check_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    prices = df[["open", "high", "low", "close"]]
    body_low = prices[["open", "close"]].min(axis=1)
    body_high = prices[["open", "close"]].max(axis=1)
    bad = (
        prices.isna().any(axis=1)
        | (prices <= 0).any(axis=1)
        | (df["low"] > body_low)
        | (df["high"] < body_high)
    )
    details = (
        "O="
        + df["open"].astype(str)
        + " H="
        + df["high"].astype(str)
        + " L="
        + df["low"].astype(str)
        + " C="
        + df["close"].astype(str)
    )
    return _rows("ohlc_invalid", df.loc[bad, "time_utc"], details[bad])


def _check_alignment(df: pd.DataFrame, timeframe: str, bar_length: pd.Timedelta) -> pd.DataFrame:
    seconds = df["time_utc"].astype("int64") // 1_000_000_000
    step = int(bar_length.total_seconds())
    bad = seconds % step != 0
    return _rows("misaligned", df.loc[bad, "time_utc"], f"not on a {timeframe} boundary")


def _check_weekend_bars(df: pd.DataFrame, bar_length: pd.Timedelta) -> pd.DataFrame:
    """Bars lying entirely inside the weekend closure (e.g. a Saturday bar)."""
    start = week_minutes(df["time_utc"].dt.tz_convert(NEW_YORK))
    end = week_minutes((df["time_utc"] + bar_length).dt.tz_convert(NEW_YORK))
    inside = (start >= WEEKEND_STRICT_FROM) & (end <= WEEKEND_STRICT_TO) & (end >= start)
    inside &= bar_length < pd.Timedelta(days=2)
    return _rows("weekend_bar", df.loc[inside, "time_utc"], "bar while market is closed")


GAP_COLUMNS = ["start_utc", "end_utc", "kind", "minutes", "details"]


def find_gaps(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Every gap between consecutive bars, classified like `validate_bars` does.

    One row per gap: start_utc (where the missing time begins, i.e. the end of the
    last bar before it), end_utc (open time of the first bar after it), kind
    (`weekend`, `daily_break`, `holiday` or `intraday`), minutes, details.
    """
    bar_length = pd.Timedelta(TIMEFRAMES[timeframe][1])
    df = bars.drop_duplicates("time_utc", keep="last").sort_values("time_utc")
    return _classify_gaps(df.reset_index(drop=True), bar_length)


def _classify_gaps(df: pd.DataFrame, bar_length: pd.Timedelta) -> pd.DataFrame:
    """Find missing bars and sort them into weekend / daily break / holiday / intraday."""
    prev_end = df["time_utc"].shift() + bar_length
    next_start = df["time_utc"]
    is_gap = next_start > prev_end
    a = prev_end[is_gap].dt.tz_convert(NEW_YORK)  # market went quiet here
    b = next_start[is_gap].dt.tz_convert(NEW_YORK)  # and came back here
    length = b - a

    a_week, b_week = week_minutes(a), week_minutes(b)
    weekend = (
        (a_week >= WEEK_CLOSE_FROM)
        & (b_week <= WEEK_OPEN_BY)
        & (b_week >= a_week)
        & (length <= pd.Timedelta(days=2, hours=2))
    )
    daily_break = ~weekend & in_closure(a) & in_closure(b) & (length <= pd.Timedelta(hours=2))

    # Trading stopped somewhere inside the last bar [a - bar_length, a]; it was an
    # early close if that bar overlaps the early-close window (H1 bars end at 15:00
    # on a 14:30 close).
    a_minute = a.dt.hour * 60 + a.dt.minute
    bar_minutes = int(bar_length.total_seconds() // 60)
    early_close = (a_minute >= EARLY_CLOSE_FROM_MIN) & (
        a_minute - bar_minutes <= EARLY_CLOSE_TO_MIN
    )
    if bar_length <= pd.Timedelta(hours=1):
        # Fine bars show exactly when trading stopped: a holiday closure starts at a
        # normal or early close and ends at a normal reopen.
        holiday = (in_closure(a) | early_close) & in_closure(b)
    else:
        # H4/D1 bars are too coarse to see close times, and 4+ hours with no trade
        # at all only happens when the market is closed. Real data holes still
        # show up on the finer timeframes.
        holiday = pd.Series(True, index=a.index)
    holiday &= ~weekend & ~daily_break

    kind = pd.Series("intraday", index=a.index)
    kind[holiday] = "holiday"
    kind[daily_break] = "daily_break"
    kind[weekend] = "weekend"
    details = (
        "no bars for "
        + _fmt_duration(length)
        + " from "
        + a.dt.strftime("%a %Y-%m-%d %H:%M")
        + " New York time"
    )
    return pd.DataFrame(
        {
            "start_utc": prev_end[is_gap].to_numpy(),
            "end_utc": next_start[is_gap].to_numpy(),
            "kind": kind.to_numpy(),
            "minutes": (length.dt.total_seconds() / 60).to_numpy(),
            "details": details.to_numpy(),
        },
        columns=GAP_COLUMNS,
    ).astype({"start_utc": "datetime64[ns, UTC]", "end_utc": "datetime64[ns, UTC]"})


def _check_gaps(df: pd.DataFrame, bar_length: pd.Timedelta) -> tuple[pd.DataFrame, dict[str, int]]:
    gaps = _classify_gaps(df, bar_length)
    logged = [
        _rows(f"gap_{kind}", g["end_utc"], g["details"], g["minutes"])
        for kind in ("holiday", "intraday")
        if not (g := gaps[gaps["kind"] == kind]).empty
    ]
    issues = pd.concat(logged, ignore_index=True) if logged else pd.DataFrame(columns=ISSUE_COLUMNS)
    counts = gaps["kind"].value_counts()
    expected = {f"gap_{k}": int(counts.get(k, 0)) for k in ("weekend", "daily_break")}
    return issues, expected


def _fmt_duration(length: pd.Series) -> pd.Series:
    minutes = (length.dt.total_seconds() // 60).astype(int)
    return (minutes // 60).astype(str) + "h" + (minutes % 60).astype(str).str.zfill(2) + "m"


def _trailing_median(values: pd.Series, window: int) -> pd.Series:
    """Median of the previous `window` values (past only), falling back to the overall median."""
    median = values.rolling(window, min_periods=min(100, window)).median().shift()
    return median.fillna(values.median())


def _check_spikes(df: pd.DataFrame, multiple: float, window: int) -> pd.DataFrame:
    prev_close = df["close"].shift()
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=1).mean()
    typical_atr = _trailing_median(atr, window)
    bar_range = df["high"] - df["low"]
    ratio = bar_range / typical_atr
    bad = ratio > multiple
    details = (
        "range "
        + bar_range.round(3).astype(str)
        + " = "
        + ratio.round(1).astype(str)
        + "x median ATR "
        + typical_atr.round(3).astype(str)
    )
    return _rows("spike", df.loc[bad, "time_utc"], details[bad], ratio[bad])


def _check_spreads(df: pd.DataFrame, multiple: float, window: int) -> pd.DataFrame:
    spread = df["spread"].astype(float)
    typical = _trailing_median(spread[spread > 0], window).reindex(df.index).ffill().bfill()
    zero = spread <= 0
    wide = ~zero & (spread > multiple * typical)
    wide_details = (
        "spread "
        + df["spread"].astype(str)
        + " pts = "
        + (spread / typical).round(1).astype(str)
        + "x median "
        + typical.astype(str)
    )
    return pd.concat(
        [
            _rows("spread_zero", df.loc[zero, "time_utc"], "spread is 0 or negative"),
            _rows(
                "spread_outlier",
                df.loc[wide, "time_utc"],
                wide_details[wide],
                (spread / typical)[wide],
            ),
        ],
        ignore_index=True,
    )
