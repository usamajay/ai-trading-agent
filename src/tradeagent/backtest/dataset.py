"""Bars for one backtest, with per-bar flags (docs/DATA_NOTES.md §3-5).

Flags added to every bar:
  in_split        bar starts inside the split period (earlier bars are warm-up
                  history for indicators only: no entries)
  excluded        an excluded window overlaps this bar or the time since the previous
                  bar (a data hole). No signal may use a lookback containing such a
                  bar, and a trade still open when one is reached is dropped
  no_new_entries  no entry may fill at this bar's open (entry_blocked or after_reopen)
  entry_blocked   hard block: warm-up, excluded, reviewed no-trade window, or
                  (scalp/intraday) after the Friday cut-off
  after_reopen    within the first minutes after a reopen (wide spreads): an entry
                  waits until the window has passed
  flatten         scalp/intraday only: at or after the Friday cut-off, close open
                  trades and cancel pending orders
  gap_before      time is missing before this bar (weekend, daily break, holiday or
                  hole): a stop the open jumps past fills at the open
  gap_minutes     how long that gap is

Excluded windows = reviewed `excluded_windows` + every unexpected gap longer than
`max_hole_minutes` found on the source timeframe (M5), unless it lies inside a
reviewed `keep_gaps` or `no_trade_windows` entry. New gaps nobody has reviewed are
excluded by default.
"""

from dataclasses import dataclass

import pandas as pd

from tradeagent.backtest.splits import split_range
from tradeagent.config import AppConfig, BacktestSettings, DataExclusions, SplitName
from tradeagent.data.market_hours import after_weekly_cutoff
from tradeagent.data.mt5_client import TIMEFRAMES
from tradeagent.data.resample import resample_ny_close
from tradeagent.data.store import BarStore
from tradeagent.data.validation import find_gaps
from tradeagent.strategies.base import Style

# A gap at least this long is a "reopen" (weekly open, daily break, holiday, hole):
# spreads are wide for the first minutes after it.
REOPEN_GAP_MINUTES = 30

WINDOW_COLUMNS = ["start_utc", "end_utc", "source", "reason"]
FLAG_COLUMNS = [
    "in_split",
    "excluded",
    "no_new_entries",
    "entry_blocked",
    "after_reopen",
    "flatten",
    "gap_before",
    "gap_minutes",
]


@dataclass(frozen=True)
class Dataset:
    symbol: str
    timeframe: str
    split: SplitName
    style: Style
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    bars: pd.DataFrame  # prices + FLAG_COLUMNS, oldest first, warm-up bars included
    exclusions: pd.DataFrame  # WINDOW_COLUMNS that were applied


def exclusion_windows(
    gaps: pd.DataFrame,
    exclusions: DataExclusions,
    symbol: str,
    max_hole_minutes: int,
) -> pd.DataFrame:
    """Reviewed excluded windows + long unexpected gaps nobody has reviewed as fine."""
    rows = [
        (_ts(w.start), _ts(w.end), "decision", f"{w.reason} (decision {w.decision})")
        for w in exclusions.excluded_windows
        if w.applies_to(symbol)
    ]
    # Gaps a human looked at and called fine (normal closure or a no-trade window).
    reviewed_fine = [
        w for w in [*exclusions.keep_gaps, *exclusions.no_trade_windows] if w.applies_to(symbol)
    ]
    decided = [w for w in exclusions.excluded_windows if w.applies_to(symbol)]
    holes = gaps[(gaps["kind"] == "intraday") & (gaps["minutes"] > max_hole_minutes)]
    for start, end, details in zip(
        holes["start_utc"], holes["end_utc"], holes["details"], strict=True
    ):
        if any(_ts(w.start) <= start and end <= _ts(w.end) for w in reviewed_fine):
            continue
        if any(start < _ts(w.end) and end > _ts(w.start) for w in decided):
            continue  # already covered by a reviewed excluded window
        rows.append((start, end, "unreviewed_gap", details))
    windows = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    windows = windows.astype({"start_utc": "datetime64[ns, UTC]", "end_utc": "datetime64[ns, UTC]"})
    return windows.sort_values("start_utc").reset_index(drop=True)


def flag_bars(
    bars: pd.DataFrame,
    timeframe: str,
    symbol: str,
    windows: pd.DataFrame,
    exclusions: DataExclusions,
    settings: BacktestSettings,
    flat_before_weekend: bool,
    period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Return a copy of `bars` (oldest first) with FLAG_COLUMNS added."""
    df = bars.sort_values("time_utc").reset_index(drop=True)
    start = df["time_utc"]
    if "end_utc" in df.columns:  # New York-close D1/H4 carry their own end time
        end = df["end_utc"]
    else:
        end = start + pd.Timedelta(TIMEFRAMES[timeframe][1])
    prev_end = end.shift().fillna(start)
    gap = ((start - prev_end).dt.total_seconds() / 60).clip(lower=0)

    if period is None:
        in_split = pd.Series(True, index=df.index)
    else:
        in_split = (start >= period[0]) & (start < period[1])

    # A window counts for this bar if it overlaps [previous bar end, this bar end):
    # that covers windows inside the bar and holes just before it.
    excluded = pd.Series(False, index=df.index)
    for w_start, w_end in zip(windows["start_utc"], windows["end_utc"], strict=True):
        excluded |= (end > w_start) & (prev_end < w_end)

    blocked = ~in_split | excluded
    for w in exclusions.no_trade_windows:
        if w.applies_to(symbol):
            blocked |= (start >= _ts(w.start)) & (start < _ts(w.end))
    # The first bar counts as a reopen too: we cannot see what came before it.
    is_reopen = (gap >= REOPEN_GAP_MINUTES) | (df.index == 0)
    reopen_at = start.where(is_reopen).ffill()
    since_reopen = start - reopen_at
    after_reopen = since_reopen < pd.Timedelta(minutes=settings.no_entry_minutes_after_open)
    if flat_before_weekend:
        flatten = after_weekly_cutoff(start, settings.friday_cutoff_minutes)
    else:
        flatten = pd.Series(False, index=df.index)
    blocked |= flatten

    return df.assign(
        in_split=in_split,
        excluded=excluded,
        no_new_entries=blocked | after_reopen,
        entry_blocked=blocked,
        after_reopen=after_reopen,
        flatten=flatten,
        gap_before=gap > 0,
        gap_minutes=gap,
    )


def load_bars(store: BarStore, symbol: str, timeframe: str) -> pd.DataFrame:
    """Stored bars, except D1/H4, which are rebuilt on a 17:00 New York close from H1."""
    if timeframe in ("D1", "H4"):
        return resample_ny_close(store.read(symbol, "H1"), timeframe)  # type: ignore[arg-type]
    return store.read(symbol, timeframe)


def load_dataset(
    cfg: AppConfig,
    store: BarStore,
    symbol: str,
    timeframe: str,
    split: SplitName,
    style: Style,
    warmup_bars: int = 500,
) -> Dataset:
    """Bars of one split plus up to `warmup_bars` earlier bars of indicator history.

    Raises SplitError for out-of-sample (locked in Phase 2) or unfrozen split dates.
    """
    period = split_range(cfg.splits, split)
    bt = cfg.settings.backtest
    bars = load_bars(store, symbol, timeframe)
    bars = bars[bars["time_utc"] < period[1]].reset_index(drop=True)
    inside = (bars["time_utc"] >= period[0]).to_numpy().nonzero()[0]
    if len(inside) == 0:
        raise ValueError(f"no {symbol} {timeframe} bars in the {split} period")
    bars = bars.iloc[max(0, inside[0] - warmup_bars) :]

    source = store.read(symbol, bt.exclusions_source_timeframe)
    gaps = find_gaps(source, bt.exclusions_source_timeframe)
    windows = exclusion_windows(gaps, cfg.exclusions, symbol, bt.max_hole_minutes)
    flagged = flag_bars(
        bars,
        timeframe,
        symbol,
        windows,
        cfg.exclusions,
        bt,
        flat_before_weekend=style in ("scalp", "intraday"),
        period=period,
    )
    return Dataset(
        symbol=symbol,
        timeframe=timeframe,
        split=split,
        style=style,
        start_utc=period[0],
        end_utc=period[1],
        bars=flagged,
        exclusions=windows,
    )


def _ts(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")  # type: ignore[arg-type]
