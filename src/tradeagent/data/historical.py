"""Bulk/incremental history download from MT5 into BarStore (Phase 1, task 1.4).

Each run fetches only what is missing on disk:
- the "tail": from the last stored bar up to now;
- the "head": from the configured start back to the first stored bar (e.g. after
  raising MT5's "Max bars in chart"). The head is fetched newest-first, so an
  interrupted run never leaves a hole in the middle.
Gaps inside the stored range are not re-fetched; validation (task 1.5) reports them.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import pandas as pd
from loguru import logger

from tradeagent.config import Settings
from tradeagent.data.mt5_client import TIMEFRAMES
from tradeagent.data.store import BarStore

# Chunk length per request: roughly 6-9k bars each, well under MT5's per-call limits.
CHUNK: dict[str, timedelta] = {
    "M1": timedelta(days=7),
    "M5": timedelta(days=30),
    "M15": timedelta(days=90),
    "H1": timedelta(days=365),
    "H4": timedelta(days=365),
    "D1": timedelta(days=365),
}


class BarSource(Protocol):
    """Anything with MT5Client.get_bars' shape (lets tests use a fake)."""

    def get_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class FetchResult:
    symbol: str
    timeframe: str
    wanted_start: datetime
    new_bars: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None


def history_start(timeframe: str, settings: Settings, now: datetime) -> datetime:
    """How far back we want data: m1_history_months for M1, history_years otherwise."""
    if timeframe == "M1":
        offset = pd.DateOffset(months=settings.m1_history_months)
    else:
        offset = pd.DateOffset(years=settings.history_years)
    return (pd.Timestamp(now) - offset).to_pydatetime()


def missing_ranges(
    stored: tuple[pd.Timestamp, pd.Timestamp] | None,
    start: datetime,
    end: datetime,
    bar_length: timedelta,
) -> list[tuple[datetime, datetime, bool]]:
    """(from, to, newest_first) ranges to download so disk covers [start, end]."""
    if stored is None:
        return [(start, end, False)]
    first, last = stored[0].to_pydatetime(), stored[1].to_pydatetime()
    ranges = []
    if start < first:
        ranges.append((start, first - timedelta(seconds=1), True))
    if last + bar_length <= end:
        ranges.append((last + bar_length, end, False))
    return ranges


def chunks(
    start: datetime, end: datetime, size: timedelta, newest_first: bool = False
) -> Iterator[tuple[datetime, datetime]]:
    """Split [start, end] into back-to-back pieces of at most `size`."""
    pieces = []
    cursor = start
    while cursor <= end:
        piece_end = min(cursor + size - timedelta(seconds=1), end)
        pieces.append((cursor, piece_end))
        cursor = piece_end + timedelta(seconds=1)
    yield from reversed(pieces) if newest_first else pieces


def fetch_history(
    source: BarSource,
    store: BarStore,
    symbol: str,
    broker_symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> FetchResult:
    """Download whatever part of [start, end] is not on disk yet, chunk by chunk."""
    bar_length = TIMEFRAMES[timeframe][1]
    new_bars = 0
    for range_start, range_end, newest_first in missing_ranges(
        store.time_range(symbol, timeframe), start, end, bar_length
    ):
        empty_in_a_row = 0
        for chunk_start, chunk_end in chunks(
            range_start, range_end, CHUNK[timeframe], newest_first
        ):
            bars = source.get_bars(broker_symbol, timeframe, chunk_start, chunk_end)
            new_bars += store.write(symbol, timeframe, bars)
            # Going back in time, several empty chunks in a row means the broker has
            # no older history (weekends/holidays are shorter than one chunk).
            empty_in_a_row = empty_in_a_row + 1 if bars.empty else 0
            if newest_first and empty_in_a_row >= 3:
                logger.info(
                    "{} {}: no data before {}, stopping", symbol, timeframe, chunk_end.date()
                )
                break

    stored = store.time_range(symbol, timeframe)
    result = FetchResult(
        symbol=symbol,
        timeframe=timeframe,
        wanted_start=start,
        new_bars=new_bars,
        first=stored[0] if stored else None,
        last=stored[1] if stored else None,
    )
    logger.debug(
        "{} {}: +{} bars, stored {} to {}", symbol, timeframe, new_bars, result.first, result.last
    )
    return result
