from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pandas as pd

from tradeagent.config import load_config
from tradeagent.data.historical import chunks, fetch_history, history_start, missing_ranges
from tradeagent.data.store import BarStore

T0 = datetime(2026, 1, 1, tzinfo=UTC)


class FakeSource:
    """Pretend broker: hourly bars from `available_from` to `available_to`."""

    def __init__(self, available_from: datetime, available_to: datetime) -> None:
        times = pd.date_range(available_from, available_to, freq="1h", tz="UTC")
        self.bars = pd.DataFrame(
            {
                "time_utc": times.astype("datetime64[ns, UTC]"),
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "tick_volume": 1,
                "spread": 3,
            }
        )
        self.calls: list[tuple[datetime, datetime]] = []

    def get_bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.calls.append((start, end))
        t = self.bars["time_utc"]
        return self.bars[(t >= start) & (t <= end)].reset_index(drop=True)


def test_chunks_cover_range_without_overlap() -> None:
    pieces = list(chunks(T0, T0 + timedelta(days=10), timedelta(days=3)))
    assert pieces[0][0] == T0 and pieces[-1][1] == T0 + timedelta(days=10)
    assert len(pieces) == 4
    for (_, a_end), (b_start, _) in pairwise(pieces):
        assert b_start == a_end + timedelta(seconds=1)
    assert (
        list(chunks(T0, T0 + timedelta(days=10), timedelta(days=3), newest_first=True))
        == (pieces[::-1])
    )


def test_missing_ranges() -> None:
    hour = timedelta(hours=1)
    end = T0 + timedelta(days=10)
    assert missing_ranges(None, T0, end, hour) == [(T0, end, False)]
    stored = (pd.Timestamp(T0 + timedelta(days=2)), pd.Timestamp(T0 + timedelta(days=5)))
    head, tail = missing_ranges(stored, T0, end, hour)
    assert head == (T0, T0 + timedelta(days=2) - timedelta(seconds=1), True)
    assert tail == (T0 + timedelta(days=5) + hour, end, False)
    fully_stored = (pd.Timestamp(T0), pd.Timestamp(end))
    assert missing_ranges(fully_stored, T0, end, hour) == []


def test_first_fetch_then_incremental_tail(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    source = FakeSource(T0, T0 + timedelta(days=800))
    end = T0 + timedelta(days=400)

    first = fetch_history(source, store, "XAUUSD", "XAUUSDm", "H1", T0, end)
    assert first.new_bars == 400 * 24 + 1
    assert len(source.calls) == 2  # 400 days in 365-day chunks

    source.calls.clear()
    later = end + timedelta(days=2)
    second = fetch_history(source, store, "XAUUSD", "XAUUSDm", "H1", T0, later)
    assert second.new_bars == 48
    assert source.calls[0][0] == end + timedelta(hours=1)  # only asks for what's new
    assert store.read("XAUUSD", "H1")["time_utc"].is_unique


def test_head_is_backfilled_newest_first(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    source = FakeSource(T0 - timedelta(days=100), T0 + timedelta(days=10))
    # Pretend an earlier run only got data from T0 (e.g. MT5 "max bars" was lower).
    store.write("XAUUSD", "H1", source.get_bars("XAUUSDm", "H1", T0, T0 + timedelta(days=10)))
    source.calls.clear()

    result = fetch_history(
        source, store, "XAUUSD", "XAUUSDm", "H1", T0 - timedelta(days=100), T0 + timedelta(days=10)
    )
    assert result.new_bars == 100 * 24
    assert result.first == pd.Timestamp(T0 - timedelta(days=100))


def test_backfill_stops_when_broker_has_no_older_data(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    source = FakeSource(T0, T0 + timedelta(days=30))
    fetch_history(source, store, "XAUUSD", "XAUUSDm", "M1", T0, T0 + timedelta(days=30))
    source.calls.clear()

    # Want a year earlier, but the broker has nothing before T0.
    result = fetch_history(
        source, store, "XAUUSD", "XAUUSDm", "M1", T0 - timedelta(days=365), T0 + timedelta(days=30)
    )
    assert result.new_bars == 0
    assert len(source.calls) == 3  # gives up after 3 empty weekly chunks, not 52


def test_history_start_per_timeframe() -> None:
    settings = load_config().settings
    now = datetime(2026, 9, 29, tzinfo=UTC)
    assert history_start("M5", settings, now) == datetime(2023, 9, 29, tzinfo=UTC)
    assert history_start("M1", settings, now) == datetime(2026, 3, 29, tzinfo=UTC)
