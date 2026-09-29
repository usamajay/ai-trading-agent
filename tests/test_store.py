import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tradeagent.data.store import TABLES, BarStore, connect_db


def make_bars(start: datetime, count: int, step: timedelta, price: float = 2000.0) -> pd.DataFrame:
    times = pd.date_range(start, periods=count, freq=step, tz="UTC").astype("datetime64[ns, UTC]")
    return pd.DataFrame(
        {
            "time_utc": times,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "tick_volume": 10,
            "spread": 20,
        }
    )


def test_write_splits_by_year_and_reads_back(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    bars = make_bars(datetime(2025, 12, 31, 22, tzinfo=UTC), 4, timedelta(hours=1))
    assert store.write("XAUUSD", "H1", bars) == 4

    assert store.years("XAUUSD", "H1") == [2025, 2026]
    assert store.path("XAUUSD", "H1", 2026) == tmp_path / "XAUUSD" / "H1" / "2026.parquet"
    back = store.read("XAUUSD", "H1")
    pd.testing.assert_frame_equal(back, bars)


def test_rewrite_is_idempotent_and_replaces_values(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    start = datetime(2026, 1, 5, tzinfo=UTC)
    store.write("XAUUSD", "M5", make_bars(start, 10, timedelta(minutes=5)))
    assert store.write("XAUUSD", "M5", make_bars(start, 10, timedelta(minutes=5))) == 0
    # overlapping write: 5 old (replaced) + 5 new
    newer = make_bars(start + timedelta(minutes=25), 10, timedelta(minutes=5), price=2100.0)
    assert store.write("XAUUSD", "M5", newer) == 5

    back = store.read("XAUUSD", "M5")
    assert len(back) == 15
    assert back["time_utc"].is_monotonic_increasing and back["time_utc"].is_unique
    assert back["close"].iloc[5] == 2100.0  # overlapping bar replaced by newer download


def test_read_filters_by_time(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    store.write("USOIL", "D1", make_bars(datetime(2024, 12, 1, tzinfo=UTC), 90, timedelta(days=1)))
    part = store.read(
        "USOIL", "D1", datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 10, tzinfo=UTC)
    )
    assert len(part) == 10
    assert part["time_utc"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")


def test_empty_store(tmp_path: Path) -> None:
    store = BarStore(tmp_path / "nothing")
    assert store.read("XAUUSD", "M5").empty
    assert store.time_range("XAUUSD", "M5") is None
    assert store.summary().empty
    assert (
        store.write("XAUUSD", "M5", make_bars(datetime(2026, 1, 1, tzinfo=UTC), 0, timedelta(1)))
        == 0
    )


def test_time_range_and_summary(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    store.write(
        "XAUUSD", "H1", make_bars(datetime(2025, 6, 1, tzinfo=UTC), 24 * 300, timedelta(hours=1))
    )
    first, last = store.time_range("XAUUSD", "H1")  # type: ignore[misc]
    assert first == pd.Timestamp("2025-06-01", tz="UTC")
    assert last == first + timedelta(hours=24 * 300 - 1)
    row = store.summary().iloc[0]
    assert (row["symbol"], row["timeframe"], row["bars"]) == ("XAUUSD", "H1", 7200)


def test_write_rejects_missing_columns(tmp_path: Path) -> None:
    bars = make_bars(datetime(2026, 1, 1, tzinfo=UTC), 3, timedelta(hours=1)).drop(columns="spread")
    with pytest.raises(ValueError, match="spread"):
        BarStore(tmp_path).write("XAUUSD", "H1", bars)


def test_no_temp_files_left_behind(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    store.write("XAUUSD", "H1", make_bars(datetime(2026, 1, 1, tzinfo=UTC), 3, timedelta(hours=1)))
    assert not list(tmp_path.rglob("*.tmp"))


def test_connect_db_creates_all_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "sub" / "tradeagent.db"
    connect_db(db_path).close()
    conn = connect_db(db_path)  # second call must not fail
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(TABLES) <= names
    for table in TABLES:
        columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if table != "approvals":  # approvals are human actions, not code results
            assert {"git_commit", "config_hash"} <= columns, table
    with pytest.raises(sqlite3.IntegrityError):  # status must be a known ladder step
        conn.execute("INSERT INTO strategies VALUES ('s1','n','1','{}','h','live_now','c','h','t')")
    conn.close()
