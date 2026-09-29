from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tradeagent.data.live import LiveUpdater, run_watch
from tradeagent.data.mt5_client import AccountRefusedError, MT5Error
from tradeagent.data.store import BarStore, connect_db

T0 = datetime(2026, 1, 7, 10, 0, tzinfo=UTC)  # a Wednesday, market open


def minute_bars(start: datetime, end: datetime, spread: int = 20) -> pd.DataFrame:
    times = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    return pd.DataFrame(
        {
            "time_utc": times.astype("datetime64[ns, UTC]"),
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.2,
            "tick_volume": 10,
            "spread": spread,
        }
    )


class FakeFeed:
    """Serves 1-minute bars up to the fake clock, including the still-forming bar."""

    def __init__(self, clock: "Clock") -> None:
        self.clock = clock
        self.calls = 0
        self.spread_override: dict[datetime, int] = {}

    def get_bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.calls += 1
        now = self.clock()
        first = pd.Timestamp(start).ceil("min").to_pydatetime()  # bars start on whole minutes
        bars = minute_bars(first, now.replace(second=0) + timedelta(minutes=1))
        for t, spread in self.spread_override.items():
            bars.loc[bars["time_utc"] == t, "spread"] = spread
        return bars[bars["time_utc"] <= end].reset_index(drop=True)


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def make_updater(tmp_path: Path, clock: Clock, history: pd.DataFrame | None = None) -> LiveUpdater:
    store = BarStore(tmp_path / "bars")
    if history is not None:
        store.write("XAUUSD", "M1", history)
    return LiveUpdater(
        store=store,
        conn=connect_db(tmp_path / "t.db"),
        symbols={"XAUUSD": "XAUUSDm"},
        timeframes=["M1"],
        run_id="run1",
        git_commit="abc",
        config_hash="cfg",
        clock=clock,
    )


def test_appends_only_closed_settled_bars(tmp_path: Path) -> None:
    clock = Clock(T0)
    updater = make_updater(tmp_path, clock, minute_bars(T0 - timedelta(hours=3), T0))
    feed = FakeFeed(clock)

    clock.now = T0 + timedelta(minutes=1, seconds=1)  # 10:00 bar closed 1s ago
    assert updater.poll_once(feed) == []  # not settled yet (2s)

    clock.now = T0 + timedelta(minutes=1, seconds=3)
    [new] = updater.poll_once(feed)
    assert list(new.bars["time_utc"]) == [pd.Timestamp(T0)]
    assert new.issues.empty

    calls = feed.calls
    assert updater.poll_once(feed) == []  # nothing new
    assert feed.calls == calls  # and MT5 was not even asked

    clock.now = T0 + timedelta(minutes=4, seconds=5)
    [new] = updater.poll_once(feed)
    assert len(new.bars) == 3  # 10:01, 10:02, 10:03; 10:04 is still forming
    stored = updater.store.read("XAUUSD", "M1")
    assert stored["time_utc"].iloc[-1] == pd.Timestamp(T0 + timedelta(minutes=3))
    assert stored["time_utc"].is_unique


def test_only_new_issues_are_logged(tmp_path: Path) -> None:
    history = minute_bars(T0 - timedelta(hours=3), T0)
    history.loc[5, "spread"] = 0  # old issue: must not be logged again
    clock = Clock(T0 + timedelta(minutes=2, seconds=5))
    updater = make_updater(tmp_path, clock, history)
    feed = FakeFeed(clock)
    feed.spread_override[T0 + timedelta(minutes=1)] = 0

    [new] = updater.poll_once(feed)
    assert list(new.issues["issue_type"]) == ["spread_zero"]
    rows = updater.conn.execute(  # type: ignore[union-attr]
        "SELECT run_id, issue_type, time_utc, git_commit, config_hash FROM data_quality_log"
    ).fetchall()
    assert rows == [("run1", "spread_zero", "2026-01-07T10:01:00+00:00", "abc", "cfg")]


def test_gap_before_new_bar_is_detected(tmp_path: Path) -> None:
    clock = Clock(T0 + timedelta(minutes=31))
    updater = make_updater(tmp_path, clock, minute_bars(T0 - timedelta(hours=3), T0))

    class HoleFeed(FakeFeed):
        def get_bars(self, *args: Any) -> pd.DataFrame:
            bars = super().get_bars(*args)
            return bars[bars["time_utc"] >= T0 + timedelta(minutes=20)]  # 20 min missing

    [new] = updater.poll_once(HoleFeed(clock))
    assert list(new.issues["issue_type"]) == ["gap_intraday"]


def test_empty_store_starts_from_recent_history(tmp_path: Path) -> None:
    clock = Clock(T0 + timedelta(seconds=5))
    updater = make_updater(tmp_path, clock)
    [new] = updater.poll_once(FakeFeed(clock))
    assert new.bars["time_utc"].iloc[0] == pd.Timestamp(
        T0 - timedelta(days=3) + timedelta(minutes=1)
    )


# --- run_watch loop -------------------------------------------------------------------


class FakeClient:
    def __init__(self, fail_polls: int = 0, refuse: bool = False) -> None:
        self.fail_polls = fail_polls
        self.refuse = refuse
        self.closed = False

    def connect(self) -> None:
        if self.refuse:
            raise AccountRefusedError("Refusing REAL account")

    def close(self) -> None:
        self.closed = True

    def get_bars(self, *args: Any) -> pd.DataFrame:
        if self.fail_polls:
            self.fail_polls -= 1
            raise MT5Error("terminal closed")
        return minute_bars(T0, T0)  # nothing new


class StubUpdater:
    def __init__(self) -> None:
        self.polls = 0

    def poll_once(self, source: Any) -> list[Any]:
        self.polls += 1
        source.get_bars()
        return []

    def heartbeat(self) -> str:
        return "alive"


def test_ctrl_c_stops_cleanly_and_disconnects() -> None:
    client = FakeClient()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise KeyboardInterrupt

    stats = run_watch(lambda: client, StubUpdater(), sleep=sleep)  # type: ignore[arg-type]
    assert stats.polls == 3 and not stats.errors
    assert client.closed


def test_reconnects_with_backoff_after_mt5_error() -> None:
    clients = [FakeClient(fail_polls=1), FakeClient(fail_polls=1), FakeClient()]
    made: list[FakeClient] = []

    def connect() -> FakeClient:
        made.append(clients[min(len(made), 2)])
        return made[-1]

    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 4:
            raise KeyboardInterrupt

    stats = run_watch(connect, StubUpdater(), interval_seconds=10, sleep=sleep)  # type: ignore[arg-type]
    assert stats.errors == ["terminal closed", "terminal closed"]
    assert sleeps[:2] == [10, 20]  # backoff doubles
    assert made[0].closed and made[1].closed  # failed connections closed before retrying
    assert stats.polls >= 1


def test_refused_account_is_never_retried() -> None:
    with pytest.raises(AccountRefusedError):
        run_watch(lambda: FakeClient(refuse=True), StubUpdater(), sleep=lambda s: None)  # type: ignore[arg-type,return-value]


def test_stop_after_duration() -> None:
    ticks = iter(range(0, 10_000, 10))
    stats = run_watch(
        lambda: FakeClient(),  # type: ignore[arg-type,return-value]
        StubUpdater(),  # type: ignore[arg-type]
        stop_after=timedelta(seconds=60),
        sleep=lambda s: None,
        monotonic=lambda: float(next(ticks)),
    )
    assert 1 <= stats.polls <= 6
