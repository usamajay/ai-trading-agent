"""Live bar updater (Phase 1, task 1.6): append each newly closed bar, validate, log.

`LiveUpdater.poll_once()` does one round for every symbol/timeframe; `run_watch()`
repeats it every few seconds, reconnects after MT5 errors, and stops on Ctrl+C.
Validation runs on the new bars plus recent history kept in memory, so gap and
spike rules have context, and only issues on the new bars are logged.
"""

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from tradeagent.data.historical import BarSource
from tradeagent.data.mt5_client import TIMEFRAMES, AccountRefusedError, MT5Client, MT5Error
from tradeagent.data.store import BarStore
from tradeagent.data.validation import log_issues, validate_bars
from tradeagent.timeutil import utc_now

CONTEXT_BARS = 2500  # recent bars kept in memory (spike rule uses a 2000-bar median)
FIRST_RUN_LOOKBACK = timedelta(days=3)  # if nothing is stored yet


@dataclass
class Series:
    symbol: str  # internal name, e.g. XAUUSD
    broker_symbol: str  # e.g. XAUUSDm
    timeframe: str
    recent: pd.DataFrame  # last CONTEXT_BARS stored bars


@dataclass(frozen=True)
class NewBars:
    symbol: str
    timeframe: str
    bars: pd.DataFrame
    issues: pd.DataFrame


@dataclass
class WatchStats:
    polls: int = 0
    new_bars: int = 0
    issues: int = 0
    reconnects: int = 0
    errors: list[str] = field(default_factory=list)


class LiveUpdater:
    """Appends newly closed bars for each symbol/timeframe to the store."""

    def __init__(
        self,
        store: BarStore,
        conn: sqlite3.Connection | None,
        symbols: dict[str, str],
        timeframes: list[str],
        run_id: str,
        git_commit: str,
        config_hash: str,
        settle_seconds: float = 2.0,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.store = store
        self.conn = conn
        self.run_id = run_id
        self.git_commit = git_commit
        self.config_hash = config_hash
        self.settle = timedelta(seconds=settle_seconds)
        self.clock = clock
        self.series = [
            Series(name, broker, tf, self._load_recent(name, tf))
            for name, broker in symbols.items()
            for tf in timeframes
        ]

    def _load_recent(self, symbol: str, timeframe: str) -> pd.DataFrame:
        stored = self.store.time_range(symbol, timeframe)
        if stored is None:
            return self.store.read(symbol, timeframe)  # empty
        bar_length = TIMEFRAMES[timeframe][1]
        # Weekends/breaks mean CONTEXT_BARS bars span more than CONTEXT_BARS * length.
        start = stored[1].to_pydatetime() - 2 * CONTEXT_BARS * bar_length
        return self.store.read(symbol, timeframe, start=start).tail(CONTEXT_BARS)

    def poll_once(self, source: BarSource) -> list[NewBars]:
        """Fetch, store, validate and log bars that closed since the last poll."""
        now = self.clock()
        results = []
        for s in self.series:
            bar_length = TIMEFRAMES[s.timeframe][1]
            if s.recent.empty:
                start = now - FIRST_RUN_LOOKBACK
            else:
                start = s.recent["time_utc"].iloc[-1].to_pydatetime() + bar_length
            if start + bar_length > now - self.settle:
                continue  # the next bar has not closed (and settled) yet

            fetched = source.get_bars(s.broker_symbol, s.timeframe, start, now)
            # Count a bar as closed only after a short settle delay, so ticks that
            # arrive just after the minute turns are not lost.
            fresh = fetched[fetched["time_utc"] + bar_length <= now - self.settle]
            if fresh.empty:
                continue

            self.store.write(s.symbol, s.timeframe, fresh)
            s.recent = pd.concat([s.recent, fresh], ignore_index=True).tail(CONTEXT_BARS)
            issues = self._validate_new(s, fresh)
            results.append(NewBars(s.symbol, s.timeframe, fresh.reset_index(drop=True), issues))
            self._log(s, fresh, issues)
        return results

    def _validate_new(self, s: Series, fresh: pd.DataFrame) -> pd.DataFrame:
        result = validate_bars(s.recent.reset_index(drop=True), s.symbol, s.timeframe)
        first_new = fresh["time_utc"].iloc[0]
        new_issues = result.issues[result.issues["time_utc"] >= first_new]
        if self.conn is not None and not new_issues.empty:
            log_issues(
                self.conn,
                replace(result, issues=new_issues),
                self.run_id,
                self.git_commit,
                self.config_hash,
            )
        return new_issues.reset_index(drop=True)

    @staticmethod
    def _log(s: Series, fresh: pd.DataFrame, issues: pd.DataFrame) -> None:
        for row in fresh.itertuples():
            logger.info(
                "{} {} {:%Y-%m-%d %H:%M} UTC  O {} H {} L {} C {}  spread {}",
                s.symbol,
                s.timeframe,
                row.time_utc,
                row.open,
                row.high,
                row.low,
                row.close,
                row.spread,
            )
        for issue_type, details in zip(issues["issue_type"], issues["details"], strict=True):
            logger.warning("{} {} data issue: {} ({})", s.symbol, s.timeframe, issue_type, details)

    def heartbeat(self) -> str:
        parts = [
            f"{s.symbol} {s.timeframe} last {s.recent['time_utc'].iloc[-1]:%H:%M}"
            for s in self.series
            if not s.recent.empty
        ]
        return "alive; " + ", ".join(parts)


def run_watch(
    connect: Callable[[], MT5Client],
    updater: LiveUpdater,
    interval_seconds: float = 10.0,
    stop_after: timedelta | None = None,
    heartbeat_every: timedelta = timedelta(minutes=5),
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> WatchStats:
    """Poll until Ctrl+C (or `stop_after`). Reconnects after MT5 errors, with backoff.

    A refused account (not DEMO) is never retried: the error is raised.
    """
    stats = WatchStats()
    client: MT5Client | None = None
    started = last_heartbeat = monotonic()
    backoff = interval_seconds
    try:
        while stop_after is None or monotonic() - started < stop_after.total_seconds():
            try:
                if client is None:
                    client = connect()
                    client.connect()
                    if stats.polls:
                        stats.reconnects += 1
                        logger.info("Reconnected to MT5")
                for new in updater.poll_once(client):
                    stats.new_bars += len(new.bars)
                    stats.issues += len(new.issues)
                stats.polls += 1
                backoff = interval_seconds
            except AccountRefusedError:
                raise
            except MT5Error as exc:
                stats.errors.append(str(exc))
                logger.warning("MT5 problem: {}. Reconnecting in {:.0f}s", exc, backoff)
                if client is not None:
                    client.close()
                client = None
                sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue

            if monotonic() - last_heartbeat >= heartbeat_every.total_seconds():
                logger.info(updater.heartbeat())
                last_heartbeat = monotonic()
            sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Ctrl+C received, stopping")
    finally:
        if client is not None:
            client.close()
    return stats
