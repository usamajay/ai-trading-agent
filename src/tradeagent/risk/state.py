"""Running account state for the account-level risk rules (SPEC §6).

- Daily loss: equity at least `max_daily_loss_pct` below the equity at the start of
  the UTC day -> no new trades until 00:00 UTC.
- Weekly loss: at least `max_weekly_loss_pct` below the start of the trading week
  (weeks start with the Sunday evening open) -> no new trades until next week.
- Drawdown: at least `max_drawdown_pct` below the equity peak -> full shutdown until
  a human restarts it (`restart`, which also resets the peak to current equity).
- Consecutive losses: `max_consecutive_losses` losing trades in a row -> pause for
  `consecutive_loss_pause_hours`.
Once triggered, a stop holds for its period even if equity recovers.
"""

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from tradeagent.config import RiskLimits
from tradeagent.data.market_hours import trading_day

STATE_KEY = "account"


def utc_day(now: datetime) -> date:
    return pd.Timestamp(now).tz_convert("UTC").date()


def trading_week(now: datetime) -> str:
    """ISO week of the trading day (17:00 New York days), e.g. '2026-W40'."""
    day = trading_day(pd.Series([pd.Timestamp(now).tz_convert("UTC")])).iloc[0]
    iso = day.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


@dataclass
class OpenPosition:
    symbol: str
    direction: str
    lots: float
    stop_loss: float


@dataclass
class RiskState:
    equity: float
    peak_equity: float
    day: date
    day_start_equity: float
    week: str
    week_start_equity: float
    consecutive_losses: int = 0
    paused_until: datetime | None = None
    daily_stop_day: date | None = None
    weekly_stop_week: str | None = None
    shutdown: bool = False
    shutdown_reason: str | None = None
    positions: list[OpenPosition] = field(default_factory=list)

    @classmethod
    def start(cls, equity: float, now: datetime) -> "RiskState":
        return cls(equity, equity, utc_day(now), equity, trading_week(now), equity)

    # --- updates ------------------------------------------------------------------

    def mark(self, equity: float, now: datetime, limits: RiskLimits) -> list[str]:
        """New equity (closed P&L + open trades valued now). Returns triggered rules."""
        today, week = utc_day(now), trading_week(now)
        if today != self.day:  # the day starts from the last equity seen before it
            self.day, self.day_start_equity = today, self.equity
        if week != self.week:
            self.week, self.week_start_equity = week, self.equity
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        triggered = []
        if self.daily_stop_day != today and _fall(self.day_start_equity, equity) >= (
            limits.max_daily_loss_pct
        ):
            self.daily_stop_day = today
            triggered.append("daily_loss")
        if self.weekly_stop_week != week and _fall(self.week_start_equity, equity) >= (
            limits.max_weekly_loss_pct
        ):
            self.weekly_stop_week = week
            triggered.append("weekly_loss")
        if not self.shutdown and self.drawdown_pct() >= limits.max_drawdown_pct:
            self.shutdown = True
            self.shutdown_reason = (
                f"drawdown {self.drawdown_pct():.2f}% from peak {self.peak_equity:,.2f} "
                f"on {now.isoformat()}"
            )
            triggered.append("max_drawdown")
        return triggered

    def record_close(self, net_pnl: float, now: datetime, limits: RiskLimits) -> list[str]:
        """A trade closed with `net_pnl`. Returns triggered rules (loss streak pause)."""
        if net_pnl >= 0:
            self.consecutive_losses = 0
            return []
        self.consecutive_losses += 1
        if self.consecutive_losses >= limits.max_consecutive_losses:
            self.paused_until = now + timedelta(hours=limits.consecutive_loss_pause_hours)
            self.consecutive_losses = 0  # a new streak is needed for the next pause
            return ["loss_streak"]
        return []

    def restart(self, reason: str) -> None:
        """Human restart after a drawdown shutdown: the peak restarts at current equity."""
        if not reason.strip():
            raise ValueError("a restart needs a written reason")
        self.shutdown, self.shutdown_reason = False, None
        self.peak_equity = self.equity

    # --- queries ------------------------------------------------------------------

    def drawdown_pct(self) -> float:
        return _fall(self.peak_equity, self.equity)

    def blocks(self, now: datetime) -> list[tuple[str, str]]:
        """Account-level stops in force at `now`: (code, detail)."""
        out = []
        if self.shutdown:
            out.append(
                ("shutdown", f"trading shut down: {self.shutdown_reason}; needs a manual restart")
            )
        if self.daily_stop_day == utc_day(now):
            out.append(
                ("daily_loss", f"daily loss limit hit on {self.daily_stop_day}; resumes 00:00 UTC")
            )
        if self.weekly_stop_week == trading_week(now):
            out.append(
                (
                    "weekly_loss",
                    f"weekly loss limit hit in {self.weekly_stop_week}; resumes next week",
                )
            )
        if self.paused_until is not None and now < self.paused_until:
            out.append(
                (
                    "loss_streak",
                    f"paused after consecutive losses until {self.paused_until.isoformat()}",
                )
            )
        return out

    # --- persistence --------------------------------------------------------------

    def to_json(self) -> str:
        data = asdict(self)
        for key in ("day", "daily_stop_day", "paused_until"):
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RiskState":
        data: dict[str, Any] = json.loads(text)
        data["day"] = date.fromisoformat(data["day"])
        if data["daily_stop_day"]:
            data["daily_stop_day"] = date.fromisoformat(data["daily_stop_day"])
        if data["paused_until"]:
            data["paused_until"] = datetime.fromisoformat(data["paused_until"])
        data["positions"] = [OpenPosition(**p) for p in data["positions"]]
        return cls(**data)


def _fall(reference: float, value: float) -> float:
    """% by which `value` is below `reference` (0 if above)."""
    if reference <= 0:
        return 0.0
    return max(0.0, (reference - value) / reference * 100)


def save_state(
    conn: sqlite3.Connection, state: RiskState, git_commit: str, config_hash: str, now: datetime
) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO risk_state (key, value_json, updated_at, git_commit, "
            "config_hash) VALUES (?, ?, ?, ?, ?)",
            (STATE_KEY, state.to_json(), now.isoformat(), git_commit, config_hash),
        )


def load_state(conn: sqlite3.Connection) -> RiskState | None:
    row = conn.execute("SELECT value_json FROM risk_state WHERE key = ?", (STATE_KEY,)).fetchone()
    return RiskState.from_json(row[0]) if row else None


def log_risk_event(
    conn: sqlite3.Connection,
    now: datetime,
    rule: str,
    value: float | None,
    limit: float | None,
    action: str,
    git_commit: str,
    config_hash: str,
) -> None:
    """Record a block, shutdown or kill in `risk_events` (action: block/shutdown/kill)."""
    with conn:
        conn.execute(
            'INSERT INTO risk_events (time, rule, value, "limit", action, git_commit, '
            "config_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now.isoformat(), rule, value, limit, action, git_commit, config_hash),
        )
