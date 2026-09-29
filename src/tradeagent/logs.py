"""Log files: one file per UTC day, UTC timestamps (CLAUDE.md: store times in UTC)."""

from datetime import UTC, date
from pathlib import Path
from typing import Any

from loguru import logger


class UtcDailyRotation:
    """Loguru rotation rule: start a new file when the UTC date changes.

    Loguru's built-in "00:00" rotation uses the PC's local midnight (00:00 PKT =
    19:00 UTC), which split a UTC-dated file in the middle of the day.
    """

    def __init__(self) -> None:
        self.day: date | None = None

    def __call__(self, message: Any, file: Any) -> bool:
        day = message.record["time"].astimezone(UTC).date()
        if self.day is None:
            self.day = day
            return False
        if day != self.day:
            self.day = day
            return True
        return False


def add_daily_log(log_dir: Path, name: str) -> int:
    """Also write logs to `log_dir/<name>_YYYY-MM-DD.log` (UTC date), kept 30 days."""
    return logger.add(
        log_dir / f"{name}_{{time:YYYY-MM-DD!UTC}}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS!UTC} UTC | {level: <7} | {message}",
        rotation=UtcDailyRotation(),
        retention="30 days",
    )
