from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from loguru import logger

from tradeagent.logs import UtcDailyRotation, add_daily_log

PKT = timezone(timedelta(hours=5))


def message_at(ts: datetime) -> SimpleNamespace:
    return SimpleNamespace(record={"time": ts})


def test_rotates_at_utc_midnight_not_local_midnight() -> None:
    rotate = UtcDailyRotation()
    assert not rotate(message_at(datetime(2026, 9, 29, 23, 46, tzinfo=PKT)), None)  # 18:46 UTC
    # 00:00 PKT = 19:00 UTC, same UTC day: keep the same file
    assert not rotate(message_at(datetime(2026, 9, 30, 0, 1, tzinfo=PKT)), None)
    # 05:00 PKT = 00:00 UTC next day: new file
    assert rotate(message_at(datetime(2026, 9, 30, 5, 0, 30, tzinfo=PKT)), None)
    assert not rotate(message_at(datetime(2026, 9, 30, 6, 0, tzinfo=PKT)), None)


def test_log_file_named_by_utc_date_with_utc_times(tmp_path: Path) -> None:
    handler = add_daily_log(tmp_path, "watch")
    try:
        logger.info("hello")
    finally:
        logger.remove(handler)
    [log_file] = tmp_path.iterdir()
    assert log_file.name == f"watch_{datetime.now(UTC):%Y-%m-%d}.log"
    line = log_file.read_text(encoding="utf-8").strip()
    assert " UTC | INFO    | hello" in line
