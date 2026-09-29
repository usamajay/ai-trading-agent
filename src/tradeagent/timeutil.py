"""Time helpers. Code and storage use UTC; PKT is for display only."""

from datetime import UTC, datetime, timedelta, timezone

PKT = timezone(timedelta(hours=5), "PKT")  # Pakistan has no daylight saving


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_utc(ts: datetime) -> datetime:
    """Convert an aware datetime to UTC; refuse naive ones (their timezone is a guess)."""
    if ts.tzinfo is None:
        raise ValueError(f"datetime {ts} has no timezone; pass a UTC-aware datetime")
    return ts.astimezone(UTC)


def fmt_utc_pkt(ts: datetime) -> str:
    """'2026-09-29 18:00 UTC (23:00 PKT)' for printing."""
    utc = ensure_utc(ts)
    return f"{utc:%Y-%m-%d %H:%M:%S} UTC ({utc.astimezone(PKT):%H:%M} PKT)"
