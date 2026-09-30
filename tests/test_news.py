"""News calendar: official-schedule parsers, live feed, blackout status (no network)."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from tradeagent.config import PROJECT_ROOT
from tradeagent.data.news import (
    NewsCalendar,
    NewsEvent,
    bls_release_days,
    build_calendar,
    fomc_statement_days,
    history_frame,
    load_history,
    parse_live_feed,
    week_bounds,
)

NY = "America/New_York"
HISTORY = PROJECT_ROOT / "config" / "news" / "usd_high_impact_history.csv"

FOMC_HTML = """
<h4><a id="1">2025 FOMC Meetings</a></h4>
<div class="fomc-meeting__month col"><strong>January</strong></div>
<div class="fomc-meeting__date col">28-29</div>
<div class="fomc-meeting__month col"><strong>April/May</strong></div>
<div class="fomc-meeting__date col">30-1*</div>
<div class="fomc-meeting__month col"><strong>August</strong></div>
<div class="fomc-meeting__date col">22 (notation vote)</div>
<div class="fomc-meeting__month col"><strong>Octember</strong></div>
<div class="fomc-meeting__date col">1-2</div>
<h4><a id="2">2024 FOMC Meetings</a></h4>
<div class="fomc-meeting__month col"><strong>March</strong></div>
<div class="fomc-meeting__date col">16 (unscheduled)</div>
"""


def test_fomc_parser() -> None:
    assert fomc_statement_days(FOMC_HTML) == [
        (date(2024, 3, 16), "FOMC unscheduled meeting (time approximate)"),
        (date(2025, 1, 29), "FOMC statement"),
        (date(2025, 5, 1), "FOMC statement"),  # a meeting spanning two months
        (date(2025, 8, 22), "FOMC notation vote (time approximate)"),
    ]  # "Octember" is not a month: skipped


def test_bls_parser() -> None:
    md = (
        "[Aug](https://www.bls.gov/news.release/archives/empsit_09042026.htm) "
        "([PDF](https://www.bls.gov/news.release/archives/empsit_09042026.pdf)) "
        "[Jul](https://www.bls.gov/news.release/archives/empsit_08072026.htm) "
        "[CPI](https://www.bls.gov/news.release/archives/cpi_09112026.htm)"
    )
    assert bls_release_days(md, "empsit") == [date(2026, 8, 7), date(2026, 9, 4)]
    assert bls_release_days(md, "cpi") == [date(2026, 9, 11)]


def test_history_frame_times_in_utc_across_daylight_saving() -> None:
    df = history_frame(
        [(date(2026, 1, 28), "FOMC statement")],
        [date(2026, 7, 2)],
        [date(2023, 1, 12)],  # outside the range: dropped
        date(2023, 9, 1),
        date(2026, 9, 27),
    )
    assert df["time_utc"].tolist() == [
        pd.Timestamp("2026-01-28 19:00", tz="UTC"),  # 14:00 New York, winter
        pd.Timestamp("2026-07-02 12:30", tz="UTC"),  # 08:30 New York, summer
    ]


def test_live_feed_keeps_high_impact_usd_only() -> None:
    raw = json.dumps(
        [
            {
                "title": "Non-Farm Employment Change",
                "country": "USD",
                "date": "2026-10-02T08:30:00-04:00",
                "impact": "High",
            },
            {
                "title": "Retail Sales",
                "country": "USD",
                "date": "2026-10-01T08:30:00-04:00",
                "impact": "Medium",
            },
            {
                "title": "CPI",
                "country": "EUR",
                "date": "2026-10-01T05:00:00-04:00",
                "impact": "High",
            },
        ]
    )
    [event] = parse_live_feed(raw)
    assert event.time_utc == pd.Timestamp("2026-10-02 12:30", tz="UTC")
    assert event.source == "forexfactory"


def test_week_bounds_sunday_to_sunday_new_york() -> None:
    start, end = week_bounds(datetime(2026, 10, 1, 12, tzinfo=UTC))  # a Thursday
    assert start == pd.Timestamp("2026-09-27 00:00", tz=NY)
    assert end == pd.Timestamp("2026-10-04 00:00", tz=NY)


def test_status_blackout_clear_unknown() -> None:
    nfp = NewsEvent(pd.Timestamp("2026-01-09 13:30", tz="UTC"), "NFP", "test")
    cal = NewsCalendar(
        (nfp,),
        ((pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-02-01", tz="UTC")),),
        30,
    )
    assert cal.status(datetime(2026, 1, 9, 13, 0, tzinfo=UTC))[0] == "blackout"  # 30 min before
    assert cal.status(datetime(2026, 1, 9, 14, 0, tzinfo=UTC))[0] == "blackout"  # 30 min after
    assert "NFP" in cal.status(datetime(2026, 1, 9, 13, 45, tzinfo=UTC))[1]
    assert cal.status(datetime(2026, 1, 9, 14, 1, tzinfo=UTC))[0] == "clear"
    assert cal.status(datetime(2026, 3, 1, tzinfo=UTC))[0] == "unknown"  # outside coverage
    assert cal.upcoming(datetime(2026, 1, 8, tzinfo=UTC)) == [nfp]
    assert cal.upcoming(datetime(2026, 1, 20, tzinfo=UTC)) == []


def test_committed_history_file() -> None:
    events, covered = load_history(HISTORY)
    assert covered == (pd.Timestamp("2023-09-01", tz="UTC"), pd.Timestamp("2026-09-27", tz="UTC"))
    titles = pd.Series([e.title for e in events])
    years = pd.Series([e.time_utc.year for e in events])
    full_2024 = titles[years == 2024].value_counts().to_dict()
    assert full_2024 == {"CPI": 12, "Employment Situation (NFP)": 12, "FOMC statement": 8}
    assert all(e.time_utc.tzinfo is not None for e in events)


def test_build_calendar_with_and_without_live_feed(tmp_path: Path) -> None:
    now = datetime(2026, 10, 1, 12, tzinfo=UTC)  # after the history ends
    history_only = build_calendar(HISTORY, None, now, 30)
    assert history_only.status(now)[0] == "unknown"  # live feed missing: not safe to trade
    assert history_only.status(datetime(2024, 6, 1, 12, tzinfo=UTC))[0] == "clear"
    raw = json.dumps(
        [{"title": "NFP", "country": "USD", "date": "2026-10-02T08:30:00-04:00", "impact": "High"}]
    )
    live = build_calendar(HISTORY, raw, now, 30)
    assert live.status(now)[0] == "clear"
    assert live.status(datetime(2026, 10, 2, 12, 30, tzinfo=UTC))[0] == "blackout"
    empty = build_calendar(tmp_path / "missing.csv", None, now, 30)
    assert empty.events == () and empty.status(now)[0] == "unknown"


def test_history_without_coverage_line(tmp_path: Path) -> None:
    path = tmp_path / "h.csv"
    path.write_text("time_utc,title,source\n2026-01-09T13:30:00Z,NFP,test\n", encoding="utf-8")
    events, covered = load_history(path)
    assert len(events) == 1 and covered is None


def test_live_feed_cache(tmp_path: Path) -> None:
    from tradeagent.data.news import cache_path, cached_live_feed, save_live_feed

    now = datetime(2026, 10, 1, 12, tzinfo=UTC)
    assert cached_live_feed(tmp_path, now) is None
    raw = json.dumps(
        [{"title": "NFP", "country": "USD", "date": "2026-10-02T08:30:00-04:00", "impact": "High"}]
    )
    path = save_live_feed(tmp_path / "news", now, raw)
    assert path == cache_path(tmp_path / "news", now) and path.name == "live_2026-09-27.json"
    assert cached_live_feed(tmp_path / "news", now) == raw
    with pytest.raises(ValueError):  # a page that is not the feed is never cached
        save_live_feed(tmp_path / "news", now, "<html>not the feed</html>")
