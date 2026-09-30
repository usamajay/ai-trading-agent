"""High-impact USD news calendar for the risk engine's blackout rule (SPEC §6).

- Live: this week's high-impact USD events from a free weekly feed (ForexFactory's
  public JSON export), cached in data/news/. `NewsCalendar.status` returns
  "unknown" outside the covered weeks, and the risk engine blocks entries then
  (strict mode) rather than trading blind.
- History (backtests): release dates taken from official schedules and committed as
  config/news/usd_high_impact_history.csv: FOMC statements (Federal Reserve
  calendar, 14:00 New York), the jobs report and CPI (BLS release archives,
  08:30 New York). Other high-impact releases (GDP, PCE, ISM, retail sales) are
  not in the history: a known limitation (docs/DECISIONS.md).
All times are stored in UTC.
"""

import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

from tradeagent.data.market_hours import NEW_YORK

LIVE_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
HISTORY_COLUMNS = ["time_utc", "title", "source"]
MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


@dataclass(frozen=True)
class NewsEvent:
    time_utc: pd.Timestamp
    title: str
    source: str


def _ny(day: date, hour: int, minute: int) -> pd.Timestamp:
    local = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hour, minute=minute)
    return local.tz_localize(NEW_YORK).tz_convert("UTC")


# --- parsers for official schedules (pure functions, tested offline) ------------------


def fomc_statement_days(html: str) -> list[tuple[date, str]]:
    """(last day, title) of each FOMC decision on the Federal Reserve calendar page.

    Regular meetings are "FOMC statement" (14:00 New York). Notation votes and
    unscheduled meetings are kept (an extra blackout is the safe side) but titled as
    such, because their announcement time is not always 14:00.
    """
    days: list[tuple[date, str]] = []
    panels = re.split(r"(\d{4}) FOMC Meetings", html)
    for year_text, body in zip(panels[1::2], panels[2::2], strict=False):
        year = int(year_text)
        pairs = re.findall(
            r"fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?"
            r"fomc-meeting__date[^>]*>\s*([^<]+?)\s*<",
            body,
            flags=re.DOTALL,
        )
        for month_text, date_text in pairs:
            last_month = month_text.split("/")[-1].strip()
            numbers = re.findall(r"\d+", date_text)
            if last_month not in MONTHS or not numbers:
                continue
            title = "FOMC statement"
            if "notation" in date_text.lower():
                title = "FOMC notation vote (time approximate)"
            elif "unscheduled" in date_text.lower():
                title = "FOMC unscheduled meeting (time approximate)"
            days.append((date(year, MONTHS[last_month], int(numbers[-1])), title))
    return sorted(set(days))


def bls_release_days(markdown: str, prefix: str) -> list[date]:
    """Release dates from BLS archive links such as `empsit_09042026` (MMDDYYYY)."""
    found = re.findall(rf"{prefix}_(\d{{2}})(\d{{2}})(\d{{4}})\.htm", markdown)
    return sorted({date(int(y), int(m), int(d)) for m, d, y in found})


def history_frame(
    fomc: list[tuple[date, str]], jobs: list[date], cpi: list[date], start: date, end: date
) -> pd.DataFrame:
    """The committed history table (UTC times) for days in [start, end]."""
    rows = [(_ny(d, 14, 0), title, "federalreserve.gov") for d, title in fomc]
    rows += [(_ny(d, 8, 30), "Employment Situation (NFP)", "bls.gov") for d in jobs]
    rows += [(_ny(d, 8, 30), "CPI", "bls.gov") for d in cpi]
    df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    days = df["time_utc"].dt.tz_convert(NEW_YORK).dt.date
    df = df[(days >= start) & (days <= end)]
    return df.sort_values("time_utc").reset_index(drop=True)


# --- live feed ------------------------------------------------------------------------


def parse_live_feed(raw: str) -> list[NewsEvent]:
    """High-impact USD events from the weekly JSON feed."""
    events = []
    for item in json.loads(raw):
        if item.get("country") != "USD" or item.get("impact") != "High":
            continue
        when = pd.Timestamp(item["date"]).tz_convert("UTC")
        events.append(NewsEvent(when, str(item.get("title", "")), "forexfactory"))
    return events


def fetch_live_feed(url: str = LIVE_FEED_URL, timeout: float = 20.0) -> str:
    """Download this week's feed (raises on network errors; callers cache and log)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 tradeagent"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return str(response.read().decode("utf-8"))


def week_bounds(now: datetime) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[Sunday 00:00, next Sunday 00:00) New York around `now`: the feed's week."""
    ny = pd.Timestamp(now).tz_convert(NEW_YORK)
    sunday = (ny - pd.Timedelta(days=(ny.dayofweek + 1) % 7)).normalize()
    return sunday.tz_convert("UTC"), (sunday + pd.Timedelta(days=7)).tz_convert("UTC")


# --- the calendar the risk engine asks ------------------------------------------------


@dataclass(frozen=True)
class NewsCalendar:
    events: tuple[NewsEvent, ...]
    covered: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]  # [start, end) ranges known
    blackout_minutes: int

    def status(self, now: datetime) -> tuple[Literal["clear", "blackout", "unknown"], str]:
        t = pd.Timestamp(now).tz_convert("UTC")
        window = pd.Timedelta(minutes=self.blackout_minutes)
        for event in self.events:
            if abs(event.time_utc - t) <= window:
                ny = event.time_utc.tz_convert(NEW_YORK)
                return (
                    "blackout",
                    f"{event.title} at {ny:%Y-%m-%d %H:%M} New York (+/-{self.blackout_minutes} min)",
                )
        if not any(start <= t < end for start, end in self.covered):
            return "unknown", f"no calendar for {t:%Y-%m-%d %H:%M} UTC"
        return "clear", "no high-impact USD event nearby"

    def upcoming(self, now: datetime, days: int = 7) -> list[NewsEvent]:
        t = pd.Timestamp(now).tz_convert("UTC")
        return [
            e
            for e in self.events
            if t - timedelta(hours=1) <= e.time_utc <= t + timedelta(days=days)
        ]


def load_history(path: Path) -> tuple[list[NewsEvent], tuple[pd.Timestamp, pd.Timestamp] | None]:
    """Events and the covered range written in the CSV header comment (`# covered: a b`)."""
    if not path.is_file():
        return [], None
    text = path.read_text(encoding="utf-8")
    match = re.search(r"# covered: (\S+) (\S+)", text)
    df = pd.read_csv(path, comment="#")
    events = [
        NewsEvent(pd.Timestamp(t).tz_convert("UTC"), str(title), str(src))
        for t, title, src in zip(df["time_utc"], df["title"], df["source"], strict=True)
    ]
    covered = None
    if match:
        covered = (pd.Timestamp(match.group(1), tz="UTC"), pd.Timestamp(match.group(2), tz="UTC"))
    return events, covered


def build_calendar(
    history_path: Path, live_raw: str | None, now: datetime, blackout_minutes: int
) -> NewsCalendar:
    """History (backtests and live) plus this week's live feed when available."""
    events, covered_history = load_history(history_path)
    covered = [covered_history] if covered_history else []
    if live_raw is not None:
        events += parse_live_feed(live_raw)
        covered.append(week_bounds(now))
    unique = {(e.time_utc, e.title): e for e in events}
    ordered = tuple(sorted(unique.values(), key=lambda e: e.time_utc))
    return NewsCalendar(ordered, tuple(covered), blackout_minutes)


# --- cache of the live feed (data/news/) -----------------------------------------------


def cache_path(cache_dir: Path, now: datetime) -> Path:
    start, _ = week_bounds(now)
    return cache_dir / f"live_{start.tz_convert(NEW_YORK):%Y-%m-%d}.json"


def cached_live_feed(cache_dir: Path, now: datetime) -> str | None:
    """This week's cached feed, if it was fetched."""
    path = cache_path(cache_dir, now)
    return path.read_text(encoding="utf-8") if path.is_file() else None


def save_live_feed(cache_dir: Path, now: datetime, raw: str) -> Path:
    parse_live_feed(raw)  # refuse to cache something that is not the expected feed
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, now)
    path.write_text(raw, encoding="utf-8")
    return path
