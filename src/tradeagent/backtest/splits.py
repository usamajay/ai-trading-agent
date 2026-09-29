"""Fixed train / validation / out-of-sample dates (SPEC §7.2).

The dates are calculated once, shown to Usama, and after his approval frozen in
config/splits.yaml. They never move: new bars arriving later are "forward" data
(for paper trading), not part of any split. Between the periods is an unused
embargo gap, so a trade or pattern at the end of one period cannot leak into the
start of the next. Periods start and end on Sunday 00:00 UTC, while the market is
closed, so no trading session is cut in half.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from tradeagent.config import DataSplits, SplitDates, SplitName, SplitPeriod

# Phase 2 never runs on out-of-sample data (CLAUDE.md rule 5). Phase 7's experiment
# manager replaces this with a split guard that logs the single allowed touch.
OOS_LOCKED = True

SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "out_of_sample")


class SplitError(Exception):
    """A split cannot be used (not frozen yet, or locked)."""


def _next_sunday(day: date) -> date:
    return day + timedelta(days=(6 - day.weekday()) % 7)


def propose_splits(
    first_bar: pd.Timestamp,
    last_bar: pd.Timestamp,
    fractions: DataSplits,
    embargo_weeks: int,
) -> dict[SplitName, SplitPeriod]:
    """Whole-week periods covering the complete weeks in [first_bar, last_bar].

    The weeks left after the two embargo gaps are shared out by `fractions`
    (rounded to whole weeks; out-of-sample gets the remainder).
    """
    start = _next_sunday(first_bar.date())
    if start == first_bar.date() and first_bar != pd.Timestamp(start, tz="UTC"):
        start += timedelta(weeks=1)  # the first week is incomplete
    end = last_bar.date() - timedelta(days=(last_bar.date().weekday() + 1) % 7)  # Sunday <=
    weeks = (end - start).days // 7
    usable = weeks - 2 * embargo_weeks
    if usable < 10:
        raise SplitError(f"only {weeks} complete weeks of data; too few to split")
    train_w = round(usable * fractions.train)
    validation_w = round(usable * fractions.validation)
    oos_w = usable - train_w - validation_w

    periods: dict[SplitName, SplitPeriod] = {}
    cursor = start
    for name, length in zip(SPLIT_NAMES, (train_w, validation_w, oos_w), strict=True):
        periods[name] = SplitPeriod(start=cursor, end=cursor + timedelta(weeks=length))
        cursor = periods[name].end + timedelta(weeks=embargo_weeks)
    return periods


def period_times(period: SplitPeriod) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) as UTC timestamps."""
    return pd.Timestamp(period.start, tz="UTC"), pd.Timestamp(period.end, tz="UTC")


def split_range(splits: SplitDates | None, split: SplitName) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The time range a backtest may use. Refuses out-of-sample and unfrozen splits."""
    if split == "out_of_sample" and OOS_LOCKED:
        raise SplitError(
            "out-of-sample data is locked in Phase 2 (touched once per candidate, Phase 7)"
        )
    if splits is None:
        raise SplitError(
            "split dates are not frozen yet: run `tradeagent backtest splits`, get Usama's "
            "approval, then `tradeagent backtest splits --freeze --approved-by NAME`"
        )
    return period_times(splits.period(split))


def label_times(times: pd.Series, periods: dict[SplitName, SplitPeriod]) -> pd.Series:
    """Which split each UTC time falls in: a split name, 'embargo', 'before' or 'after'."""
    labels = pd.Series("embargo", index=times.index, dtype="object")
    first, _ = period_times(periods["train"])
    _, last = period_times(periods["out_of_sample"])
    for name in SPLIT_NAMES:
        start, end = period_times(periods[name])
        labels[(times >= start) & (times < end)] = name
    labels[times < first] = "before"
    labels[times >= last] = "after"
    return labels


def freeze_splits(
    periods: dict[SplitName, SplitPeriod], approved_by: str, path: Path, today: date
) -> SplitDates:
    """Write config/splits.yaml once. Changing it later is a human edit + DECISIONS.md entry."""
    if path.exists():
        raise SplitError(f"{path.name} already exists; split dates are frozen and never recomputed")
    frozen = SplitDates(approved_by=approved_by, approved_on=today, **periods)
    header = (
        "# Fixed data splits (SPEC §7.2), approved by a human. Dates are UTC, [start, end).\n"
        "# Never recompute or edit without a docs/DECISIONS.md entry. Bars after the\n"
        "# out_of_sample end are forward data (paper trading), not part of any split.\n"
    )
    body = yaml.safe_dump(frozen.model_dump(mode="json"), sort_keys=False)
    path.write_text(header + body, encoding="utf-8")
    return frozen
