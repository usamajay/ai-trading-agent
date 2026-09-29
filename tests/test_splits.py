"""Fixed split dates with embargo gaps, and the Phase 2 out-of-sample lock."""

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tradeagent.backtest.splits import (
    SplitError,
    freeze_splits,
    label_times,
    propose_splits,
    split_range,
)
from tradeagent.config import (
    DEFAULT_CONFIG_DIR,
    SPLITS_FILE,
    ConfigError,
    DataSplits,
    SplitDates,
    load_config,
)

FRACTIONS = DataSplits(train=0.6, validation=0.2, out_of_sample=0.2)


@pytest.fixture
def periods() -> dict:
    # The real stored range: 2023-09-29 (a Friday) to 2026-09-29 (a Tuesday).
    return propose_splits(
        pd.Timestamp("2023-09-29 18:25", tz="UTC"),
        pd.Timestamp("2026-09-29 19:50", tz="UTC"),
        FRACTIONS,
        embargo_weeks=2,
    )


def test_proposal_uses_whole_weeks_and_embargo_gaps(periods: dict) -> None:
    assert (periods["train"].start, periods["train"].end) == (date(2023, 10, 1), date(2025, 6, 29))
    assert (periods["validation"].start, periods["validation"].end) == (
        date(2025, 7, 13),
        date(2026, 2, 8),
    )
    assert (periods["out_of_sample"].start, periods["out_of_sample"].end) == (
        date(2026, 2, 22),
        date(2026, 9, 27),
    )
    weeks = [(p.end - p.start).days // 7 for p in periods.values()]
    assert weeks == [91, 30, 31]  # 152 usable weeks: 60% / 20% / 20%
    for p in periods.values():
        assert p.start.weekday() == 6 and p.end.weekday() == 6  # Sunday: market closed
    assert (periods["validation"].start - periods["train"].end).days == 14
    assert (periods["out_of_sample"].start - periods["validation"].end).days == 14


def test_too_little_data_is_refused() -> None:
    with pytest.raises(SplitError, match="too few"):
        propose_splits(
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-03-01", tz="UTC"),
            FRACTIONS,
            embargo_weeks=2,
        )


def test_label_times(periods: dict) -> None:
    times = pd.Series(
        pd.to_datetime(
            ["2023-09-30", "2024-05-01", "2025-07-01", "2025-10-01", "2026-03-01", "2026-09-28"],
            utc=True,
        )
    )
    assert label_times(times, periods).tolist() == [
        "before",
        "train",
        "embargo",
        "validation",
        "out_of_sample",
        "after",
    ]


def frozen(periods: dict) -> SplitDates:
    return SplitDates(approved_by="Usama", approved_on=date(2026, 9, 30), **periods)


def test_out_of_sample_is_locked_in_phase_2(periods: dict) -> None:
    with pytest.raises(SplitError, match="locked"):
        split_range(frozen(periods), "out_of_sample")


def test_unfrozen_splits_cannot_be_used() -> None:
    with pytest.raises(SplitError, match="not frozen"):
        split_range(None, "train")


def test_split_range(periods: dict) -> None:
    start, end = split_range(frozen(periods), "validation")
    assert (start, end) == (
        pd.Timestamp("2025-07-13", tz="UTC"),
        pd.Timestamp("2026-02-08", tz="UTC"),
    )


def test_freeze_once_then_load(tmp_path: Path, periods: dict) -> None:
    config_dir = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, config_dir)
    (config_dir / SPLITS_FILE).unlink(missing_ok=True)
    before = load_config(config_dir)
    assert before.splits is None

    freeze_splits(periods, "Usama", config_dir / SPLITS_FILE, today=date(2026, 9, 30))
    after = load_config(config_dir)
    assert after.splits == frozen(periods)
    assert after.config_hash != before.config_hash  # split dates are part of the config

    with pytest.raises(SplitError, match="already exists"):
        freeze_splits(periods, "Usama", config_dir / SPLITS_FILE, today=date(2026, 9, 30))


def test_splits_without_embargo_are_rejected(tmp_path: Path, periods: dict) -> None:
    config_dir = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, config_dir)
    (config_dir / SPLITS_FILE).unlink(missing_ok=True)
    touching = dict(periods)
    touching["validation"] = type(periods["validation"])(
        start=periods["train"].end, end=periods["validation"].end
    )
    freeze_splits(touching, "Usama", config_dir / SPLITS_FILE, today=date(2026, 9, 30))
    with pytest.raises(ConfigError, match="embargo"):
        load_config(config_dir)
