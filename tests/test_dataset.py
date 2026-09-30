"""Per-bar backtest flags: exclusions, no-trade windows, reopens, Friday cut-off."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from synthetic_bars import NY, trading_bars

from tradeagent.backtest.dataset import (
    FLAG_COLUMNS,
    exclusion_windows,
    flag_bars,
    load_dataset,
)
from tradeagent.backtest.splits import SplitError
from tradeagent.config import (
    BacktestSettings,
    DataExclusions,
    SplitDates,
    SplitPeriod,
    TimeWindow,
    load_config,
)
from tradeagent.data.store import BarStore
from tradeagent.data.validation import find_gaps

SETTINGS = BacktestSettings(
    embargo_weeks=2,
    max_hole_minutes=60,
    no_entry_minutes_after_open=15,
    friday_cutoff_ny="16:30",
    exclusions_source_timeframe="M5",
    spread_margin_multiple=1.0,
    spread_margin_points=0,
    slippage_spread_multiple=0.2,
    commission_per_lot_usd=0.0,
    starting_balance=10_000,
    cost_stress_multiple=1.5,
)
NONE = DataExclusions(excluded_windows=[], no_trade_windows=[], keep_gaps=[])


def ny(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def window(start: str, end: str, symbols: list[str] | None = None) -> TimeWindow:
    return TimeWindow(
        start=ny(start).to_pydatetime(),
        end=ny(end).to_pydatetime(),
        symbols=symbols,
        reason="test",
        decision=date(2026, 9, 29),
    )


def flags(
    bars: pd.DataFrame,
    exclusions: DataExclusions = NONE,
    flat_before_weekend: bool = False,
    period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    windows = exclusion_windows(find_gaps(bars, "M5"), exclusions, "XAUUSD", 60)
    out = flag_bars(
        bars, "M5", "XAUUSD", windows, exclusions, SETTINGS, flat_before_weekend, period
    )
    return out.set_index(out["time_utc"].dt.tz_convert(NY).dt.tz_localize(None))


def remove(bars: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    t = bars["time_utc"]
    return bars[(t < ny(start)) | (t >= ny(end))].reset_index(drop=True)


@pytest.fixture
def bars() -> pd.DataFrame:
    return trading_bars("2026-01-04", weeks=1)  # one clean M5 week, US winter time


def test_clean_week(bars: pd.DataFrame) -> None:
    f = flags(bars)
    assert list(f.columns[-len(FLAG_COLUMNS) :]) == FLAG_COLUMNS
    assert not f["excluded"].any()
    # Gaps only at the four daily breaks (60 min each).
    gaps = f[f["gap_before"]]
    assert gaps["gap_minutes"].tolist() == [60.0] * 4
    assert (gaps.index.hour == 18).all()


def test_no_entries_in_first_15_minutes_after_a_reopen(bars: pd.DataFrame) -> None:
    f = flags(bars)["no_new_entries"]
    assert f["2026-01-04 18:00":"2026-01-04 18:10"].all()  # weekly open (first bar)
    assert not f["2026-01-04 18:15"]
    assert f["2026-01-06 18:00":"2026-01-06 18:10"].all()  # after the daily break
    assert not f["2026-01-06 18:15"]
    assert not f["2026-01-06 16:55"]


def test_friday_cutoff_only_for_flat_before_weekend(bars: pd.DataFrame) -> None:
    swing = flags(bars)["no_new_entries"]
    intraday = flags(bars, flat_before_weekend=True)["no_new_entries"]
    assert not intraday["2026-01-09 16:25"]
    assert intraday["2026-01-09 16:30":"2026-01-09 16:55"].all()
    assert not swing["2026-01-09 16:30":"2026-01-09 16:55"].any()
    assert not intraday["2026-01-08 16:30"]  # Thursday is not the weekly cut-off


def test_long_unreviewed_hole_is_excluded(bars: pd.DataFrame) -> None:
    holed = remove(bars, "2026-01-07 10:00", "2026-01-07 12:00")
    f = flags(holed)
    excluded = f[f["excluded"]]
    assert excluded.index.tolist() == [pd.Timestamp("2026-01-07 12:00")]  # first bar after
    assert not f.loc["2026-01-07 09:55", "excluded"]  # the bar before the hole is fine
    assert f.loc["2026-01-07 12:00", "gap_minutes"] == 120
    assert f["2026-01-07 12:00":"2026-01-07 12:10"]["no_new_entries"].all()


def test_short_hole_is_a_gap_not_an_exclusion(bars: pd.DataFrame) -> None:
    f = flags(remove(bars, "2026-01-07 10:00", "2026-01-07 10:45"))
    assert not f["excluded"].any()
    assert f.loc["2026-01-07 10:45", "gap_before"]
    assert f.loc["2026-01-07 10:45", "no_new_entries"]  # 45 min >= 30: counts as a reopen
    assert not f.loc["2026-01-07 11:00", "no_new_entries"]


def test_reviewed_keep_gap_is_not_excluded(bars: pd.DataFrame) -> None:
    holed = remove(bars, "2026-01-07 10:00", "2026-01-07 12:00")
    keep = DataExclusions(
        excluded_windows=[],
        no_trade_windows=[],
        keep_gaps=[window("2026-01-07 09:00", "2026-01-07 13:00")],
    )
    f = flags(holed, keep)
    assert not f["excluded"].any()
    assert f.loc["2026-01-07 12:00", "gap_before"]  # still a gap: stops fill at the open


def test_no_trade_window_blocks_entries_but_does_not_exclude(bars: pd.DataFrame) -> None:
    holed = remove(bars, "2026-01-07 10:00", "2026-01-07 12:00")
    rules = DataExclusions(
        excluded_windows=[],
        no_trade_windows=[window("2026-01-07 08:00", "2026-01-07 14:00")],
        keep_gaps=[],
    )
    f = flags(holed, rules)
    assert not f["excluded"].any()  # reviewed: open trades are managed normally
    assert f["2026-01-07 08:00":"2026-01-07 13:55"]["no_new_entries"].all()
    assert not f.loc["2026-01-07 14:00", "no_new_entries"]
    assert not f.loc["2026-01-07 07:55", "no_new_entries"]


def test_explicit_excluded_window_over_existing_bars(bars: pd.DataFrame) -> None:
    rules = DataExclusions(
        excluded_windows=[window("2026-01-06 10:00", "2026-01-06 11:00")],
        no_trade_windows=[],
        keep_gaps=[],
    )
    f = flags(bars, rules)
    excluded = f[f["excluded"]].index
    assert excluded.min() == pd.Timestamp("2026-01-06 10:00")
    assert excluded.max() == pd.Timestamp("2026-01-06 10:55")
    assert len(excluded) == 12
    assert f[f["excluded"]]["no_new_entries"].all()


def test_hole_covered_by_a_decision_is_listed_once(bars: pd.DataFrame) -> None:
    holed = remove(bars, "2026-01-07 10:00", "2026-01-07 12:00")
    rules = DataExclusions(
        excluded_windows=[window("2026-01-07 10:00", "2026-01-07 11:55")],  # ends a bit early
        no_trade_windows=[],
        keep_gaps=[],
    )
    windows = exclusion_windows(find_gaps(holed, "M5"), rules, "XAUUSD", 60)
    assert windows["source"].tolist() == ["decision"]
    f = flags(holed, rules)
    assert f[f["excluded"]].index.tolist() == [pd.Timestamp("2026-01-07 12:00")]


def test_windows_for_another_symbol_are_ignored(bars: pd.DataFrame) -> None:
    rules = DataExclusions(
        excluded_windows=[window("2026-01-06 10:00", "2026-01-06 11:00", ["USOIL"])],
        no_trade_windows=[window("2026-01-07 08:00", "2026-01-07 14:00", ["USOIL"])],
        keep_gaps=[],
    )
    f = flags(bars, rules)
    assert not f["excluded"].any()
    assert not f.loc["2026-01-07 10:00", "no_new_entries"]


def test_warmup_bars_are_history_only(bars: pd.DataFrame) -> None:
    period = (ny("2026-01-06 00:00"), ny("2026-01-08 00:00"))
    f = flags(bars, period=period)
    warmup = f[~f["in_split"]]
    assert warmup["no_new_entries"].all()
    assert f["in_split"].sum() > 0
    assert f[f["in_split"]].index.min() >= pd.Timestamp("2026-01-06 00:00")


# --- load_dataset ------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> BarStore:
    s = BarStore(tmp_path / "bars")
    week = trading_bars("2026-01-04", weeks=3)
    s.write("XAUUSD", "M5", week)
    s.write("XAUUSD", "H1", trading_bars("2026-01-04", weeks=3, freq="1h"))
    return s


def config_with_splits():  # type: ignore[no-untyped-def]
    cfg = load_config()
    splits = SplitDates(
        approved_by="test",
        approved_on=date(2026, 9, 30),
        train=SplitPeriod(start=date(2026, 1, 4), end=date(2026, 1, 11)),
        validation=SplitPeriod(start=date(2026, 1, 25), end=date(2026, 2, 1)),
        out_of_sample=SplitPeriod(start=date(2026, 2, 15), end=date(2026, 3, 1)),
    )
    return cfg.model_copy(update={"splits": splits})


def test_load_dataset_train_split(store: BarStore) -> None:
    ds = load_dataset(config_with_splits(), store, "XAUUSD", "M5", "train", "intraday")
    b = ds.bars
    assert b["in_split"].all()  # train starts at the first bar: no warm-up exists
    assert b["time_utc"].max() < ds.end_utc
    assert b.loc[b["time_utc"].dt.tz_convert(NY).dt.dayofweek == 4, "no_new_entries"].any()


def test_load_dataset_warmup_and_d1(store: BarStore) -> None:
    cfg = config_with_splits().model_copy(
        update={
            "splits": config_with_splits().splits.model_copy(  # type: ignore[union-attr]
                update={"train": SplitPeriod(start=date(2026, 1, 11), end=date(2026, 1, 18))}
            )
        }
    )
    ds = load_dataset(cfg, store, "XAUUSD", "M5", "train", "swing", warmup_bars=100)
    assert (~ds.bars["in_split"]).sum() == 100
    d1 = load_dataset(cfg, store, "XAUUSD", "D1", "train", "swing", warmup_bars=5)
    assert "end_utc" in d1.bars.columns  # New York-close bars, not broker D1
    assert (d1.bars["time_utc"].dt.tz_convert(NY).dt.hour == 17).all()
    assert d1.bars["in_split"].sum() == 5


def test_load_dataset_refuses_oos_and_unfrozen(store: BarStore) -> None:
    with pytest.raises(SplitError, match="locked"):
        load_dataset(config_with_splits(), store, "XAUUSD", "M5", "out_of_sample", "swing")
    with pytest.raises(SplitError, match="not frozen"):
        load_dataset(
            load_config().model_copy(update={"splits": None}),
            store,
            "XAUUSD",
            "M5",
            "train",
            "swing",
        )


def test_entry_flags_say_why(bars: pd.DataFrame) -> None:
    f = flags(bars, flat_before_weekend=True)
    reopen = f.loc["2026-01-06 18:05"]
    assert reopen["after_reopen"] and not reopen["entry_blocked"]  # wait, not a hard block
    friday = f.loc["2026-01-09 16:40"]
    assert friday["flatten"] and friday["entry_blocked"] and not friday["after_reopen"]
    assert (f["no_new_entries"] == (f["entry_blocked"] | f["after_reopen"])).all()
    assert not flags(bars)["flatten"].any()  # swing: never flattened for the weekend


def test_intraday_flattens_before_holiday_closures() -> None:
    def span(start: str, end: str) -> pd.DataFrame:
        times = pd.date_range(ny(start), ny(end), freq="5min", inclusive="left")
        return pd.DataFrame(
            {
                "time_utc": times.astype("datetime64[ns, UTC]"),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "tick_volume": 1,
                "spread": 20,
            }
        )

    # Black Friday: early close 13:00; Good Friday: closed, so the week ends Thursday 17:00.
    black_friday = pd.concat(
        [
            span("2026-11-27 09:00", "2026-11-27 13:00"),
            span("2026-11-29 18:00", "2026-11-29 19:00"),
        ],
        ignore_index=True,
    )
    good_friday = pd.concat(
        [
            span("2026-04-02 14:00", "2026-04-02 17:00"),
            span("2026-04-05 18:00", "2026-04-05 19:00"),
        ],
        ignore_index=True,
    )
    for bars, first_flat in ((black_friday, "2026-11-27 12:30"), (good_friday, "2026-04-02 16:30")):
        f = flags(bars, flat_before_weekend=True)
        flat = f[f["flatten"]].index
        assert flat.min() == pd.Timestamp(first_flat), first_flat
        assert not flags(bars)["flatten"].any()  # swing styles hold

    # A data hole of a day or more is not a closure: no early flattening before it.
    hole = pd.concat(
        [
            span("2026-01-07 09:00", "2026-01-07 12:00"),
            span("2026-01-08 14:00", "2026-01-08 15:00"),
        ],
        ignore_index=True,
    )
    w = window("2026-01-07 12:00", "2026-01-08 14:00")
    rules = DataExclusions(excluded_windows=[w], no_trade_windows=[], keep_gaps=[])
    f = flags(hole, rules, flat_before_weekend=True)
    assert not f.loc[:"2026-01-07 11:55", "flatten"].any()
