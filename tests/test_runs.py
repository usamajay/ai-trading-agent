"""Recorded backtest runs (Task 2.8): determinism, run counter, fingerprint, report files."""

import json
import math
import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic_bars import trading_bars

from tradeagent.backtest.costs import CostModel
from tradeagent.backtest.records import clean, list_runs, run_counts
from tradeagent.backtest.runner import bars_fingerprint, stressed
from tradeagent.backtest.runs import execute_run
from tradeagent.backtest.splits import SplitError
from tradeagent.config import AppConfig, SplitDates, SplitPeriod, load_config
from tradeagent.data.store import BarStore, connect_db
from tradeagent.features.indicators import atr
from tradeagent.strategies import registry
from tradeagent.strategies.base import BaseStrategy, MarketContext, Signal
from tradeagent.strategies.variants import VariantSpec

NAME = "test_seeded_random"


class SeededRandom(BaseStrategy):
    """M15, 1-in-8 random entries, stop 1.5 ATR, target 2x the stop from the entry side."""

    name, version, style = NAME, "1", "intraday"
    timeframes = ("M15",)
    lookback_bars = 15

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out = dict(frames)
        out["M15"] = out["M15"].assign(atr14=atr(out["M15"], 14))
        return out

    def generate(self, ctx: MarketContext) -> list[Signal]:
        draw, side = self.rng.random(), self.rng.random()
        view = ctx.bars("M15")
        a = view.last("atr14")
        if draw > 0.125 or not np.isfinite(a):
            return []
        long = side < 0.5
        entry = view.last("close") + (view.last("spread") * 0.001 * 1.1 if long else 0.0)
        stop = 1.5 * a
        sl, tp = (entry - stop, entry + 2 * stop) if long else (entry + stop, entry - 2 * stop)
        return [Signal(ctx.symbol, "long" if long else "short", sl, tp, why="seeded random")]


def aggregate(m5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    out = m5.groupby(m5["time_utc"].dt.floor(f"{minutes}min")).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        tick_volume=("tick_volume", "sum"),
        spread=("spread", "max"),
    )
    return out.rename_axis("time_utc").reset_index()


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> BarStore:
    s = BarStore(tmp_path_factory.mktemp("bars"))
    m5 = trading_bars("2026-01-04", weeks=5)  # 5 weeks of M5, US winter time
    s.write("XAUUSD", "M5", m5)
    s.write("XAUUSD", "M15", aggregate(m5, 15))
    s.write("XAUUSD", "H1", aggregate(m5, 60))
    return s


@pytest.fixture
def cfg() -> AppConfig:
    splits = SplitDates(
        approved_by="test",
        approved_on=date(2026, 9, 30),
        train=SplitPeriod(start=date(2026, 1, 11), end=date(2026, 1, 25)),
        validation=SplitPeriod(start=date(2026, 1, 25), end=date(2026, 2, 8)),
        out_of_sample=SplitPeriod(start=date(2026, 2, 22), end=date(2026, 3, 1)),
    )
    return load_config().model_copy(update={"splits": splits})


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect_db(tmp_path / "test.db")
    yield connection
    connection.close()


@pytest.fixture(autouse=True)
def registered() -> Iterator[None]:
    registry.register(NAME, lambda seed, params: SeededRandom(seed))
    yield
    registry.unregister(NAME)


def run(
    cfg: AppConfig,
    store: BarStore,
    conn: sqlite3.Connection,
    out: Path,
    split: str = "train",
    seed: int = 1,
):  # type: ignore[no-untyped-def]
    return execute_run(cfg, store, conn, NAME, "XAUUSD", split, seed, {}, out)  # type: ignore[arg-type]


def test_same_command_twice_gives_identical_results(
    cfg: AppConfig, store: BarStore, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    first = run(cfg, store, conn, tmp_path)
    second = run(cfg, store, conn, tmp_path)
    assert first.result.trades["trade_id"].size > 0
    assert first.run["metrics_json"] == second.run["metrics_json"]
    assert first.run["stress_metrics_json"] == second.run["stress_metrics_json"]
    assert first.run["risk_flags_json"] == second.run["risk_flags_json"]
    pd.testing.assert_frame_equal(first.result.trades, second.result.trades)
    assert first.run["data_hash"] == second.run["data_hash"]
    assert first.run["config_hash"] == second.run["config_hash"] == cfg.config_hash
    assert first.run["run_id"] != second.run["run_id"]


def test_run_counter_per_strategy_and_split(
    cfg: AppConfig, store: BarStore, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    assert run(cfg, store, conn, tmp_path).run["run_number"] == 1
    assert run(cfg, store, conn, tmp_path, seed=2).run["run_number"] == 2  # a new variant
    last = run(cfg, store, conn, tmp_path, split="validation")
    assert last.run["run_number"] == 1
    assert last.counts == {"train": 2, "validation": 1, "out_of_sample": 0}
    assert run_counts(conn, "someone_else") == {"train": 0, "validation": 0, "out_of_sample": 0}
    report = (last.output_dir / "report.md").read_text(encoding="utf-8")
    assert "run **#1**" in report and "**train 2**" in report and "**validation 1**" in report
    listed = list_runs(conn)
    assert len(listed) == 3 and listed["strategy"].eq(NAME).all()


def test_out_of_sample_is_refused(
    cfg: AppConfig, store: BarStore, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    with pytest.raises(SplitError, match="locked"):
        run(cfg, store, conn, tmp_path, split="out_of_sample")
    assert list_runs(conn).empty  # nothing was saved


def test_report_and_files(
    cfg: AppConfig, store: BarStore, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    rec = run(cfg, store, conn, tmp_path)
    for name in ("report.md", "equity.png", "trades.parquet", "daily.parquet", "metrics.json"):
        assert (rec.output_dir / name).is_file(), name
    assert len(pd.read_parquet(rec.output_dir / "trades.parquet")) == len(rec.result.trades)
    saved = json.loads((rec.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["run"]["data_hash"] == rec.run["data_hash"]
    report = (rec.output_dir / "report.md").read_text(encoding="utf-8")
    for section in (
        "## Provenance",
        "## Multiple testing",
        "## Summary: normal costs vs cost",
        "## 95% confidence intervals",
        "## Account risk limits",
        "## Costs (% of gross profit)",
        "## Minimum balance",
        "## Conventions",
    ):
        assert section in report, section
    assert rec.run["data_hash"] in report and rec.run["git_commit"] in report
    verdict = "PASS" if rec.run["stress_pass"] else "FAIL"
    assert f"**Cost stress: {verdict}**" in report
    if rec.metrics["trades"]["insufficient_sample"]:
        assert "INSUFFICIENT SAMPLE" in report


def test_stress_run_charges_more(
    cfg: AppConfig, store: BarStore, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    rec = run(cfg, store, conn, tmp_path)
    normal = rec.metrics["costs"]["spread_usd"] / max(rec.metrics["trades"]["trades"], 1)
    stress = rec.stress_metrics["costs"]["spread_usd"] / max(
        rec.stress_metrics["trades"]["trades"], 1
    )
    assert stress > normal
    assert rec.run["stress_pass"] == int((rec.run["stress_expectancy_r"] or 0) > 0)


def test_stressed_costs_scale_spread_and_so_slippage() -> None:
    costs = CostModel.from_config(load_config(), "XAUUSD")
    hard = stressed(costs, 1.5)
    assert hard.spread(200) == pytest.approx(costs.spread(200) * 1.5)
    assert hard.slippage(200) == pytest.approx(costs.slippage(200) * 1.5)  # not x2.25
    assert hard.spec == costs.spec  # swap and contract unchanged


def test_fingerprint_changes_with_any_price() -> None:
    bars = trading_bars("2026-01-04", weeks=1, freq="1h")
    base = bars_fingerprint({"H1": bars})
    assert bars_fingerprint({"H1": bars.copy()}) == base
    nudged = bars.copy()
    nudged.loc[10, "close"] += 0.001
    assert bars_fingerprint({"H1": nudged}) != base
    assert bars_fingerprint({"H1": bars.iloc[:-1]}) != base  # a bar fewer
    assert bars_fingerprint({"M15": bars}) != base  # a different role


def test_clean_makes_json_safe() -> None:
    value = {
        "pf": math.inf,
        "x": float("nan"),
        "n": np.int64(3),
        "f": np.float64(1.5),
        "d": date(2026, 1, 5),
        "l": [math.inf, 1],
    }
    assert clean(value) == {
        "pf": "inf",
        "x": None,
        "n": 3,
        "f": 1.5,
        "d": "2026-01-05",
        "l": ["inf", 1],
    }


def test_baseline_distribution_one_row_per_seed(
    cfg: AppConfig, store: BarStore, tmp_path: Path
) -> None:
    from tradeagent.backtest.baseline import baseline_distribution, save_distribution

    df, data_hash = baseline_distribution(
        cfg, store, "XAUUSD", range(1, 4), params={"p_entry": 0.1}
    )
    assert df["seed"].tolist() == [1, 2, 3]
    assert (df["trades"] > 0).all()
    assert df["expectancy_r"].nunique() == 3  # different seeds, different trades
    assert len(data_hash) == 64
    again, _ = baseline_distribution(cfg, store, "XAUUSD", range(1, 2), params={"p_entry": 0.1})
    assert again.iloc[0].equals(df.iloc[0])  # same seed, same row
    path = save_distribution(df, tmp_path, "XAUUSD", "M15", "train")
    assert pd.read_parquet(path).shape == df.shape


def test_account_limit_mode_is_recorded(
    cfg: AppConfig, store: BarStore, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    flags = run(cfg, store, conn, tmp_path)
    assert json.loads(flags.run["params_json"])["__account_limits"] == "flags"
    enforced = execute_run(
        cfg, store, conn, NAME, "XAUUSD", "train", 1, {}, tmp_path, enforce_account_limits=True
    )
    assert json.loads(enforced.run["params_json"])["__account_limits"] == "enforced"
    report = (enforced.output_dir / "report.md").read_text(encoding="utf-8")
    assert "**Enforced** in this run" in report


def test_variants_are_counted_under_their_base_strategy(
    cfg: AppConfig, store: BarStore, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    plain = run(cfg, store, conn, tmp_path)
    longs = VariantSpec(NAME, direction="long")
    rec = execute_run(cfg, store, conn, NAME, "XAUUSD", "train", 1, {}, tmp_path, variant=longs)
    assert rec.run["strategy"] == f"{NAME}[dir=long]"
    assert rec.run["run_number"] == 2  # the plain run and this variant: two tries
    assert rec.counts["train"] == 2 and run_counts(conn, NAME)["train"] == 2
    assert set(rec.result.trades["direction"]) == {"long"}
    assert len(rec.result.trades) < len(plain.result.trades)
    params = json.loads(rec.run["params_json"])
    assert VariantSpec.from_dict(params["__variant"]) == longs
    assert "__variant" not in json.loads(plain.run["params_json"])


def test_timeframe_variant_runs_on_other_bars(
    cfg: AppConfig, store: BarStore, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    spec = VariantSpec(NAME, timeframes=("H1",), style="swing")
    rec = execute_run(cfg, store, conn, NAME, "XAUUSD", "train", 1, {}, tmp_path, variant=spec)
    assert rec.run["timeframe"] == "H1" and rec.run["strategy"] == f"{NAME}[tf=H1,style=swing]"
    assert len(rec.result.trades) > 0


def test_variant_for_another_strategy_is_refused(
    cfg: AppConfig, store: BarStore, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="variant is for"):
        execute_run(
            cfg,
            store,
            conn,
            NAME,
            "XAUUSD",
            "train",
            1,
            {},
            tmp_path,
            variant=VariantSpec("other"),
        )


def test_direction_breakdown_in_metrics_and_report(
    cfg: AppConfig, store: BarStore, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    rec = run(cfg, store, conn, tmp_path)
    by_dir = rec.metrics["by_direction"]
    assert set(by_dir) == {"long", "short"}
    assert sum(g["trades"] for g in by_dir.values()) == rec.metrics["trades"]["trades"]
    assert all(g["avg_cost_r"] > 0 for g in by_dir.values())
    report = (rec.output_dir / "report.md").read_text(encoding="utf-8")
    assert "| Direction | Trades" in report and "Swap paid (R)" in report
