# ruff: noqa: F811  (pytest fixtures shared from test_runs are re-bound as arguments)
"""Experiment manager and split guard (Phase 6.1/6.2)."""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from test_runs import NAME, cfg, conn, registered, store  # noqa: F401 (pytest fixtures)

from tradeagent.backtest.baseline import distribution_path
from tradeagent.config import AppConfig
from tradeagent.data.store import BarStore
from tradeagent.research.experiments import (
    Criterion,
    ExperimentError,
    add_hypothesis,
    conclude,
    evaluate,
    list_experiments,
    register,
    research_run_total,
    run_experiment,
    set_hypothesis_status,
)
from tradeagent.research.guard import SplitGuardError
from tradeagent.strategies.variants import VariantSpec

PLAIN = VariantSpec(NAME)
EASY = Criterion(min_trades=1)


def _hyp(conn: sqlite3.Connection, cfg: AppConfig, hid: str = "H1") -> None:
    add_hypothesis(conn, cfg, hid, "test idea", "because", "human")


def _status(conn: sqlite3.Connection, hid: str) -> str:
    return conn.execute("SELECT status FROM hypotheses WHERE hypothesis_id = ?", (hid,)).fetchone()[
        0
    ]


def _run(conn, cfg, store, eid, tmp_path: Path):  # type: ignore[no-untyped-def]
    return run_experiment(conn, cfg, store, eid, tmp_path / "bt", tmp_path / "base")


# --- criterion ----------------------------------------------------------------------


def test_evaluate_every_check() -> None:
    c = Criterion(
        min_trades=100,
        expectancy_gt=0.0,
        expectancy_ci_low_gt=0.0,
        baseline_percentile_ge=95,
        stress_pass=True,
        beats_experiment="E0001",
    )
    good = {"trades": 150, "expectancy_r": 0.2, "ci_low": 0.05, "stress_pass": True}
    assert all(ch.ok for ch in evaluate(c, good, 97.0, 0.01))
    bad = {"trades": 20, "expectancy_r": None, "ci_low": None, "stress_pass": False}
    checks = evaluate(c, bad, None, None)
    assert [ch.name for ch in checks] == [
        "trades",
        "expectancy_r",
        "ci_low",
        "vs_random_pct",
        "stress",
        "beats E0001",
    ]
    assert not any(ch.ok for ch in checks)
    assert [ch.name for ch in evaluate(Criterion(), good, None, None)] == ["trades"]


def test_criterion_is_strict() -> None:
    with pytest.raises(ValidationError):
        Criterion(min_trade=5)  # type: ignore[call-arg]  # typo is refused
    with pytest.raises(ValidationError):
        Criterion(baseline_percentile_ge=120)


# --- hypotheses ---------------------------------------------------------------------


def test_hypotheses(conn: sqlite3.Connection, cfg: AppConfig) -> None:
    assert add_hypothesis(conn, cfg, "H1", "idea", "why", "human")
    assert not add_hypothesis(conn, cfg, "H1", "again", "why", "human")  # kept once
    assert _status(conn, "H1") == "proposed"
    with pytest.raises(ExperimentError):
        add_hypothesis(conn, cfg, "H2", "idea", "why", "guess")
    with pytest.raises(ExperimentError):
        set_hypothesis_status(conn, "H1", "maybe")
    with pytest.raises(ExperimentError):
        set_hypothesis_status(conn, "H9", "falsified")
    with pytest.raises(ExperimentError, match="no finished experiment"):
        conclude(conn, cfg, "H1", "falsified", "nothing ran")


# --- split guard --------------------------------------------------------------------


def test_register_refuses_bad_requests(conn: sqlite3.Connection, cfg: AppConfig) -> None:
    with pytest.raises(ExperimentError, match="no hypothesis"):
        register(conn, cfg, "H1", PLAIN, "XAUUSD", "train", EASY)
    _hyp(conn, cfg)
    with pytest.raises(KeyError):
        register(conn, cfg, "H1", VariantSpec("nope"), "XAUUSD", "train", EASY)
    for split in ("out_of_sample", "walk_forward"):
        with pytest.raises(SplitGuardError, match="train and validation only"):
            register(conn, cfg, "H1", PLAIN, "XAUUSD", split, EASY)
    with pytest.raises(SplitGuardError, match="needs the train experiment"):
        register(conn, cfg, "H1", PLAIN, "XAUUSD", "validation", EASY)
    with pytest.raises(SplitGuardError, match="no experiment"):
        register(conn, cfg, "H1", PLAIN, "XAUUSD", "validation", EASY, parent_experiment_id="E9")
    eid = register(conn, cfg, "H1", PLAIN, "XAUUSD", "train", EASY)
    assert eid == "E0001" and _status(conn, "H1") == "testing"
    with pytest.raises(SplitGuardError, match="only after the train experiment passed"):
        register(conn, cfg, "H1", PLAIN, "XAUUSD", "validation", EASY, parent_experiment_id=eid)


def test_validation_once_after_a_train_pass(
    conn: sqlite3.Connection, cfg: AppConfig, store: BarStore, tmp_path: Path
) -> None:
    _hyp(conn, cfg)
    train = register(conn, cfg, "H1", PLAIN, "XAUUSD", "train", EASY)
    assert _run(conn, cfg, store, train, tmp_path).verdict == "pass"
    longs = VariantSpec(NAME, direction="long")
    with pytest.raises(SplitGuardError, match="same variant"):
        register(conn, cfg, "H1", longs, "XAUUSD", "validation", EASY, parent_experiment_id=train)
    with pytest.raises(SplitGuardError, match="same variant"):
        register(conn, cfg, "H1", PLAIN, "USOIL", "validation", EASY, parent_experiment_id=train)
    val = register(conn, cfg, "H1", PLAIN, "XAUUSD", "validation", EASY, parent_experiment_id=train)
    with pytest.raises(SplitGuardError, match="one attempt only"):
        register(conn, cfg, "H1", PLAIN, "XAUUSD", "validation", EASY, parent_experiment_id=train)
    res = _run(conn, cfg, store, val, tmp_path)
    assert res.record.run["split"] == "validation"
    assert research_run_total(conn) == 2


# --- running ------------------------------------------------------------------------


def test_run_records_verdict_lesson_and_counts(
    conn: sqlite3.Connection, cfg: AppConfig, store: BarStore, tmp_path: Path
) -> None:
    _hyp(conn, cfg)
    hard = Criterion(min_trades=10_000)
    eid = register(conn, cfg, "H1", PLAIN, "XAUUSD", "train", hard)
    res = _run(conn, cfg, store, eid, tmp_path)
    assert res.verdict == "fail" and [c.ok for c in res.checks] == [False]
    row = list_experiments(conn).iloc[0]
    assert row["status"] == "done" and row["verdict"] == "fail"
    assert row["run_id"] == res.record.run["run_id"]
    saved = json.loads(row["metrics_json"])
    assert saved["metrics"]["trades"] == res.record.metrics["trades"]["trades"]
    lesson = conn.execute("SELECT text FROM lessons").fetchone()[0]
    assert lesson.startswith(f"{eid} (H1) {NAME} XAUUSD train: FAIL") and "needs >= 10000" in lesson
    with pytest.raises(ExperimentError, match="run once"):
        _run(conn, cfg, store, eid, tmp_path)
    with pytest.raises(ExperimentError, match="no experiment"):
        _run(conn, cfg, store, "E0404", tmp_path)

    # a second variant of the same strategy is counted too (strict multiple testing)
    e2 = register(conn, cfg, "H1", VariantSpec(NAME, direction="short"), "XAUUSD", "train", EASY)
    assert _run(conn, cfg, store, e2, tmp_path).record.run["run_number"] == 2
    assert research_run_total(conn) == 2
    conclude(conn, cfg, "H1", "falsified", "no edge")
    assert _status(conn, "H1") == "falsified"
    with pytest.raises(ExperimentError):
        conclude(conn, cfg, "H1", "proposed", "x")
    with pytest.raises(ExperimentError):
        conclude(conn, cfg, "H1", "falsified", " ")


def test_inputs_are_checked_before_the_run(
    conn: sqlite3.Connection, cfg: AppConfig, store: BarStore, tmp_path: Path
) -> None:
    _hyp(conn, cfg)
    needs_base = register(
        conn, cfg, "H1", PLAIN, "XAUUSD", "train", Criterion(baseline_percentile_ge=95)
    )
    with pytest.raises(ExperimentError, match="random baseline missing"):
        _run(conn, cfg, store, needs_base, tmp_path)
    needs_other = register(
        conn, cfg, "H1", PLAIN, "XAUUSD", "train", Criterion(beats_experiment=needs_base)
    )
    with pytest.raises(ExperimentError, match="must be run first"):
        _run(conn, cfg, store, needs_other, tmp_path)
    assert conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0] == 0  # none burned

    # with the baseline in place both run, and "beats" compares with the other result
    base = distribution_path(tmp_path / "base", "XAUUSD", "M15", "intraday", "train")
    base.parent.mkdir(parents=True)
    pd.DataFrame({"expectancy_r": [-1.0, -0.5, 0.0, 0.5]}).to_parquet(base)
    first = _run(conn, cfg, store, needs_base, tmp_path)
    pct = json.loads(list_experiments(conn).iloc[0]["metrics_json"])["metrics"]
    assert pct["baseline_percentile"] is not None and first.checks[-1].name == "vs_random_pct"
    second = _run(conn, cfg, store, needs_other, tmp_path)
    assert second.checks[-1].name == f"beats {needs_base}"
