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
    withdraw,
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


def test_withdraw(
    conn: sqlite3.Connection, cfg: AppConfig, store: BarStore, tmp_path: Path
) -> None:
    _hyp(conn, cfg)
    eid = register(conn, cfg, "H1", PLAIN, "XAUUSD", "train", EASY)
    with pytest.raises(ExperimentError, match="written reason"):
        withdraw(conn, cfg, eid, " ")
    with pytest.raises(ExperimentError, match="no experiment"):
        withdraw(conn, cfg, "E0404", "x")
    withdraw(conn, cfg, eid, "confounded by drift")
    row = list_experiments(conn).iloc[0]
    assert row["status"] == "withdrawn" and research_run_total(conn) == 0
    assert "confounded by drift" in json.loads(row["metrics_json"])["withdrawn_reason"]
    with pytest.raises(ExperimentError, match="only registered"):
        withdraw(conn, cfg, eid, "again")
    with pytest.raises(ExperimentError, match="run once"):
        _run(conn, cfg, store, eid, tmp_path)
    conclude(conn, cfg, "H1", "inconclusive", "answered by measurement instead")
    assert _status(conn, "H1") == "inconclusive"


def test_mixed_baseline_weights_long_and_short_per_seed(tmp_path: Path) -> None:
    from tradeagent.research.experiments import mixed_baseline

    lp, sp = tmp_path / "l.parquet", tmp_path / "s.parquet"
    pd.DataFrame({"seed": [1, 2, 3], "expectancy_r": [0.3, 0.1, 0.2]}).to_parquet(lp)
    pd.DataFrame({"seed": [2, 3, 4], "expectancy_r": [-0.1, -0.3, 9.0]}).to_parquet(sp)
    mix = mixed_baseline(lp, sp, 0.75)
    assert mix.to_dict() == pytest.approx({2: 0.05, 3: 0.075})  # seeds in both only


def test_baseline_mix_criterion(
    conn: sqlite3.Connection, cfg: AppConfig, store: BarStore, tmp_path: Path
) -> None:
    _hyp(conn, cfg)
    crit = Criterion(baseline_percentile_ge=95, baseline_mix=True)
    eid = register(conn, cfg, "H1", PLAIN, "XAUUSD", "train", crit)
    with pytest.raises(ExperimentError, match="_long.parquet, XAUUSD_M15_intraday_train_short"):
        _run(conn, cfg, store, eid, tmp_path)
    for side, e in (("long", 0.1), ("short", -0.2)):
        path = distribution_path(tmp_path / "base", "XAUUSD", "M15", "intraday", "train", side)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"seed": [1, 2], "expectancy_r": [e, e]}).to_parquet(path)
    res = _run(conn, cfg, store, eid, tmp_path)
    m = json.loads(list_experiments(conn).iloc[0]["metrics_json"])["metrics"]
    share = float((res.record.result.trades["direction"] == "long").mean())
    assert m["long_share"] == pytest.approx(share)
    assert m["baseline_mean"] == pytest.approx(share * 0.1 - (1 - share) * 0.2)


def test_exit_stage_labels_and_report_table() -> None:
    from tradeagent.backtest.metrics import exit_stage
    from tradeagent.backtest.report import _exit_stage_table

    trades = pd.DataFrame(
        {
            "exit_reason": ["tp", "sl", "sl_gap", "sl", "exit_by"],
            "stop_stage": [2, 0, 1, 2, 1],
        }
    )
    assert exit_stage(trades).tolist() == [
        "target",
        "first stop",
        "stop after move 1",
        "stop after move 2",
        "exit_by",
    ]
    assert exit_stage(trades.drop(columns="stop_stage")).iloc[2] == "first stop"
    assert _exit_stage_table({"target": {}}) == []  # no stop moves: no table
