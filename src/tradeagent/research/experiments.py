"""Experiment manager (SPEC §8, docs/PHASE_6_TASKS.md 6.1).

hypothesis -> register experiments (variant, symbol, split, success criterion written
**before** the run) -> run (a normal recorded backtest) -> verdict computed by code
from the stored criterion -> lesson row. The criterion is never edited after
registration; a new idea is a new experiment and adds to the run counters.
"""

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tradeagent.backtest.baseline import distribution_path
from tradeagent.backtest.compare import baseline_percentile
from tradeagent.backtest.runs import RunRecord, execute_run
from tradeagent.config import AppConfig, SplitName
from tradeagent.data.store import BarStore
from tradeagent.provenance import git_commit
from tradeagent.research.guard import check_split
from tradeagent.strategies.variants import VariantSpec, build
from tradeagent.timeutil import utc_now

HYPOTHESIS_STATUSES = ("proposed", "testing", "supported", "falsified", "inconclusive")
SOURCES = ("llm", "scan", "human")


class ExperimentError(ValueError):
    """Raised when an experiment cannot be registered or run."""


class Criterion(BaseModel):
    """Success criterion, all checks must pass. Written before the run, never edited."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_trades: int = Field(default=30, ge=1)
    expectancy_gt: float | None = None  # mean R after costs
    expectancy_ci_low_gt: float | None = None  # 95% bootstrap CI lower bound
    baseline_percentile_ge: float | None = Field(default=None, ge=0, le=100)
    stress_pass: bool = False  # still positive with spread and slippage x 1.5
    beats_experiment: str | None = None  # CI lower bound > that experiment's expectancy
    # Compare with random trades of the SAME long/short mix: per seed, the long-only and
    # short-only baselines weighted by this run's share of long and short trades.
    baseline_mix: bool = False


@dataclass(frozen=True)
class Check:
    name: str
    required: str
    actual: str
    ok: bool


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:+.3f}"


def evaluate(
    c: Criterion,
    m: dict[str, Any],
    percentile: float | None,
    other_expectancy: float | None,
) -> list[Check]:
    """Checks of `c` against run metrics `m` (keys: trades, expectancy_r, ci_low, stress_pass)."""
    exp, low = m["expectancy_r"], m["ci_low"]
    checks = [Check("trades", f">= {c.min_trades}", str(m["trades"]), m["trades"] >= c.min_trades)]
    if c.expectancy_gt is not None:
        ok = exp is not None and exp > c.expectancy_gt
        checks.append(Check("expectancy_r", f"> {c.expectancy_gt:+.3f}", _fmt(exp), ok))
    if c.expectancy_ci_low_gt is not None:
        ok = low is not None and low > c.expectancy_ci_low_gt
        checks.append(Check("ci_low", f"> {c.expectancy_ci_low_gt:+.3f}", _fmt(low), ok))
    if c.baseline_percentile_ge is not None:
        ok = percentile is not None and percentile >= c.baseline_percentile_ge
        shown = "-" if percentile is None else f"{percentile:.0f}"
        checks.append(Check("vs_random_pct", f">= {c.baseline_percentile_ge:.0f}", shown, ok))
    if c.stress_pass:
        checks.append(
            Check("stress", "PASS", "PASS" if m["stress_pass"] else "FAIL", m["stress_pass"])
        )
    if c.beats_experiment is not None:
        ok = low is not None and other_expectancy is not None and low > other_expectancy
        checks.append(
            Check(
                f"beats {c.beats_experiment}",
                f"ci_low > {_fmt(other_expectancy)}",
                _fmt(low),
                ok,
            )
        )
    return checks


# --- hypotheses ------------------------------------------------------------------------


def add_hypothesis(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    hypothesis_id: str,
    text: str,
    rationale: str,
    source: str,
) -> bool:
    """Insert a hypothesis (status `proposed`); False if that id already exists."""
    if source not in SOURCES:
        raise ExperimentError(f"source must be one of {SOURCES}")
    with conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO hypotheses (hypothesis_id, text, rationale, source, status, "
            "git_commit, config_hash, created_at) VALUES (?, ?, ?, ?, 'proposed', ?, ?, ?)",
            (
                hypothesis_id,
                text,
                rationale,
                source,
                git_commit(),
                cfg.config_hash,
                utc_now().isoformat(),
            ),
        )
    return cur.rowcount == 1


def set_hypothesis_status(conn: sqlite3.Connection, hypothesis_id: str, status: str) -> None:
    if status not in HYPOTHESIS_STATUSES:
        raise ExperimentError(f"status must be one of {HYPOTHESIS_STATUSES}")
    with conn:
        cur = conn.execute(
            "UPDATE hypotheses SET status = ? WHERE hypothesis_id = ?", (status, hypothesis_id)
        )
    if cur.rowcount == 0:
        raise ExperimentError(f"no hypothesis {hypothesis_id!r}")


def add_lesson(
    conn: sqlite3.Connection, cfg: AppConfig, experiment_id: str, text: str, evidence: Any
) -> None:
    n = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0] + 1
    with conn:
        conn.execute(
            "INSERT INTO lessons (lesson_id, experiment_id, text, evidence_json, git_commit, "
            "config_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"L{n:04d}",
                experiment_id,
                text,
                json.dumps(evidence, sort_keys=True),
                git_commit(),
                cfg.config_hash,
                utc_now().isoformat(),
            ),
        )


def conclude(
    conn: sqlite3.Connection, cfg: AppConfig, hypothesis_id: str, status: str, lesson: str
) -> None:
    """A human's conclusion on a hypothesis, with a lesson on its latest experiment."""
    if status not in ("supported", "falsified", "inconclusive"):
        raise ExperimentError("conclude with supported, falsified or inconclusive")
    if not lesson.strip():
        raise ExperimentError("a conclusion needs a written lesson")
    last = conn.execute(
        "SELECT experiment_id FROM experiments WHERE hypothesis_id = ? "
        "AND status IN ('done', 'withdrawn') "
        "ORDER BY experiment_id DESC LIMIT 1",
        (hypothesis_id,),
    ).fetchone()
    if last is None:
        raise ExperimentError(f"{hypothesis_id} has no finished experiment to conclude from")
    set_hypothesis_status(conn, hypothesis_id, status)
    add_lesson(conn, cfg, last[0], f"{hypothesis_id} {status}: {lesson}", {"status": status})


# --- experiments -----------------------------------------------------------------------


def variant_key(spec: VariantSpec) -> str:
    return json.dumps(spec.to_dict(), sort_keys=True)


def register(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    hypothesis_id: str,
    spec: VariantSpec,
    symbol: str,
    split: str,
    criterion: Criterion,
    seed: int = 1,
    parent_experiment_id: str | None = None,
) -> str:
    """Store the experiment and its criterion before anything runs; returns its id."""
    if (
        conn.execute(
            "SELECT 1 FROM hypotheses WHERE hypothesis_id = ?", (hypothesis_id,)
        ).fetchone()
        is None
    ):
        raise ExperimentError(f"no hypothesis {hypothesis_id!r}")
    build(spec, seed)  # unknown strategy or wrong timeframes fail here, not later
    key = variant_key(spec)
    check_split(conn, split, key, symbol, parent_experiment_id)
    n = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] + 1
    experiment_id = f"E{n:04d}"
    with conn:
        conn.execute(
            "INSERT INTO experiments (experiment_id, hypothesis_id, dataset_split, date_range, "
            "git_commit, config_hash, seed, created_at, symbol, variant_json, "
            "success_criterion, status, parent_experiment_id) "
            "VALUES (?, ?, ?, 'not run', ?, ?, ?, ?, ?, ?, ?, 'registered', ?)",
            (
                experiment_id,
                hypothesis_id,
                split,
                git_commit(),
                cfg.config_hash,
                seed,
                utc_now().isoformat(),
                symbol,
                key,
                criterion.model_dump_json(),
                parent_experiment_id,
            ),
        )
        conn.execute(
            "UPDATE hypotheses SET status = 'testing' WHERE hypothesis_id = ? "
            "AND status = 'proposed'",
            (hypothesis_id,),
        )
    return experiment_id


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    name: str
    verdict: str
    checks: list[Check]
    record: RunRecord


Runner = Callable[..., RunRecord]


def run_experiment(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    store: BarStore,
    experiment_id: str,
    out_root: Path,
    baselines_dir: Path,
    runner: Runner = execute_run,
) -> ExperimentResult:
    row = conn.execute(
        "SELECT hypothesis_id, dataset_split, seed, symbol, variant_json, success_criterion, "
        "status FROM experiments WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    if row is None:
        raise ExperimentError(f"no experiment {experiment_id!r}")
    hypothesis_id, split, seed, symbol, key, criterion_json, status = row
    if status != "registered":
        raise ExperimentError(f"{experiment_id} is {status}; experiments run once")
    spec = VariantSpec.from_dict(json.loads(key))
    criterion = Criterion.model_validate_json(criterion_json)
    strategy = build(spec, seed)

    # Everything the verdict needs is checked before the run, so a missing input
    # never costs a run on the counter.
    tf, style = strategy.timeframes[0], strategy.style
    if criterion.baseline_mix:
        needed = [
            distribution_path(baselines_dir, symbol, tf, style, split, side)
            for side in ("long", "short")
        ]
    else:
        needed = [distribution_path(baselines_dir, symbol, tf, style, split, spec.direction)]
    missing = [p.name for p in needed if not p.is_file()]
    if criterion.baseline_percentile_ge is not None and missing:
        raise ExperimentError(f"random baseline missing: {', '.join(missing)} (run it first)")
    other = None
    if criterion.beats_experiment is not None:
        o = conn.execute(
            "SELECT status, metrics_json FROM experiments WHERE experiment_id = ?",
            (criterion.beats_experiment,),
        ).fetchone()
        if o is None or o[0] != "done":
            raise ExperimentError(f"{criterion.beats_experiment} must be run first")
        other = json.loads(o[1])["metrics"]["expectancy_r"]

    split_name: SplitName = "train" if split == "train" else "validation"
    rec = runner(
        cfg,
        store,
        conn,
        spec.strategy,
        symbol,
        split_name,
        seed,
        dict(spec.params),
        out_root,
        variant=spec,
    )
    t = rec.metrics["trades"]
    ci = rec.metrics["confidence_95"]["expectancy_r"]
    m = {
        "trades": t["trades"],
        "expectancy_r": t["expectancy_r"],
        "ci_low": ci[0] if ci else None,
        "ci_high": ci[1] if ci else None,
        "stress_expectancy_r": rec.run["stress_expectancy_r"],
        "stress_pass": bool(rec.run["stress_pass"]),
        "avg_cost_r": rec.metrics["costs"]["avg_cost_r"],
        "run_number": rec.run["run_number"],
    }
    percentile = None
    if not missing and t["expectancy_r"] is not None:
        if criterion.baseline_mix:
            long_share = float((rec.result.trades["direction"] == "long").mean())
            base = mixed_baseline(needed[0], needed[1], long_share)
            m["long_share"] = long_share
        else:
            base = pd.read_parquet(needed[0])["expectancy_r"]
        m["baseline_mean"] = float(base.mean())
        percentile = baseline_percentile(float(t["expectancy_r"]), base)
    m["baseline_percentile"] = percentile
    checks = evaluate(criterion, m, percentile, other)
    verdict = "pass" if all(c.ok for c in checks) else "fail"
    with conn:
        conn.execute(
            "UPDATE experiments SET status = 'done', verdict = ?, run_id = ?, metrics_json = ?, "
            "date_range = ?, git_commit = ? WHERE experiment_id = ?",
            (
                verdict,
                rec.run["run_id"],
                json.dumps({"metrics": m, "checks": [c.__dict__ for c in checks]}, sort_keys=True),
                f"{rec.run['date_start']} to {rec.run['date_end']}",
                rec.run["git_commit"],
                experiment_id,
            ),
        )
    failed = [c for c in checks if not c.ok]
    text = (
        f"{experiment_id} ({hypothesis_id}) {spec.name()} {symbol} {split}: {verdict.upper()}; "
        f"expectancy {_fmt(m['expectancy_r'])} R over {m['trades']} trades"
        + (
            "; failed " + ", ".join(f"{c.name} {c.actual} (needs {c.required})" for c in failed)
            if failed
            else ""
        )
    )
    add_lesson(conn, cfg, experiment_id, text, {"metrics": m})
    return ExperimentResult(experiment_id, spec.name(), verdict, checks, rec)


def withdraw(conn: sqlite3.Connection, cfg: AppConfig, experiment_id: str, reason: str) -> None:
    """Withdraw a registered experiment without running it. It never ran, so it does not
    count toward the multiple-testing total; the reason is kept as a lesson."""
    if not reason.strip():
        raise ExperimentError("withdrawing needs a written reason")
    row = conn.execute(
        "SELECT status FROM experiments WHERE experiment_id = ?", (experiment_id,)
    ).fetchone()
    if row is None:
        raise ExperimentError(f"no experiment {experiment_id!r}")
    if row[0] != "registered":
        raise ExperimentError(f"{experiment_id} is {row[0]}; only registered ones can be withdrawn")
    with conn:
        conn.execute(
            "UPDATE experiments SET status = 'withdrawn', verdict = 'withdrawn', "
            "metrics_json = ? WHERE experiment_id = ?",
            (json.dumps({"metrics": {}, "withdrawn_reason": reason}), experiment_id),
        )
    add_lesson(conn, cfg, experiment_id, f"{experiment_id} withdrawn (not run): {reason}", {})


def mixed_baseline(long_path: Path, short_path: Path, long_share: float) -> pd.Series:
    """Per seed: long_share x long-only expectancy + (1 - long_share) x short-only."""
    longs = pd.read_parquet(long_path).set_index("seed")["expectancy_r"]
    shorts = pd.read_parquet(short_path).set_index("seed")["expectancy_r"]
    seeds = longs.index.intersection(shorts.index)
    return long_share * longs[seeds] + (1 - long_share) * shorts[seeds]


def research_run_total(conn: sqlite3.Connection) -> int:
    """Every experiment run so far (all hypotheses, variants, symbols, splits)."""
    return int(conn.execute("SELECT COUNT(*) FROM experiments WHERE status = 'done'").fetchone()[0])


def list_experiments(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT experiment_id, hypothesis_id, dataset_split, symbol, variant_json, status, "
        "verdict, metrics_json, run_id FROM experiments ORDER BY experiment_id",
        conn,
    )
