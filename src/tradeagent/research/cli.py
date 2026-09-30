"""`tradeagent research ...` commands (Phase 6): hypotheses and pre-registered experiments."""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Annotated

import typer

from tradeagent.config import AppConfig

research_app = typer.Typer(
    help="Research: hypotheses and pre-registered experiments (Phase 6).", no_args_is_help=True
)

# The default success criterion: the bar `backtest compare` uses for "worth a look".
STANDARD_CRITERION = (
    '{"min_trades": 100, "expectancy_gt": 0, "expectancy_ci_low_gt": 0, '
    '"baseline_percentile_ge": 95, "stress_pass": true}'
)


def _console_safe() -> None:
    """Hypothesis text may hold characters the Windows console cannot show."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def _db() -> tuple[AppConfig, sqlite3.Connection]:
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import connect_db

    _console_safe()
    cfg = load_config()
    return cfg, connect_db(project_path(cfg.settings.storage.sqlite_path))


@research_app.command("load-leads")
def load_leads() -> None:
    """Load H1..Hn from docs/RESEARCH_HYPOTHESES.md as `human` hypotheses (once each)."""
    from tradeagent.config import project_path
    from tradeagent.research.experiments import add_hypothesis
    from tradeagent.research.leads import read_leads

    cfg, conn = _db()
    try:
        for lead in read_leads(project_path(Path("docs/RESEARCH_HYPOTHESES.md"))):
            added = add_hypothesis(
                conn, cfg, lead.hypothesis_id, lead.text, lead.rationale, "human"
            )
            typer.echo(f"{lead.hypothesis_id}: {'added' if added else 'already loaded'}")
    finally:
        conn.close()


@research_app.command("register")
def register_cmd(
    hypothesis: Annotated[str, typer.Option("--hypothesis", "-H")],
    strategy: Annotated[str, typer.Option("--strategy")],
    symbol: Annotated[str, typer.Option("--symbol", "-s")] = "XAUUSD",
    split: Annotated[str, typer.Option("--split")] = "train",
    timeframes: Annotated[
        str, typer.Option("--timeframes", help="Replaces the strategy's own, e.g. H4,D1")
    ] = "",
    direction: Annotated[str, typer.Option("--direction", help="both, long or short")] = "both",
    style: Annotated[str, typer.Option("--style", help="intraday or swing (default: own)")] = "",
    param: Annotated[list[str] | None, typer.Option("--param", help="name=value")] = None,
    regimes: Annotated[
        str, typer.Option("--regimes", help="Trade only in these regimes, or 'suited'")
    ] = "",
    criterion: Annotated[
        str, typer.Option("--criterion", help="JSON success criterion (default: standard)")
    ] = STANDARD_CRITERION,
    parent: Annotated[
        str | None, typer.Option("--parent", help="Train experiment that passed (validation)")
    ] = None,
    seed: Annotated[int, typer.Option("--seed")] = 1,
) -> None:
    """Register an experiment and its success criterion BEFORE it runs."""
    from pydantic import ValidationError

    from tradeagent.cli import _parse_params
    from tradeagent.research.experiments import Criterion, ExperimentError, register
    from tradeagent.research.guard import SplitGuardError
    from tradeagent.strategies import registry
    from tradeagent.strategies.variants import VariantSpec

    registry.load_builtins()
    cfg, conn = _db()
    try:
        spec = VariantSpec(
            strategy,
            timeframes=tuple(t.strip() for t in timeframes.split(",") if t.strip()),
            direction=direction,  # type: ignore[arg-type]
            style=style or None,  # type: ignore[arg-type]
            params=_parse_params(param),
            regimes=tuple(r.strip() for r in regimes.split(",") if r.strip()),
        )
        crit = Criterion.model_validate_json(criterion)
        eid = register(conn, cfg, hypothesis, spec, symbol, split, crit, seed, parent)
    except (ExperimentError, SplitGuardError, ValidationError, KeyError, ValueError) as exc:
        typer.secho(f"Cannot register: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(f"{eid}: {spec.name()} {symbol} {split}; criterion {crit.model_dump_json()}")


@research_app.command("run")
def run_cmd(
    experiment: Annotated[list[str], typer.Argument(help="Experiment ids, e.g. E0001 E0002")],
) -> None:
    """Run registered experiments once each; the verdict comes from the stored criterion."""
    from tradeagent.backtest.costs import CostError
    from tradeagent.backtest.splits import SplitError
    from tradeagent.config import project_path
    from tradeagent.data.store import BarStore
    from tradeagent.research.experiments import (
        ExperimentError,
        research_run_total,
        run_experiment,
    )
    from tradeagent.strategies import registry

    registry.load_builtins()
    cfg, conn = _db()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    failed = False
    try:
        for eid in experiment:
            try:
                res = run_experiment(
                    conn,
                    cfg,
                    store,
                    eid,
                    project_path(Path("data/backtests")),
                    project_path(Path("data/baselines")),
                )
            except (ExperimentError, SplitError, CostError, KeyError, ValueError) as exc:
                typer.secho(f"{eid}: cannot run: {exc}", fg=typer.colors.RED, err=True)
                failed = True
                continue
            checks = "; ".join(
                f"{c.name} {c.actual} ({'ok' if c.ok else 'needs ' + c.required})"
                for c in res.checks
            )
            typer.echo(f"{eid} {res.name}: {res.verdict.upper()}  {checks}")
        typer.echo(f"research runs so far (all hypotheses): {research_run_total(conn)}")
    finally:
        conn.close()
    if failed:
        raise typer.Exit(code=1)


@research_app.command("list")
def list_cmd() -> None:
    """Hypotheses and experiments, with the strict multiple-testing total."""
    import pandas as pd

    from tradeagent.research.experiments import list_experiments, research_run_total
    from tradeagent.strategies.variants import VariantSpec

    _, conn = _db()
    try:
        hyps = pd.read_sql_query(
            "SELECT hypothesis_id, source, status, text FROM hypotheses ORDER BY hypothesis_id",
            conn,
        )
        exps = list_experiments(conn)
        total = research_run_total(conn)
    finally:
        conn.close()
    for h in hyps.to_dict("records"):
        typer.echo(f"{h['hypothesis_id']:<5} {h['source']:<6} {h['status']:<12} {h['text'][:70]}")
    typer.echo("")
    for e in exps.to_dict("records"):
        spec = json.loads(e["variant_json"])
        raw = e["metrics_json"]  # NULL (not run yet) arrives from pandas as NaN
        m = json.loads(raw)["metrics"] if isinstance(raw, str) else {}
        exp = m.get("expectancy_r")
        shown = "-" if exp is None else f"{exp:+.3f} R"
        typer.echo(
            f"{e['experiment_id']} {e['hypothesis_id']:<4} {e['dataset_split']:<10} "
            f"{e['symbol']:<7} {e['status']:<10} {e['verdict'] if isinstance(e['verdict'], str) else '-':<5} {shown:>9} "
            f"{VariantSpec.from_dict(spec).name()}"
        )
    typer.echo(f"\nresearch runs so far (strict multiple-testing total): {total}")


@research_app.command("conclude")
def conclude_cmd(
    hypothesis: Annotated[str, typer.Argument()],
    status: Annotated[str, typer.Option("--status", help="supported, falsified, inconclusive")],
    lesson: Annotated[str, typer.Option("--lesson", help="What was learned (required)")],
) -> None:
    """Record a conclusion on a hypothesis with a written lesson."""
    from tradeagent.research.experiments import ExperimentError, conclude

    cfg, conn = _db()
    try:
        conclude(conn, cfg, hypothesis, status, lesson)
    except ExperimentError as exc:
        typer.secho(f"Cannot conclude: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(f"{hypothesis}: {status}")


@research_app.command("withdraw")
def withdraw_cmd(
    experiment: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option("--reason", help="Why it will not be run (required)")],
) -> None:
    """Withdraw a registered experiment before it runs (not counted as a test)."""
    from tradeagent.research.experiments import ExperimentError, withdraw

    cfg, conn = _db()
    try:
        withdraw(conn, cfg, experiment, reason)
    except ExperimentError as exc:
        typer.secho(f"Cannot withdraw: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(f"{experiment}: withdrawn")
