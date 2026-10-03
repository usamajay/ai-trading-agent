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


@research_app.command("scan")
def scan_cmd(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show flags without adding hypotheses")
    ] = False,
) -> None:
    """Scan the latest train run of each strategy/variant for groups that differ beyond
    noise (Bonferroni over every group tested); flags become `scan` hypotheses."""
    import pandas as pd

    from tradeagent.backtest.compare import latest_runs
    from tradeagent.backtest.dataset import load_bars
    from tradeagent.config import project_path
    from tradeagent.data.store import BarStore
    from tradeagent.features.regime import trade_regimes
    from tradeagent.research.experiments import add_hypothesis
    from tradeagent.research.scans import hypothesis_for, run_labels, scan

    cfg, conn = _db()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    try:
        runs = []
        bars_cache: dict[tuple[str, str], pd.DataFrame] = {}
        for run in latest_runs(conn, "train").to_dict("records"):
            path = Path(run["output_dir"]) / "trades.parquet"
            if not path.is_file():
                continue
            trades = pd.read_parquet(path)
            if trades.empty:
                continue
            key = (run["symbol"], run["timeframe"])
            if key not in bars_cache:
                bars_cache[key] = load_bars(store, *key)
            regimes = trade_regimes(trades, bars_cache[key], run["timeframe"])
            runs.append((run, trades, run_labels(trades, regimes)))
        result = scan(runs)
        typer.echo(
            f"{len(runs)} runs, {len(result.tests)} groups tested; "
            f"flag threshold p < {result.threshold:.2e} (0.05 / groups)"
        )
        for t in sorted(result.flags, key=lambda x: x.p):
            hid, text, rationale = hypothesis_for(t, result.threshold)
            added = False if dry_run else add_hypothesis(conn, cfg, hid, text, rationale, "scan")
            status = "dry run" if dry_run else ("added" if added else "already known")
            typer.echo(f"{hid} p={t.p:.1e} [{status}] {text}")
        if not result.flags:
            typer.echo("No group differs beyond noise after the correction.")
        typer.echo("Closest groups (for information, not flagged unless marked above):")
        for t in sorted(result.tests, key=lambda x: x.p)[:5]:
            typer.echo(
                f"  p={t.p:.1e} {t.strategy} {t.symbol} {t.dimension}={t.group}: "
                f"{t.mean_r:+.3f} R (n={t.n}) vs {t.rest_mean_r:+.3f} R (n={t.rest_n})"
            )
    finally:
        conn.close()


@research_app.command("propose")
def propose_cmd(
    live: Annotated[
        bool,
        typer.Option("--live", help="Send ONE call to the Claude API (default: dry run only)"),
    ] = False,
    show_prompt: Annotated[
        bool, typer.Option("--show-prompt/--no-prompt", help="Print the full prompt")
    ] = True,
) -> None:
    """Ask Claude for up to 5 new hypotheses (stored as `llm`, status proposed).
    Dry run by default: prints the prompt and its worst-case cost, sends nothing."""
    from tradeagent.research.hypothesis import (
        LlmError,
        anthropic_client,
        dry_run,
        propose,
    )

    cfg, conn = _db()
    llm = cfg.settings.llm
    try:
        prompt, budget = dry_run(conn, cfg)
        if show_prompt and not live:
            typer.echo("=== system ===")
            typer.echo(prompt.system)
            typer.echo("=== user ===")
            typer.echo(prompt.user)
            typer.echo("=== end of prompt ===")
        typer.echo(
            f"model {llm.model}, effort {llm.effort}; prompt ~{budget.input_tokens} tokens "
            f"(estimate), output cap {llm.max_output_tokens} tokens"
        )
        typer.echo(
            f"worst case ${budget.worst_case_usd:.3f} (cap ${llm.max_usd_per_run:.2f}/run); "
            f"spent this month ${budget.month_spent_usd:.3f} (cap ${llm.max_usd_per_month:.2f})"
        )
        if not live:
            verdict = "within caps" if budget.allowed else f"would be refused: {budget.refusal}"
            typer.echo(f"DRY RUN: nothing sent ({verdict}). Add --live to send one call.")
            return
        try:
            result = propose(conn, cfg, anthropic_client())
        except LlmError as e:
            typer.echo(f"Refused or failed: {e}", err=True)
            raise typer.Exit(code=1) from e
        typer.echo(f"{result.call_id}: cost ${result.cost_usd:.4f}")
        for hid in result.stored:
            typer.echo(f"  {hid} added (proposed)")
        for hid in result.duplicates:
            typer.echo(f"  {hid} already known (not added again)")
        if not result.proposals:
            typer.echo("  The model proposed no hypotheses.")
        typer.echo(
            "Read them with `tradeagent research show <ID>`; register only the ones you want."
        )
    finally:
        conn.close()


@research_app.command("show")
def show_cmd(
    hypothesis: Annotated[str, typer.Argument(help="Hypothesis id, e.g. H5 or L1a2b3c")],
) -> None:
    """One hypothesis in full (LLM proposals: rationale, params, how to falsify)."""
    _, conn = _db()
    try:
        row = conn.execute(
            "SELECT hypothesis_id, source, status, text, rationale, created_at FROM hypotheses "
            "WHERE hypothesis_id = ?",
            (hypothesis,),
        ).fetchone()
        if row is None:
            typer.echo(f"No hypothesis {hypothesis}", err=True)
            raise typer.Exit(code=1)
        hid, source, status, text, rationale, created = row
        typer.echo(f"{hid} [{source}, {status}] created {created}")
        typer.echo(text)
        try:
            details = json.loads(rationale or "")
        except json.JSONDecodeError:
            details = None
        if isinstance(details, dict):
            for key, value in details.items():
                typer.echo(f"- {key}: {value}")
        elif rationale:
            typer.echo(f"- rationale: {rationale}")
    finally:
        conn.close()
