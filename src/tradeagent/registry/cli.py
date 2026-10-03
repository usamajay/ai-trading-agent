"""`tradeagent gate ...` commands (Phase 7): registry and promotion ladder."""

import sqlite3
from pathlib import Path
from typing import Annotated

import typer

from tradeagent.config import AppConfig

gate_app = typer.Typer(
    help="Strategy registry and promotion gates (SPEC §7.4, §8.1).", no_args_is_help=True
)


def _db() -> tuple[AppConfig, sqlite3.Connection]:
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import connect_db
    from tradeagent.strategies import registry

    registry.load_builtins()
    cfg = load_config()
    return cfg, connect_db(project_path(cfg.settings.storage.sqlite_path))


def _fail(message: str) -> typer.Exit:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


@gate_app.command("register")
def register_cmd(
    strategy: Annotated[str, typer.Option("--strategy")],
    symbol: Annotated[str, typer.Option("--symbol", "-s")] = "XAUUSD",
    timeframes: Annotated[str, typer.Option("--timeframes", help="e.g. H1,H4")] = "",
    direction: Annotated[str, typer.Option("--direction", help="both, long or short")] = "both",
    style: Annotated[str, typer.Option("--style", help="intraday or swing")] = "",
    param: Annotated[list[str] | None, typer.Option("--param", help="name=value")] = None,
    regimes: Annotated[str, typer.Option("--regimes", help="'suited' or a list")] = "",
    seed: Annotated[int, typer.Option("--seed")] = 1,
) -> None:
    """Register a candidate (variant + symbol) at `research`."""
    from tradeagent.cli import _parse_params
    from tradeagent.registry.registry import register
    from tradeagent.strategies.variants import VariantSpec

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
        entry = register(conn, cfg, spec, symbol, seed)
    except (KeyError, ValueError) as exc:
        raise _fail(f"Cannot register: {exc}") from exc
    finally:
        conn.close()
    typer.echo(f"{entry.strategy_id}: {entry.status}")


@gate_app.command("status")
def status_cmd() -> None:
    """Every registered strategy and its step on the ladder."""
    from tradeagent.registry.registry import list_entries

    _, conn = _db()
    try:
        rows = list_entries(conn)
    finally:
        conn.close()
    if not rows:
        typer.echo("No strategies registered (`tradeagent gate register --strategy NAME`).")
    for r in rows:
        typer.echo(f"{r['strategy_id']:<52} {r['status']:<11} updated {r['updated_at']}")


@gate_app.command("show")
def show_cmd(strategy_id: Annotated[str, typer.Argument()]) -> None:
    """One strategy's gate history: every check, its requirement and its value."""
    from tradeagent.registry.gates import gate_history, oos_touches
    from tradeagent.registry.registry import RegistryError, get

    _, conn = _db()
    try:
        try:
            entry = get(conn, strategy_id)
        except RegistryError as exc:
            raise _fail(str(exc)) from exc
        typer.echo(f"{entry.strategy_id}: {entry.status} (seed {entry.seed})")
        for g in gate_history(conn, strategy_id):
            verdict = "PASS" if g["passed"] else "FAIL"
            typer.echo(f"{g['check_id']} {g['from_status']} -> {g['to_status']}: {verdict}")
            for c in g["checks"]:
                mark = "ok  " if c["ok"] else "FAIL"
                typer.echo(f"   {mark} {c['name']}: {c['actual']} (needs {c['required']})")
        touches = oos_touches(conn, strategy_id)
        typer.echo(f"out-of-sample touches: {len(touches)} {' '.join(touches)}")
    finally:
        conn.close()


@gate_app.command("check")
def check_cmd(
    strategy_id: Annotated[str, typer.Argument()],
    touch_oos: Annotated[
        bool,
        typer.Option(
            "--touch-oos", help="Allow the OOS gate to open out-of-sample data (once per candidate)"
        ),
    ] = False,
    override_reason: Annotated[
        str | None,
        typer.Option("--override-reason", help="Human reason for a second OOS touch"),
    ] = None,
) -> None:
    """Run the next automatic gate (research -> candidate -> validated -> oos_passed ->
    paper). Backtests use enforced account limits; results go to gate_checks."""
    from tradeagent.config import project_path
    from tradeagent.data.store import BarStore
    from tradeagent.registry.gates import GateError, check_next
    from tradeagent.registry.registry import RegistryError

    cfg, conn = _db()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    try:
        result = check_next(
            conn,
            cfg,
            store,
            strategy_id,
            project_path(Path("data/backtests")),
            touch_oos=touch_oos,
            override_reason=override_reason,
        )
    except (GateError, RegistryError) as exc:
        raise _fail(f"Gate not run: {exc}") from exc
    finally:
        conn.close()
    verdict = "PASS" if result.passed else "FAIL"
    typer.echo(f"{result.from_status} -> {result.to_status}: {verdict}")
    for c in result.checks:
        typer.echo(f"   {'ok  ' if c.ok else 'FAIL'} {c.name}: {c.actual} (needs {c.required})")


@gate_app.command("approve")
def approve_cmd(
    strategy_id: Annotated[str, typer.Argument()],
    by: Annotated[str, typer.Option("--by", help="Your name (human approval)")],
    reason: Annotated[str, typer.Option("--reason", help="Why, with the evidence you checked")],
    evidence: Annotated[str, typer.Option("--evidence", help="Link or file")] = "",
) -> None:
    """HUMAN step: paper -> approved, approved -> production (recorded in approvals)."""
    from tradeagent.registry.registry import RegistryError, approve

    _, conn = _db()
    try:
        new = approve(conn, strategy_id, by, reason, evidence)
    except RegistryError as exc:
        raise _fail(f"Not approved: {exc}") from exc
    finally:
        conn.close()
    typer.echo(f"{strategy_id}: {new} (approved by {by})")


@gate_app.command("retire")
def retire_cmd(
    strategy_id: Annotated[str, typer.Argument()],
    by: Annotated[str, typer.Option("--by")],
    reason: Annotated[str, typer.Option("--reason")],
) -> None:
    """HUMAN step: retire a strategy from any status."""
    from tradeagent.registry.registry import RegistryError, retire

    _, conn = _db()
    try:
        retire(conn, strategy_id, by, reason)
    except RegistryError as exc:
        raise _fail(f"Not retired: {exc}") from exc
    finally:
        conn.close()
    typer.echo(f"{strategy_id}: retired")
