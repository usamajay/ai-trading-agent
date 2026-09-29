"""Command-line entry point. Commands are added phase by phase."""

from typing import Annotated

import typer

app = typer.Typer(help="AI Trading Agent (paper mode by default).", no_args_is_help=True)
data_app = typer.Typer(help="Market data commands (read-only).", no_args_is_help=True)
app.add_typer(data_app, name="data")


@app.callback()
def main() -> None:
    """AI Trading Agent (paper mode by default). See docs/SPEC.md."""
    # A callback keeps Typer in "group" mode, so `tradeagent --help` lists all
    # commands even while there is only one.


@app.command()
def version() -> None:
    """Print the version."""
    from tradeagent import __version__

    typer.echo(__version__)


@app.command("check-config")
def check_config() -> None:
    """Load and validate config/*.yaml, then print a short summary."""
    from tradeagent.config import ConfigError, load_config

    try:
        cfg = load_config()
    except ConfigError as exc:
        typer.secho(f"Config problem: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho("Config OK", fg=typer.colors.GREEN)
    typer.echo(f"  mode:          {cfg.settings.mode}")
    typer.echo(f"  live enabled:  {cfg.live.enabled}")
    typer.echo(f"  symbols:       {', '.join(cfg.settings.symbols.values())}")
    typer.echo(f"  risk/trade:    {cfg.risk.risk_per_trade_pct}%")
    typer.echo(f"  min RR:        {cfg.risk.min_reward_risk}")
    typer.echo(f"  config_hash:   {cfg.config_hash[:12]}")


@data_app.command("ping")
def data_ping() -> None:
    """Connect to MT5 (demo only), show the account and the latest price per symbol."""
    from datetime import timedelta

    from tradeagent.config import load_config
    from tradeagent.data.mt5_client import MT5Client, MT5Credentials, MT5Error
    from tradeagent.timeutil import fmt_utc_pkt, utc_now

    cfg = load_config()
    try:
        with MT5Client(mode=cfg.settings.mode, credentials=MT5Credentials.from_env()) as client:
            account = client.account
            typer.secho(f"Account type: {account.trade_mode}", fg=typer.colors.GREEN)
            typer.echo(f"Login:        {account.login} ({account.server})")
            typer.echo(f"Balance:      {account.balance:,.2f} {account.currency}")
            typer.echo(f"Equity:       {account.equity:,.2f} {account.currency}")
            for symbol in cfg.settings.symbols.values():
                tick = client.last_tick(symbol)
                if tick is not None:
                    typer.echo(
                        f"{symbol:<9} bid {tick.bid}  ask {tick.ask}  at {fmt_utc_pkt(tick.time_utc)}"
                    )
                    continue
                now = utc_now()
                bars = client.get_bars(symbol, "M1", now - timedelta(days=7), now)
                if bars.empty:
                    typer.echo(f"{symbol:<9} no price available")
                else:
                    last = bars.iloc[-1]
                    typer.echo(
                        f"{symbol:<9} last M1 close {last['close']}  "
                        f"at {fmt_utc_pkt(last['time_utc'].to_pydatetime())} (no live tick)"
                    )
    except MT5Error as exc:
        typer.secho(f"MT5 problem: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@data_app.command("fetch")
def data_fetch(
    symbol: Annotated[
        list[str] | None,
        typer.Option("--symbol", "-s", help="Internal symbol, e.g. XAUUSD (default: all)."),
    ] = None,
    timeframe: Annotated[
        list[str] | None,
        typer.Option("--timeframe", "-t", help="e.g. M5 (default: all in settings)."),
    ] = None,
) -> None:
    """Download missing history from MT5 into data/bars (incremental)."""
    from tradeagent.config import load_config, project_path
    from tradeagent.data.historical import fetch_history, history_start
    from tradeagent.data.mt5_client import MT5Client, MT5Credentials, MT5Error
    from tradeagent.data.store import BarStore, connect_db
    from tradeagent.timeutil import utc_now

    cfg = load_config()
    symbols = cfg.settings.symbols
    wanted_symbols = symbol or list(symbols)
    wanted_tfs = timeframe or list(cfg.settings.timeframes)
    unknown = [s for s in wanted_symbols if s not in symbols] + [
        t for t in wanted_tfs if t not in cfg.settings.timeframes
    ]
    if unknown:
        typer.secho(f"Unknown symbol/timeframe: {unknown}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    connect_db(project_path(cfg.settings.storage.sqlite_path)).close()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    try:
        with MT5Client(mode=cfg.settings.mode, credentials=MT5Credentials.from_env()) as client:
            now = utc_now()
            for name in wanted_symbols:
                for tf in wanted_tfs:
                    start = history_start(tf, cfg.settings, now)
                    r = fetch_history(client, store, name, symbols[name], tf, start, now)
                    note = ""
                    if r.first is not None and r.first > start + (now - start) * 0.02:
                        note = "  (broker history starts later than wanted)"
                    typer.echo(
                        f"{name:<7} {tf:<4} +{r.new_bars:>7} new  "
                        f"{_fmt_day(r.first)} -> {_fmt_day(r.last)}{note}"
                    )
    except MT5Error as exc:
        typer.secho(f"MT5 problem: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@data_app.command("summary")
def data_summary() -> None:
    """Show stored bar counts and date ranges (UTC)."""
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import BarStore

    cfg = load_config()
    table = BarStore(project_path(cfg.settings.storage.bars_dir)).summary()
    if table.empty:
        typer.echo("No bars stored yet. Run: uv run tradeagent data fetch")
        return
    typer.echo(f"{'symbol':<7} {'tf':<4} {'bars':>9}  first (UTC)        last (UTC)")
    for row in table.itertuples():
        typer.echo(
            f"{row.symbol:<7} {row.timeframe:<4} {row.bars:>9,}  "
            f"{row.first_utc:%Y-%m-%d %H:%M}   {row.last_utc:%Y-%m-%d %H:%M}"
        )


@data_app.command("validate")
def data_validate(
    symbol: Annotated[
        list[str] | None,
        typer.Option("--symbol", "-s", help="Internal symbol, e.g. XAUUSD (default: all)."),
    ] = None,
    timeframe: Annotated[
        list[str] | None,
        typer.Option("--timeframe", "-t", help="e.g. M5 (default: all in settings)."),
    ] = None,
    examples: Annotated[
        int, typer.Option("--examples", "-n", help="Worst examples to show per issue type.")
    ] = 3,
    save: Annotated[
        bool, typer.Option("--save/--no-save", help="Write issues to data_quality_log.")
    ] = True,
) -> None:
    """Check stored bars for gaps, spikes, duplicates, bad prices and spread outliers."""
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import BarStore, connect_db
    from tradeagent.data.validation import ISSUE_TYPES, log_issues, validate_bars
    from tradeagent.provenance import git_commit, new_run_id
    from tradeagent.timeutil import fmt_utc_pkt

    cfg = load_config()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    run_id, commit = new_run_id(), git_commit()
    short = {
        "duplicate": "dup",
        "not_monotonic": "order",
        "ohlc_invalid": "ohlc",
        "misaligned": "align",
        "weekend_bar": "wkndbar",
        "gap_intraday": "gap!",
        "gap_holiday": "holiday",
        "spike": "spike",
        "spread_zero": "spr=0",
        "spread_outlier": "spr>10x",
    }

    results = []
    conn = connect_db(project_path(cfg.settings.storage.sqlite_path)) if save else None
    for name in symbol or list(cfg.settings.symbols):
        for tf in timeframe or list(cfg.settings.timeframes):
            bars = store.read(name, tf)
            if bars.empty:
                continue
            result = validate_bars(bars, name, tf)
            results.append(result)
            if conn is not None:
                log_issues(conn, result, run_id, commit, cfg.config_hash)
    if conn is not None:
        conn.close()

    typer.echo(f"Validation run {run_id}  (commit {commit})")
    typer.echo("\nISSUES (need review; '.' = none)")
    typer.echo(
        f"{'symbol':<7}{'tf':<5}{'bars':>9} " + "".join(f"{short[t]:>8}" for t in ISSUE_TYPES)
    )
    for r in results:
        counts = r.counts()
        typer.echo(
            f"{r.symbol:<7}{r.timeframe:<5}{r.bars:>9,} "
            + "".join(f"{counts[t] or '.':>8}" for t in ISSUE_TYPES)
        )

    typer.echo("\nEXPECTED market behaviour (counted, not logged)")
    typer.echo(
        f"{'symbol':<7}{'tf':<5}{'weekend gaps':>14}{'daily breaks':>14}{'Sunday stubs':>14}"
    )
    for r in results:
        e = r.expected
        typer.echo(
            f"{r.symbol:<7}{r.timeframe:<5}{e['gap_weekend']:>14}"
            f"{e['gap_daily_break']:>14}{e['sunday_stub']:>14}"
        )

    if examples > 0:
        typer.echo(f"\nWORST EXAMPLES (up to {examples} per type)")
        for r in results:
            for issue_type, group in r.issues.groupby("issue_type"):
                typer.echo(f"  {r.symbol} {r.timeframe} {issue_type} ({len(group)}):")
                worst = group.nlargest(examples, "severity")
                for time_utc, details in zip(worst["time_utc"], worst["details"], strict=True):
                    typer.echo(f"      {fmt_utc_pkt(time_utc.to_pydatetime())}  {details}")
    if save:
        typer.echo(f"\nSaved to data_quality_log with run_id {run_id}")


def _fmt_day(ts: object) -> str:
    return "-" if ts is None else f"{ts:%Y-%m-%d}"


if __name__ == "__main__":
    app()
