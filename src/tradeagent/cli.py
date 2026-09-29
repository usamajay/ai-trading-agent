"""Command-line entry point. Commands are added phase by phase."""

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


if __name__ == "__main__":
    app()
