"""Command-line entry point. Commands are added phase by phase."""

import typer

app = typer.Typer(help="AI Trading Agent (paper mode by default).", no_args_is_help=True)


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


if __name__ == "__main__":
    app()
