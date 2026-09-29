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


if __name__ == "__main__":
    app()
