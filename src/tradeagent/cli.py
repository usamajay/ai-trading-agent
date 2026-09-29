"""Command-line entry point. Commands are added phase by phase."""

import typer

app = typer.Typer(help="AI Trading Agent (paper mode by default).")


@app.command()
def version() -> None:
    """Print the version."""
    from tradeagent import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
