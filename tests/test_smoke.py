from pathlib import Path

import yaml
from typer.testing import CliRunner

from tradeagent.cli import app

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports() -> None:
    import tradeagent

    assert tradeagent.__version__


def test_cli_help_lists_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_live_trading_locked() -> None:
    live = yaml.safe_load((ROOT / "config" / "live.yaml").read_text())
    assert live["enabled"] is False


def test_default_mode_is_paper() -> None:
    settings = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())
    assert settings["mode"] == "paper"
