"""Kill-switch and risk-status commands, run in a temporary folder and database."""

import functools
from pathlib import Path

import pytest
from typer.testing import CliRunner

import tradeagent.config as config_module
import tradeagent.risk.killswitch as killswitch_module
from tradeagent.cli import app
from tradeagent.config import StorageSettings, load_config
from tradeagent.data.store import connect_db
from tradeagent.risk.killswitch import KillSwitch


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Kill file and SQLite database inside tmp_path, never the real ones."""
    kill_file = tmp_path / "KILL"
    monkeypatch.setattr(killswitch_module, "KillSwitch", functools.partial(KillSwitch, kill_file))
    real = load_config()
    storage = StorageSettings(
        bars_dir=real.settings.storage.bars_dir, sqlite_path=tmp_path / "test.db"
    )
    settings = real.settings.model_copy(update={"storage": storage})
    patched = real.model_copy(update={"settings": settings})
    monkeypatch.setattr(config_module, "load_config", lambda *a, **k: patched)
    return tmp_path


def test_kill_activate_status_clear(sandbox: Path) -> None:
    runner = CliRunner()
    assert "off" in runner.invoke(app, ["kill", "--status"]).output
    missing = runner.invoke(app, ["kill"])
    assert missing.exit_code == 1 and not (sandbox / "KILL").exists()

    on = runner.invoke(app, ["kill", "--reason", "drill"])
    assert on.exit_code == 0 and (sandbox / "KILL").exists()
    assert "ON" in runner.invoke(app, ["kill", "--status"]).output
    assert "Kill switch: ON" in runner.invoke(app, ["risk", "status"]).output

    off = runner.invoke(app, ["kill", "--clear", "--reason", "drill over"])
    assert off.exit_code == 0 and not (sandbox / "KILL").exists()
    conn = connect_db(sandbox / "test.db")
    rules = [r[0] for r in conn.execute("SELECT rule FROM risk_events ORDER BY id")]
    conn.close()
    assert rules == ["kill_switch: drill", "kill_switch_cleared: drill over"]


def test_risk_status_shows_limits_and_no_state(sandbox: Path) -> None:
    out = CliRunner().invoke(app, ["risk", "status"]).output
    assert "risk per trade 0.5%" in out and "max drawdown 10.0%" in out
    assert "none saved yet" in out
