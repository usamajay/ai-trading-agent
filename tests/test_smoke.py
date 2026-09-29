import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    import tradeagent

    assert tradeagent.__version__


def test_live_trading_locked():
    live = yaml.safe_load((ROOT / "config" / "live.yaml").read_text())
    assert live["enabled"] is False


def test_default_mode_is_paper():
    settings = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())
    assert settings["mode"] == "paper"
