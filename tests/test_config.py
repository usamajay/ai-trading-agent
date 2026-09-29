import shutil
from pathlib import Path
from typing import Any

import pydantic
import pytest
import yaml

from tradeagent.config import DEFAULT_CONFIG_DIR, ConfigError, load_config


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A throwaway copy of the real config/ folder that tests may break."""
    dest = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, dest)
    return dest


def edit(config_dir: Path, name: str, **changes: Any) -> None:
    path = config_dir / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_real_config_loads() -> None:
    cfg = load_config()
    assert cfg.settings.mode == "paper"
    assert cfg.live.enabled is False
    assert cfg.risk.min_reward_risk == 2.0
    assert cfg.settings.symbols["XAUUSD"] == "XAUUSDm"
    assert len(cfg.config_hash) == 64


def test_config_is_read_only() -> None:
    cfg = load_config()
    with pytest.raises(pydantic.ValidationError):
        cfg.risk.risk_per_trade_pct = 5.0  # type: ignore[misc]


def test_hash_is_stable_and_ignores_comments(config_dir: Path) -> None:
    before = load_config(config_dir).config_hash
    path = config_dir / "risk.yaml"
    path.write_text("# extra comment\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_config(config_dir).config_hash == before


def test_hash_changes_when_a_value_changes(config_dir: Path) -> None:
    before = load_config(config_dir).config_hash
    edit(config_dir, "settings", history_years=4)
    assert load_config(config_dir).config_hash != before


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("risk", {"risk_per_trade_pct": -0.5}),  # negative risk
        ("risk", {"risk_per_trade_pct": 3.0}),  # bigger than the daily loss limit
        ("risk", {"sl_atr_min": 4.0}),  # min stop wider than max stop
        ("risk", {"correlation_limit": 1.5}),
        ("risk", {"typo_limit": 1}),  # unknown key
        ("settings", {"mode": "pepar"}),
        ("settings", {"mode": "live"}),  # live while live.yaml is locked
        ("settings", {"timeframes": ["M5", "M7"]}),
        ("settings", {"history_years": 0}),
        ("settings", {"data_splits": {"train": 0.7, "validation": 0.2, "out_of_sample": 0.2}}),
        ("live", {"enabled": True, "account_login": None}),
    ],
)
def test_bad_values_are_rejected(config_dir: Path, name: str, changes: dict[str, Any]) -> None:
    edit(config_dir, name, **changes)
    with pytest.raises(ConfigError):
        load_config(config_dir)


def test_missing_key_is_rejected(config_dir: Path) -> None:
    path = config_dir / "risk.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["max_daily_loss_pct"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="max_daily_loss_pct"):
        load_config(config_dir)


def test_missing_file_is_rejected(config_dir: Path) -> None:
    (config_dir / "live.yaml").unlink()
    with pytest.raises(ConfigError, match="not found"):
        load_config(config_dir)
