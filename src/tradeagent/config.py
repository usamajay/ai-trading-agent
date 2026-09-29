"""Load and validate config/*.yaml (SPEC §6, §9, §13).

Every file maps to a frozen pydantic model: unknown keys, missing keys and
out-of-range values fail loudly at startup, and nothing can change a limit
while the program runs. `config_hash` fingerprints the loaded config so every
result row can record exactly which settings produced it.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"

Mode = Literal["research", "paper", "live"]
Timeframe = Literal["M1", "M5", "M15", "H1", "H4", "D1"]
M = TypeVar("M", bound=BaseModel)


class ConfigError(Exception):
    """Raised when a config file is missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StorageSettings(_Strict):
    bars_dir: Path
    sqlite_path: Path


class EntryRules(_Strict):
    min_ev_r: float = Field(ge=0)
    min_p_win_lower: float = Field(gt=0, lt=1)
    min_sample_trades: int = Field(ge=1)


class DataSplits(_Strict):
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    out_of_sample: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _sum_to_one(self) -> Self:
        total = self.train + self.validation + self.out_of_sample
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"data_splits must add up to 1.0, got {total}")
        return self


class Settings(_Strict):
    mode: Mode
    timezone_display: str
    symbols: dict[str, str] = Field(min_length=1)  # internal name -> broker symbol
    timeframes: list[Timeframe] = Field(min_length=1)
    history_years: int = Field(ge=1, le=20)
    storage: StorageSettings
    entry_rules: EntryRules
    data_splits: DataSplits


class RiskLimits(_Strict):
    risk_per_trade_pct: float = Field(gt=0, le=100)
    max_daily_loss_pct: float = Field(gt=0, le=100)
    max_weekly_loss_pct: float = Field(gt=0, le=100)
    max_drawdown_pct: float = Field(gt=0, le=100)
    max_open_positions: int = Field(ge=1)
    max_open_per_symbol: int = Field(ge=1)
    max_consecutive_losses: int = Field(ge=1)
    consecutive_loss_pause_hours: float = Field(gt=0)
    min_reward_risk: float = Field(gt=0)
    sl_atr_min: float = Field(gt=0)
    sl_atr_max: float = Field(gt=0)
    max_spread_multiple: float = Field(gt=0)
    news_blackout_minutes: int = Field(ge=0)
    correlation_limit: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not (
            self.risk_per_trade_pct
            <= self.max_daily_loss_pct
            <= self.max_weekly_loss_pct
            <= self.max_drawdown_pct
        ):
            raise ValueError("need risk_per_trade <= daily loss <= weekly loss <= max drawdown")
        if self.max_open_per_symbol > self.max_open_positions:
            raise ValueError("max_open_per_symbol cannot exceed max_open_positions")
        if self.sl_atr_min >= self.sl_atr_max:
            raise ValueError("sl_atr_min must be smaller than sl_atr_max")
        return self


class LiveLock(_Strict):
    enabled: bool
    account_login: int | None

    @model_validator(mode="after")
    def _login_when_enabled(self) -> Self:
        if self.enabled and self.account_login is None:
            raise ValueError("live.enabled is true but account_login is not set")
        return self


class AppConfig(_Strict):
    settings: Settings
    risk: RiskLimits
    live: LiveLock
    config_hash: str

    @model_validator(mode="after")
    def _live_mode_needs_unlock(self) -> Self:
        # Only the first of the SPEC §9 live checks; the rest run at startup.
        if self.settings.mode == "live" and not self.live.enabled:
            raise ValueError("mode is 'live' but config/live.yaml has enabled: false")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain key: value settings")
    return data


def compute_config_hash(raw: dict[str, Any]) -> str:
    """sha256 of the config in a canonical form (key order and comments don't matter)."""
    canonical = json.dumps(raw, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_file(path: Path, model: type[M]) -> M:
    try:
        return model.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"Invalid {path.name}:\n{exc}") from exc


def load_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> AppConfig:
    """Load settings.yaml, risk.yaml and live.yaml; raise ConfigError if anything is wrong."""
    settings = _load_file(config_dir / "settings.yaml", Settings)
    risk = _load_file(config_dir / "risk.yaml", RiskLimits)
    live = _load_file(config_dir / "live.yaml", LiveLock)

    config_hash = compute_config_hash(
        {
            "settings": settings.model_dump(mode="json"),
            "risk": risk.model_dump(mode="json"),
            "live": live.model_dump(mode="json"),
        }
    )
    try:
        return AppConfig(settings=settings, risk=risk, live=live, config_hash=config_hash)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config combination:\n{exc}") from exc
