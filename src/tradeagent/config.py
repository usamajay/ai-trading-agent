"""Load and validate config/*.yaml (SPEC §6, §9, §13).

Every file maps to a frozen pydantic model: unknown keys, missing keys and
out-of-range values fail loudly at startup, and nothing can change a limit
while the program runs. `config_hash` fingerprints the loaded config so every
result row can record exactly which settings produced it.
"""

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Self, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
SPLITS_FILE = "splits.yaml"
COSTS_FILE = "costs.yaml"

Mode = Literal["research", "paper", "live"]
Timeframe = Literal["M1", "M5", "M15", "H1", "H4", "D1"]
M = TypeVar("M", bound=BaseModel)


def project_path(path: Path) -> Path:
    """Resolve a config path such as `data/bars` against the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    prior_strength: float = Field(gt=0)
    credible_level: float = Field(gt=0, lt=1)


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


class BacktestSettings(_Strict):
    embargo_weeks: int = Field(ge=0, le=12)
    max_hole_minutes: int = Field(ge=1)
    no_entry_minutes_after_open: int = Field(ge=0, le=240)
    friday_cutoff_ny: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    exclusions_source_timeframe: Timeframe
    spread_margin_multiple: float = Field(ge=1.0, le=5.0)
    spread_margin_points: float = Field(ge=0)
    slippage_spread_multiple: float = Field(ge=0, le=5.0)
    commission_per_lot_usd: float = Field(ge=0)  # round turn (open + close)
    starting_balance: float = Field(gt=0)  # account currency
    cost_stress_multiple: float = Field(ge=1.0, le=5.0)  # stress re-run: spread x this

    @property
    def friday_cutoff_minutes(self) -> int:
        hours, minutes = self.friday_cutoff_ny.split(":")
        return int(hours) * 60 + int(minutes)


class Settings(_Strict):
    mode: Mode
    timezone_display: str
    symbols: dict[str, str] = Field(min_length=1)  # internal name -> broker symbol
    timeframes: list[Timeframe] = Field(min_length=1)
    history_years: int = Field(ge=1, le=20)
    m1_history_months: int = Field(ge=1, le=240)
    storage: StorageSettings
    entry_rules: EntryRules
    data_splits: DataSplits
    backtest: BacktestSettings


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


class TimeWindow(_Strict):
    """A reviewed time window in UTC, [start, end). `symbols` None means all symbols."""

    start: datetime
    end: datetime
    symbols: list[str] | None = None
    reason: str = Field(min_length=1)
    decision: date  # date of the docs/DECISIONS.md entry that approved it

    @field_validator("start", "end")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("times must include a timezone, e.g. 2025-06-19T08:50:00Z")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"window ends before it starts: {self.start} -> {self.end}")
        return self

    def applies_to(self, symbol: str) -> bool:
        return self.symbols is None or symbol in self.symbols


class DataExclusions(_Strict):
    """config/data_exclusions.yaml: reviewed data decisions (docs/DATA_NOTES.md §5)."""

    excluded_windows: list[TimeWindow]  # no signals or open trades across these
    no_trade_windows: list[TimeWindow]  # no new entries; open trades managed normally
    keep_gaps: list[TimeWindow]  # unexpected gaps inside these count as normal closures


class SplitPeriod(_Strict):
    """[start, end) in whole UTC days."""

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"period ends before it starts: {self.start} -> {self.end}")
        return self


SplitName = Literal["train", "validation", "out_of_sample"]


class SplitDates(_Strict):
    """config/splits.yaml: fixed, human-approved split dates (SPEC §7.2)."""

    approved_by: str = Field(min_length=1)
    approved_on: date
    train: SplitPeriod
    validation: SplitPeriod
    out_of_sample: SplitPeriod

    @model_validator(mode="after")
    def _chronological(self) -> Self:
        if not (
            self.train.end <= self.validation.start
            and self.validation.end <= self.out_of_sample.start
        ):
            raise ValueError("splits must be in order: train, validation, out_of_sample")
        return self

    def period(self, split: SplitName) -> SplitPeriod:
        periods = {
            "train": self.train,
            "validation": self.validation,
            "out_of_sample": self.out_of_sample,
        }
        return periods[split]


class SymbolCosts(_Strict):
    """Contract and swap values for one symbol, copied from MT5 symbol_info."""

    broker_symbol: str
    digits: int = Field(ge=0)
    point: float = Field(gt=0)
    tick_size: float = Field(gt=0)
    tick_value: float = Field(gt=0)  # account currency per tick_size move, 1 lot
    contract_size: float = Field(gt=0)
    volume_min: float = Field(gt=0)
    volume_step: float = Field(gt=0)
    volume_max: float = Field(gt=0)
    swap_mode: Literal[1]  # only "swap in points" is supported (both symbols use it)
    swap_long: float  # points per lot per night (negative = you pay)
    swap_short: float
    swap_rollover3days: int = Field(ge=0, le=7)  # MT5: 0=Sunday..6=Saturday, 7=none

    @property
    def value_per_point_per_lot(self) -> float:
        """Account currency gained/lost per 1-point price move with 1 lot."""
        return self.tick_value * self.point / self.tick_size

    @property
    def triple_swap_weekday(self) -> int | None:
        """Python weekday (Monday=0) of the x3 swap rollover, or None if there is none."""
        if self.swap_rollover3days == 7:
            return None
        return (self.swap_rollover3days - 1) % 7


class CostSnapshot(_Strict):
    """config/costs.yaml: a dated copy of broker costs so backtests are repeatable."""

    taken_utc: datetime
    source: str
    symbols: dict[str, SymbolCosts]  # internal symbol -> costs


class AppConfig(_Strict):
    settings: Settings
    risk: RiskLimits
    live: LiveLock
    exclusions: DataExclusions
    splits: SplitDates | None  # None until Usama approves the split dates
    costs: CostSnapshot | None  # None until `tradeagent backtest costs --snapshot`
    config_hash: str

    @model_validator(mode="after")
    def _live_mode_needs_unlock(self) -> Self:
        # Only the first of the SPEC §9 live checks; the rest run at startup.
        if self.settings.mode == "live" and not self.live.enabled:
            raise ValueError("mode is 'live' but config/live.yaml has enabled: false")
        return self

    @model_validator(mode="after")
    def _splits_have_embargo(self) -> Self:
        if self.splits is None:
            return self
        embargo = self.settings.backtest.embargo_weeks * 7
        gaps = [
            (self.splits.validation.start - self.splits.train.end).days,
            (self.splits.out_of_sample.start - self.splits.validation.end).days,
        ]
        if min(gaps) < embargo:
            raise ValueError(f"splits need an embargo of at least {embargo} days, got {gaps}")
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
    """Load config/*.yaml; raise ConfigError if anything is wrong.

    splits.yaml is optional: it only exists once Usama has approved the split dates.
    """
    settings = _load_file(config_dir / "settings.yaml", Settings)
    risk = _load_file(config_dir / "risk.yaml", RiskLimits)
    live = _load_file(config_dir / "live.yaml", LiveLock)
    exclusions = _load_file(config_dir / "data_exclusions.yaml", DataExclusions)
    splits_path = config_dir / SPLITS_FILE
    splits = _load_file(splits_path, SplitDates) if splits_path.is_file() else None
    costs_path = config_dir / COSTS_FILE
    costs = _load_file(costs_path, CostSnapshot) if costs_path.is_file() else None

    parts: dict[str, Any] = {
        "settings": settings.model_dump(mode="json"),
        "risk": risk.model_dump(mode="json"),
        "live": live.model_dump(mode="json"),
        "exclusions": exclusions.model_dump(mode="json"),
    }
    if splits is not None:
        parts["splits"] = splits.model_dump(mode="json")
    if costs is not None:
        parts["costs"] = costs.model_dump(mode="json")
    config_hash = compute_config_hash(parts)
    try:
        return AppConfig(
            settings=settings,
            risk=risk,
            live=live,
            exclusions=exclusions,
            splits=splits,
            costs=costs,
            config_hash=config_hash,
        )
    except ValidationError as exc:
        raise ConfigError(f"Invalid config combination:\n{exc}") from exc
