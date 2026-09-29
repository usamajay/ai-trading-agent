"""Read-only MetaTrader 5 client (SPEC §2.3, §9; CLAUDE.md safety rule 1).

- Refuses to run unless the connected account is DEMO. A REAL account is
  refused in every mode until the SPEC §9 live checks exist (Phase 10).
- Has no order functions: placing orders lives only in `execution/`, and
  tests/test_no_orders_outside_execution.py enforces it.
- Returns bars in UTC. Exness server time is UTC (checked 2026-09-29, see
  docs/DECISIONS.md), so MT5 bar timestamps are used as UTC directly.
"""

import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Self

import pandas as pd
from dotenv import dotenv_values
from loguru import logger

from tradeagent.config import PROJECT_ROOT, Mode
from tradeagent.timeutil import ensure_utc, utc_now

# Our timeframe name -> (MetaTrader5 constant name, bar length)
TIMEFRAMES: dict[str, tuple[str, timedelta]] = {
    "M1": ("TIMEFRAME_M1", timedelta(minutes=1)),
    "M5": ("TIMEFRAME_M5", timedelta(minutes=5)),
    "M15": ("TIMEFRAME_M15", timedelta(minutes=15)),
    "H1": ("TIMEFRAME_H1", timedelta(hours=1)),
    "H4": ("TIMEFRAME_H4", timedelta(hours=4)),
    "D1": ("TIMEFRAME_D1", timedelta(days=1)),
}

TIME_DTYPE = pd.DatetimeTZDtype(unit="ns", tz="UTC")
BAR_COLUMNS = ["time_utc", "open", "high", "low", "close", "tick_volume", "spread"]

# account_info().trade_mode values (MetaTrader5 ACCOUNT_TRADE_MODE_*)
TRADE_MODE_NAMES = {0: "DEMO", 1: "CONTEST", 2: "REAL"}
DEMO = 0


class MT5Error(Exception):
    """MT5 terminal/connection problem."""


class AccountRefusedError(MT5Error):
    """The connected account is not one we are allowed to use."""


@dataclass(frozen=True)
class MT5Credentials:
    login: int
    password: str = field(repr=False)  # never printed in logs or tracebacks
    server: str
    path: str | None = None

    @classmethod
    def from_env(cls, env_file: Path = PROJECT_ROOT / ".env") -> Self | None:
        """Read MT5_* values from .env (or real env vars). None if login is not set."""
        values = {**dotenv_values(env_file), **os.environ}
        login = (values.get("MT5_LOGIN") or "").strip()
        if not login:
            return None
        try:
            return cls(
                login=int(login),
                password=values.get("MT5_PASSWORD") or "",
                server=values.get("MT5_SERVER") or "",
                path=values.get("MT5_PATH") or None,
            )
        except ValueError as exc:
            raise MT5Error("MT5_LOGIN in .env must be a number") from exc


@dataclass(frozen=True)
class AccountSummary:
    login: int
    server: str
    trade_mode: str
    balance: float
    equity: float
    currency: str
    leverage: int


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    digits: int
    point: float  # smallest price step
    tick_size: float
    tick_value: float  # account-currency value of one tick_size move for 1 lot
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    spread_points: int  # current spread
    typical_spread_points: float  # median over the last day of M1 bars


@dataclass(frozen=True)
class Tick:
    time_utc: datetime
    bid: float
    ask: float


def check_account_allowed(trade_mode: int, mode: Mode) -> None:
    """Raise AccountRefusedError unless this account may be used in this mode."""
    name = TRADE_MODE_NAMES.get(trade_mode, f"UNKNOWN({trade_mode})")
    if trade_mode == DEMO:
        return
    if name == "REAL" and mode != "live":
        raise AccountRefusedError(f"Refusing REAL account: mode is '{mode}'. Use the demo account.")
    if name == "REAL":
        raise AccountRefusedError(
            "Refusing REAL account: live-mode checks (SPEC §9) are not built until Phase 10."
        )
    raise AccountRefusedError(f"Refusing {name} account: only DEMO accounts are allowed.")


class MT5Client:
    """Read-only access to one MT5 terminal. Use as `with MT5Client() as c: ...`."""

    def __init__(
        self,
        mode: Mode = "paper",
        credentials: MT5Credentials | None = None,
        mt5_module: ModuleType | Any | None = None,
    ) -> None:
        if mt5_module is None:
            import MetaTrader5  # Windows-only package

            mt5_module = MetaTrader5
        self._mt5: Any = mt5_module
        self._mode = mode
        self._credentials = credentials
        self._connected = False
        self._account: AccountSummary | None = None

    @property
    def account(self) -> AccountSummary:
        """The account checked by the last successful connect()."""
        if self._account is None:
            raise MT5Error("Not connected; call connect() first")
        return self._account

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def connect(self) -> AccountSummary:
        """Start/attach to the terminal, then refuse anything but a DEMO account."""
        creds = self._credentials
        if creds is None:
            # No .env login: attach to the terminal that is already logged in.
            ok = self._mt5.initialize()
        else:
            kwargs: dict[str, Any] = {
                "login": creds.login,
                "password": creds.password,
                "server": creds.server,
            }
            if creds.path:
                kwargs["path"] = creds.path
            ok = self._mt5.initialize(**kwargs)
        if not ok:
            raise MT5Error(f"MT5 initialize failed: {self._mt5.last_error()}")
        self._connected = True

        info = self._mt5.account_info()
        if info is None:
            self.close()
            raise MT5Error(f"Could not read account info: {self._mt5.last_error()}")
        account = AccountSummary(
            login=int(info.login),
            server=str(info.server),
            trade_mode=TRADE_MODE_NAMES.get(info.trade_mode, f"UNKNOWN({info.trade_mode})"),
            balance=float(info.balance),
            equity=float(info.equity),
            currency=str(info.currency),
            leverage=int(info.leverage),
        )
        logger.info(
            "MT5 account: {} login {} on {}", account.trade_mode, account.login, account.server
        )

        try:
            check_account_allowed(int(info.trade_mode), self._mode)
            if creds is not None and account.login != creds.login:
                raise AccountRefusedError(
                    f"Terminal is logged in to {account.login}, not MT5_LOGIN from .env"
                )
        except AccountRefusedError:
            self.close()
            raise
        self._account = account
        return account

    def close(self) -> None:
        if self._connected:
            self._mt5.shutdown()
            self._connected = False
        self._account = None

    def _require_connection(self) -> None:
        if not self._connected:
            raise MT5Error("Not connected; call connect() first")

    def _select(self, symbol: str) -> None:
        if not self._mt5.symbol_select(symbol, True):
            raise MT5Error(f"Symbol {symbol} not available: {self._mt5.last_error()}")

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        closed_only: bool = True,
    ) -> pd.DataFrame:
        """Bars with open time in [start, end], as a DataFrame with UTC `time_utc`.

        With closed_only (default) the still-forming latest bar is dropped.
        """
        self._require_connection()
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Unknown timeframe {timeframe!r}; use one of {list(TIMEFRAMES)}")
        const_name, bar_length = TIMEFRAMES[timeframe]
        start_utc, end_utc = ensure_utc(start), ensure_utc(end)
        self._select(symbol)

        rates = self._mt5.copy_rates_range(
            symbol, getattr(self._mt5, const_name), start_utc, end_utc
        )
        if rates is None:
            raise MT5Error(
                f"copy_rates_range failed for {symbol} {timeframe}: {self._mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        if df.empty:
            return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}).astype(
                {"time_utc": TIME_DTYPE, "tick_volume": "int64", "spread": "int64"}
            )
        df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True).astype(TIME_DTYPE)
        df = df[BAR_COLUMNS].astype({"tick_volume": "int64", "spread": "int64"})
        # MT5 returns a stray bar from outside the range when asked for dates older
        # than its history, so keep only what was asked for.
        df = df[(df["time_utc"] >= start_utc) & (df["time_utc"] <= end_utc)]
        if closed_only:
            df = df[df["time_utc"] + bar_length <= utc_now()]
        return df.sort_values("time_utc").reset_index(drop=True)

    def last_tick(self, symbol: str, wait_seconds: float = 2.0) -> Tick | None:
        """Latest bid/ask. A just-selected symbol can take a moment to get its first tick."""
        self._require_connection()
        self._select(symbol)
        deadline = time.monotonic() + wait_seconds
        while True:
            tick = self._mt5.symbol_info_tick(symbol)
            if tick is not None and tick.time > 0:
                return Tick(
                    time_utc=datetime.fromtimestamp(tick.time, UTC),
                    bid=float(tick.bid),
                    ask=float(tick.ask),
                )
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.2)

    def symbol_info(self, symbol: str) -> SymbolSpec:
        """Contract specs used for sizing and cost estimates."""
        self._require_connection()
        self._select(symbol)
        info = self._mt5.symbol_info(symbol)
        if info is None:
            raise MT5Error(f"symbol_info failed for {symbol}: {self._mt5.last_error()}")

        # The live spread can read 0 right after selecting a symbol, so also keep the
        # median of the last day of M1 bars as the "typical" spread.
        recent = self._mt5.copy_rates_from_pos(symbol, self._mt5.TIMEFRAME_M1, 0, 1440)
        if recent is not None and len(recent) > 0:
            typical = float(pd.DataFrame(recent)["spread"].median())
        else:
            typical = float(info.spread)

        return SymbolSpec(
            symbol=symbol,
            digits=int(info.digits),
            point=float(info.point),
            tick_size=float(info.trade_tick_size),
            tick_value=float(info.trade_tick_value),
            contract_size=float(info.trade_contract_size),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            spread_points=int(info.spread),
            typical_spread_points=typical,
        )
