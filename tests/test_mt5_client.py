"""MT5 client tests against a fake MetaTrader5 module (no terminal needed)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tradeagent.data.mt5_client import (
    AccountRefusedError,
    MT5Client,
    MT5Credentials,
    MT5Error,
    check_account_allowed,
)

RATE_DTYPE = [
    ("time", "<i8"),
    ("open", "<f8"),
    ("high", "<f8"),
    ("low", "<f8"),
    ("close", "<f8"),
    ("tick_volume", "<u8"),
    ("spread", "<i4"),
    ("real_volume", "<u8"),
]


def make_rates(start: datetime, count: int, step: timedelta) -> np.ndarray:
    rows = []
    for i in range(count):
        t = int((start + i * step).timestamp())
        price = 2000.0 + i
        rows.append((t, price, price + 2, price - 1, price + 1, 100 + i, 200, 0))
    return np.array(rows, dtype=RATE_DTYPE)


class FakeMT5:
    """Just enough of the MetaTrader5 module for the client."""

    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15 = 1, 5, 15
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 16385, 16388, 16408

    def __init__(self, trade_mode: int = 0, login: int = 111, rates: Any = None) -> None:
        self.trade_mode = trade_mode
        self.login = login
        self.rates = rates
        self.initialize_kwargs: dict[str, Any] | None = None
        self.shutdown_called = False
        self.ticks: list[Any] = []

    def initialize(self, **kwargs: Any) -> bool:
        self.initialize_kwargs = kwargs
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def last_error(self) -> tuple[int, str]:
        return (1, "Success")

    def account_info(self) -> SimpleNamespace:
        return SimpleNamespace(
            login=self.login,
            server="Exness-MT5Trial15",
            trade_mode=self.trade_mode,
            balance=10_000.0,
            equity=10_000.0,
            currency="USD",
            leverage=200,
        )

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return symbol != "NOPE"

    def copy_rates_range(self, symbol: str, tf: int, start: datetime, end: datetime) -> Any:
        self.last_range = (symbol, tf, start, end)
        return self.rates

    def copy_rates_from_pos(self, symbol: str, tf: int, pos: int, count: int) -> Any:
        return self.rates

    def symbol_info_tick(self, symbol: str) -> Any:
        return self.ticks.pop(0) if self.ticks else None

    def symbol_info(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(
            digits=3,
            point=0.001,
            trade_tick_size=0.001,
            trade_tick_value=0.1,
            trade_contract_size=100.0,
            volume_min=0.01,
            volume_max=200.0,
            volume_step=0.01,
            spread=0,
        )


CREDS = MT5Credentials(login=111, password="secret", server="Exness-MT5Trial15")


# --- account safety -------------------------------------------------------------


def test_demo_account_is_allowed() -> None:
    fake = FakeMT5(trade_mode=0)
    client = MT5Client(credentials=CREDS, mt5_module=fake)
    account = client.connect()
    assert account.trade_mode == "DEMO"
    assert client.account.login == 111
    assert fake.initialize_kwargs == {"login": 111, "password": "secret", "server": CREDS.server}


@pytest.mark.parametrize("mode", ["research", "paper", "live"])
def test_real_account_is_refused_in_every_mode(mode: Any) -> None:
    fake = FakeMT5(trade_mode=2)
    client = MT5Client(mode=mode, credentials=CREDS, mt5_module=fake)
    with pytest.raises(AccountRefusedError, match="REAL"):
        client.connect()
    assert fake.shutdown_called  # disconnected straight away
    with pytest.raises(MT5Error):
        _ = client.account


def test_contest_and_unknown_accounts_are_refused() -> None:
    for trade_mode in (1, 7):
        with pytest.raises(AccountRefusedError):
            check_account_allowed(trade_mode, "paper")


def test_wrong_login_is_refused() -> None:
    fake = FakeMT5(login=999)
    with pytest.raises(AccountRefusedError, match="999"):
        MT5Client(credentials=CREDS, mt5_module=fake).connect()
    assert fake.shutdown_called


def test_attaches_to_running_terminal_without_credentials() -> None:
    fake = FakeMT5()
    MT5Client(credentials=None, mt5_module=fake).connect()
    assert fake.initialize_kwargs == {}


def test_password_is_hidden_in_repr() -> None:
    assert "secret" not in repr(CREDS)


def test_credentials_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("MT5_LOGIN=123\nMT5_PASSWORD=pw\nMT5_SERVER=srv\nMT5_PATH=\n")
    creds = MT5Credentials.from_env(env)
    assert creds == MT5Credentials(login=123, password="pw", server="srv", path=None)
    assert MT5Credentials.from_env(tmp_path / "missing.env") is None


# --- data -----------------------------------------------------------------------


def test_get_bars_returns_utc_and_drops_forming_bar() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    start = now - timedelta(minutes=4)
    fake = FakeMT5(rates=make_rates(start, 5, timedelta(minutes=1)))  # last bar opens "now"
    with MT5Client(credentials=CREDS, mt5_module=fake) as client:
        bars = client.get_bars("XAUUSDm", "M1", start, now)
        all_bars = client.get_bars("XAUUSDm", "M1", start, now, closed_only=False)

    assert list(bars.columns) == [
        "time_utc",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
    ]
    assert str(bars["time_utc"].dtype) == "datetime64[ns, UTC]"
    assert bars["time_utc"].iloc[0] == pd.Timestamp(start)
    assert len(bars) == 4 and len(all_bars) == 5  # forming bar dropped
    assert fake.last_range[1] == FakeMT5.TIMEFRAME_M1


def test_get_bars_rejects_naive_datetimes_and_bad_timeframes() -> None:
    with MT5Client(credentials=CREDS, mt5_module=FakeMT5(rates=[])) as client:
        with pytest.raises(ValueError, match="timezone"):
            naive = datetime(2026, 1, 1)  # noqa: DTZ001 - deliberately naive
            client.get_bars("XAUUSDm", "M1", naive, naive)
        with pytest.raises(ValueError, match="timeframe"):
            client.get_bars("XAUUSDm", "M7", datetime(2026, 1, 1, tzinfo=UTC), datetime.now(UTC))


def test_get_bars_empty_and_error() -> None:
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)
    with MT5Client(credentials=CREDS, mt5_module=FakeMT5(rates=[])) as client:
        empty = client.get_bars("XAUUSDm", "H1", start, end)
        assert empty.empty and str(empty["time_utc"].dtype) == "datetime64[ns, UTC]"
    with MT5Client(credentials=CREDS, mt5_module=FakeMT5(rates=None)) as client:
        with pytest.raises(MT5Error):
            client.get_bars("XAUUSDm", "H1", start, end)
        with pytest.raises(MT5Error, match="NOPE"):
            client.get_bars("NOPE", "H1", start, end)


def test_calls_before_connect_fail() -> None:
    client = MT5Client(credentials=CREDS, mt5_module=FakeMT5())
    with pytest.raises(MT5Error, match="connect"):
        client.symbol_info("XAUUSDm")


def test_symbol_info_uses_median_spread_when_live_spread_is_zero() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    fake = FakeMT5(rates=make_rates(start, 10, timedelta(minutes=1)))
    with MT5Client(credentials=CREDS, mt5_module=fake) as client:
        spec = client.symbol_info("XAUUSDm")
    assert spec.spread_points == 0
    assert spec.typical_spread_points == 200
    assert spec.volume_min == 0.01 and spec.volume_step == 0.01


def test_last_tick_waits_for_first_tick() -> None:
    fake = FakeMT5()
    fake.ticks = [
        SimpleNamespace(time=0, bid=0.0, ask=0.0),  # just selected, no tick yet
        SimpleNamespace(time=1_790_000_000, bid=89.1, ask=89.12),
    ]
    with MT5Client(credentials=CREDS, mt5_module=fake) as client:
        tick = client.last_tick("USOILm", wait_seconds=2)
        assert tick is not None and tick.bid == 89.1
        assert tick.time_utc.tzinfo is UTC
        assert client.last_tick("USOILm", wait_seconds=0) is None
