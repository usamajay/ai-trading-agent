"""Risk module (SPEC §6): every rule, the token gate, the kill switch, account state.
Target: 100% coverage of src/tradeagent/risk (Phase 4 done-criterion)."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tradeagent.config import RiskLimits, SymbolCosts, load_config
from tradeagent.data.store import connect_db
from tradeagent.risk.engine import OrderRequest, RiskDecision, RiskEngine
from tradeagent.risk.killswitch import KillSwitch
from tradeagent.risk.sizing import loss_per_lot, min_balance, position_size
from tradeagent.risk.state import (
    OpenPosition,
    RiskState,
    load_state,
    log_risk_event,
    save_state,
    trading_date,
    trading_week,
)
from tradeagent.risk.tokens import TokenSigner

LIMITS = load_config().risk  # the real risk.yaml, read only
GOLD = SymbolCosts(
    broker_symbol="XAUUSDm",
    digits=3,
    point=0.001,
    tick_size=0.001,
    tick_value=0.1,
    contract_size=100.0,
    volume_min=0.01,
    volume_step=0.01,
    volume_max=200.0,
    swap_mode=1,
    swap_long=-560.0,
    swap_short=0.0,
    swap_rollover3days=3,
)
OIL = GOLD.model_copy(
    update={"broker_symbol": "USOILm", "tick_value": 1.0, "contract_size": 1000.0}
)
NOW = datetime(2026, 1, 6, 15, 0, tzinfo=UTC)  # Tuesday


def test_limits_are_the_real_risk_yaml() -> None:
    assert (
        LIMITS.risk_per_trade_pct,
        LIMITS.max_daily_loss_pct,
        LIMITS.max_weekly_loss_pct,
        LIMITS.max_drawdown_pct,
        LIMITS.min_reward_risk,
    ) == (0.5, 2.0, 4.0, 10.0, 2.0)


# --- sizing -----------------------------------------------------------------------------


def test_sizing() -> None:
    assert loss_per_lot(5.0, GOLD) == pytest.approx(500.0)  # $5 x 100 oz
    assert position_size(10_000, 0.5, 5.0, GOLD) == pytest.approx(0.10)
    assert position_size(10_000, 0.5, 7.3, GOLD) == pytest.approx(0.06)  # rounded down
    assert position_size(1e12, 0.5, 5.0, GOLD) == GOLD.volume_max
    for bad in ((0, 0.5, 5.0), (10_000, 0, 5.0), (10_000, 0.5, 0.0)):
        assert position_size(*bad, GOLD) == 0.0
    assert min_balance(60.0, GOLD, 0.5) == pytest.approx(12_000)


# --- kill switch ------------------------------------------------------------------------


def test_kill_switch(tmp_path: Path) -> None:
    ks = KillSwitch(tmp_path / "KILL")
    assert not ks.active() and ks.info() is None
    ks.activate("test", "pytest", NOW)
    assert ks.active()
    assert ks.info() == {"reason": "test", "by": "pytest", "time_utc": NOW.isoformat()}
    with pytest.raises(ValueError, match="reason"):
        ks.clear("  ")
    assert ks.active()
    ks.clear("tested")
    assert not ks.active()
    ks.activate("no time given", "pytest")
    assert ks.info()["reason"] == "no time given"  # type: ignore[index]


def test_kill_switch_file_written_by_hand(tmp_path: Path) -> None:
    ks = KillSwitch(tmp_path / "KILL")
    ks.path.write_text("stop!", encoding="utf-8")  # not JSON: still active
    assert ks.active() and ks.info()["by"] == "?"  # type: ignore[index]
    ks.path.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert ks.active() and ks.info() is None


# --- tokens -----------------------------------------------------------------------------


def test_token_approves_exactly_one_order_once() -> None:
    signer = TokenSigner()
    token = signer.issue("o1", "XAUUSD", "long", 0.1, 1995.0, NOW)
    assert signer.verify(token, "o1", "XAUUSD", "long", 0.1, 1995.0, NOW) is None
    assert (
        signer.verify(token, "o1", "XAUUSD", "long", 0.1, 1995.0, NOW) == "token was already used"
    )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"order_id": "o2"}, "order id"),
        ({"symbol": "USOIL"}, "symbol"),
        ({"direction": "short"}, "direction"),
        ({"lots": 0.2}, "lots"),
        ({"stop_loss": 1990.0}, "stop-loss"),
        ({"now": NOW + timedelta(seconds=61)}, "expired"),
    ],
)
def test_token_refused_when_anything_differs(changes: dict[str, object], reason: str) -> None:
    signer = TokenSigner()
    token = signer.issue("o1", "XAUUSD", "long", 0.1, 1995.0, NOW)
    args: dict[str, object] = {
        "order_id": "o1",
        "symbol": "XAUUSD",
        "direction": "long",
        "lots": 0.1,
        "stop_loss": 1995.0,
        "now": NOW,
    }
    args.update(changes)
    result = signer.verify(token, **args)  # type: ignore[arg-type]
    assert result is not None and reason in result


def test_token_from_another_signer_or_missing_is_refused() -> None:
    forged = TokenSigner().issue("o1", "XAUUSD", "long", 0.1, 1995.0, NOW)
    signer = TokenSigner()
    assert "signature" in signer.verify(forged, "o1", "XAUUSD", "long", 0.1, 1995.0, NOW)  # type: ignore[operator]
    assert signer.verify(None, "o1", "XAUUSD", "long", 0.1, 1995.0, NOW) == "no approval token"
    same_secret = TokenSigner(secret=b"k" * 32)
    token = TokenSigner(secret=b"k" * 32).issue("o1", "XAUUSD", "long", 0.1, 1995.0, NOW)
    assert same_secret.verify(token, "o1", "XAUUSD", "long", 0.1, 1995.0, NOW) is None


# --- account state ----------------------------------------------------------------------


def test_trading_week_starts_with_the_sunday_evening_open() -> None:
    sunday_open = datetime(2026, 1, 4, 23, 0, tzinfo=UTC)  # 18:00 New York
    friday = datetime(2026, 1, 9, 21, 0, tzinfo=UTC)
    before_open = datetime(2026, 1, 4, 21, 0, tzinfo=UTC)
    assert trading_week(sunday_open) == trading_week(friday) == "2026-W02"
    assert trading_week(before_open) == "2026-W01"


def test_daily_loss_stops_until_17_new_york() -> None:
    s = RiskState.start(10_000, NOW)
    assert s.mark(9_850, NOW, LIMITS) == []  # -1.5%
    assert s.mark(9_790, NOW + timedelta(hours=1), LIMITS) == ["daily_loss"]  # -2.1%
    s.mark(10_100, NOW + timedelta(hours=2), LIMITS)  # recovered: still stopped today
    assert [c for c, _ in s.blocks(NOW + timedelta(hours=2))] == ["daily_loss"]
    before_close = datetime(2026, 1, 6, 21, 55, tzinfo=UTC)  # 16:55 New York (winter)
    assert [c for c, _ in s.blocks(before_close)] == ["daily_loss"]
    new_day = datetime(2026, 1, 6, 22, 5, tzinfo=UTC)  # 17:05 New York: same UTC day
    s.mark(10_100, new_day, LIMITS)
    assert s.blocks(new_day) == [] and s.day_start_equity == 10_100


def test_trading_date_follows_new_york_summer_time() -> None:
    assert trading_date(datetime(2026, 7, 7, 20, 55, tzinfo=UTC)) == date(2026, 7, 7)
    assert trading_date(datetime(2026, 7, 7, 21, 5, tzinfo=UTC)) == date(2026, 7, 8)


def test_weekly_loss_stops_until_next_week() -> None:
    monday = datetime(2026, 1, 5, 12, tzinfo=UTC)
    s = RiskState.start(10_000, monday)
    for day, equity in ((5, 9_850), (6, 9_700), (7, 9_560)):  # -1.5% a day, never 2%
        triggered = s.mark(equity, datetime(2026, 1, day, 14, tzinfo=UTC), LIMITS)
    assert triggered == ["weekly_loss"]  # -4.4% on the week
    assert "weekly_loss" in [c for c, _ in s.blocks(datetime(2026, 1, 8, 14, tzinfo=UTC))]
    next_week = datetime(2026, 1, 12, 9, tzinfo=UTC)
    s.mark(9_560, next_week, LIMITS)
    assert s.blocks(next_week) == []


def test_drawdown_shutdown_needs_a_human_restart() -> None:
    s = RiskState.start(10_000, NOW)
    s.mark(11_000, NOW, LIMITS)
    for hour, equity in ((1, 10_500), (26, 10_000), (50, 9_950)):  # -9.5% from the peak
        s.mark(equity, NOW + timedelta(hours=hour), LIMITS)
    assert not s.shutdown
    later = NOW + timedelta(hours=74)
    assert "max_drawdown" in s.mark(9_890, later, LIMITS)  # -10.1%
    assert s.shutdown and "shutdown" in [c for c, _ in s.blocks(later)]
    with pytest.raises(ValueError):
        s.restart("")
    s.restart("reviewed the losses")
    assert not s.shutdown and s.peak_equity == 9_890
    assert s.mark(9_880, later, LIMITS) == []  # no instant re-trigger


def test_loss_streak_pause() -> None:
    s = RiskState.start(10_000, NOW)
    assert s.record_close(-50, NOW, LIMITS) == []
    assert s.record_close(10, NOW, LIMITS) == []  # a win resets the count
    for k in range(3):
        assert s.record_close(-50, NOW, LIMITS) == []
    assert s.record_close(-50, NOW, LIMITS) == ["loss_streak"]
    assert s.blocks(NOW + timedelta(hours=23))[0][0] == "loss_streak"
    assert s.blocks(NOW + timedelta(hours=24)) == []
    assert s.consecutive_losses == 0


def test_state_persistence(tmp_path: Path) -> None:
    s = RiskState.start(10_000, NOW)
    s.positions.append(OpenPosition("XAUUSD", "long", 0.1, 1995.0))
    s.mark(9_700, NOW, LIMITS)
    s.record_close(-1, NOW, LIMITS)
    s.paused_until = NOW
    assert RiskState.from_json(s.to_json()) == s
    conn = connect_db(tmp_path / "db.sqlite")
    assert load_state(conn) is None
    save_state(conn, s, "abc", "hash", NOW)
    save_state(conn, s, "abc", "hash", NOW)  # replaces, one row
    assert load_state(conn) == s
    log_risk_event(conn, NOW, "daily_loss", 3.0, 2.0, "block", "abc", "hash")
    assert conn.execute("SELECT rule, action FROM risk_events").fetchall() == [
        ("daily_loss", "block")
    ]
    conn.close()


def test_drawdown_with_no_equity() -> None:
    s = RiskState.start(0.0, NOW)
    assert s.drawdown_pct() == 0.0


# --- the rule engine ---------------------------------------------------------------------


class News:
    def __init__(self, status: str) -> None:
        self.value = status

    def status(self, now: datetime):  # type: ignore[no-untyped-def]
        return self.value, f"calendar says {self.value}"


def request(**changes: object) -> OrderRequest:
    fields: dict[str, object] = {
        "order_id": "o1",
        "symbol": "XAUUSD",
        "direction": "long",
        "entry_ref": 2000.0,
        "stop_loss": 1995.0,
        "take_profit": 2010.0,
        "time_utc": NOW,
        "atr": 5.0,
        "spread": 190.0,
        "median_spread": 190.0,
    }
    fields.update(changes)
    return OrderRequest(**fields)  # type: ignore[arg-type]


def engine(**changes: object) -> RiskEngine:
    fields: dict[str, object] = {
        "limits": LIMITS,
        "specs": {"XAUUSD": GOLD, "USOIL": OIL},
        "signer": TokenSigner(),
    }
    fields.update(changes)
    return RiskEngine(**fields)  # type: ignore[arg-type]


def state(**changes: object) -> RiskState:
    s = RiskState.start(10_000, NOW)
    for key, value in changes.items():
        setattr(s, key, value)
    return s


def test_approved_order_gets_a_size_and_a_token() -> None:
    e = engine()
    d = e.evaluate(request(), state())
    assert d.approved and d.lots == pytest.approx(0.10) and d.codes == []
    assert d.explanation() == "approved: 0.1 lots"
    assert e.signer is not None and d.token is not None
    assert e.signer.verify(d.token, "o1", "XAUUSD", "long", 0.1, 1995.0, NOW) is None
    assert e.decisions == 1


def test_short_order_and_no_signer() -> None:
    d = engine(signer=None).evaluate(
        request(direction="short", stop_loss=2005.0, take_profit=1990.0), state()
    )
    assert d.approved and d.token is None


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"stop_loss": None}, "no_stop"),
        ({"stop_loss": 2001.0}, "no_stop"),  # above a long's entry
        ({"stop_loss": float("nan")}, "no_stop"),
        ({"atr": float("nan")}, "sl_atr"),
        ({"atr": 12.0}, "sl_atr"),  # 0.42 x ATR
        ({"atr": 1.5, "take_profit": 2020.0}, "sl_atr"),  # 3.33 x ATR
        ({"take_profit": 2009.9}, "rr"),
        ({"spread": 381.0}, "spread"),
    ],
)
def test_trade_rules(changes: dict[str, object], code: str) -> None:
    d = engine().evaluate(request(**changes), state())
    assert not d.approved and code in d.codes and d.token is None


def test_boundaries_pass() -> None:
    e = engine()
    assert e.evaluate(request(spread=380.0), state()).approved  # exactly 2 x median
    assert e.evaluate(request(atr=10.0), state()).approved  # exactly 0.5 x ATR
    assert e.evaluate(request(median_spread=0.0, spread=999.0), state()).approved  # no median


def test_news() -> None:
    assert engine(news=News("clear")).evaluate(request(), state()).approved
    assert engine(news=News("blackout")).evaluate(request(), state()).codes == ["news"]
    assert engine(news=News("unknown")).evaluate(request(), state()).codes == ["news_unknown"]
    assert engine(news=News("unknown"), strict_news=False).evaluate(request(), state()).approved


def test_open_positions() -> None:
    held = [OpenPosition("USOIL", "long", 0.1, 60.0)]
    assert engine().evaluate(request(), state(positions=held)).codes == ["max_positions"]
    two = LIMITS.model_copy(update={"max_open_positions": 2})
    same = [OpenPosition("XAUUSD", "short", 0.1, 2010.0)]
    d = engine(limits=two).evaluate(request(), state(positions=same))
    assert d.codes == ["max_positions"] and "on XAUUSD" in d.rejections[0].detail
    assert engine(limits=two).evaluate(request(), state(positions=held)).approved


def test_correlation() -> None:
    two = LIMITS.model_copy(update={"max_open_positions": 2})
    oil_long = [OpenPosition("USOIL", "long", 0.1, 60.0)]
    e = engine(limits=two)
    assert e.evaluate(request(), state(positions=oil_long), correlation=0.7).codes == [
        "correlation"
    ]
    assert e.evaluate(request(), state(positions=oil_long), correlation=0.5).approved
    assert e.evaluate(request(), state(positions=oil_long)).approved  # correlation unknown
    oil_short = [OpenPosition("USOIL", "short", 0.1, 60.0)]
    assert e.evaluate(request(), state(positions=oil_short), correlation=0.9).approved


def test_min_lot_and_unknown_symbol() -> None:
    d = engine().evaluate(request(), state(equity=900.0))
    assert d.codes == ["min_lot"] and d.lots == 0.0
    assert engine().evaluate(request(symbol="BTCUSD"), state()).codes == ["min_lot"]


def test_account_rules(tmp_path: Path) -> None:
    ks = KillSwitch(tmp_path / "KILL")
    ks.activate("test", "pytest", NOW)
    assert engine(kill_switch=ks).evaluate(request(), state()).codes == ["kill_switch"]
    blocked = state(
        shutdown=True,
        shutdown_reason="test",
        daily_stop_day=NOW.date(),
        weekly_stop_week=trading_week(NOW),
        paused_until=NOW + timedelta(hours=1),
    )
    d = engine().evaluate(request(), blocked)
    assert d.codes == ["shutdown", "daily_loss", "weekly_loss", "loss_streak"]
    # Backtests may report account limits instead of enforcing them.
    assert engine(account_rules=False).evaluate(request(), blocked).approved
    # The kill switch applies either way.
    assert engine(account_rules=False, kill_switch=ks).evaluate(request(), blocked).codes == [
        "kill_switch"
    ]


def test_every_failed_rule_is_listed() -> None:
    d = engine(news=News("blackout")).evaluate(
        request(take_profit=2005.0, spread=999.0), state(equity=500.0)
    )
    assert d.codes == ["rr", "spread", "news", "min_lot"]
    assert d.explanation().startswith("rejected: rr (")
    assert isinstance(d, RiskDecision)


def test_limits_type() -> None:
    assert isinstance(LIMITS, RiskLimits)


def test_invalid_stop_is_not_sized() -> None:
    d = engine().evaluate(request(stop_loss=float("nan")), state())
    assert d.codes == ["no_stop"]  # no extra min_lot noise from sizing a broken stop
    assert position_size(10_000, 0.5, float("nan"), GOLD) == 0.0


def test_fresh_state_round_trip() -> None:
    fresh = RiskState.start(10_000, NOW)  # no stops, no pause, no positions
    assert RiskState.from_json(fresh.to_json()) == fresh
