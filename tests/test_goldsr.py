"""GoldSR EA v2.4 port (H5): MT5 indicators, levels, H4 trend, gates, orders, re-entry."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from tradeagent.features.indicators import adx_mt5, atr_mt5
from tradeagent.strategies.base import TradeEvent, check_strategy
from tradeagent.strategies.goldsr import (
    GoldSR,
    friday_close,
    h4_trend_columns,
    pivot_levels,
)


def _ohlc(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    h, lo, c = zip(*rows, strict=True)
    return pd.DataFrame({"open": c, "high": h, "low": lo, "close": c})


# --- MT5 indicators -----------------------------------------------------------------


def test_atr_mt5_is_a_simple_average_of_true_range() -> None:
    bars = _ohlc([(10, 8, 9), (12, 9, 11), (11, 7, 8)])  # TR: 2, 3, 4
    assert atr_mt5(bars, 2).tolist()[1:] == [2.5, 3.5]
    assert np.isnan(atr_mt5(bars, 2).iloc[0])


def test_adx_mt5_by_hand() -> None:
    # period 3 -> EMA factor 0.5, starting from 0.
    # bar1: +DM 2, -DM 0, TR 3 -> +di 66.67 -> +DI 33.33, -DI 0, DX 100, ADX 50
    # bar2: +DM 0, -DM 2, TR 4 -> -di 50 -> +DI 16.67, -DI 25, DX 20, ADX 35
    bars = _ohlc([(10, 8, 9), (12, 9, 11), (11, 7, 8)])
    assert adx_mt5(bars, 3).tolist() == pytest.approx([0.0, 50.0, 35.0])
    same = _ohlc([(10, 8, 9), (11, 7, 9)])  # +DM = -DM = 1 -> both 0 -> ADX 0
    assert adx_mt5(same, 3).tolist() == [0.0, 0.0]
    assert adx_mt5(_ohlc([(10, 8, 9)]), 3).isna().all()
    flat = _ohlc([(9, 9, 9), (9, 9, 9)])  # TR 0 -> DI 0 -> DX 0
    assert adx_mt5(flat, 3).tolist() == [0.0, 0.0]


# --- levels, H4 trend, Friday close ---------------------------------------------------


def test_pivot_levels_confirm_after_five_bars_with_ea_comparisons() -> None:
    low = np.array([10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
    high = low + 1
    floor, ceiling = pivot_levels(high, low)
    assert np.isnan(floor[:10]).all()  # pivot at bar 5 is known at bar 10
    assert floor[10] == 5 and floor[-1] == 5
    assert np.isnan(ceiling).all()
    # a newer bar EQUAL to the pivot low still allows it; an equal OLDER bar does not
    eq_new = low.copy()
    eq_new[7] = 5
    assert pivot_levels(high, eq_new)[0][10] == 5
    eq_old = low.copy()
    eq_old[3] = 5
    assert np.isnan(pivot_levels(high, eq_old)[0][10])
    peak = 20 - low  # mirror: a pivot high at bar 5
    assert pivot_levels(peak, peak - 1)[1][10] == 15


def test_h4_trend_uses_only_closed_utc_h4_bars() -> None:
    t = pd.date_range("2026-01-05 00:00", periods=260, freq="1h", tz="UTC").as_unit("ns")
    h1 = pd.DataFrame({"time_utc": t, "close": np.arange(260, dtype=float)})
    cols = h4_trend_columns(h1)
    # H1 bar 06:00 ends 07:00: last closed H4 is 00:00-04:00 (close = H1 03:00 = 3)
    assert cols["h4_close"].iloc[6] == 3
    # H1 bar 07:00 ends 08:00, the same moment H4 04:00-08:00 closes (close 7)
    assert cols["h4_close"].iloc[7] == 7
    assert np.isnan(cols["h4_close"].iloc[2])  # no H4 closed yet
    assert np.isnan(cols["h4_ema_slow"].iloc[150]) and np.isfinite(cols["h4_ema_slow"].iloc[-1])
    # truncating the future changes nothing already known
    part = h4_trend_columns(h1.iloc[:101])
    pd.testing.assert_frame_equal(part, cols.iloc[:101])


def test_friday_close_winter_summer_and_weekend() -> None:
    tue_winter = pd.Timestamp("2026-01-06 10:00", tz="UTC")
    assert friday_close(tue_winter) == pd.Timestamp("2026-01-09 21:00", tz="UTC")
    tue_summer = pd.Timestamp("2026-07-07 10:00", tz="UTC")
    assert friday_close(tue_summer) is None  # market closes at 21:00 UTC: EA holds
    sunday = pd.Timestamp("2026-01-04 23:15", tz="UTC")
    assert friday_close(sunday) == pd.Timestamp("2026-01-09 21:00", tz="UTC")


# --- orders and gates -------------------------------------------------------------------


@dataclass
class View:
    values: dict[str, list[float]]

    def __len__(self) -> int:
        return len(next(iter(self.values.values())))

    def last(self, name: str, back: int = 0) -> float:
        return float(self.values[name][-1 - back])


def _now() -> pd.Timestamp:
    return pd.Timestamp("2026-01-06 10:00", tz="UTC")


@dataclass
class Ctx:
    """A hand-set context: bars per timeframe, bid/ask and time."""

    m15: dict[str, list[float]]
    h1: dict[str, list[float]]
    bid: float = 2000.0
    ask: float = 2000.2
    now: pd.Timestamp = field(default_factory=_now)
    symbol: str = "XAUUSD"

    def bars(self, tf: str) -> View:
        return View(self.m15 if tf == "M15" else self.h1)

    def expected_entry(self, direction: str) -> float:
        return self.ask if direction == "long" else self.bid


UP_H1 = {
    "h4_ema_fast": [2000.0],
    "h4_ema_slow": [1990.0],
    "h4_close": [2001.0],
    "adx_mt5": [30.0],
}


def buy_ctx(**kw: object) -> Ctx:
    """Uptrend; bar2 closed at the 1995 ceiling, bar1 breaks out strongly to 1999.9."""
    m15 = {
        "open": [1994.0, 1995.0],
        "high": [1995.5, 2000.0],
        "low": [1993.0, 1994.8],
        "close": [1995.0, 1999.9],
        "ema_fast": [1990.0, 1991.0],
        "ema_slow": [1985.0, 1986.0],
        "atr_mt5": [3.0, 3.0],
        "floor": [1980.0, 1980.0],
        "ceiling": [1995.0, 1995.0],
    }
    return Ctx(m15, dict(UP_H1), **kw)  # type: ignore[arg-type]


def gen(s: GoldSR, ctx: Ctx) -> list:  # type: ignore[type-arg]
    return s.generate(ctx)  # type: ignore[arg-type]


def test_breakout_buy_order_matches_the_ea() -> None:
    s = GoldSR()
    check_strategy(s)
    (sig,) = gen(s, buy_ctx())
    # SL = min(bar1 low 1994.8, ceiling 1995) - 1 x ATR 3 = 1991.8; entry ask 2000.2
    risk = 2000.2 - 1991.8
    assert sig.direction == "long" and sig.stop_loss == pytest.approx(1991.8)
    assert sig.take_profit == pytest.approx(2000.2 + 2 * risk)
    (t1, be), (t2, lock) = sig.stop_moves
    assert (t1, be) == pytest.approx((2000.2 + risk, 2000.5))
    assert (t2, lock) == pytest.approx((2000.2 + 1.5 * risk, t1))
    assert sig.exit_by == pd.Timestamp("2026-01-09 21:00", tz="UTC") and sig.tag == "goldsr:0"


def test_filters_block_the_signal() -> None:
    cases = {
        "higher trend down": {**UP_H1, "h4_close": [1980.0]},
        "adx too low": {**UP_H1, "adx_mt5": [15.0]},
        "adx too high": {**UP_H1, "adx_mt5": [55.0]},
        "h4 unknown": {**UP_H1, "h4_ema_slow": [float("nan")]},
    }
    for name, h1 in cases.items():
        ctx = buy_ctx()
        ctx.h1 = h1
        assert gen(GoldSR(), ctx) == [], name
    weak = buy_ctx()
    weak.m15["open"][-1] = 1999.0  # body 0.9 of a 5.2 range
    assert gen(GoldSR(), weak) == []
    near = buy_ctx()
    near.m15["close"][-1] = 1995.2  # only 0.2 beyond the level (< 0.1 x ATR = 0.3)
    near.m15["open"][-1] = 1993.0
    near.m15["low"][-1] = 1992.9
    assert gen(GoldSR(), near) == []
    no_adx = buy_ctx()
    no_adx.h1 = {**UP_H1, "adx_mt5": [float("nan")]}  # MT5: no ADX value -> no check
    assert len(gen(GoldSR(), no_adx)) == 1


def test_gates_hours_spread_friday_and_daily_count() -> None:
    assert gen(GoldSR(), buy_ctx(now=pd.Timestamp("2026-01-06 06:45", tz="UTC"))) == []
    assert gen(GoldSR(), buy_ctx(ask=2001.0)) == []  # spread $1 > $0.80
    assert gen(GoldSR(), buy_ctx(now=pd.Timestamp("2026-01-09 21:00", tz="UTC"))) == []
    s = GoldSR()
    day = pd.Timestamp("2026-01-06 08:00", tz="UTC")
    for k in range(8):
        s.on_trade_opened(TradeEvent("long", f"goldsr:{k}", day, 2000.0, 1990.0))
    assert gen(s, buy_ctx()) == []
    s.entries = [day - pd.Timedelta(days=1)] * 8  # yesterday's do not count
    assert len(gen(s, buy_ctx())) == 1


def test_stop_distance_rules() -> None:
    close = buy_ctx()
    close.m15.update(
        {
            "atr_mt5": [0.1, 0.1],
            "ceiling": [1999.5, 1999.5],
            "close": [1999.5, 1999.9],
            "open": [1999.4, 1999.55],
            "high": [1999.6, 1999.95],
            "low": [1999.3, 1999.5],
        }
    )
    (sig,) = gen(GoldSR(), close)
    assert sig.stop_loss == pytest.approx(2000.2 - 2.0)  # widened to the $2 minimum
    far = buy_ctx()
    far.m15["atr_mt5"] = [15.0, 15.0]  # stop 1994.8 - 15 = 1979.8: risk > $20
    assert gen(GoldSR(), far) == []


def test_sell_breakdown() -> None:
    m15 = {
        "open": [2006.0, 2005.0],
        "high": [2006.5, 2005.2],
        "low": [2004.8, 2000.0],
        "close": [2005.0, 2000.1],
        "ema_fast": [2010.0, 2009.0],
        "ema_slow": [2015.0, 2014.0],
        "atr_mt5": [3.0, 3.0],
        "floor": [2005.0, 2005.0],
        "ceiling": [2030.0, 2030.0],
    }
    h1 = {
        "h4_ema_fast": [2000.0],
        "h4_ema_slow": [2010.0],
        "h4_close": [1999.0],
        "adx_mt5": [25.0],
    }
    (sig,) = gen(GoldSR(), Ctx(m15, h1))
    # SL = max(bar1 high 2005.2, floor 2005) + 3 = 2008.2; entry bid 2000
    assert sig.direction == "short" and sig.stop_loss == pytest.approx(2008.2)
    assert sig.take_profit == pytest.approx(2000 - 2 * 8.2)
    assert sig.stop_moves[0][1] == pytest.approx(1999.7)  # breakeven - $0.30 for a sell


# --- re-entry -------------------------------------------------------------------------


def _closed(s: GoldSR, tag: str, net: float, at: pd.Timestamp) -> None:
    opened = at - pd.Timedelta(hours=1)
    s.on_trade_opened(TradeEvent("long", tag, opened, 2000.2, 1991.8))
    s.on_trade_closed(TradeEvent("long", tag, opened, 2000.2, 1991.8, at, 2016.0, "tp", net, 2))


def quiet_ctx(ask: float, now: pd.Timestamp) -> Ctx:
    """Uptrend, close 1999 above the 1995 level, but no fresh breakout."""
    ctx = buy_ctx(ask=ask, bid=ask - 0.2, now=now)
    ctx.m15["close"] = [1998.0, 1999.0]
    return ctx


def test_reentry_after_a_win_market_or_limit() -> None:
    s = GoldSR()
    (first,) = gen(s, buy_ctx())
    t = pd.Timestamp("2026-01-06 11:00", tz="UTC")
    _closed(s, first.tag, 50.0, t)
    assert s.armed is not None and s.armed.level == 1995.0
    # price back inside the $0.50 zone around the first entry 2000.2 -> market order
    (re,) = gen(s, quiet_ctx(2000.5, t + pd.Timedelta(minutes=15)))
    assert re.order_type == "market" and re.tag == "goldsr:0:re"
    assert re.stop_loss == pytest.approx(1991.8)  # the first stop is reused
    # above the zone -> a buy limit at 2000.7 for one bar
    (lim,) = gen(s, quiet_ctx(2003.0, t + pd.Timedelta(minutes=30)))
    assert (lim.order_type, lim.entry_price, lim.expiry_bars) == ("limit", 2000.7, 1)
    assert lim.take_profit == pytest.approx(2000.7 + 2 * (2000.7 - 1991.8))
    # price at/below the stop: nothing, still armed
    assert gen(s, quiet_ctx(1991.0, t + pd.Timedelta(minutes=45))) == []
    assert s.armed is not None
    # window of 10 bars passed -> disarmed
    assert gen(s, quiet_ctx(2000.5, t + pd.Timedelta(minutes=151))) == []
    assert s.armed is None


def test_reentry_limits_and_cancellation() -> None:
    s = GoldSR()
    (first,) = gen(s, buy_ctx())
    t = pd.Timestamp("2026-01-06 11:00", tz="UTC")
    _closed(s, first.tag, -5.0, t)  # a loss never re-arms
    assert s.armed is None
    _closed(s, first.tag, 5.0, t)
    for k in range(2):  # two re-entries, both winners
        s.on_trade_opened(TradeEvent("long", "goldsr:0:re", t, 2000.2, 1991.8))
        assert s.re_count == k + 1
        s.on_trade_closed(
            TradeEvent("long", "goldsr:0:re", t, 2000.2, 1991.8, t, 2010.0, "tp", 5.0)
        )
        assert (s.armed is not None) == (k == 0)  # re-armed after the 1st, not the 2nd
    assert s.re_count == 0  # EA: count reset once the maximum is used

    s2 = GoldSR()
    gen(s2, buy_ctx())
    _closed(s2, "goldsr:0", 5.0, t)
    broken = quiet_ctx(2000.5, t + pd.Timedelta(minutes=15))
    broken.m15["close"][-1] = 1994.0  # closed back below the level: setup invalid
    assert gen(s2, broken) == [] and s2.armed is None

    off = GoldSR(params={"reentry": 0.0})
    gen(off, buy_ctx())
    _closed(off, "goldsr:0", 50.0, t)
    assert off.armed is None
    with pytest.raises(ValueError):
        GoldSR(params={"reentry": 0.5})


def test_fresh_signal_replaces_a_pending_reentry() -> None:
    s = GoldSR()
    gen(s, buy_ctx())
    t = pd.Timestamp("2026-01-06 11:00", tz="UTC")
    _closed(s, "goldsr:0", 5.0, t)
    (sig,) = gen(s, buy_ctx(now=t + pd.Timedelta(minutes=15)))
    assert sig.tag == "goldsr:1" and s.armed is None


def test_prepare_adds_every_column() -> None:
    t = pd.date_range("2026-01-05", periods=400, freq="15min", tz="UTC").as_unit("ns")
    close = 2000 + np.sin(np.arange(400) / 7) * 5
    m15 = pd.DataFrame(
        {"time_utc": t, "open": close, "high": close + 1, "low": close - 1, "close": close}
    )
    h1 = m15.iloc[::4].reset_index(drop=True)
    out = GoldSR().prepare({"M15": m15, "H1": h1})
    for col in ("ema_fast", "ema_slow", "atr_mt5", "floor", "ceiling"):
        assert col in out["M15"] and out["M15"][col].notna().any(), col
    for col in ("h4_close", "h4_ema_fast", "h4_ema_slow", "adx_mt5"):
        assert col in out["H1"], col


def test_not_enough_data_or_price_beyond_the_stop() -> None:
    short = buy_ctx()
    short.m15 = {k: v[-1:] for k, v in short.m15.items()}
    assert gen(GoldSR(), short) == []
    warm = buy_ctx()
    warm.m15["ema_slow"][-1] = float("nan")
    assert gen(GoldSR(), warm) == []
    beyond = buy_ctx(ask=1991.5, bid=1991.3)  # ask already below the 1991.8 stop
    assert gen(GoldSR(), beyond) == []


def test_short_reentry_market_limit_and_beyond_stop() -> None:
    s = GoldSR()
    t = pd.Timestamp("2026-01-06 11:00", tz="UTC")
    s.levels["goldsr:0"] = 2005.0
    s.on_trade_closed(TradeEvent("short", "goldsr:0", t, 2000.0, 2008.2, t, 1990.0, "tp", 40.0, 2))
    m15 = {
        "open": [2001.0, 2000.5],
        "high": [2001.5, 2001.0],
        "low": [2000.0, 1999.5],
        "close": [2000.5, 2000.0],
        "ema_fast": [2009.0, 2008.0],
        "ema_slow": [2014.0, 2013.0],
        "atr_mt5": [3.0, 3.0],
        "floor": [2005.0, 2005.0],
        "ceiling": [2030.0, 2030.0],
    }
    h1 = {"h4_ema_fast": [2000.0], "h4_ema_slow": [2010.0], "h4_close": [1999.0], "adx_mt5": [25.0]}
    later = t + pd.Timedelta(minutes=15)
    (mkt,) = gen(s, Ctx(dict(m15), h1, bid=1999.8, ask=2000.0, now=later))
    assert mkt.order_type == "market" and mkt.direction == "short"
    (lim,) = gen(s, Ctx(dict(m15), h1, bid=1997.0, ask=1997.2, now=later))
    assert (lim.order_type, lim.entry_price) == ("limit", 1999.5)
    assert gen(s, Ctx(dict(m15), h1, bid=2009.0, ask=2009.2, now=later)) == []
