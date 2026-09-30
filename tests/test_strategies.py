"""Phase 3 strategies: decision rules on hand-set indicator values, look-ahead checks
with the real indicators, and engine runs whose signals pass the risk rules."""

import numpy as np
import pandas as pd
import pytest
from synthetic_bars import NY, trading_bars

from tradeagent.backtest.costs import CostModel
from tradeagent.backtest.dataset import WINDOW_COLUMNS, Dataset, flag_bars
from tradeagent.backtest.engine import run_backtest
from tradeagent.backtest.lookahead import check_lookahead
from tradeagent.config import DataExclusions, load_config
from tradeagent.data.resample import resample_ny_close
from tradeagent.strategies import registry
from tradeagent.strategies.base import Frames, check_strategy
from tradeagent.strategies.breakout_compression import BreakoutCompression
from tradeagent.strategies.mean_reversion_bb import MeanReversionBB
from tradeagent.strategies.session_breakout import SessionBreakout
from tradeagent.strategies.structure_retest import StructureRetest
from tradeagent.strategies.trend_ema_pullback import TrendEmaPullback

ALL = [
    "trend_ema_pullback",
    "breakout_compression",
    "mean_reversion_bb",
    "session_breakout",
    "structure_retest",
]


def bars(
    n: int, start: str = "2026-01-06 03:00", minutes: int = 15, **columns: object
) -> pd.DataFrame:
    """n consecutive bars (New York start time) with constant prices unless overridden."""
    times = pd.date_range(pd.Timestamp(start, tz=NY), periods=n, freq=f"{minutes}min")
    df = pd.DataFrame(
        {
            "time_utc": times.tz_convert("UTC").astype("datetime64[ns, UTC]"),
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "tick_volume": 100,
            "spread": 0,
        }
    )
    for name, value in columns.items():
        df[name] = value
    return df


def decide(strategy, frames: dict[str, pd.DataFrame]):  # type: ignore[no-untyped-def]
    """generate() at the close of the decision timeframe's last bar (no prepare())."""
    tf = strategy.timeframes[0]
    last = frames[tf]["time_utc"].iloc[-1]
    length = pd.Timedelta(minutes=15 if tf == "M15" else 60)
    return strategy.generate(Frames("XAUUSD", frames).context(last + length))


# --- every strategy -------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_registered_and_valid(name: str) -> None:
    registry.load_builtins()
    strategy = registry.create(name, seed=1)
    check_strategy(strategy)
    assert len(strategy.param_specs) <= 5


@pytest.fixture(scope="module")
def market() -> dict[str, pd.DataFrame]:
    m5 = trading_bars("2026-01-04", weeks=6, freq="5min").assign(spread=200)

    def agg(minutes: int) -> pd.DataFrame:
        g = m5.groupby(m5["time_utc"].dt.floor(f"{minutes}min"))
        return (
            g.agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                tick_volume=("tick_volume", "sum"),
                spread=("spread", "max"),
            )
            .rename_axis("time_utc")
            .reset_index()
        )

    h1 = agg(60)
    return {"M5": m5, "M15": agg(15), "H1": h1, "H4": resample_ny_close(h1, "H4")}


@pytest.mark.parametrize("name", ALL)
def test_no_lookahead(name: str, market: dict[str, pd.DataFrame]) -> None:
    registry.load_builtins()
    strategy = registry.create(name)
    frames = {tf: market[tf] for tf in strategy.timeframes}
    assert check_lookahead(lambda: registry.create(name), "XAUUSD", frames) == []


@pytest.mark.parametrize("name", ALL)
def test_engine_run_signals_pass_the_risk_rules(name: str, market: dict[str, pd.DataFrame]) -> None:
    registry.load_builtins()
    cfg = load_config()
    strategy = registry.create(name)
    tf = strategy.timeframes[0]
    none = DataExclusions(excluded_windows=[], no_trade_windows=[], keep_gaps=[])
    windows = pd.DataFrame(columns=WINDOW_COLUMNS).astype(
        {"start_utc": "datetime64[ns, UTC]", "end_utc": "datetime64[ns, UTC]"}
    )
    flat = strategy.style in ("scalp", "intraday")
    data = flag_bars(market[tf], tf, "XAUUSD", windows, none, cfg.settings.backtest, flat)
    dataset = Dataset(
        "XAUUSD",
        tf,
        "train",
        strategy.style,
        data["time_utc"].iloc[0],
        data["time_utc"].iloc[-1],
        data,
        windows,
    )
    exec_m5 = flag_bars(market["M5"], "M5", "XAUUSD", windows, none, cfg.settings.backtest, flat)
    result = run_backtest(
        strategy,
        dataset,
        CostModel.from_config(cfg, "XAUUSD"),
        cfg.risk,
        cfg.settings.backtest,
        context={t: market[t] for t in strategy.timeframes[1:]},
        exec_bars=exec_m5,
    )
    assert result.counts.get("rejected_rr", 0) == 0
    assert result.counts.get("rejected_invalid_signal", 0) == 0
    assert result.counts.get("rejected_entry_side", 0) == 0


# --- 3.3 trend EMA pullback -----------------------------------------------------------


def trend_frames(
    direction: str = "long", rsi_before: float = 40, rsi_now: float = 50, touched: bool = True
) -> dict[str, pd.DataFrame]:
    """H1 trend + 8 M15 bars whose lows (highs for shorts) stay clear of EMA20 = 100,
    except one bar inside the 5-bar pullback window when `touched`."""
    up = direction == "long"
    h1 = bars(
        3,
        start="2026-01-06 00:00",
        minutes=60,
        close=102.0 if up else 98.0,
        ema20=101.0 if up else 99.0,
        ema50=100.5 if up else 99.5,
        ema200=100.0,
    )
    m15 = bars(
        8,
        close=100.2 if up else 99.8,
        ema20=100.0,
        atr14=0.5,
        low=100.1 if up else 99.5,
        high=100.5 if up else 99.9,
        rsi14=[50] * 6 + [rsi_before, rsi_now],
    )
    if touched:  # one bar inside the pullback window reaches EMA20
        m15.loc[5, "low" if up else "high"] = 99.9 if up else 100.1
    return {"M15": m15, "H1": h1}


def test_trend_pullback_long_and_short() -> None:
    [long] = decide(TrendEmaPullback(), trend_frames("long", 40, 50))
    assert long.direction == "long"
    assert long.take_profit - 100.2 == pytest.approx(2 * (100.2 - long.stop_loss))
    assert 100.2 - long.stop_loss == pytest.approx(0.75)  # 1.5 x ATR 0.5
    [short] = decide(TrendEmaPullback(), trend_frames("short", 60, 50))
    assert short.direction == "short"


def test_trend_pullback_needs_every_condition() -> None:
    assert decide(TrendEmaPullback(), trend_frames("long", 50, 52)) == []  # no RSI cross
    assert decide(TrendEmaPullback(), trend_frames("long", touched=False)) == []  # no pullback
    no_trend = trend_frames("long")
    no_trend["H1"]["ema50"] = 102.0  # EMA20 below EMA50: not an uptrend
    assert decide(TrendEmaPullback(), no_trend) == []


# --- 3.4 compression breakout ---------------------------------------------------------


def squeeze_frames(rank: float = 0.1, volume: float = 300, close: float = 101.5):  # type: ignore[no-untyped-def]
    return {
        "M15": bars(
            3,
            close=close,
            bb_upper=101.0,
            bb_lower=99.0,
            atr14=0.5,
            width_rank=[0.5, rank, 0.6],
            avg_vol_before=100.0,
            volume=volume,
        )
    }


def test_compression_breakout() -> None:
    [signal] = decide(BreakoutCompression(), squeeze_frames())
    assert signal.direction == "long"
    [short] = decide(BreakoutCompression(), squeeze_frames(close=98.5))
    assert short.direction == "short"
    assert decide(BreakoutCompression(), squeeze_frames(rank=0.3)) == []  # no squeeze
    assert decide(BreakoutCompression(), squeeze_frames(volume=140)) == []  # weak volume
    assert decide(BreakoutCompression(), squeeze_frames(close=100.5)) == []  # inside bands


# --- 3.5 mean reversion ---------------------------------------------------------------


def reversion_frames(adx_value: float = 15, mid: float = 102.0, close_before: float = 98.8):  # type: ignore[no-untyped-def]
    return {
        "M15": bars(
            2,
            close=[close_before, 99.2],
            bb_lower=99.0,
            bb_upper=105.0,
            bb_mid=mid,
            adx=adx_value,
            atr14=1.0,
        )
    }


def test_mean_reversion() -> None:
    [signal] = decide(MeanReversionBB(), reversion_frames())
    assert signal.direction == "long"
    assert signal.take_profit == 102.0  # the middle band
    assert signal.stop_loss == pytest.approx(99.2 - 1.0)  # 1 x ATR below the entry
    assert decide(MeanReversionBB(), reversion_frames(adx_value=25)) == []  # trending
    assert decide(MeanReversionBB(), reversion_frames(mid=100.5)) == []  # reward < 2R
    assert decide(MeanReversionBB(), reversion_frames(close_before=99.1)) == []  # never outside


# --- 3.6 session breakout -------------------------------------------------------------


def session_frames(hour: float = 4.0, close: float = 101.2, asia=(101.0, 99.0)):  # type: ignore[no-untyped-def]
    return {
        "M15": bars(
            2,
            close=close,
            asia_high=asia[0],
            asia_low=asia[1],
            atr14=0.5,
            ny_hour=hour,
            day_id=20460.0,
        )
    }


def test_session_breakout() -> None:
    strategy = SessionBreakout()
    [signal] = decide(strategy, session_frames())
    assert signal.direction == "long"
    # stop at the range midpoint (100) would be 1.2 = 2.4 x ATR: inside 0.55-2.95 x ATR
    assert signal.stop_loss == pytest.approx(100.0)
    assert decide(strategy, session_frames()) == []  # one signal per trading day
    assert decide(SessionBreakout(), session_frames(hour=12.0)) == []  # outside 03:00-11:00
    assert decide(SessionBreakout(), session_frames(close=101.02)) == []  # inside the buffer
    assert decide(SessionBreakout(), session_frames(asia=(104.0, 99.0))) == []  # range > 6 ATR
    [short] = decide(SessionBreakout(), session_frames(close=98.5))
    assert short.direction == "short"


def test_session_breakout_stop_is_kept_within_atr_limits() -> None:
    [signal] = decide(SessionBreakout(), session_frames(close=101.2, asia=(101.0, 98.4)))
    # midpoint 99.7 would be 1.5 = 3 x ATR: capped at 2.95 x ATR
    assert 101.2 - signal.stop_loss == pytest.approx(2.95 * 0.5)


# --- 3.7 structure retest -------------------------------------------------------------


def retest_frames(up: float = 1.0, down: float = 0.0, h4_close: float = 105.0):  # type: ignore[no-untyped-def]
    h1 = bars(
        2,
        start="2026-01-06 10:00",
        minutes=60,
        close=103.0,
        bos_up=up,
        bos_down=down,
        bos_level=102.0,
        atr14=1.0,
    )
    h4 = bars(1, start="2026-01-06 05:00", minutes=240, close=h4_close, ema50=100.0)
    return {"H1": h1, "H4": h4}


def test_structure_retest_places_a_limit_at_the_broken_level() -> None:
    [signal] = decide(StructureRetest(), retest_frames())
    assert (signal.order_type, signal.entry_price, signal.expiry_bars) == ("limit", 102.0, 12)
    assert (signal.stop_loss, signal.take_profit) == (101.0, 104.0)
    assert decide(StructureRetest(), retest_frames(h4_close=99.0)) == []  # against H4 trend
    assert decide(StructureRetest(), retest_frames(up=0.0)) == []  # no break
    [short] = decide(StructureRetest(), retest_frames(up=0.0, down=1.0, h4_close=95.0))
    assert short.direction == "short" and short.entry_price == 102.0


def test_signals_explain_themselves(market: dict[str, pd.DataFrame]) -> None:
    [signal] = decide(TrendEmaPullback(), trend_frames())
    assert "EMA20>50>200" in signal.why and "RSI" in signal.why
    assert np.isfinite(signal.stop_loss)
