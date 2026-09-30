"""GoldSR EA v2.4 port (hypothesis H5): Usama's MT5 EA, defaults FROZEN (no tuning).

Source: docs/external/goldsr/GoldSR_EA_v2_4.mq5 (the reference) and GOLDSR_RULES.md.
The EA's default inputs are copied below as constants; they may already be fitted by
the MT5 optimizer, so they are never changed here. Only `reentry` (on/off) varies,
as Usama asked (two experiments).

What is ported (defaults ON in the EA):
- M15 signal on the closed bar; market entry at the next open.
- Levels: most recent confirmed pivot low (floor) / high (ceiling), 5 bars each side
  (a pivot is known 5 bars after it forms). Same strict/equal comparisons as the EA.
- M15 trend: EMA20/EMA50 and the close vs EMA50.
- SELL: trend DOWN, bar1 closes below the floor, bar2 closed at/above it.
  BUY: trend UP, bar1 closes above the ceiling, bar2 closed at/below it.
- Filters: H4 trend agrees (EMA20/50 + close, last closed H4 bar, H4 built from H1 on
  UTC 4-hour boundaries like the broker's); H1 ADX(14) in [20, 50] (MT5 `iADX`
  formula); strong candle (body >= 50% of range, closes in the trade direction,
  >= 0.1 x ATR beyond the level). ATR is MT5 `iATR` (simple average).
- Stop: beyond max(bar1 high, floor) / min(bar1 low, ceiling) by 1 x ATR; widened to
  $2 if closer, skipped if farther than $20. Targets 1R/1.5R/2R; TP3 is the order's
  take-profit. At TP1 the stop moves to entry +/- $0.30, at TP2 to TP1 (engine
  `stop_moves`, settled on M5 bars).
- Re-entry after a WIN only (net P&L > 0): up to 2 per signal, within 10 M15 bars of
  the close, when price is back within $0.50 of the first entry, the setup is still
  valid (same trend, close beyond the level, H4 agrees); reuses the first stop.
- Gates: spread <= $0.80; server hours 07-24 (Exness server = UTC, checked
  2026-09-30); no entries Friday from 21:00; at most 8 entries per server day.
- Friday close at 21:00 server time. In US summer time the market closes at 21:00
  UTC, so this never fires and trades are held over the weekend, as in the EA.

Not ported (our system decides instead): risk 1% and hard cap 5% (ours: 0.5%,
risk engine), daily loss 5% (ours: 2%), free-margin check, and the news filter (ours
applies the historical news blackout; the EA's is off inside the MT5 tester).
Differences from tick-by-tick MT5: decisions happen at M15 bar closes, so a re-entry
is a market order (price already in the zone) or a limit order at the zone edge
valid for one bar, instead of a tick-level fill.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradeagent.data.market_hours import NEW_YORK
from tradeagent.features.indicators import adx_mt5, atr_mt5, ema
from tradeagent.strategies import registry
from tradeagent.strategies.base import (
    BaseStrategy,
    Direction,
    MarketContext,
    ParamSpec,
    Signal,
    Style,
    TradeEvent,
)

NAME = "goldsr"

# EA v2.4 default inputs (frozen).
FAST_EMA, SLOW_EMA, PIV_LEN, ATR_LEN = 20, 50, 5, 14
SL_ATR_MULT, MIN_SL, MAX_SL = 1.0, 2.0, 20.0
TP1_R, TP2_R, TP3_R, BE_BUFFER = 1.0, 1.5, 2.0, 0.3
MAX_REENTRIES, REENTRY_BARS, REENTRY_ZONE = 2, 10, 0.5
HTF_FAST, HTF_SLOW = 20, 50
ADX_PERIOD, ADX_MIN, ADX_MAX = 14, 20.0, 50.0
MIN_BODY_PCT, BREAK_BUF_ATR = 50.0, 0.1
MAX_TRADES_DAY, MAX_SPREAD, START_HOUR, END_HOUR, FRIDAY_HOUR = 8, 0.8, 7, 24, 21
SERVER_UTC_OFFSET_HOURS = 0  # Exness MT5 server time = UTC all year (DECISIONS 2026-09-30)
BAR = pd.Timedelta(minutes=15)


def pivot_levels(high: np.ndarray, low: np.ndarray, n: int = PIV_LEN) -> tuple[np.ndarray, ...]:
    """(floor, ceiling) known at the close of each bar: the latest confirmed pivot.

    Pivot low at p: the n older bars have strictly higher lows, the n newer bars lows
    >= low[p] (EA `IsPivotLow`); highs mirror it. It is confirmed at bar p + n.
    """
    lo, hi = pd.Series(low, dtype=float), pd.Series(high, dtype=float)
    is_low = pd.Series(True, index=lo.index)
    is_high = pd.Series(True, index=hi.index)
    for k in range(1, n + 1):
        is_low &= (lo.shift(k) > lo) & (lo.shift(-k) >= lo)
        is_high &= (hi.shift(k) < hi) & (hi.shift(-k) <= hi)
    floor = lo.where(is_low).shift(n).ffill()
    ceiling = hi.where(is_high).shift(n).ffill()
    return floor.to_numpy(), ceiling.to_numpy()


def h4_trend_columns(h1: pd.DataFrame) -> pd.DataFrame:
    """H4 close/EMA20/EMA50 of the last H4 bar closed by the end of each H1 bar.

    H4 bars are built from H1 on UTC 4-hour boundaries (00, 04, 08, ... = the broker's
    H4 with server time UTC), each closing at its start + 4 hours.
    """
    t = h1.set_index("time_utc")["close"]
    h4 = t.resample("4h", origin="epoch", label="left", closed="left").last().dropna()
    table = pd.DataFrame(
        {
            "end": (h4.index + pd.Timedelta(hours=4)).as_unit("ns"),
            "h4_close": h4.to_numpy(),
            "h4_ema_fast": ema(h4, HTF_FAST).to_numpy(),
            "h4_ema_slow": ema(h4, HTF_SLOW).to_numpy(),
        }
    )
    h1_end = pd.DataFrame(
        {"end": (h1["time_utc"] + pd.Timedelta(hours=1)).dt.as_unit("ns"), "row": h1.index}
    )
    merged = pd.merge_asof(h1_end.sort_values("end"), table, on="end", direction="backward")
    return merged.set_index("row").drop(columns="end").reindex(h1.index)


def friday_close(now: pd.Timestamp) -> pd.Timestamp | None:
    """The EA's Friday close for a trade opened at `now`: Friday 21:00 server time, or
    None when that is not before the weekly close (US summer: both are 21:00 UTC)."""
    days = (4 - now.weekday()) % 7
    friday = (now + pd.Timedelta(days=days)).normalize()
    close_at = friday + pd.Timedelta(hours=FRIDAY_HOUR - SERVER_UTC_OFFSET_HOURS)
    weekly = pd.Timestamp(
        year=friday.year, month=friday.month, day=friday.day, hour=17, tz=NEW_YORK
    ).tz_convert("UTC")
    return close_at if close_at < weekly else None


@dataclass
class _Rearm:
    direction: Direction
    price: float  # first entry
    stop: float  # first stop (reused)
    level: float
    expiry: pd.Timestamp
    tag: str  # the first trade's signal, which re-entries belong to


class GoldSR(BaseStrategy):
    name = NAME
    version = "2.4-port1"
    style: Style = "swing"  # holds overnight; Friday close handled by `exit_by`
    timeframes: Sequence[str] = ("M15", "H1")
    suited_regimes: Sequence[str] = ("trending",)
    lookback_bars = 600  # the EA scans 600 M15 bars for pivots
    param_specs: Sequence[ParamSpec] = (
        ParamSpec("reentry", 0.0, 1.0, "1 = re-enter after a win (EA default), 0 = never"),
    )

    def __init__(self, seed: int = 0, params: dict[str, float] | None = None) -> None:
        super().__init__(**{"reentry": 1.0, **(params or {})})
        if self.params["reentry"] not in (0.0, 1.0):
            raise ValueError("reentry must be 0 or 1")
        self.armed: _Rearm | None = None
        self.re_count = 0
        self.entries: list[pd.Timestamp] = []
        self.levels: dict[str, float] = {}
        self.n_signals = 0

    # --- indicators -----------------------------------------------------------------

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out = dict(frames)
        m = out["M15"]
        floor, ceiling = pivot_levels(m["high"].to_numpy(float), m["low"].to_numpy(float))
        out["M15"] = m.assign(
            ema_fast=ema(m["close"], FAST_EMA),
            ema_slow=ema(m["close"], SLOW_EMA),
            atr_mt5=atr_mt5(m, ATR_LEN),
            floor=floor,
            ceiling=ceiling,
        )
        h1 = out["H1"]
        out["H1"] = pd.concat([h1, h4_trend_columns(h1)], axis=1).assign(
            adx_mt5=adx_mt5(h1, ADX_PERIOD)
        )
        return out

    # --- trade hooks (the engine calls these as fills and exits happen) ------------

    def on_trade_opened(self, event: TradeEvent) -> None:
        self.entries.append(event.entry_time)
        self.re_count = self.re_count + 1 if event.tag.endswith(":re") else 0
        self.armed = None

    def on_trade_closed(self, event: TradeEvent) -> None:
        won = event.net_pnl is not None and event.net_pnl > 0
        if won and self.params["reentry"] == 1.0 and self.re_count < MAX_REENTRIES:
            assert event.exit_time is not None
            base = event.tag.removesuffix(":re")
            self.armed = _Rearm(
                event.direction,
                event.entry_price,
                event.stop_loss,
                self.levels[base],
                event.exit_time + REENTRY_BARS * BAR,
                base,
            )
        else:
            self.armed = None
            self.re_count = 0

    # --- decisions --------------------------------------------------------------------

    def generate(self, ctx: MarketContext) -> list[Signal]:
        m, h = ctx.bars("M15"), ctx.bars("H1")
        if len(m) < 2 or len(h) < 1:  # the EA reads bars 1 and 2 (bar 0 is forming)
            return []
        fast, slow, a = m.last("ema_fast"), m.last("ema_slow"), m.last("atr_mt5")
        if not np.isfinite([fast, slow, a]).all():
            return []
        c1, c2 = m.last("close"), m.last("close", 1)
        down = fast < slow and c1 < slow
        up = fast > slow and c1 > slow
        floor, ceiling = m.last("floor"), m.last("ceiling")

        fresh: tuple[Direction, float, float] | None = None
        if down and np.isfinite(floor) and c1 < floor and c2 >= floor:
            fresh = ("short", floor, max(m.last("high"), floor) + a * SL_ATR_MULT)
        elif up and np.isfinite(ceiling) and c1 > ceiling and c2 <= ceiling:
            fresh = ("long", ceiling, min(m.last("low"), ceiling) - a * SL_ATR_MULT)
        if fresh is not None and self._filters_pass(ctx, fresh[0], fresh[1], a):
            self.armed = None  # a fresh signal replaces any pending re-entry
            direction, level, stop = fresh
            sig = self._order(ctx, direction, stop, level)
            return [sig] if sig is not None else []
        return self._reentry(ctx, down, up, c1)

    def _htf_allows(self, ctx: MarketContext, direction: Direction) -> bool:
        h = ctx.bars("H1")
        f, s, c = h.last("h4_ema_fast"), h.last("h4_ema_slow"), h.last("h4_close")
        if not np.isfinite([f, s, c]).all():
            return False  # unclear higher trend blocks (InpHTFAllowNeutral = false)
        trend = 1 if f > s and c > s else -1 if f < s and c < s else 0
        return trend == (1 if direction == "long" else -1)

    def _filters_pass(
        self, ctx: MarketContext, direction: Direction, level: float, a: float
    ) -> bool:
        if not self._htf_allows(ctx, direction):
            return False
        adx = ctx.bars("H1").last("adx_mt5")
        if np.isfinite(adx) and not ADX_MIN <= adx <= ADX_MAX:
            return False
        m = ctx.bars("M15")
        o, hi, lo, c = m.last("open"), m.last("high"), m.last("low"), m.last("close")
        rng, body = hi - lo, abs(c - o)
        dir_ok = c < o if direction == "short" else c > o
        if rng <= 0 or not dir_ok or body / rng * 100 < MIN_BODY_PCT:
            return False
        beyond = level - c if direction == "short" else c - level
        return beyond >= a * BREAK_BUF_ATR

    def _can_open(self, ctx: MarketContext) -> bool:
        """The EA's CanOpen gates (risk and daily loss are the risk engine's)."""
        spread = ctx.expected_entry("long") - ctx.expected_entry("short")
        if spread > MAX_SPREAD:
            return False
        server = ctx.now + pd.Timedelta(hours=SERVER_UTC_OFFSET_HOURS)
        if not START_HOUR <= server.hour < END_HOUR:
            return False
        if server.weekday() == 4 and server.hour >= FRIDAY_HOUR:
            return False
        day = server.normalize()
        offset = pd.Timedelta(hours=SERVER_UTC_OFFSET_HOURS)
        today = sum(1 for t in self.entries if (t + offset).normalize() == day)
        return today < MAX_TRADES_DAY

    def _order(
        self,
        ctx: MarketContext,
        direction: Direction,
        stop: float,
        level: float,
        reentry_of: str | None = None,
        limit_price: float | None = None,
    ) -> Signal | None:
        """The order the EA would send; `reentry_of` = tag of the signal re-entered."""
        reentry = reentry_of is not None
        if not self._can_open(ctx):
            return None
        short = direction == "short"
        entry = limit_price if limit_price is not None else ctx.expected_entry(direction)
        risk = stop - entry if short else entry - stop
        if risk <= 0:
            return None
        if risk < MIN_SL:
            stop, risk = (entry + MIN_SL, MIN_SL) if short else (entry - MIN_SL, MIN_SL)
        if risk > MAX_SL:
            return None
        sign = -1.0 if short else 1.0
        t1, t2, t3 = (entry + sign * r * risk for r in (TP1_R, TP2_R, TP3_R))
        if reentry_of is None:
            base = f"{NAME}:{self.n_signals}"
            self.n_signals += 1
            self.levels[base] = level
        else:
            base = reentry_of
        kind = "limit" if limit_price is not None else "market"
        return Signal(
            symbol=ctx.symbol,
            direction=direction,
            stop_loss=stop,
            take_profit=t3,
            why=(
                f"GoldSR {'re-entry' if reentry else 'breakout'} {direction} "
                f"level {level:.2f}, risk {risk:.2f}"
            ),
            order_type=kind,
            entry_price=limit_price,
            expiry_bars=1 if limit_price is not None else None,
            stop_moves=((t1, entry + sign * BE_BUFFER), (t2, t1)),
            exit_by=friday_close(ctx.now),
            tag=f"{base}:re" if reentry else base,
        )

    def _reentry(self, ctx: MarketContext, down: bool, up: bool, c1: float) -> list[Signal]:
        r = self.armed
        if r is None:
            return []
        if ctx.now > r.expiry:
            self.armed, self.re_count = None, 0
            return []
        short = r.direction == "short"
        valid = (down and c1 < r.level) if short else (up and c1 > r.level)
        if not (valid and self._htf_allows(ctx, r.direction)):
            self.armed, self.re_count = None, 0
            return []
        price = ctx.expected_entry(r.direction)  # bid for shorts, ask for longs
        if short:
            zone = r.price - REENTRY_ZONE
            if zone <= price < r.stop:
                sig = self._order(ctx, "short", r.stop, r.level, r.tag)
            elif price < zone:
                sig = self._order(ctx, "short", r.stop, r.level, r.tag, limit_price=zone)
            else:
                return []
        else:
            zone = r.price + REENTRY_ZONE
            if r.stop < price <= zone:
                sig = self._order(ctx, "long", r.stop, r.level, r.tag)
            elif price > zone:
                sig = self._order(ctx, "long", r.stop, r.level, r.tag, limit_price=zone)
            else:
                return []
        return [sig] if sig is not None else []


registry.register(NAME, lambda seed, params: GoldSR(seed, params))
