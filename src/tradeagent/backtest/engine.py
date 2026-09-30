"""Event-driven backtest engine (SPEC §7.1, docs/PHASE_2_TASKS.md 2.5).

How one run works, bar by bar on the strategy's decision timeframe:

1. **Manage** the open trade / pending order on decision bar t. For strategies
   above M5 this is done on the M5 bars inside bar t ("execution bars"), so a
   stop-loss and take-profit inside the same bar are settled by M5, the Friday
   cut-off happens at 16:30 exactly, and an entry right after a reopen waits for
   the first M5 bar after the no-entry window. If the M5 bars are missing or do not
   match bar t's high/low, bar t itself is used (counted as a fallback). Inside one
   execution bar, SL and TP both touched -> the stop-loss is assumed first.
2. After bar t closes, **ask the strategy** (it sees only closed bars) and check
   each signal: interface rules, one slot (a position or pending order), the
   temporary risk basics, and position size. An accepted market order fills at
   the open of bar t+1, never at bar t's close; limit/stop orders are active from
   bar t+1 until they expire.

Fills: bars are bid prices; `CostModel` turns them into ask/bid sides, adds the
spread margin and slippage. A bar that opens beyond the stop fills at that open
(the loss can exceed 1R). Take-profits fill at their price, never better.
Trades still open when an excluded bar (data hole) is reached are dropped and
counted. Nothing here places orders: it is a simulation on stored bars.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from tradeagent.backtest.costs import CostModel
from tradeagent.backtest.dataset import Dataset
from tradeagent.backtest.risk_basics import check_signal, min_balance, position_size
from tradeagent.config import BacktestSettings, RiskLimits
from tradeagent.data.market_hours import NEW_YORK
from tradeagent.data.mt5_client import TIMEFRAMES
from tradeagent.features.indicators import atr
from tradeagent.strategies.base import (
    Frames,
    InvalidSignal,
    Signal,
    Strategy,
    check_strategy,
)

ATR_PERIOD = 14  # for the stop-distance check (SPEC §6: SL 0.5-3 x ATR)
SPREAD_MEDIAN_BARS = 2000  # trailing window for "spread <= 2 x median"

TRADE_COLUMNS = [
    "trade_id",
    "symbol",
    "direction",
    "order_type",
    "signal_time",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "lots",
    "stop_loss",
    "take_profit",
    "exit_reason",  # tp, sl, sl_gap, weekend_close, time, end_of_data
    "gross_pnl",  # price P&L incl. spread and slippage, before swap/commission
    "spread_cost",
    "slippage_cost",
    "swap",  # negative = paid
    "commission",
    "net_pnl",
    "risk_usd",  # 1R: the planned loss if the stop fills exactly
    "r_multiple",
    "tie",  # none, m5 (settled by M5 bars), sl_first, fill_bar
    "m1_check",  # info only: sl / tp / same_m1_bar / no_m1 / "" (no tie)
    "held_over_weekend",
    "min_balance",  # account needed to take it at the minimum lot and the risk %
    "bars_held",
    "why",
]


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    timeframe: str
    start_balance: float
    final_balance: float
    trades: pd.DataFrame  # TRADE_COLUMNS
    equity: pd.DataFrame  # time_utc, equity (after each closed trade)
    counts: dict[str, int]  # rejections, skips, cancellations, tie resolutions


class _Bars:
    """numpy columns of flagged bars, for fast access inside the loop."""

    def __init__(self, df: pd.DataFrame, timeframe: str) -> None:
        start = df["time_utc"]
        end = df["end_utc"] if "end_utc" in df.columns else start + TIMEFRAMES[timeframe][1]
        self.n = len(df)
        self.time = list(start)
        self.end = list(end)
        self.time_ns = start.to_numpy("datetime64[ns]")
        self.end_ns = end.to_numpy("datetime64[ns]")
        self.open = df["open"].to_numpy(float)
        self.high = df["high"].to_numpy(float)
        self.low = df["low"].to_numpy(float)
        self.close = df["close"].to_numpy(float)
        self.spread = df["spread"].to_numpy(float)
        self.in_split = _flag(df, "in_split", True)
        self.excluded = _flag(df, "excluded", False)
        self.entry_blocked = _flag(df, "entry_blocked", False)
        self.after_reopen = _flag(df, "after_reopen", False)
        self.flatten = _flag(df, "flatten", False)
        self.partial = _flag(df, "partial", False)


def _flag(df: pd.DataFrame, name: str, default: bool) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(bool)
    return np.full(len(df), default)


@dataclass
class _Order:
    signal: Signal
    lots: float
    signal_time: pd.Timestamp
    bars_left: int | None  # pending orders: decision bars left before expiry


@dataclass
class _Position:
    order: _Order
    entry_time: pd.Timestamp
    entry_price: float
    entry_bar: int
    spread_cost: float
    slippage_cost: float


@dataclass
class _Run:
    strategy: Strategy
    symbol: str
    timeframe: str
    costs: CostModel
    limits: RiskLimits
    risk_basics: bool
    decision: _Bars
    exec_bars: _Bars | None
    exec_range: tuple[np.ndarray, np.ndarray] | None
    m1: _Bars | None
    frames: Frames
    windows: list[tuple[pd.Timestamp, pd.Timestamp]]
    atr: np.ndarray
    median_spread: np.ndarray
    equity: float
    order: _Order | None = None
    position: _Position | None = None
    trades: list[dict[str, object]] = field(default_factory=list)
    curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)

    # --- main loop -----------------------------------------------------------------

    def run(self) -> None:
        d = self.decision
        for i in range(d.n):
            if self.order is not None or self.position is not None:
                self._manage(i)
            if self.position is not None:
                limit = self.position.order.signal.max_hold_bars
                if limit is not None and i - self.position.entry_bar + 1 >= limit:
                    self._close_at_bar_close(i, "time")
            last = i == d.n - 1 or not d.in_split[i + 1]
            if last:
                if d.in_split[i]:
                    if self.position is not None:
                        self._close_at_bar_close(i, "end_of_data")
                    if self.order is not None:
                        self.counts["cancelled_end_of_data"] += 1
                        self.order = None
                continue
            self._decide(i)

    # --- managing orders and positions ----------------------------------------------

    def _manage(self, i: int) -> None:
        bars, lo, hi = self._execution_bars(i)
        for j in range(lo, hi):
            self._process(bars, j, i)
            if self.order is None and self.position is None:
                break
        if self.order is not None:
            if self.order.signal.order_type == "market":
                self.counts["entry_skipped_blocked"] += 1  # no allowed bar to fill in
                self.order = None
            else:
                assert self.order.bars_left is not None
                self.order.bars_left -= 1
                if self.order.bars_left <= 0:
                    self.counts["orders_expired"] += 1
                    self.order = None

    def _execution_bars(self, i: int) -> tuple[_Bars, int, int]:
        """The M5 bars inside decision bar i, or bar i itself (M5 strategies/fallback)."""
        if self.exec_bars is None or self.exec_range is None:
            return self.decision, i, i + 1
        lo, hi = int(self.exec_range[0][i]), int(self.exec_range[1][i])
        e, d = self.exec_bars, self.decision
        tolerance = self.costs.spec.point / 2
        if (
            hi > lo
            and abs(e.high[lo:hi].max() - d.high[i]) <= tolerance
            and abs(e.low[lo:hi].min() - d.low[i]) <= tolerance
        ):
            return e, lo, hi
        self.counts["exec_fallback_bars"] += 1
        return self.decision, i, i + 1

    def _process(self, b: _Bars, j: int, i: int) -> None:
        if b.excluded[j]:
            if self.position is not None:
                self.counts["dropped_excluded"] += 1
                self.position = None
            if self.order is not None:
                self.counts["cancelled_excluded"] += 1
                self.order = None
            return
        if b.flatten[j]:
            if self.position is not None:
                p = self.position
                price = self.costs.exit_side(p.order.signal.direction, b.open[j], b.spread[j])
                self._close(price, "weekend_close", b, j, i, slip=True)
            if self.order is not None:
                self.counts["cancelled_weekend"] += 1
                self.order = None
            return
        fill = None
        if self.order is not None and self.position is None:
            fill = self._try_fill(b, j, i)
        if self.position is not None:
            self._check_exit(b, j, i, fill)

    def _try_fill(self, b: _Bars, j: int, i: int) -> str | None:
        """Fill the order on execution bar j if possible: 'open', 'intrabar' or None."""
        assert self.order is not None
        if b.entry_blocked[j] or b.after_reopen[j]:
            return None
        sig = self.order.signal
        d, sp, c = sig.direction, b.spread[j], self.costs
        at_open = c.entry_side(d, b.open[j], sp)
        kind: str | None
        if sig.order_type == "market":
            price, slip, kind = at_open, True, "open"
        else:
            level = sig.entry_price
            assert level is not None
            high, low = c.entry_range(d, b.high[j], b.low[j], sp)
            buy = d == "long"
            if sig.order_type == "limit":
                slip = False  # limit orders fill at their price, or better at a gap
                opened_beyond = at_open <= level if buy else at_open >= level
                touched = low <= level if buy else high >= level
            else:
                slip = True  # stop orders become market orders when touched
                opened_beyond = at_open >= level if buy else at_open <= level
                touched = high >= level if buy else low <= level
            if opened_beyond:
                price, kind = at_open, "open"
            elif touched:
                price, kind = level, "intrabar"
            else:
                return None
        filled = c.entry_fill(d, price, sp, slip)
        lots = self.order.lots
        value = c.spec.value_per_point_per_lot * lots / c.spec.point
        spread_cost = c.spread(sp) * value if d == "long" else 0.0  # longs pay it on entry
        self.position = _Position(
            order=self.order,
            entry_time=b.time[j],
            entry_price=filled,
            entry_bar=i,
            spread_cost=spread_cost,
            slippage_cost=abs(filled - price) * value,
        )
        self.order = None
        self.counts["filled"] += 1
        return kind

    def _check_exit(self, b: _Bars, j: int, i: int, fill: str | None) -> None:
        assert self.position is not None
        sig = self.position.order.signal
        d, sp, c = sig.direction, b.spread[j], self.costs
        sl, tp = sig.stop_loss, sig.take_profit
        long = d == "long"
        if fill is None:  # the position was already open when this bar opened
            at_open = c.exit_side(d, b.open[j], sp)
            if (at_open <= sl) if long else (at_open >= sl):
                self._close(at_open, "sl_gap", b, j, i, slip=True)
                return
            if (at_open >= tp) if long else (at_open <= tp):
                self._close(tp, "tp", b, j, i, slip=False)  # never better than TP
                return
        high, low = c.exit_range(d, b.high[j], b.low[j], sp)
        sl_hit = low <= sl if long else high >= sl
        tp_hit = high >= tp if long else low <= tp
        if fill == "intrabar":
            # Entered inside this bar: the order of events is unknown, so a stop
            # touched in this bar counts, a target touched in this bar does not.
            if sl_hit:
                self._close(sl, "sl", b, j, i, slip=True, tie="fill_bar")
            return
        if sl_hit:
            self._close(sl, "sl", b, j, i, slip=True, tie="sl_first" if tp_hit else "none")
        elif tp_hit:
            self._close(tp, "tp", b, j, i, slip=False)

    def _close_at_bar_close(self, i: int, reason: str) -> None:
        assert self.position is not None
        d = self.decision
        direction = self.position.order.signal.direction
        price = self.costs.exit_side(direction, d.close[i], d.spread[i])
        self._close(price, reason, d, i, i, slip=True, at_end=True)

    def _close(
        self,
        price: float,
        reason: str,
        b: _Bars,
        j: int,
        i: int,
        slip: bool,
        tie: str = "none",
        at_end: bool = False,
    ) -> None:
        p = self.position
        assert p is not None
        sig = p.order.signal
        c, d, lots = self.costs, sig.direction, p.order.lots
        exit_price = c.exit_fill(d, price, b.spread[j], slip)
        exit_time = b.end[j] if at_end else b.time[j]
        value = c.spec.value_per_point_per_lot * lots / c.spec.point
        spread_cost = p.spread_cost + (c.spread(b.spread[j]) * value if d == "short" else 0.0)
        slippage_cost = p.slippage_cost + abs(exit_price - price) * value

        if tie == "none" and reason in ("sl", "tp") and b is not self.decision:
            dec = self.decision
            high, low = c.exit_range(d, dec.high[i], dec.low[i], dec.spread[i])
            long = d == "long"
            both = (low <= sig.stop_loss if long else high >= sig.stop_loss) and (
                high >= sig.take_profit if long else low <= sig.take_profit
            )
            if both:
                tie = "m5"  # the decision bar touched both; its M5 bars settled it
        if tie != "none":
            self.counts[f"tie_{tie}"] += 1
        m1 = self._m1_check(p, i) if tie in ("m5", "sl_first") else ""
        if m1:
            agrees = (m1 == "sl") == (reason == "sl")
            self.counts[
                "m1_check_"
                + ("unavailable" if m1 == "no_m1" else "agrees" if agrees else "differs")
            ] += 1

        gross = c.pnl(d, p.entry_price, exit_price, lots)
        swap = c.swap(d, lots, p.entry_time, exit_time)
        commission = c.commission(lots)
        net = gross + swap - commission
        sl_distance = abs(p.entry_price - sig.stop_loss)
        risk_usd = sl_distance * value
        self.equity += net
        self.curve.append((exit_time, self.equity))
        self.trades.append(
            {
                "trade_id": len(self.trades) + 1,
                "symbol": self.symbol,
                "direction": d,
                "order_type": sig.order_type,
                "signal_time": p.order.signal_time,
                "entry_time": p.entry_time,
                "entry_price": p.entry_price,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "lots": lots,
                "stop_loss": sig.stop_loss,
                "take_profit": sig.take_profit,
                "exit_reason": reason,
                "gross_pnl": gross,
                "spread_cost": spread_cost,
                "slippage_cost": slippage_cost,
                "swap": swap,
                "commission": commission,
                "net_pnl": net,
                "risk_usd": risk_usd,
                "r_multiple": net / risk_usd if risk_usd > 0 else float("nan"),
                "tie": tie,
                "m1_check": m1,
                "held_over_weekend": held_over_weekend(p.entry_time, exit_time),
                "min_balance": min_balance(sl_distance, c.spec, self.limits.risk_per_trade_pct),
                "bars_held": i - p.entry_bar + 1,
                "why": sig.why,
            }
        )
        self.position = None

    def _m1_check(self, p: _Position, i: int) -> str:
        """Info only: which level M1 bars say was touched first inside decision bar i."""
        m1 = self.m1
        if m1 is None or m1.n == 0:
            return "no_m1"
        start = max(p.entry_time, self.decision.time[i]).to_datetime64()
        lo = int(np.searchsorted(m1.time_ns, start, side="left"))
        hi = int(np.searchsorted(m1.time_ns, self.decision.end_ns[i], side="left"))
        if hi <= lo:
            return "no_m1"
        sig = p.order.signal
        long = sig.direction == "long"
        for k in range(lo, hi):
            high, low = self.costs.exit_range(sig.direction, m1.high[k], m1.low[k], m1.spread[k])
            sl_hit = low <= sig.stop_loss if long else high >= sig.stop_loss
            tp_hit = high >= sig.take_profit if long else low <= sig.take_profit
            if sl_hit and tp_hit:
                return "same_m1_bar"
            if sl_hit:
                return "sl"
            if tp_hit:
                return "tp"
        return "no_m1"

    # --- decisions ------------------------------------------------------------------

    def _decide(self, i: int) -> None:
        d = self.decision
        if d.partial[i]:
            self.counts["skipped_partial_bar"] += 1
            return
        first = max(0, i - self.strategy.lookback_bars + 1)
        t0, t1 = d.time[first], d.end[i]
        if any(ws < t1 and we > t0 for ws, we in self.windows):
            self.counts["skipped_lookback_excluded"] += 1
            return
        try:
            signals = self.strategy.generate(self.frames.context(d.end[i]))
        except InvalidSignal:
            self.counts["rejected_invalid_signal"] += 1
            return
        for signal in signals:
            self.counts["signals"] += 1
            reason = self._consider(signal, i)
            if reason is not None:
                self.counts[f"rejected_{reason}"] += 1

    def _consider(self, sig: Signal, i: int) -> str | None:
        if sig.symbol != self.symbol:
            return "wrong_symbol"
        if self.order is not None or self.position is not None:
            return "position_open"
        d = self.decision
        at_close = self.costs.entry_side(sig.direction, d.close[i], d.spread[i])
        if sig.order_type == "market":
            entry_ref = at_close
        else:
            if sig.entry_side_error(at_close) is not None:
                return "entry_side"
            assert sig.entry_price is not None
            entry_ref = sig.entry_price
        if self.risk_basics:
            reason, lots = check_signal(
                entry_ref=entry_ref,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
                atr=float(self.atr[i]),
                bar_spread=float(d.spread[i]),
                median_spread=float(self.median_spread[i]),
                equity=self.equity,
                limits=self.limits,
                spec=self.costs.spec,
            )
        else:
            lots = position_size(
                self.equity,
                self.limits.risk_per_trade_pct,
                abs(entry_ref - sig.stop_loss),
                self.costs.spec,
            )
            reason = "min_lot" if lots < self.costs.spec.volume_min else None
        if reason is not None:
            return reason
        self.order = _Order(
            signal=sig,
            lots=lots,
            signal_time=d.end[i],
            bars_left=sig.expiry_bars if sig.order_type != "market" else None,
        )
        self.counts["orders"] += 1
        return None


def held_over_weekend(entry: pd.Timestamp, exit: pd.Timestamp) -> bool:
    """True if a Friday 17:00 New York weekly close lies between entry and exit."""
    day = entry.tz_convert(NEW_YORK).date()
    last = exit.tz_convert(NEW_YORK).date()
    while day <= last:
        if day.weekday() == 4:
            close = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=17, tz=NEW_YORK)
            if entry < close < exit:
                return True
        day += timedelta(days=1)
    return False


def run_backtest(
    strategy: Strategy,
    dataset: Dataset,
    costs: CostModel,
    limits: RiskLimits,
    settings: BacktestSettings,
    context: dict[str, pd.DataFrame] | None = None,
    exec_bars: pd.DataFrame | None = None,
    m1_bars: pd.DataFrame | None = None,
    risk_basics: bool = True,
) -> BacktestResult:
    """Run one strategy over one dataset (a split of one symbol/timeframe).

    context:   bars of the strategy's other timeframes (timeframes[1:])
    exec_bars: flagged M5 bars covering the dataset (needed above M5; without them
               every bar falls back to SL-first on the decision bar)
    m1_bars:   M1 bars for the side-by-side tie check only (never changes results)
    risk_basics: apply the temporary risk checks (min RR, SL vs ATR, spread)
    """
    check_strategy(strategy)
    if strategy.timeframes[0] != dataset.timeframe:
        raise ValueError(
            f"dataset is {dataset.timeframe} but the strategy decides on {strategy.timeframes[0]}"
        )
    missing = [tf for tf in strategy.timeframes[1:] if tf not in (context or {})]
    if missing:
        raise ValueError(f"missing context bars for {missing}")

    bars = dataset.bars.reset_index(drop=True)
    frames = {dataset.timeframe: bars, **(context or {})}
    prepared = strategy.prepare(dict(frames))
    decision = _Bars(bars, dataset.timeframe)

    exec_view: _Bars | None = None
    exec_range: tuple[np.ndarray, np.ndarray] | None = None
    if dataset.timeframe != "M5" and exec_bars is not None:
        e = exec_bars.sort_values("time_utc").reset_index(drop=True)
        exec_view = _Bars(e, "M5")
        exec_range = (
            np.searchsorted(exec_view.time_ns, decision.time_ns, side="left"),
            np.searchsorted(exec_view.time_ns, decision.end_ns, side="left"),
        )

    spread = bars["spread"].astype(float)
    run = _Run(
        strategy=strategy,
        symbol=dataset.symbol,
        timeframe=dataset.timeframe,
        costs=costs,
        limits=limits,
        risk_basics=risk_basics,
        decision=decision,
        exec_bars=exec_view,
        exec_range=exec_range,
        m1=_Bars(m1_bars.sort_values("time_utc").reset_index(drop=True), "M1")
        if m1_bars is not None
        else None,
        frames=Frames(dataset.symbol, prepared),
        windows=list(
            zip(dataset.exclusions["start_utc"], dataset.exclusions["end_utc"], strict=True)
        ),
        atr=atr(bars, ATR_PERIOD).to_numpy(),
        median_spread=spread.rolling(SPREAD_MEDIAN_BARS, min_periods=1).median().to_numpy(),
        equity=settings.starting_balance,
    )
    first_in_split = int(np.argmax(decision.in_split)) if decision.in_split.any() else 0
    run.curve.append((decision.time[first_in_split], settings.starting_balance))
    run.run()

    trades = pd.DataFrame(run.trades, columns=TRADE_COLUMNS)
    equity = pd.DataFrame(run.curve, columns=["time_utc", "equity"])
    return BacktestResult(
        symbol=dataset.symbol,
        timeframe=dataset.timeframe,
        start_balance=settings.starting_balance,
        final_balance=run.equity,
        trades=trades,
        equity=equity,
        counts=dict(run.counts),
    )
