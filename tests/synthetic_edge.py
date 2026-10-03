"""A synthetic market with a planted, known edge, for Phase 7 pipeline tests.

The bars are a random walk (so random trades lose about their costs), except that at
02:00, 08:00 and 14:00 UTC the price drifts for 3 hours **in the direction of the
previous hour's move**. That direction is known when the trade is entered (it is the
last closed hour), so a strategy that follows it has a real edge without looking
ahead. Real market data and real out-of-sample data are never used.
"""

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
from synthetic_bars import NY, market_open

from tradeagent.config import AppConfig, SplitDates, SplitPeriod
from tradeagent.data.store import BarStore
from tradeagent.features.indicators import atr
from tradeagent.strategies.base import BaseStrategy, MarketContext, ParamSpec, Signal
from tradeagent.strategies.common import finite, market_signal

PLANTED = "test_planted_momentum"
DRIFT_HOURS = (2, 8, 14)  # UTC hours where the planted move starts
DRIFT_BARS = 36  # 3 hours of M5
DRIFT_PER_BAR = 0.06  # vs noise 0.5 per M5 bar: a modest edge (train PF ~2)
SYMBOL = "XAUUSD"


def edge_bars(first_sunday: str, weeks: int, seed: int = 11) -> pd.DataFrame:
    start = pd.Timestamp(f"{first_sunday} 18:00", tz=NY)
    ny = pd.date_range(start, start + pd.Timedelta(weeks=weeks), freq="5min", inclusive="left")
    ny = ny[market_open(ny)]
    utc = ny.tz_convert("UTC")
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.5, len(utc))
    close = 2000 + np.cumsum(steps)
    hour, minute = utc.hour.to_numpy(), utc.minute.to_numpy()
    starts = np.flatnonzero(np.isin(hour, DRIFT_HOURS) & (minute == 0))
    for i in starts:
        if i < 12:
            continue
        sign = np.sign(close[i - 1] - close[i - 13])  # the last closed hour's move
        end = min(i + DRIFT_BARS, len(close))
        close[i:end] += sign * DRIFT_PER_BAR * np.arange(1, end - i + 1)
        close[end:] += sign * DRIFT_PER_BAR * (end - i)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "time_utc": utc.astype("datetime64[ns, UTC]"),
            "open": open_,
            "high": np.maximum(open_, close) + 0.3,
            "low": np.minimum(open_, close) - 0.3,
            "close": close,
            "tick_volume": 100,
            "spread": 20,
        }
    )


def _aggregate(m5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    out = m5.groupby(m5["time_utc"].dt.floor(f"{minutes}min")).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        tick_volume=("tick_volume", "sum"),
        spread=("spread", "max"),
    )
    return out.rename_axis("time_utc").reset_index()


def edge_store(root: object, first_sunday: str = "2023-12-31", weeks: int = 29) -> BarStore:
    store = BarStore(root)  # type: ignore[arg-type]
    m5 = edge_bars(first_sunday, weeks)
    store.write(SYMBOL, "M5", m5)
    store.write(SYMBOL, "M15", _aggregate(m5, 15))
    store.write(SYMBOL, "H1", _aggregate(m5, 60))
    return store


# train 12 weeks | 2-week embargo | validation 6 weeks | embargo | OOS 5 weeks (2024)
EDGE_SPLITS = SplitDates(
    approved_by="test",
    approved_on=date(2026, 10, 4),
    train=SplitPeriod(start=date(2024, 1, 7), end=date(2024, 3, 31)),
    validation=SplitPeriod(start=date(2024, 4, 14), end=date(2024, 5, 26)),
    out_of_sample=SplitPeriod(start=date(2024, 6, 9), end=date(2024, 7, 14)),
)


def edge_config(base: AppConfig, **validation: float) -> AppConfig:
    """Real config, synthetic split dates, walk-forward windows sized for 20 weeks."""
    v = base.settings.validation.model_copy(
        update={"wf_history_days": 14, "wf_test_days": 14, "wf_step_days": 14, **validation}
    )
    settings = base.settings.model_copy(update={"validation": v})
    return base.model_copy(update={"splits": EDGE_SPLITS, "settings": settings})


class PlantedMomentum(BaseStrategy):
    """At 02:00/08:00/14:00 UTC, trade in the direction of the last closed hour."""

    name, version, style = PLANTED, "1", "intraday"
    timeframes: Sequence[str] = ("M15",)
    lookback_bars = 20
    param_specs: Sequence[ParamSpec] = (
        ParamSpec("atr_mult", 1.0, 3.0, "stop distance in ATR(14) M15"),
        ParamSpec("rr", 2.0, 4.0, "target as a multiple of the stop distance"),
    )

    def __init__(self, seed: int = 0, params: dict[str, float] | None = None) -> None:
        super().__init__(**{"atr_mult": 1.5, "rr": 2.0, **(params or {})})

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out = dict(frames)
        out["M15"] = out["M15"].assign(atr14=atr(out["M15"], 14))
        return out

    def generate(self, ctx: MarketContext) -> list[Signal]:
        view = ctx.bars("M15")
        if len(view) < 6:
            return []
        closes_at = pd.Timestamp(view.time_utc[-1]).tz_localize(None) + pd.Timedelta(minutes=15)
        if closes_at.minute != 0 or closes_at.hour not in DRIFT_HOURS:
            return []
        move = view.last("close") - view.last("close", 4)  # the last closed hour
        a = view.last("atr14")
        if not finite(move, a) or a <= 0 or move == 0:
            return []
        direction = "long" if move > 0 else "short"
        return [
            market_signal(ctx, direction, self.params["atr_mult"] * a, self.params["rr"], "planted")
        ]
