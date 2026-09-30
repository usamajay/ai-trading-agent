"""Look-ahead protection (SPEC §7.1): planted leaks must be caught, honest code must pass.

These run in every `pytest` run (CLAUDE.md: keep the look-ahead test passing).
"""

import numpy as np
import pandas as pd
import pytest
from synthetic_bars import NY, trading_bars

from tradeagent.backtest.costs import CostModel
from tradeagent.backtest.dataset import WINDOW_COLUMNS, Dataset, flag_bars
from tradeagent.backtest.engine import run_backtest
from tradeagent.backtest.lookahead import check_lookahead, truncate
from tradeagent.config import DataExclusions, load_config
from tradeagent.data.resample import resample_ny_close
from tradeagent.features.indicators import atr
from tradeagent.strategies.base import BaseStrategy, MarketContext, Signal


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    m15 = trading_bars("2026-01-04", weeks=2, freq="15min")
    h1 = trading_bars("2026-01-04", weeks=2, freq="1h")
    return {"M15": m15, "H1": h1, "D1": resample_ny_close(h1, "D1")}


class Honest(BaseStrategy):
    """Uses only causal indicators: ATR and an exponential average (past bars only)."""

    name, version, style = "honest", "1", "intraday"
    timeframes = ("M15", "H1")
    lookback_bars = 20

    def prepare(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out = dict(frames)
        out["M15"] = self.add_features(out["M15"])
        out["H1"] = out["H1"].assign(ema=out["H1"]["close"].ewm(span=10, adjust=False).mean())
        return out

    def add_features(self, m15: pd.DataFrame) -> pd.DataFrame:
        return m15.assign(atr14=atr(m15, 14))

    def generate(self, ctx: MarketContext) -> list[Signal]:
        m15, h1 = ctx.bars("M15"), ctx.bars("H1")
        if len(m15) < 15 or len(h1) < 1 or len(m15) % 7:
            return []
        close, a = m15.last("close"), m15.last("atr14")
        direction = "long" if close > h1.last("ema") else "short"
        sl, tp = (close - a, close + 2 * a) if direction == "long" else (close + a, close - 2 * a)
        return [Signal(ctx.symbol, direction, sl, tp, why="close vs H1 EMA")]


class NextBarFeature(Honest):
    """Leak: an indicator column built from the NEXT bar's close."""

    def add_features(self, m15: pd.DataFrame) -> pd.DataFrame:
        return super().add_features(m15).assign(next_close=m15["close"].shift(-1))


class CentredAverage(Honest):
    """Leak: a centred moving average (half of its window is in the future)."""

    def add_features(self, m15: pd.DataFrame) -> pd.DataFrame:
        centred = m15["close"].rolling(9, center=True).mean()
        return super().add_features(m15).assign(centred=centred)


class WholeHistoryScaling(Honest):
    """Leak: scaling by the mean of the WHOLE series (includes future bars)."""

    def add_features(self, m15: pd.DataFrame) -> pd.DataFrame:
        z = (m15["close"] - m15["close"].mean()) / m15["close"].std()
        return super().add_features(m15).assign(z=z)


class ReadsBehindTheView(Honest):
    """Leak: digs the full array out from behind the read-only view."""

    def generate(self, ctx: MarketContext) -> list[Signal]:
        view = ctx.bars("M15")
        full = view.close.base
        while getattr(full, "base", None) is not None:
            full = full.base
        if len(view) < 15 or full is None or len(full) <= len(view):
            return []
        close, a = view.last("close"), view.last("atr14")
        if full[len(view)] > close:  # tomorrow's price
            return [Signal(ctx.symbol, "long", close - a, close + 2 * a, why="peeked")]
        return []


class SeededRandom(Honest):
    """Honest randomness: a fixed seed and one draw per decision."""

    def __init__(self) -> None:
        super().__init__()
        self.rng = np.random.default_rng(42)

    def generate(self, ctx: MarketContext) -> list[Signal]:
        draw = self.rng.random()
        m15 = ctx.bars("M15")
        if draw > 0.1 or len(m15) < 15:
            return []
        close, a = m15.last("close"), m15.last("atr14")
        return [Signal(ctx.symbol, "long", close - a, close + 2 * a, why="random")]


def test_truncate_keeps_only_closed_bars(frames: dict[str, pd.DataFrame]) -> None:
    cut = pd.Timestamp("2026-01-06 10:15", tz=NY)
    cut_frames = truncate(frames, cut)
    assert cut_frames["M15"]["time_utc"].max() == pd.Timestamp("2026-01-06 10:00", tz=NY)
    assert cut_frames["H1"]["time_utc"].max() == pd.Timestamp("2026-01-06 09:00", tz=NY)
    assert len(cut_frames["D1"]) == 1  # only Monday's day has closed


@pytest.mark.parametrize("strategy", [Honest, SeededRandom])
def test_honest_strategies_pass(frames: dict[str, pd.DataFrame], strategy: type) -> None:
    assert check_lookahead(strategy, "XAUUSD", frames) == []


@pytest.mark.parametrize(
    ("strategy", "what"),
    [
        (NextBarFeature, "feature"),
        (CentredAverage, "feature"),
        (WholeHistoryScaling, "feature"),
        (ReadsBehindTheView, "signal"),
    ],
)
def test_planted_leaks_are_caught(
    frames: dict[str, pd.DataFrame], strategy: type, what: str
) -> None:
    problems = check_lookahead(strategy, "XAUUSD", frames)
    assert problems, f"{strategy.__name__} peeked at the future but was not caught"
    assert any(p.what == what for p in problems)


# --- the engine itself --------------------------------------------------------------------


def engine_run(strategy: BaseStrategy, frames: dict[str, pd.DataFrame]) -> None:
    cfg = load_config()
    none = DataExclusions(excluded_windows=[], no_trade_windows=[], keep_gaps=[])
    windows = pd.DataFrame(columns=WINDOW_COLUMNS).astype(
        {"start_utc": "datetime64[ns, UTC]", "end_utc": "datetime64[ns, UTC]"}
    )
    decision_tf = strategy.timeframes[0]
    bars = flag_bars(
        frames[decision_tf],
        decision_tf,
        "XAUUSD",
        windows,
        none,
        cfg.settings.backtest,
        flat_before_weekend=False,
    )
    dataset = Dataset(
        "XAUUSD",
        decision_tf,
        "train",
        strategy.style,
        bars["time_utc"].iloc[0],
        bars["time_utc"].iloc[-1],
        bars,
        windows,
    )
    context = {tf: frames[tf] for tf in strategy.timeframes[1:]}
    m5 = trading_bars("2026-01-04", weeks=2, freq="5min")
    m1 = trading_bars("2026-01-04", weeks=1, freq="1min")
    exec_bars = flag_bars(m5, "M5", "XAUUSD", windows, none, cfg.settings.backtest, False)
    run_backtest(
        strategy,
        dataset,
        CostModel.from_config(cfg, "XAUUSD"),
        cfg.risk,
        cfg.settings.backtest,
        context=context,
        exec_bars=exec_bars,
        m1_bars=m1,
    )


class Spy(BaseStrategy):
    """Records anything it can see that had not closed by the decision time."""

    name, version, style = "spy", "1", "swing"
    timeframes = ("M15", "H1", "D1")
    lookback_bars = 1

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.problems: list[str] = []
        self.lengths = {
            "M15": pd.Timedelta(minutes=15),
            "H1": pd.Timedelta(hours=1),
            "D1": pd.Timedelta(hours=24),
        }

    def generate(self, ctx: MarketContext) -> list[Signal]:
        self.calls += 1
        now = ctx.now.tz_convert("UTC").tz_localize(None).to_datetime64()
        for tf, length in self.lengths.items():
            view = ctx.bars(tf)
            if len(view) and view.time_utc[-1] + length.to_timedelta64() > now:
                self.problems.append(f"{tf} bar not closed at {ctx.now}")
        if ctx.bars("M15").time_utc[-1] + self.lengths["M15"].to_timedelta64() != now:
            self.problems.append(f"latest M15 bar is not the one that just closed at {ctx.now}")
        for hidden in ("M5", "M1"):  # execution / tie-check bars: never for strategies
            try:
                ctx.bars(hidden)
                self.problems.append(f"{hidden} bars visible to the strategy")
            except KeyError:
                pass
        return []


def test_engine_shows_only_closed_bars_and_never_the_execution_bars(
    frames: dict[str, pd.DataFrame],
) -> None:
    spy = Spy()
    engine_run(spy, frames)
    assert spy.calls > 400
    assert spy.problems == []


class AsksForTheFuture(Honest):
    def generate(self, ctx: MarketContext) -> list[Signal]:
        view = ctx.bars("M15")
        view.close[len(view)]  # one bar past the latest closed bar
        return []


def test_asking_the_context_for_a_future_bar_fails(frames: dict[str, pd.DataFrame]) -> None:
    with pytest.raises(IndexError):
        engine_run(AsksForTheFuture(), frames)  # the engine does not hide the error


def test_one_cut_misses_the_view_escape_so_the_default_uses_many(
    frames: dict[str, pd.DataFrame],
) -> None:
    # Documents why DEFAULT_CUTS is large: this cheat only shows at a cut boundary.
    assert check_lookahead(ReadsBehindTheView, "XAUUSD", frames, cuts=1) == []
    assert check_lookahead(ReadsBehindTheView, "XAUUSD", frames) != []
