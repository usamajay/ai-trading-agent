"""Walk-forward check (SPEC §7.2, §7.4; docs/PHASE_7_TASKS.md 7.2).

Rolling test windows over **train + validation only** (never out-of-sample): the first
window starts `wf_history_days` after the train start, each is `wf_test_days` long, and
they step by `wf_step_days`; a last partial window is dropped. Strategies keep fixed
parameters (nothing is re-fitted per window), so the strategy runs once over the whole
span with enforced account limits, and its trades are cut into windows by entry time.
Trades entering in the embargo gap between train and validation are dropped.

Pass: at least 3 windows, and at least `wf_min_positive_share` of **all** windows
(a window with no trades does not count as working) have expectancy > 0 after costs.
"""

from dataclasses import dataclass

import pandas as pd

from tradeagent.backtest.engine import BacktestResult
from tradeagent.backtest.metrics import profit_factor
from tradeagent.backtest.runner import load_inputs, run_inputs
from tradeagent.backtest.splits import split_range
from tradeagent.config import AppConfig
from tradeagent.data.store import BarStore
from tradeagent.strategies.base import check_strategy
from tradeagent.strategies.variants import VariantSpec, build

MIN_WINDOWS = 3


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive


@dataclass(frozen=True)
class WindowResult:
    window: Window
    trades: int
    expectancy_r: float | None
    profit_factor: float | None

    @property
    def positive(self) -> bool:
        return self.expectancy_r is not None and self.expectancy_r > 0


@dataclass(frozen=True)
class WalkForwardResult:
    windows: list[WindowResult]
    min_positive_share: float

    @property
    def positive_share(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.positive for w in self.windows) / len(self.windows)

    @property
    def passed(self) -> bool:
        return len(self.windows) >= MIN_WINDOWS and self.positive_share >= self.min_positive_share


def wf_windows(
    span_start: pd.Timestamp,
    span_end: pd.Timestamp,
    history_days: int,
    test_days: int,
    step_days: int,
) -> list[Window]:
    windows = []
    start = span_start + pd.Timedelta(days=history_days)
    test, step = pd.Timedelta(days=test_days), pd.Timedelta(days=step_days)
    while start + test <= span_end:
        windows.append(Window(start, start + test))
        start += step
    return windows


def window_results(
    trades: pd.DataFrame,
    windows: list[Window],
    gap: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> list[WindowResult]:
    """Trades grouped by entry time into windows; entries inside `gap` are dropped."""
    entry = trades["entry_time"]
    keep = pd.Series(True, index=trades.index)
    if gap is not None:
        keep &= ~((entry >= gap[0]) & (entry < gap[1]))
    out = []
    for w in windows:
        t = trades[keep & (entry >= w.start) & (entry < w.end)]
        n = len(t)
        out.append(
            WindowResult(
                w,
                n,
                float(t["r_multiple"].mean()) if n else None,
                profit_factor(t["net_pnl"].to_numpy(float)) if n else None,
            )
        )
    return out


def train_val_span(
    cfg: AppConfig,
) -> tuple[pd.Timestamp, pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]]:
    """(train start, validation end, embargo gap between them)."""
    train = split_range(cfg.splits, "train")
    validation = split_range(cfg.splits, "validation")
    return train[0], validation[1], (train[1], validation[0])


def walk_forward(
    cfg: AppConfig, store: BarStore, spec: VariantSpec, symbol: str, seed: int
) -> tuple[WalkForwardResult, BacktestResult]:
    """One run over train + validation (account limits enforced), cut into windows."""
    start, end, gap = train_val_span(cfg)
    strategy = build(spec, seed)
    check_strategy(strategy)
    inputs = load_inputs(cfg, store, strategy, symbol, "train", period=(start, end))
    result = run_inputs(cfg, strategy, inputs, enforce_account_limits=True)
    v = cfg.settings.validation
    windows = wf_windows(start, end, v.wf_history_days, v.wf_test_days, v.wf_step_days)
    return WalkForwardResult(
        window_results(result.trades, windows, gap), v.wf_min_positive_share
    ), result
