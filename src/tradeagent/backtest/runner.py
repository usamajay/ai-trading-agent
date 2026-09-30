"""Load everything one backtest needs from disk, run it, and fingerprint the data."""

import dataclasses
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradeagent.backtest.costs import CostModel
from tradeagent.backtest.dataset import Dataset, flag_bars, load_bars, load_dataset
from tradeagent.backtest.engine import BacktestResult, run_backtest
from tradeagent.config import AppConfig, SplitName
from tradeagent.data.store import BarStore
from tradeagent.strategies.base import Strategy

FINGERPRINT_COLUMNS = ["open", "high", "low", "close", "tick_volume", "spread"]


@dataclass(frozen=True)
class RunInputs:
    dataset: Dataset
    context: dict[str, pd.DataFrame]  # the strategy's other timeframes
    exec_bars: pd.DataFrame | None  # flagged M5 bars (strategies above M5)
    m1: pd.DataFrame | None  # side-by-side tie check only
    costs: CostModel
    data_hash: str  # fingerprint of every bar used


def bars_fingerprint(frames: dict[str, pd.DataFrame]) -> str:
    """sha256 over the bars' times and prices, in a fixed order. Flags and derived
    columns are left out: they follow from the config, which config_hash covers."""
    digest = hashlib.sha256()
    for name in sorted(frames):
        df = frames[name]
        digest.update(f"{name}:{len(df)};".encode())
        digest.update(df["time_utc"].to_numpy("datetime64[ns]").view("int64").tobytes())
        for column in FINGERPRINT_COLUMNS:
            digest.update(df[column].to_numpy(np.float64).tobytes())
    return digest.hexdigest()


def load_inputs(
    cfg: AppConfig, store: BarStore, strategy: Strategy, symbol: str, split: SplitName
) -> RunInputs:
    """Out-of-sample is refused in Phase 2 (SplitError from load_dataset)."""
    decision_tf = strategy.timeframes[0]
    warmup = max(200, 2 * strategy.lookback_bars)
    dataset = load_dataset(cfg, store, symbol, decision_tf, split, strategy.style, warmup)
    first, end = dataset.bars["time_utc"].iloc[0], dataset.end_utc

    def window(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["time_utc"] >= first) & (df["time_utc"] < end)].reset_index(drop=True)

    context = {tf: window(load_bars(store, symbol, tf)) for tf in strategy.timeframes[1:]}
    exec_bars = None
    if decision_tf != "M5":
        exec_bars = flag_bars(
            window(store.read(symbol, "M5")),
            "M5",
            symbol,
            dataset.exclusions,
            cfg.exclusions,
            cfg.settings.backtest,
            flat_before_weekend=strategy.style in ("scalp", "intraday"),
            period=(dataset.start_utc, dataset.end_utc),
        )
    m1 = window(store.read(symbol, "M1"))
    frames = {f"decision_{decision_tf}": dataset.bars, **context}
    if exec_bars is not None:
        frames["exec_M5"] = exec_bars
    if not m1.empty:
        frames["M1"] = m1
    return RunInputs(
        dataset=dataset,
        context=context,
        exec_bars=exec_bars,
        m1=m1 if not m1.empty else None,
        costs=CostModel.from_config(cfg, symbol),
        data_hash=bars_fingerprint(frames),
    )


def stressed(costs: CostModel, multiple: float) -> CostModel:
    """Spread x `multiple` on top of the normal margin. Slippage is a fraction of the
    spread, so it grows by the same multiple; swap and commission are unchanged.
    Used only for fills: signal checks and sizing keep the normal costs."""
    return dataclasses.replace(
        costs,
        spread_margin_multiple=costs.spread_margin_multiple * multiple,
        spread_margin_points=costs.spread_margin_points * multiple,
    )


def run_inputs(
    cfg: AppConfig,
    strategy: Strategy,
    inputs: RunInputs,
    costs: CostModel | None = None,
    risk_basics: bool = True,
) -> BacktestResult:
    return run_backtest(
        strategy,
        inputs.dataset,
        costs or inputs.costs,
        cfg.risk,
        cfg.settings.backtest,
        context=inputs.context,
        exec_bars=inputs.exec_bars,
        m1_bars=inputs.m1,
        risk_basics=risk_basics,
        decision_costs=inputs.costs,
    )


def run_from_store(
    cfg: AppConfig,
    store: BarStore,
    strategy: Strategy,
    symbol: str,
    split: SplitName,
    risk_basics: bool = True,
) -> BacktestResult:
    """Backtest `strategy` on one split of `symbol` using the stored bars."""
    return run_inputs(
        cfg, strategy, load_inputs(cfg, store, strategy, symbol, split), risk_basics=risk_basics
    )
