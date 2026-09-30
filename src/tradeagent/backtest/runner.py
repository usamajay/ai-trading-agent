"""Load everything one backtest needs from disk and run it (used by the CLI)."""

import pandas as pd

from tradeagent.backtest.costs import CostModel
from tradeagent.backtest.dataset import flag_bars, load_bars, load_dataset
from tradeagent.backtest.engine import BacktestResult, run_backtest
from tradeagent.config import AppConfig, SplitName
from tradeagent.data.store import BarStore
from tradeagent.strategies.base import Strategy


def run_from_store(
    cfg: AppConfig,
    store: BarStore,
    strategy: Strategy,
    symbol: str,
    split: SplitName,
    risk_basics: bool = True,
) -> BacktestResult:
    """Backtest `strategy` on one split of `symbol` using the stored bars.

    Out-of-sample is refused in Phase 2 (SplitError from load_dataset).
    """
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
    return run_backtest(
        strategy,
        dataset,
        CostModel.from_config(cfg, symbol),
        cfg.risk,
        cfg.settings.backtest,
        context=context,
        exec_bars=exec_bars,
        m1_bars=m1 if not m1.empty else None,
        risk_basics=risk_basics,
    )
