"""Look-ahead check (SPEC §7.1, docs/PHASE_2_TASKS.md 2.6): does a strategy peek?

The context a strategy gets holds only closed bars, so asking it for a future bar
raises an error. But Python cannot make the future physically unreachable (a
strategy could read the full array behind a view, or compute an indicator in
`prepare()` with a centred window). The **truncation test** catches all of these:

    run the strategy on the full data, and again on data cut at time T.
    Everything it produced up to T must be identical in both runs.

If adding future bars changes a past signal or a past indicator value, the
strategy used the future. Cost: one extra run per cut. Checked here:
  - indicator columns added by `prepare()` (rows closed by T), and
  - the signals from `generate()` at every decision bar closed by T.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradeagent.data.mt5_client import TIMEFRAMES
from tradeagent.strategies.base import Frames, Signal, Strategy

SignalLog = list[tuple[pd.Timestamp, tuple[Signal, ...]]]

# Indicator leaks show up at almost any cut. A strategy that reads the array behind
# its view only differs at the bar right at a cut, and only when the peek changes
# its decision (roughly half the time), so it needs many cuts: measured on test
# data it was missed with 1-3 cuts and caught with 6+. 20 cuts leave about a
# one-in-a-million chance of missing it.
DEFAULT_CUTS = 20


@dataclass(frozen=True)
class LookaheadProblem:
    cut_utc: pd.Timestamp
    what: str  # "feature" or "signal"
    where: str  # e.g. "M15 column ema_20 at 2026-01-06 10:15" or a decision time
    detail: str


def _end_times(df: pd.DataFrame, timeframe: str) -> pd.Series:
    if "end_utc" in df.columns:
        return df["end_utc"]
    return df["time_utc"] + pd.Timedelta(TIMEFRAMES[timeframe][1])


def truncate(frames: dict[str, pd.DataFrame], cut: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Only the bars that had closed by `cut` (what a live system would have)."""
    return {tf: df[_end_times(df, tf) <= cut].reset_index(drop=True) for tf, df in frames.items()}


def signal_log(
    strategy: Strategy, symbol: str, frames: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], SignalLog]:
    """prepare() once, then generate() after every decision bar; returns both outputs."""
    prepared = strategy.prepare({tf: df.copy() for tf, df in frames.items()})
    view = Frames(symbol, prepared)
    decision_tf = strategy.timeframes[0]
    log: SignalLog = []
    for now in _end_times(frames[decision_tf], decision_tf):
        log.append((now, tuple(strategy.generate(view.context(now)))))
    return prepared, log


def check_lookahead(
    make_strategy: Callable[[], Strategy],
    symbol: str,
    frames: dict[str, pd.DataFrame],
    cuts: Sequence[pd.Timestamp] | int = DEFAULT_CUTS,
) -> list[LookaheadProblem]:
    """Truncation test. `make_strategy` must build a fresh strategy each call (same
    seed/params), so both runs start from the same state. Returns [] if clean.

    `cuts`: times to cut at, or a number of evenly spaced decision-bar close times.
    """
    decision_tf = make_strategy().timeframes[0]
    ends = _end_times(frames[decision_tf], decision_tf).reset_index(drop=True)
    if isinstance(cuts, int):
        picks = np.linspace(len(ends) // 4, len(ends) - 2, num=cuts).astype(int)
        cut_times = [ends[int(k)] for k in sorted(set(picks))]
    else:
        cut_times = list(cuts)

    full_features, full_log = signal_log(make_strategy(), symbol, frames)
    problems: list[LookaheadProblem] = []
    for cut in cut_times:
        cut_features, cut_log = signal_log(make_strategy(), symbol, truncate(frames, cut))
        problems += _compare_features(cut, full_features, cut_features)
        problems += _compare_signals(cut, full_log, cut_log)
    return problems


def _compare_features(
    cut: pd.Timestamp, full: dict[str, pd.DataFrame], truncated: dict[str, pd.DataFrame]
) -> list[LookaheadProblem]:
    problems = []
    for tf, cut_df in truncated.items():
        head = full[tf].iloc[: len(cut_df)]
        for column in cut_df.columns:
            if column not in head.columns or not pd.api.types.is_numeric_dtype(cut_df[column]):
                continue
            a = head[column].to_numpy(dtype=float)
            b = cut_df[column].to_numpy(dtype=float)
            same = np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)
            if not same.all():
                k = int(np.argmin(same))
                problems.append(
                    LookaheadProblem(
                        cut_utc=cut,
                        what="feature",
                        where=f"{tf} column {column} at {cut_df['time_utc'].iloc[k]}",
                        detail=f"{a[k]!r} with future data vs {b[k]!r} without",
                    )
                )
    return problems


def _compare_signals(
    cut: pd.Timestamp, full: SignalLog, truncated: SignalLog
) -> list[LookaheadProblem]:
    before = [(now, sigs) for now, sigs in full if now <= cut]
    problems = []
    if len(before) != len(truncated):
        problems.append(LookaheadProblem(cut, "signal", str(cut), "different number of decisions"))
    for (now, a), (_, b) in zip(before, truncated, strict=False):
        if a != b:
            problems.append(
                LookaheadProblem(
                    cut_utc=cut,
                    what="signal",
                    where=str(now),
                    detail=f"{len(a)} signal(s) with future data vs {len(b)} without: {a} / {b}",
                )
            )
            break  # the first difference is enough to show the leak
    return problems
