"""Backtest metrics (SPEC §7.3). Pure functions over a BacktestResult.

Definitions (the same text is in SPEC §7.3):
- Trade stats use net P&L (after spread, slippage, swap, commission). A win is
  net P&L > 0. Profit factor = sum of winning net P&L / |sum of losing net P&L|
  (infinite when there are no losses). Expectancy (R) = mean R multiple;
  expectancy ($) = mean net P&L per trade.
- Fewer than MIN_TRADES (30) trades: the result is flagged "insufficient sample".
- 95% confidence intervals for win rate, expectancy (R) and profit factor: trade
  bootstrap (resample the trades with replacement, BOOTSTRAP_RUNS times, fixed
  seed), 2.5th and 97.5th percentiles without interpolation (an upper profit-factor
  bound can be infinite when many resamples contain no losing trade).
- Equity for drawdown, Sharpe, Sortino and Calmar is the **daily mark-to-market**
  equity: one value per trading day (17:00 -> 17:00 New York) at its last bar's
  close, with an open trade valued on its exit side plus swap so far, less
  commission. Closed-trade drawdown is also reported, for comparison.
- Max drawdown = largest fall from a running peak (starting balance included),
  in % of that peak and in $. Drawdown duration = most trading days from a peak
  until equity is back at or above it (or until the end, if never recovered).
- Daily return = equity_d / equity_(d-1) - 1 (first day vs the starting balance),
  over every trading day in the test, flat days included.
  **Annualisation convention:** 252 trading days per year, risk-free rate 0.
  Sharpe = mean(daily) / sample std (ddof 1) x sqrt(252).
  Sortino = mean(daily) / downside deviation x sqrt(252), where downside deviation
  = sqrt(mean(min(daily, 0)^2)) over all days (target 0).
  CAGR = (final / start)^(252 / trading days) - 1. Calmar = CAGR / max drawdown
  (mark-to-market). Recovery factor = net profit / max drawdown $.
- Costs as % of gross profit: gross profit = sum over trades of max(pre-cost
  P&L, 0), where pre-cost P&L = gross P&L + spread cost + slippage cost (the price
  move before any cost). Spread, slippage, swap paid (-swap) and commission are
  each shown as % of that gross profit, plus their total.
- Exposure = decision bars with a trade open / decision bars in the split.
- Sessions by entry time, New York: Asia 18:00-03:00, London 03:00-08:00,
  New York 08:00-17:00.
"""

import math
from typing import Any

import numpy as np
import pandas as pd

from tradeagent.backtest.engine import BacktestResult
from tradeagent.data.market_hours import NEW_YORK

TRADING_DAYS_PER_YEAR = 252
MIN_TRADES = 30
BOOTSTRAP_RUNS = 2000
BOOTSTRAP_SEED = 20260930
CONFIDENCE = 0.95

CONVENTIONS = {
    "equity": "daily mark-to-market at each trading day's close (17:00 New York)",
    "annualisation": "252 trading days per year, risk-free rate 0; "
    "Sharpe = mean/std(ddof 1) x sqrt(252); Sortino = mean/downside deviation x sqrt(252)",
    "confidence_intervals": f"trade bootstrap, {BOOTSTRAP_RUNS} runs, seed {BOOTSTRAP_SEED}, "
    "2.5th-97.5th percentiles",
    "insufficient_sample": f"fewer than {MIN_TRADES} trades",
    "costs": "% of gross profit = sum of positive pre-cost trade P&L",
}


# --- trade statistics ----------------------------------------------------------------


def profit_factor(net: np.ndarray) -> float | None:
    if len(net) == 0:
        return None
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    if losses == 0:
        return math.inf if wins > 0 else None
    return float(wins / losses)


def longest_losing_streak(net: np.ndarray) -> int:
    longest = current = 0
    for value in net:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def trade_stats(trades: pd.DataFrame) -> dict[str, Any]:
    net = trades["net_pnl"].to_numpy(float)
    r = trades["r_multiple"].to_numpy(float)
    n = len(trades)
    wins, losses = r[net > 0], r[net < 0]
    return {
        "trades": n,
        "insufficient_sample": n < MIN_TRADES,
        "win_rate_pct": float((net > 0).mean() * 100) if n else None,
        "profit_factor": profit_factor(net),
        "expectancy_r": float(r.mean()) if n else None,
        "expectancy_usd": float(net.mean()) if n else None,
        "avg_win_r": float(wins.mean()) if len(wins) else None,
        "avg_loss_r": float(losses.mean()) if len(losses) else None,
        "net_pnl_usd": float(net.sum()),
        "longest_losing_streak": longest_losing_streak(net),
    }


def bootstrap_ci(
    trades: pd.DataFrame, runs: int = BOOTSTRAP_RUNS, seed: int = BOOTSTRAP_SEED
) -> dict[str, list[float] | None]:
    """95% intervals for win rate (%), expectancy (R) and profit factor."""
    n = len(trades)
    if n == 0:
        return {"win_rate_pct": None, "expectancy_r": None, "profit_factor": None}
    net = trades["net_pnl"].to_numpy(float)
    r = trades["r_multiple"].to_numpy(float)
    rng = np.random.default_rng(seed)
    win_rate, expectancy, pf = [], [], []
    for start in range(0, runs, 250):  # chunks keep memory small for long trade lists
        idx = rng.integers(0, n, size=(min(250, runs - start), n))
        sample_net, sample_r = net[idx], r[idx]
        win_rate.append((sample_net > 0).mean(axis=1) * 100)
        expectancy.append(sample_r.mean(axis=1))
        gains = np.where(sample_net > 0, sample_net, 0).sum(axis=1)
        losses = -np.where(sample_net < 0, sample_net, 0).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pf.append(np.where(losses > 0, gains / losses, np.inf))
    tail = (1 - CONFIDENCE) / 2 * 100

    def interval(values: list[np.ndarray]) -> list[float]:
        # No interpolation: each bound is an actual resampled value, so an infinite
        # profit factor (a resample without losses) stays inf instead of inf - inf = nan.
        v = np.concatenate(values)
        low = np.percentile(v, tail, method="inverted_cdf")
        high = np.percentile(v, 100 - tail, method="inverted_cdf")
        return [float(low), float(high)]

    return {
        "win_rate_pct": interval(win_rate),
        "expectancy_r": interval(expectancy),
        "profit_factor": interval(pf),
    }


# --- equity ----------------------------------------------------------------------------


def drawdown(equity: np.ndarray, start: float) -> dict[str, float | int]:
    """Max fall from the running peak (the start included): % of peak, $, and the
    longest time under water in periods (until back at or above the peak)."""
    values = np.concatenate([[start], np.asarray(equity, float)])
    peak, peak_at = values[0], 0
    max_pct = max_usd = 0.0
    longest, under = 0, False
    for k in range(1, len(values)):
        v = values[k]
        if v >= peak:
            if under:
                longest = max(longest, k - peak_at)
            peak, peak_at, under = v, k, False
        else:
            under = True
            max_usd = max(max_usd, peak - v)
            max_pct = max(max_pct, (peak - v) / peak * 100)
    if under:
        longest = max(longest, len(values) - 1 - peak_at)
    return {"max_dd_pct": float(max_pct), "max_dd_usd": float(max_usd), "max_dd_duration": longest}


def daily_returns(equity: np.ndarray, start: float) -> np.ndarray:
    values = np.concatenate([[start], np.asarray(equity, float)])
    return values[1:] / values[:-1] - 1


def sharpe_sortino(returns: np.ndarray) -> tuple[float | None, float | None]:
    if len(returns) < 2:
        return None, None
    scale = math.sqrt(TRADING_DAYS_PER_YEAR)
    mean = returns.mean()
    std = returns.std(ddof=1)
    downside = math.sqrt(float((np.minimum(returns, 0) ** 2).mean()))
    sharpe = float(mean / std * scale) if std > 0 else None
    sortino = float(mean / downside * scale) if downside > 0 else None
    return sharpe, sortino


def cagr(start: float, final: float, trading_days: int) -> float | None:
    if trading_days <= 0 or start <= 0 or final <= 0:
        return None
    return (final / start) ** (TRADING_DAYS_PER_YEAR / trading_days) - 1


# --- costs, balances and breakdowns ------------------------------------------------------


def cost_breakdown(trades: pd.DataFrame) -> dict[str, float | None]:
    pre_cost = trades["gross_pnl"] + trades["spread_cost"] + trades["slippage_cost"]
    gross_profit = float(pre_cost.clip(lower=0).sum())
    costs = {
        "spread": float(trades["spread_cost"].sum()),
        "slippage": float(trades["slippage_cost"].sum()),
        "swap": float(-trades["swap"].sum()),  # paid swap is a cost; a credit is negative
        "commission": float(trades["commission"].sum()),
    }
    costs["total"] = sum(costs.values())
    out: dict[str, float | None] = {"gross_profit_usd": gross_profit}
    for name, usd in costs.items():
        out[f"{name}_usd"] = usd
        out[f"{name}_pct_of_gross_profit"] = usd / gross_profit * 100 if gross_profit > 0 else None
    per_trade_r = (
        trades["spread_cost"] + trades["slippage_cost"] - trades["swap"] + trades["commission"]
    ) / trades["risk_usd"]
    out["avg_cost_r"] = float(per_trade_r.mean()) if len(trades) else None
    return out


def session_of(times: pd.Series) -> pd.Series:
    hour = times.dt.tz_convert(NEW_YORK).dt.hour
    return pd.Series(
        np.select([(hour >= 18) | (hour < 3), hour < 8], ["asia", "london"], "new_york"),
        index=times.index,
    )


def breakdown(trades: pd.DataFrame, keys: pd.Series) -> dict[str, dict[str, Any]]:
    return {
        str(key): trade_stats(group) for key, group in trades.groupby(keys.to_numpy(), sort=True)
    }


def direction_breakdown(trades: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per direction: trade stats plus average cost and swap per trade in R (H2, swap)."""
    out = {}
    for key, group in trades.groupby(trades["direction"].to_numpy(), sort=True):
        stats = trade_stats(group)
        risk = group["risk_usd"]
        stats["avg_cost_r"] = float(
            (
                (
                    group["spread_cost"]
                    + group["slippage_cost"]
                    - group["swap"]
                    + group["commission"]
                )
                / risk
            ).mean()
        )
        stats["avg_swap_paid_r"] = float((-group["swap"] / risk).mean())
        out[str(key)] = stats
    return out


EXIT_STAGES = ("target", "first stop", "stop after move 1", "stop after move 2")


def exit_stage(trades: pd.DataFrame) -> pd.Series:
    """How each trade ended, counting stop moves: target / stop at stage 0, 1, 2 / other."""
    reason = trades["exit_reason"]
    stage = trades["stop_stage"] if "stop_stage" in trades else pd.Series(0, index=trades.index)
    stopped = reason.isin(["sl", "sl_gap"])
    return pd.Series(
        np.select(
            [
                reason == "tp",
                stopped & (stage == 0),
                stopped & (stage == 1),
                stopped & (stage >= 2),
            ],
            list(EXIT_STAGES),
            reason.astype(str),
        ),
        index=trades.index,
    )


def regime_breakdown(trades: pd.DataFrame, bars: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Trade stats by the research regime labels of each signal's decision bar."""
    from tradeagent.features.regime import DETECTOR_VERSION, trade_regimes

    if not len(trades):
        return {"detector_version": DETECTOR_VERSION, "trend": {}, "vol": {}}
    labels = trade_regimes(trades, bars, timeframe)
    return {
        "detector_version": DETECTOR_VERSION,
        "trend": breakdown(trades, labels["trend"]),
        "vol": breakdown(trades, labels["vol"]),
    }


def min_balance_summary(result: BacktestResult) -> dict[str, float | int | None]:
    values = result.trades["min_balance"]
    return {
        "max": float(values.max()) if len(values) else None,
        "p95": float(values.quantile(0.95)) if len(values) else None,
        "median": float(values.median()) if len(values) else None,
        "trades_above_start_balance": int((values > result.start_balance).sum()),
        "signals_too_small_for_min_lot": int(result.counts.get("rejected_min_lot", 0)),
    }


# --- everything together -----------------------------------------------------------------


def compute_metrics(result: BacktestResult) -> dict[str, Any]:
    trades = result.trades
    start = float(result.start_balance)
    final = float(result.final_balance)
    daily = result.daily["equity"].to_numpy(float)
    closed = result.equity["equity"].to_numpy(float)[1:]  # after each closed trade
    returns = daily_returns(daily, start)
    sharpe, sortino = sharpe_sortino(returns)
    mtm = drawdown(daily, start)
    closed_dd = drawdown(closed, start)
    growth = cagr(start, final, len(daily))
    net_profit = final - start

    return {
        "conventions": CONVENTIONS,
        "trades": trade_stats(trades),
        "confidence_95": bootstrap_ci(trades),
        "equity": {
            "start_balance": start,
            "final_balance": final,
            "net_profit_usd": net_profit,
            "return_pct": net_profit / start * 100,
            "cagr_pct": float(growth * 100) if growth is not None else None,
            "trading_days": len(daily),
            "max_dd_pct": mtm["max_dd_pct"],
            "max_dd_usd": mtm["max_dd_usd"],
            "max_dd_duration_days": mtm["max_dd_duration"],
            "closed_trade_max_dd_pct": closed_dd["max_dd_pct"],
            "closed_trade_max_dd_usd": closed_dd["max_dd_usd"],
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": float(growth / (mtm["max_dd_pct"] / 100))
            if growth is not None and mtm["max_dd_pct"] > 0
            else None,
            "recovery_factor": float(net_profit / mtm["max_dd_usd"])
            if mtm["max_dd_usd"] > 0
            else None,
            "exposure_pct": float(trades["bars_held"].sum()) / result.bars_in_split * 100
            if result.bars_in_split
            else 0.0,
        },
        "costs": cost_breakdown(trades),
        "min_balance": min_balance_summary(result),
        "by_session": breakdown(trades, session_of(trades["entry_time"])) if len(trades) else {},
        "by_year": breakdown(trades, trades["entry_time"].map(lambda t: t.year))
        if len(trades)
        else {},
        "by_weekend_hold": breakdown(
            trades, trades["held_over_weekend"].map({True: "held", False: "not_held"})
        )
        if len(trades)
        else {},
        "by_direction": direction_breakdown(trades) if len(trades) else {},
        "by_exit_stage": breakdown(trades, exit_stage(trades)) if len(trades) else {},
        "by_exit_reason": {str(k): int(v) for k, v in trades["exit_reason"].value_counts().items()},
        "by_regime": None,  # needs the regime detector (Phase 8)
        "counts": dict(sorted(result.counts.items())),
    }
