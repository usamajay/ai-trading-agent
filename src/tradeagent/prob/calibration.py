"""Calibration of p_win / ev_r on a run's trades, walked forward in time (SPEC §5).

Each trade is predicted from the trades of the same run that **closed before it
entered** (what would have been known at the time), then compared with what happened:
- Brier score of p_win vs the outcome (1 = take-profit hit), next to two references:
  always 50% and the run's final hit rate (hindsight, the best constant forecast);
- reliability table: predicted p_win in 10 buckets vs the actual hit rate;
- predicted ev_r vs realised R, and what the SPEC §5 entry gate would have allowed.
Out-of-sample runs are refused: OOS is opened only by the Phase 7 gates.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradeagent.config import EntryRules
from tradeagent.prob.estimator import WIN, estimate, outcome
from tradeagent.prob.gate import entry_rejections


class SplitRefused(ValueError):
    """Raised for runs on a split the probability engine may not look at."""


def planned_rr(trades: pd.DataFrame) -> pd.Series:
    reward = (trades["take_profit"] - trades["entry_price"]).abs()
    risk = (trades["entry_price"] - trades["stop_loss"]).abs()
    return reward / risk


def entry_cost_r(trades: pd.DataFrame) -> pd.Series:
    """Typical cost of each trade in R, as known at entry: spread + slippage + commission."""
    return (trades["spread_cost"] + trades["slippage_cost"] + trades["commission"]) / trades[
        "risk_usd"
    ]


def walk_forward(trades: pd.DataFrame, rules: EntryRules) -> pd.DataFrame:
    """One row per trade: prediction from earlier closed trades, gate result, outcome."""
    t = trades.sort_values("entry_time").reset_index(drop=True)
    rr = planned_rr(t)
    cost = entry_cost_r(t)
    rows = []
    for i, trade in t.iterrows():
        known = t[t["exit_time"] < trade["entry_time"]]
        est = estimate(
            known["exit_reason"].tolist(),
            known["r_multiple"].tolist(),
            prior_strength=rules.prior_strength,
            credible_level=rules.credible_level,
            planned_rr=float(rr.iloc[i]),
            cost_r=float(cost.iloc[i]),
        )
        codes = [r.code for r in entry_rejections(est, rules)]
        rows.append(
            {
                "entry_time": trade["entry_time"],
                "n_known": est.n,
                "p_win": est.p_win,
                "p_win_low": est.p_win_low,
                "ev_r": est.ev_r,
                "gate_pass": not codes,
                "gate_codes": ",".join(codes),
                "win": int(outcome(trade["exit_reason"]) == WIN),
                "r_multiple": float(trade["r_multiple"]),
            }
        )
    return pd.DataFrame(rows)


def brier(p: pd.Series, y: pd.Series) -> float:
    return float(((p - y) ** 2).mean())


@dataclass(frozen=True)
class CalibrationSummary:
    trades: int
    hit_rate: float
    brier: float
    brier_half: float  # always predict 50%
    brier_hindsight: float  # always predict the final hit rate
    mean_p_win: float
    mean_ev_r: float
    realised_r: float
    gate_passed: int
    gate_realised_r: float | None


def summarise(walk: pd.DataFrame) -> CalibrationSummary:
    if walk.empty:
        raise ValueError("no trades to calibrate")
    y = walk["win"]
    rate = float(y.mean())
    passed = walk[walk["gate_pass"]]
    return CalibrationSummary(
        trades=len(walk),
        hit_rate=rate,
        brier=brier(walk["p_win"], y),
        brier_half=brier(pd.Series(0.5, index=y.index), y),
        brier_hindsight=brier(pd.Series(rate, index=y.index), y),
        mean_p_win=float(walk["p_win"].mean()),
        mean_ev_r=float(walk["ev_r"].mean()),
        realised_r=float(walk["r_multiple"].mean()),
        gate_passed=len(passed),
        gate_realised_r=float(passed["r_multiple"].mean()) if len(passed) else None,
    )


def reliability(walk: pd.DataFrame, buckets: int = 10) -> pd.DataFrame:
    """Predicted p_win in equal-width buckets vs the actual hit rate (empty buckets dropped)."""
    edges = np.linspace(0, 1, buckets + 1)
    idx = np.clip(np.digitize(walk["p_win"], edges[1:-1]), 0, buckets - 1)
    g = walk.assign(bucket=idx).groupby("bucket")
    out = pd.DataFrame(
        {
            "trades": g.size(),
            "mean_p_win": g["p_win"].mean(),
            "hit_rate": g["win"].mean(),
        }
    )
    out.index = [f"{edges[b]:.1f}-{edges[b + 1]:.1f}" for b in out.index]
    return out


def check_split(split: str) -> None:
    if split == "out_of_sample":
        raise SplitRefused("out-of-sample runs are opened only by the Phase 7 gates")


def _r(x: float | None) -> str:
    return "-" if x is None else f"{x:+.3f}"


def markdown_section(title: str, s: CalibrationSummary, rel: pd.DataFrame) -> str:
    lines = [
        f"### {title}",
        "",
        (
            f"- Trades {s.trades}, take-profit hit rate {s.hit_rate:.1%}; "
            f"mean predicted p_win {s.mean_p_win:.1%}"
        ),
        (
            f"- Brier {s.brier:.4f} (always 50%: {s.brier_half:.4f}; "
            f"hindsight constant: {s.brier_hindsight:.4f}; lower is better)"
        ),
        f"- Mean predicted ev_r {_r(s.mean_ev_r)} R vs realised {_r(s.realised_r)} R",
        (
            f"- Entry gate would have allowed {s.gate_passed} trades "
            f"(realised {_r(s.gate_realised_r)} R)"
        ),
        "",
        "| predicted p_win | trades | mean predicted | actual hit rate |",
        "|---|---|---|---|",
    ]
    for bucket, row in rel.iterrows():
        lines.append(
            f"| {bucket} | {int(row['trades'])} | {row['mean_p_win']:.1%} | {row['hit_rate']:.1%} |"
        )
    return "\n".join(lines) + "\n"
