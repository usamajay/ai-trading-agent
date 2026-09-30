"""p_win and expected value per signal from a history of comparable closed trades (SPEC §5).

- Outcome of a closed trade: `tp` exit = win, `sl` exit = loss, anything else
  (weekend, Friday or time exits) = timeout.
- p_win = mean of the Beta posterior with prior Beta(k/2, k/2): with few trades the
  estimate is pulled toward 50%, with many it approaches the observed rate.
- The credible interval on p_win comes from the same posterior; its lower bound is
  what the entry rule checks. confidence = 1 - interval width.
- History R multiples are net of costs, so ev_r needs no separate cost term;
  `extra_cost_r` covers a signal that costs more than the history did.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

WIN, LOSS, TIMEOUT = "win", "loss", "timeout"


def outcome(exit_reason: str) -> str:
    if exit_reason == "tp":
        return WIN
    if exit_reason == "sl":
        return LOSS
    return TIMEOUT


@dataclass(frozen=True)
class Estimate:
    n: int
    wins: int
    losses: int
    timeouts: int
    p_win: float
    p_win_low: float
    p_win_high: float
    p_loss: float
    p_timeout: float
    avg_win_r: float
    avg_loss_r: float  # size of the average loss, as a positive number
    avg_timeout_r: float
    ev_r: float

    @property
    def confidence(self) -> float:
        return 1.0 - (self.p_win_high - self.p_win_low)


def estimate(
    exit_reasons: Sequence[str],
    r_multiples: Sequence[float],
    *,
    prior_strength: float,
    credible_level: float,
    planned_rr: float,
    extra_cost_r: float = 0.0,
) -> Estimate:
    """Estimate for a new signal from closed trades (net R) of the same setup."""
    if len(exit_reasons) != len(r_multiples):
        raise ValueError("exit_reasons and r_multiples must have the same length")
    kinds = np.array([outcome(x) for x in exit_reasons], dtype=object)
    r = np.asarray(r_multiples, dtype=float)
    n = len(r)
    wins, losses = int((kinds == WIN).sum()), int((kinds == LOSS).sum())
    timeouts = n - wins - losses

    a = prior_strength / 2 + wins
    b = prior_strength / 2 + (n - wins)
    tail = (1 - credible_level) / 2
    p_win = a / (a + b)
    p_timeout = timeouts / n if n else 0.0
    p_loss = 1.0 - p_win - p_timeout

    avg_win = float(r[kinds == WIN].mean()) if wins else planned_rr
    avg_loss = float(-r[kinds == LOSS].mean()) if losses else 1.0
    avg_timeout = float(r[kinds == TIMEOUT].mean()) if timeouts else 0.0
    ev = p_win * avg_win - p_loss * avg_loss + p_timeout * avg_timeout - extra_cost_r
    return Estimate(
        n=n,
        wins=wins,
        losses=losses,
        timeouts=timeouts,
        p_win=float(p_win),
        p_win_low=float(beta.ppf(tail, a, b)),
        p_win_high=float(beta.ppf(1 - tail, a, b)),
        p_loss=float(p_loss),
        p_timeout=float(p_timeout),
        avg_win_r=avg_win,
        avg_loss_r=avg_loss,
        avg_timeout_r=avg_timeout,
        ev_r=float(ev),
    )
