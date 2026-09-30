"""SPEC §5 entry rule: every failed check is listed with a reason code.

RR >= 2 is not repeated here; the risk engine checks it (`rr`).
"""

from tradeagent.config import EntryRules
from tradeagent.prob.estimator import Estimate
from tradeagent.risk.engine import Rejection

CODES = ("sample_small", "p_win_low", "ev_low")


def entry_rejections(est: Estimate, rules: EntryRules) -> list[Rejection]:
    out = []
    if est.n < rules.min_sample_trades:
        out.append(
            Rejection("sample_small", f"{est.n} comparable trades < {rules.min_sample_trades}")
        )
    if est.p_win_low < rules.min_p_win_lower:
        out.append(
            Rejection(
                "p_win_low",
                f"p_win lower bound {est.p_win_low:.3f} < {rules.min_p_win_lower:.2f}",
            )
        )
    if est.ev_r < rules.min_ev_r:
        out.append(Rejection("ev_low", f"ev {est.ev_r:+.3f} R < {rules.min_ev_r:.2f} R"))
    return out
