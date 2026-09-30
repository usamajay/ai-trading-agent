"""Statistical scans over train trades (docs/PHASE_6_TASKS.md 6.6).

For each recorded train run (latest per strategy/variant and symbol), trades are split
into groups by what is **known when the trade is entered**: entry hour (UTC), session,
weekday, direction, and the research regime labels (trend, volatility) of the signal
bar. Holding time and "held over the weekend" are left out on purpose: they are only
known after the exit (fast exits are mostly stop-outs), so they "explain" results
without being usable as a rule. A group is **flagged** only when:
- it has at least `MIN_GROUP_TRADES` trades and so does the rest of that run, and
- its expectancy differs from the rest of the run by more than noise: Welch z-test
  (normal approximation), p below 0.05 / (number of groups tested in the whole scan)
  (Bonferroni: the more groups looked at, the stronger a difference must be).
Each flag becomes a `scan` hypothesis (status `proposed`): an idea to test, not a result.
It was found on train, so a later train pass is not independent evidence; validation is.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tradeagent.backtest.metrics import session_of

MIN_GROUP_TRADES = 30
FAMILY_ALPHA = 0.05

DIMENSIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "hour_utc": lambda t: t["entry_time"].dt.tz_convert("UTC").dt.hour.astype(str),
    "session": lambda t: session_of(t["entry_time"]),
    "weekday": lambda t: t["entry_time"].dt.tz_convert("UTC").dt.day_name(),
    "direction": lambda t: t["direction"].astype(str),
}


@dataclass(frozen=True)
class GroupTest:
    run_id: str
    strategy: str
    symbol: str
    timeframe: str
    dimension: str
    group: str
    n: int
    mean_r: float
    rest_n: int
    rest_mean_r: float
    z: float
    p: float


def welch(group: np.ndarray, rest: np.ndarray) -> tuple[float, float]:
    """(z, two-sided p) for the difference in means, normal approximation."""
    se = np.sqrt(group.var(ddof=1) / len(group) + rest.var(ddof=1) / len(rest))
    if not np.isfinite(se) or se == 0:
        return 0.0, 1.0
    z = (group.mean() - rest.mean()) / se
    return float(z), float(2 * norm.sf(abs(z)))


def group_tests(
    run: dict[str, Any], trades: pd.DataFrame, labels: dict[str, pd.Series]
) -> list[GroupTest]:
    """Every group with enough trades on both sides, in one run."""
    r = trades["r_multiple"].to_numpy(float)
    out = []
    for dimension, keys in labels.items():
        values = keys.to_numpy()
        for group in sorted(set(values)):
            mask = values == group
            n, rest_n = int(mask.sum()), int((~mask).sum())
            if n < MIN_GROUP_TRADES or rest_n < MIN_GROUP_TRADES:
                continue
            z, p = welch(r[mask], r[~mask])
            out.append(
                GroupTest(
                    run["run_id"],
                    run["strategy"],
                    run["symbol"],
                    run["timeframe"],
                    dimension,
                    str(group),
                    n,
                    float(r[mask].mean()),
                    rest_n,
                    float(r[~mask].mean()),
                    z,
                    p,
                )
            )
    return out


def run_labels(trades: pd.DataFrame, regimes: pd.DataFrame | None) -> dict[str, pd.Series]:
    labels = {name: f(trades) for name, f in DIMENSIONS.items()}
    if regimes is not None:
        labels["trend_regime"] = regimes["trend"]
        labels["vol_regime"] = regimes["vol"]
    return labels


@dataclass(frozen=True)
class ScanResult:
    tests: list[GroupTest]
    threshold: float  # Bonferroni p threshold for the whole scan

    @property
    def flags(self) -> list[GroupTest]:
        return [t for t in self.tests if t.p < self.threshold]


def scan(runs: list[tuple[dict[str, Any], pd.DataFrame, dict[str, pd.Series]]]) -> ScanResult:
    tests = [t for run, trades, labels in runs for t in group_tests(run, trades, labels)]
    threshold = FAMILY_ALPHA / len(tests) if tests else 0.0
    return ScanResult(tests, threshold)


def hypothesis_for(t: GroupTest, threshold: float) -> tuple[str, str, str]:
    """(id, text, rationale) for a flagged group; the id is stable across re-scans."""
    key = f"{t.strategy}|{t.symbol}|{t.timeframe}|{t.dimension}|{t.group}"
    hid = "S" + hashlib.sha256(key.encode()).hexdigest()[:6]
    better = "better" if t.mean_r > t.rest_mean_r else "worse"
    text = (
        f"{t.strategy} {t.symbol} {t.timeframe}: trades with {t.dimension} = {t.group} do "
        f"{better} ({t.mean_r:+.3f} R, n={t.n}) than the rest ({t.rest_mean_r:+.3f} R, "
        f"n={t.rest_n})"
    )
    rationale = json.dumps(
        {
            "scan": "welch_bonferroni",
            "run_id": t.run_id,
            "z": round(t.z, 3),
            "p": t.p,
            "threshold": threshold,
            "note": "found on train: validation is the independent test",
        }
    )
    return hid, text, rationale
