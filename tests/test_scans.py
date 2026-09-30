"""Statistical scans (Phase 6.6): entry-time groups, Welch test, Bonferroni, stable ids."""

import json

import numpy as np
import pandas as pd
import pytest

from tradeagent.research.scans import (
    DIMENSIONS,
    MIN_GROUP_TRADES,
    GroupTest,
    group_tests,
    hypothesis_for,
    run_labels,
    scan,
    welch,
)

RUN = {"run_id": "r1", "strategy": "s", "symbol": "XAUUSD", "timeframe": "M15"}


def trades(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-06", periods=n, freq="7h", tz="UTC")
    return pd.DataFrame(
        {
            "entry_time": t,
            "direction": np.where(np.arange(n) % 2 == 0, "long", "short"),
            "r_multiple": rng.normal(0, 1, n),
            "bars_held": np.arange(n),
            "held_over_weekend": False,
        }
    )


def test_only_entry_time_dimensions() -> None:
    assert set(DIMENSIONS) == {"hour_utc", "session", "weekday", "direction"}
    labels = run_labels(trades(), None)
    assert set(labels) == set(DIMENSIONS)
    regimes = pd.DataFrame({"trend": ["trending"] * 120, "vol": ["low_vol"] * 120})
    assert {"trend_regime", "vol_regime"} <= set(run_labels(trades(), regimes))


def test_welch() -> None:
    a, b = np.array([1.0, 1.2, 0.8] * 20), np.array([0.0, 0.2, -0.2] * 20)
    z, p = welch(a, b)
    assert z > 10 and p < 1e-10
    assert welch(np.ones(5), np.ones(5)) == (0.0, 1.0)  # no spread at all


def test_small_groups_are_not_tested() -> None:
    t = trades(60)
    tests = group_tests(RUN, t, {"direction": t["direction"]})
    assert [(x.group, x.n) for x in tests] == [("long", 30), ("short", 30)]
    assert group_tests(RUN, t.iloc[:50], {"direction": t["direction"].iloc[:50]}) == []
    assert MIN_GROUP_TRADES == 30


def test_bonferroni_flags_only_a_real_difference() -> None:
    noise = trades(200, seed=1)
    real = trades(200, seed=2)
    real.loc[real["direction"] == "long", "r_multiple"] += 1.5
    runs = [
        ({**RUN, "run_id": "noise"}, noise, run_labels(noise, None)),
        ({**RUN, "run_id": "real"}, real, run_labels(real, None)),
    ]
    result = scan(runs)
    assert result.threshold == pytest.approx(0.05 / len(result.tests))
    assert {(f.run_id, f.dimension) for f in result.flags} == {("real", "direction")}
    assert scan([]).flags == [] and scan([]).threshold == 0.0


def test_hypothesis_text_and_stable_id() -> None:
    t = GroupTest("r1", "s", "XAUUSD", "M15", "session", "london", 40, 0.3, 90, -0.2, 3.1, 1e-5)
    hid, text, rationale = hypothesis_for(t, 2e-4)
    assert hid.startswith("S") and len(hid) == 7
    assert hypothesis_for(t, 1e-3)[0] == hid  # same group -> same id on a re-scan
    assert "session = london do better (+0.300 R, n=40)" in text
    assert json.loads(rationale)["note"].startswith("found on train")
    worse = GroupTest("r1", "s", "XAUUSD", "M15", "session", "asia", 40, -0.5, 90, 0.1, -3, 1e-5)
    assert "do worse" in hypothesis_for(worse, 2e-4)[1]
