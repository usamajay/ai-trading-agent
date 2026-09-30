"""Probability / EV engine (SPEC §5): estimator, entry gate, calibration."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tradeagent.config import EntryRules
from tradeagent.prob.calibration import (
    SplitRefused,
    brier,
    check_split,
    entry_cost_r,
    markdown_section,
    planned_rr,
    reliability,
    summarise,
    walk_forward,
)
from tradeagent.prob.estimator import LOSS, TIMEOUT, WIN, break_even_p_win, estimate, outcome
from tradeagent.prob.gate import CODES, entry_rejections

RULES = EntryRules(
    min_ev_r=0.15,
    min_p_win_lower=0.40,
    min_sample_trades=100,
    prior_strength=20,
    credible_level=0.90,
)
KW = {"prior_strength": 20, "credible_level": 0.90, "planned_rr": 2.0}


def test_outcome_classes() -> None:
    assert outcome("tp") == WIN and outcome("sl") == LOSS
    assert outcome("weekend_close") == outcome("friday_close") == TIMEOUT


def test_break_even_win_rate() -> None:
    assert break_even_p_win(2.0, 0.0) == pytest.approx(1 / 3)
    assert break_even_p_win(2.0, 0.1) == pytest.approx(1.1 / 3)  # costs raise it
    assert break_even_p_win(0.01, 0.5) == 0.99 and break_even_p_win(1000, 0) == 0.01
    for rr, cost in ((0, 0.1), (2.0, -0.1)):
        with pytest.raises(ValueError):
            break_even_p_win(rr, cost)


def test_no_history_is_break_even() -> None:
    e = estimate([], [], **KW)
    assert e.n == 0 and e.p_win == pytest.approx(1 / 3) and e.p_timeout == 0.0
    assert e.avg_win_r == 2.0 and e.avg_loss_r == 1.0 and e.ev_r == pytest.approx(0)
    assert e.p_win_low < 1 / 3 < e.p_win_high
    costly = estimate([], [], **KW, cost_r=0.1)
    assert costly.p_win == pytest.approx(1.1 / 3)
    assert costly.avg_win_r == pytest.approx(1.9) and costly.avg_loss_r == pytest.approx(1.1)
    assert costly.ev_r == pytest.approx(0)  # the prior alone never claims an edge


def test_small_samples_shrink_toward_break_even() -> None:
    e = estimate(["tp"] * 3, [2.0] * 3, **KW)
    assert e.p_win == pytest.approx((3 + 20 / 3) / 23)  # not 100%
    assert e.avg_loss_r == 1.0  # no losses yet: fallback


def test_large_samples_approach_the_observed_rate() -> None:
    reasons = ["tp"] * 300 + ["sl"] * 700
    e = estimate(reasons, [1.9] * 300 + [-1.05] * 700, **KW)
    assert e.p_win == pytest.approx((300 + 20 / 3) / 1020)
    assert e.avg_win_r == pytest.approx(1.9) and e.avg_loss_r == pytest.approx(1.05)
    assert e.ev_r == pytest.approx(e.p_win * 1.9 - e.p_loss * 1.05)


def test_interval_narrows_with_more_trades() -> None:
    small = estimate(["tp", "sl"] * 10, [2.0, -1.0] * 10, **KW)
    big = estimate(["tp", "sl"] * 500, [2.0, -1.0] * 500, **KW)
    assert small.p_win_low < big.p_win_low < 0.5 < big.p_win_high < small.p_win_high
    assert big.confidence > small.confidence


def test_timeouts_and_extra_cost() -> None:
    e = estimate(["tp", "sl", "weekend_close", "sl"], [2.0, -1.0, 0.4, -1.2], **KW)
    assert e.timeouts == 1 and e.p_timeout == 0.25
    assert e.p_loss == pytest.approx(1 - e.p_win - 0.25)
    assert e.avg_loss_r == pytest.approx(1.1) and e.avg_timeout_r == pytest.approx(0.4)
    base = e.p_win * 2.0 - e.p_loss * 1.1 + 0.25 * 0.4
    assert e.ev_r == pytest.approx(base)
    cost = estimate(
        ["tp", "sl", "weekend_close", "sl"], [2.0, -1.0, 0.4, -1.2], **KW, extra_cost_r=0.1
    )
    assert cost.ev_r == pytest.approx(base - 0.1)


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError):
        estimate(["tp"], [], **KW)


def test_gate_lists_every_failed_rule() -> None:
    thin = estimate(["tp", "sl", "sl"], [2.0, -1.0, -1.0], **KW)
    assert [r.code for r in entry_rejections(thin, RULES)] == list(CODES)  # near break-even
    losing = estimate(["tp"] * 60 + ["sl"] * 140, [2.0] * 60 + [-1.0] * 140, **KW)
    assert [r.code for r in entry_rejections(losing, RULES)] == ["p_win_low", "ev_low"]
    good = estimate(["tp"] * 500 + ["sl"] * 500, [2.0] * 500 + [-1.0] * 500, **KW)
    assert entry_rejections(good, RULES) == []


def _trades(rows: list[tuple[int, int, str, float]]) -> pd.DataFrame:
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    return pd.DataFrame(
        {
            "entry_time": [t0 + timedelta(hours=e) for e, _, _, _ in rows],
            "exit_time": [t0 + timedelta(hours=x) for _, x, _, _ in rows],
            "exit_reason": [r for _, _, r, _ in rows],
            "r_multiple": [m for _, _, _, m in rows],
            "entry_price": 100.0,
            "stop_loss": 99.0,
            "take_profit": 102.0,
            "spread_cost": 3.0,
            "slippage_cost": 1.0,
            "commission": 0.0,
            "risk_usd": 40.0,
        }
    )


def test_walk_forward_uses_only_trades_closed_before_entry() -> None:
    # trade 2 enters at 5h while trade 1 is still open (exits 10h): it must not see it
    t = _trades([(0, 2, "tp", 2.0), (1, 10, "sl", -1.0), (5, 6, "sl", -1.0), (11, 12, "tp", 2.0)])
    w = walk_forward(t, RULES)
    assert w["n_known"].tolist() == [0, 0, 1, 3]
    p0 = 1.1 / 3  # RR 2, cost (3 + 1) / 40 = 0.1 R
    assert w["p_win"].iloc[0] == pytest.approx(p0)
    assert w["p_win"].iloc[2] == pytest.approx((1 + 20 * p0) / 21)
    assert w["win"].tolist() == [1, 0, 0, 1]
    assert not w["gate_pass"].any() and "sample_small" in w["gate_codes"].iloc[3]


def test_planned_rr_and_entry_cost() -> None:
    t = _trades([(0, 1, "tp", 2.0)])
    assert planned_rr(t).iloc[0] == pytest.approx(2.0)
    assert entry_cost_r(t).iloc[0] == pytest.approx(0.1)


def test_summary_and_reliability() -> None:
    t = _trades(
        [
            (i * 2, i * 2 + 1, "tp" if i % 3 == 0 else "sl", 2.0 if i % 3 == 0 else -1.0)
            for i in range(30)
        ]
    )
    w = walk_forward(t, RULES)
    s = summarise(w)
    assert s.trades == 30 and s.hit_rate == pytest.approx(10 / 30)
    assert s.brier_half == pytest.approx(0.25)
    assert s.brier_hindsight == pytest.approx((1 / 3) * (2 / 3))
    assert s.gate_passed == 0 and s.gate_realised_r is None
    rel = reliability(w)
    assert rel["trades"].sum() == 30 and set(rel.index) <= {"0.2-0.3", "0.3-0.4", "0.4-0.5"}


def test_summary_counts_gate_passes() -> None:
    w = pd.DataFrame(
        {
            "p_win": [0.6, 0.6],
            "win": [1, 0],
            "ev_r": [0.3, 0.3],
            "gate_pass": [True, False],
            "r_multiple": [2.0, -1.0],
        }
    )
    s = summarise(w)
    assert s.gate_passed == 1 and s.gate_realised_r == 2.0
    with pytest.raises(ValueError):
        summarise(w.iloc[0:0])


def test_brier_perfect_is_zero() -> None:
    assert brier(pd.Series([1.0, 0.0]), pd.Series([1, 0])) == 0.0


def test_out_of_sample_refused() -> None:
    check_split("train")
    check_split("validation")
    with pytest.raises(SplitRefused):
        check_split("out_of_sample")


def test_markdown_section() -> None:
    t = _trades([(i * 2, i * 2 + 1, "tp" if i % 3 == 0 else "sl", 2.0) for i in range(6)])
    w = walk_forward(t, RULES)
    text = markdown_section("demo", summarise(w), reliability(w))
    assert text.startswith("### demo") and "Brier" in text and "| 0." in text
    assert "(realised - R)" in text  # nothing passed the gate
