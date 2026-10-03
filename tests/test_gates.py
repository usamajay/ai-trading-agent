"""Phase 7 pipeline on a synthetic market with a planted edge (no real data).

The planted strategy must climb research -> candidate -> validated -> oos_passed ->
paper and stop there; the random baseline must fail the first gate. OOS opens only
through the OOS gate, once per candidate unless a human gives a written reason.
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from synthetic_edge import PLANTED, SYMBOL, PlantedMomentum, edge_config, edge_store

from tradeagent.backtest.runs import execute_run
from tradeagent.backtest.splits import SplitError
from tradeagent.config import AppConfig, load_config
from tradeagent.data.store import connect_db
from tradeagent.registry.gates import GateError, _enforced, check_next, gate_history, oos_touches
from tradeagent.registry.registry import RegistryError, approve, get, register, retire
from tradeagent.research.guard import SplitGuardError, check_split
from tradeagent.strategies import registry as strategy_registry
from tradeagent.strategies.variants import VariantSpec

SRC = Path(__file__).resolve().parents[1] / "src" / "tradeagent"
SEEDS = 20  # p < 0.05 needs at least 20 random seeds (1 / 21 = 0.048)


@pytest.fixture(scope="module", autouse=True)
def planted() -> Iterator[None]:
    strategy_registry.load_builtins()
    strategy_registry.register(PLANTED, lambda seed, params: PlantedMomentum(seed, params))
    yield
    strategy_registry.unregister(PLANTED)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
    root = tmp_path_factory.mktemp("gates")
    conn = connect_db(root / "test.db")
    yield SimpleNamespace(
        cfg=edge_config(load_config()), store=edge_store(root / "bars"), conn=conn, out=root / "bt"
    )
    conn.close()


def _show(result: Any) -> str:
    return "\n".join(f"{c.name}: {c.actual} (needs {c.required}) {c.ok}" for c in result.checks)


def _next(env: SimpleNamespace, sid: str, cfg: AppConfig | None = None, **kw: Any) -> Any:
    return check_next(env.conn, cfg or env.cfg, env.store, sid, env.out, baseline_seeds=SEEDS, **kw)


@pytest.fixture(scope="module")
def ladder(env: SimpleNamespace) -> SimpleNamespace:
    sid = register(env.conn, env.cfg, VariantSpec(PLANTED), SYMBOL).strategy_id
    steps = [_next(env, sid), _next(env, sid)]  # -> candidate, -> validated
    with pytest.raises(GateError, match="touch_oos"):
        _next(env, sid)
    assert oos_touches(env.conn, sid) == []  # refused before anything was logged
    steps.append(_next(env, sid, touch_oos=True))  # -> oos_passed
    steps.append(_next(env, sid))  # -> paper (automatic)
    return SimpleNamespace(sid=sid, steps=steps)


def test_planted_edge_climbs_to_paper(env: SimpleNamespace, ladder: SimpleNamespace) -> None:
    for step in ladder.steps:
        assert step.passed, f"{step.from_status} -> {step.to_status}\n{_show(step)}"
    assert [s.to_status for s in ladder.steps] == ["candidate", "validated", "oos_passed", "paper"]
    assert get(env.conn, ladder.sid).status == "paper"
    names = [c.name for c in ladder.steps[2].checks]
    for needed in ("OOS trades", "random baseline p", "walk-forward windows positive"):
        assert needed in names
    for prefix in ("Monte Carlo", "deflated Sharpe", "sensitivity worst PF", "train+validation"):
        assert any(n.startswith(prefix) for n in names), prefix
    assert len(gate_history(env.conn, ladder.sid)) == 4


def test_every_gate_run_enforced_account_limits(
    env: SimpleNamespace, ladder: SimpleNamespace
) -> None:
    rows = env.conn.execute(
        "SELECT split, params_json FROM backtest_runs WHERE strategy = ?", (PLANTED,)
    ).fetchall()
    assert {r[0] for r in rows} >= {"train", "validation", "out_of_sample"}
    assert all(json.loads(r[1])["__account_limits"] == "enforced" for r in rows)


def test_oos_touch_logged_once_in_experiments(
    env: SimpleNamespace, ladder: SimpleNamespace
) -> None:
    (eid,) = oos_touches(env.conn, ladder.sid)
    status, verdict, run_id = env.conn.execute(
        "SELECT status, verdict, run_id FROM experiments WHERE experiment_id = ?", (eid,)
    ).fetchone()
    assert (status, verdict) == ("done", "pass") and run_id


def test_no_automatic_step_past_paper(env: SimpleNamespace, ladder: SimpleNamespace) -> None:
    with pytest.raises(GateError, match="human"):
        _next(env, ladder.sid)


def test_human_steps_need_name_and_reason(env: SimpleNamespace, ladder: SimpleNamespace) -> None:
    with pytest.raises(RegistryError, match="written reason"):
        approve(env.conn, ladder.sid, "Usama", " ")
    assert approve(env.conn, ladder.sid, "Usama", "test: 4 weeks paper") == "approved"
    assert approve(env.conn, ladder.sid, "Usama", "test: go") == "production"
    with pytest.raises(RegistryError):
        approve(env.conn, ladder.sid, "Usama", "nothing after production")
    retire(env.conn, ladder.sid, "Usama", "test over")
    n = env.conn.execute("SELECT COUNT(*) FROM approvals WHERE strategy_id = ?", (ladder.sid,))
    assert n.fetchone()[0] == 3


def test_human_cannot_skip_the_gates(env: SimpleNamespace) -> None:
    spec = VariantSpec(PLANTED, params={"rr": 3.0})
    sid = register(env.conn, env.cfg, spec, SYMBOL).strategy_id
    with pytest.raises(RegistryError, match="decided by the gates"):
        approve(env.conn, sid, "Usama", "looks good")


def test_random_baseline_fails_first_gate_once(env: SimpleNamespace) -> None:
    sid = register(env.conn, env.cfg, VariantSpec("random_baseline"), SYMBOL).strategy_id
    result = _next(env, sid)
    assert not result.passed and get(env.conn, sid).status == "research"
    with pytest.raises(GateError, match="already failed"):
        _next(env, sid)


def test_failed_oos_reopens_only_with_written_override(env: SimpleNamespace) -> None:
    spec = VariantSpec(PLANTED, params={"atr_mult": 1.6})
    sid = register(env.conn, env.cfg, spec, SYMBOL).strategy_id
    assert _next(env, sid).passed and _next(env, sid).passed
    strict = edge_config(load_config(), min_pf_oos=1000.0)  # cannot pass
    assert not _next(env, sid, strict, touch_oos=True).passed
    assert get(env.conn, sid).status == "validated"
    with pytest.raises(GateError, match="already touched"):
        _next(env, sid, strict, touch_oos=True)
    _next(env, sid, strict, touch_oos=True, override_reason="test: human re-open")
    touches = oos_touches(env.conn, sid)
    assert len(touches) == 2
    criterion = env.conn.execute(
        "SELECT success_criterion FROM experiments WHERE experiment_id = ?", (touches[1],)
    ).fetchone()[0]
    assert json.loads(criterion)["override_reason"] == "test: human re-open"


def test_changed_code_is_refused(env: SimpleNamespace) -> None:
    spec = VariantSpec(PLANTED, params={"atr_mult": 1.4})
    sid = register(env.conn, env.cfg, spec, SYMBOL).strategy_id
    with env.conn:
        env.conn.execute("UPDATE strategies SET code_hash = 'old' WHERE strategy_id = ?", (sid,))
    with pytest.raises(GateError, match="code changed"):
        _next(env, sid)


# --- OOS stays locked everywhere else ---------------------------------------------------


def test_flag_mode_runs_fail_the_enforced_check() -> None:
    record: Any = SimpleNamespace(
        run={"split": "train", "params_json": json.dumps({"__account_limits": "flags"})}
    )
    assert not _enforced(record).ok


def test_oos_refused_without_the_gate(env: SimpleNamespace) -> None:
    with pytest.raises(SplitError, match="out-of-sample"):
        execute_run(env.cfg, env.store, env.conn, PLANTED, SYMBOL, "out_of_sample", 1, {}, env.out)


def test_research_guard_still_refuses_oos(env: SimpleNamespace) -> None:
    with pytest.raises(SplitGuardError):
        check_split(env.conn, "out_of_sample", "{}", SYMBOL, None)


def test_only_the_gate_code_opens_oos() -> None:
    """`allow_oos=True` may appear in src only in registry/gates.py."""
    users = [
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if re.search(r"allow_oos\s*=\s*True", p.read_text(encoding="utf-8"))
    ]
    assert users == ["registry/gates.py"]


def test_gate_cli_lists_commands() -> None:
    from typer.testing import CliRunner

    from tradeagent.cli import app

    out = CliRunner().invoke(app, ["gate", "--help"])
    assert out.exit_code == 0
    for command in ("register", "status", "show", "check", "approve", "retire"):
        assert command in out.output
