"""Promotion gates (SPEC §7.4, §8.1; docs/PHASE_7_TASKS.md 7.4).

`check_next()` runs the next automatic gate for a registered strategy, records every
check in `gate_checks`, and promotes one step only if all checks pass:

- research -> candidate: train run, profit factor >= `min_pf_candidate`.
- candidate -> validated: validation run (one attempt), expectancy > 0, profit
  factor >= `min_pf_validation`, cost stress > 0.
- validated -> oos_passed: **the only code that opens out-of-sample data.** Needs
  `touch_oos=True`; refused if this candidate already touched OOS, unless a human gives
  an override reason. The touch is logged in `experiments` before the run. Checks:
  every SPEC §7.4 item (see `_oos_checks`).
- oos_passed -> paper: automatic, no new evidence.

Every backtest a gate uses runs with account limits enforced, and each recorded run is
re-checked for `"__account_limits": "enforced"` before its numbers are used. A failed
gate cannot be retried for the same strategy id (a new idea is a new registration),
except the OOS gate: a human may re-open it, each time with a written override reason.
"""

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tradeagent.backtest.baseline import baseline_distribution
from tradeagent.backtest.metrics import daily_returns, profit_factor
from tradeagent.backtest.robustness import (
    baseline_p_value,
    deflated_sharpe,
    monte_carlo_dd,
    sensitivity_values,
)
from tradeagent.backtest.runner import load_inputs, run_inputs
from tradeagent.backtest.runs import RunRecord, execute_run
from tradeagent.backtest.walkforward import walk_forward
from tradeagent.config import AppConfig, SplitName
from tradeagent.data.store import BarStore
from tradeagent.provenance import git_commit
from tradeagent.registry.registry import Entry, code_hash, get, next_status, promote_automatic
from tradeagent.strategies.variants import VariantSpec, base_name, build
from tradeagent.timeutil import utc_now

GATE_SPLIT: dict[str, SplitName] = {"candidate": "train", "validated": "validation"}


class GateError(ValueError):
    """The gate may not run (wrong status, OOS not confirmed, repeat attempt, ...)."""


@dataclass(frozen=True)
class Check:
    name: str
    required: str
    actual: str
    ok: bool


@dataclass(frozen=True)
class GateResult:
    strategy_id: str
    from_status: str
    to_status: str
    checks: list[Check]
    run_ids: list[str]

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)


def _num(x: float | None, fmt: str = "+.3f") -> str:
    if x is None:
        return "-"
    return "inf" if math.isinf(x) else format(x, fmt)


def _ge(name: str, actual: float | None, minimum: float, fmt: str = ".2f") -> Check:
    ok = actual is not None and actual >= minimum
    return Check(name, f">= {minimum:{fmt}}", _num(actual, fmt), ok)


def _gt0(name: str, actual: float | None) -> Check:
    return Check(name, "> 0", _num(actual), actual is not None and actual > 0)


def _enforced(record: RunRecord) -> Check:
    mode = json.loads(record.run["params_json"]).get("__account_limits")
    return Check(f"{record.run['split']} account limits", "enforced", str(mode), mode == "enforced")


def _record(conn: sqlite3.Connection, cfg: AppConfig, result: GateResult) -> None:
    n = conn.execute("SELECT COUNT(*) FROM gate_checks").fetchone()[0] + 1
    with conn:
        conn.execute(
            "INSERT INTO gate_checks (check_id, strategy_id, from_status, to_status, passed, "
            "checks_json, run_ids_json, git_commit, config_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"G{n:04d}",
                result.strategy_id,
                result.from_status,
                result.to_status,
                int(result.passed),
                json.dumps([asdict(c) for c in result.checks]),
                json.dumps(result.run_ids),
                git_commit(),
                cfg.config_hash,
                utc_now().isoformat(),
            ),
        )


def gate_history(conn: sqlite3.Connection, sid: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT check_id, from_status, to_status, passed, checks_json, run_ids_json, created_at "
        "FROM gate_checks WHERE strategy_id = ? ORDER BY check_id",
        (sid,),
    ).fetchall()
    keys = ("check_id", "from_status", "to_status", "passed", "checks", "run_ids", "created_at")
    return [
        {**dict(zip(keys, r, strict=True)), "checks": json.loads(r[4]), "run_ids": json.loads(r[5])}
        for r in rows
    ]


def _passed_run(conn: sqlite3.Connection, sid: str, to_status: str) -> str:
    """Run id of the passed gate that led to `to_status` (its first run)."""
    for g in gate_history(conn, sid):
        if g["to_status"] == to_status and g["passed"]:
            return str(g["run_ids"][0])
    raise GateError(f"no passed {to_status} gate for {sid}")


def _run_trades(conn: sqlite3.Connection, run_id: str) -> int:
    row = conn.execute("SELECT trades FROM backtest_runs WHERE run_id = ?", (run_id,)).fetchone()
    return int(row[0]) if row else 0


def _run(
    cfg: AppConfig,
    store: BarStore,
    conn: sqlite3.Connection,
    entry: Entry,
    split: SplitName,
    out_root: Path,
    allow_oos: bool = False,
) -> RunRecord:
    return execute_run(
        cfg,
        store,
        conn,
        entry.spec.strategy,
        entry.symbol,
        split,
        entry.seed,
        dict(entry.spec.params),
        out_root,
        enforce_account_limits=True,
        variant=entry.spec,
        allow_oos=allow_oos,
    )


# --- OOS touch log (SPEC §7.2: each touch in `experiments`) --------------------------


def oos_touches(conn: sqlite3.Connection, sid: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT experiment_id FROM experiments WHERE strategy_id = ? "
            "AND dataset_split = 'out_of_sample' ORDER BY experiment_id",
            (sid,),
        )
    ]


def _log_oos_touch(
    conn: sqlite3.Connection, cfg: AppConfig, entry: Entry, override_reason: str | None
) -> str:
    n = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] + 1
    eid = f"E{n:04d}"
    oos = cfg.splits.out_of_sample if cfg.splits else None
    criterion = {"gate": "validated -> oos_passed", **cfg.settings.validation.model_dump()}
    if override_reason:
        criterion["override_reason"] = override_reason
    with conn:
        conn.execute(
            "INSERT INTO experiments (experiment_id, hypothesis_id, strategy_id, dataset_split, "
            "date_range, git_commit, config_hash, seed, created_at, symbol, variant_json, "
            "success_criterion, status) VALUES (?, NULL, ?, 'out_of_sample', ?, ?, ?, ?, ?, ?, "
            "?, ?, 'running')",
            (
                eid,
                entry.strategy_id,
                f"{oos.start}..{oos.end}" if oos else "-",
                git_commit(),
                cfg.config_hash,
                entry.seed,
                utc_now().isoformat(),
                entry.symbol,
                json.dumps(entry.spec.to_dict(), sort_keys=True),
                json.dumps(criterion, sort_keys=True),
            ),
        )
    return eid


def _close_oos_touch(conn: sqlite3.Connection, eid: str, result: GateResult) -> None:
    with conn:
        conn.execute(
            "UPDATE experiments SET status = 'done', verdict = ?, run_id = ?, metrics_json = ? "
            "WHERE experiment_id = ?",
            (
                "pass" if result.passed else "fail",
                result.run_ids[0] if result.run_ids else None,
                json.dumps({"checks": [asdict(c) for c in result.checks]}),
                eid,
            ),
        )


# --- the OOS gate checks ---------------------------------------------------------------


def _trial_sharpes(conn: sqlite3.Connection, name: str) -> list[float]:
    """Per-day Sharpe of every recorded run of the base strategy (all splits/variants)."""
    base = base_name(name)
    out = []
    for strategy, metrics_json in conn.execute(
        "SELECT strategy, metrics_json FROM backtest_runs"
    ).fetchall():
        if base_name(strategy) != base:
            continue
        sharpe = json.loads(metrics_json).get("equity", {}).get("sharpe")
        if sharpe is not None and math.isfinite(sharpe):
            out.append(sharpe / math.sqrt(252))
    return out


def _oos_checks(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    store: BarStore,
    entry: Entry,
    oos: RunRecord,
    baseline_seeds: int,
) -> tuple[list[Check], list[str]]:
    v = cfg.settings.validation
    strategy = build(entry.spec, entry.seed)
    om, sm = oos.metrics["trades"], oos.stress_metrics["trades"]
    train_id = _passed_run(conn, entry.strategy_id, "candidate")
    val_id = _passed_run(conn, entry.strategy_id, "validated")
    checks = [
        _enforced(oos),
        _ge(
            "train+validation trades",
            _run_trades(conn, train_id) + _run_trades(conn, val_id),
            v.min_trades_train_val,
            ".0f",
        ),
        _ge("OOS trades", om["trades"], v.min_trades_oos, ".0f"),
        _ge("OOS profit factor", om["profit_factor"], v.min_pf_oos),
        _gt0("OOS expectancy R", om["expectancy_r"]),
        _gt0("OOS cost-stress expectancy R", sm["expectancy_r"]),
    ]

    # Random baseline on the same OOS period, timeframe, style and direction.
    base, _ = baseline_distribution(
        cfg,
        store,
        entry.symbol,
        range(1, baseline_seeds + 1),
        split="out_of_sample",
        timeframe=strategy.timeframes[0],
        style=strategy.style,
        direction=entry.spec.direction,
        allow_oos=True,
    )
    exp = om["expectancy_r"]
    p = baseline_p_value(exp, base["expectancy_r"]) if exp is not None else 1.0
    checks.append(
        Check("random baseline p", f"< {v.baseline_p_max}", f"{p:.4f}", p < v.baseline_p_max)
    )

    # Parameter sensitivity on train (each parameter x0.8 / x1.2 within its range).
    inputs = load_inputs(cfg, store, strategy, entry.symbol, "train")
    worst: tuple[str, float] | None = None
    for name, value in sensitivity_values(
        dict(strategy.params), strategy.param_specs, v.sensitivity_pct
    ):
        spec = VariantSpec.from_dict(
            {**entry.spec.to_dict(), "params": {**strategy.params, name: value}}
        )
        result = run_inputs(cfg, build(spec, entry.seed), inputs, enforce_account_limits=True)
        pf = profit_factor(result.trades["net_pnl"].to_numpy(float)) if len(result.trades) else 0.0
        if worst is None or pf < worst[1]:
            worst = (f"{name}={value:g}", pf)
    if worst is None:
        checks.append(Check("parameter sensitivity", "parameters to move", "none", True))
    else:
        checks.append(_ge(f"sensitivity worst PF ({worst[0]})", worst[1], v.min_pf_sensitivity))

    # Walk-forward over train + validation; its full run also feeds Monte Carlo and DSR.
    wf, full = walk_forward(cfg, store, entry.spec, entry.symbol, entry.seed)
    checks.append(
        Check(
            "walk-forward windows positive",
            f">= {v.wf_min_positive_share:.0%} of >= 3 windows",
            f"{sum(w.positive for w in wf.windows)}/{len(wf.windows)}",
            wf.passed,
        )
    )
    net = full.trades["net_pnl"].to_numpy(float)
    dd, dd95 = monte_carlo_dd(net, full.start_balance, v.mc_runs)
    limit = min(v.mc_dd_multiple * dd, cfg.risk.max_drawdown_pct)
    checks.append(
        Check(
            "Monte Carlo 95th pct drawdown %",
            f"<= {limit:.2f} (min of {v.mc_dd_multiple:g}x {dd:.2f}, limit "
            f"{cfg.risk.max_drawdown_pct:g})",
            f"{dd95:.2f}",
            dd95 <= limit,
        )
    )
    sharpes = _trial_sharpes(conn, entry.name)
    returns = daily_returns(full.daily["equity"].to_numpy(float), full.start_balance)
    dsr = deflated_sharpe(returns, sharpes, max(len(sharpes), 1))
    checks.append(_ge(f"deflated Sharpe (N={len(sharpes)} runs)", dsr, v.dsr_min))
    return checks, [oos.run["run_id"], train_id, val_id]


# --- entry point ----------------------------------------------------------------------


def check_next(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    store: BarStore,
    sid: str,
    out_root: Path,
    touch_oos: bool = False,
    override_reason: str | None = None,
    baseline_seeds: int | None = None,
) -> GateResult:
    entry = get(conn, sid)
    to_status = next_status(entry.status)
    if to_status is None or (entry.status, to_status) not in {
        ("research", "candidate"),
        ("candidate", "validated"),
        ("validated", "oos_passed"),
        ("oos_passed", "paper"),
    }:
        raise GateError(f"{sid} is {entry.status}: no automatic gate (human steps: `gate approve`)")
    if code_hash(entry.spec) != entry.code_hash:
        raise GateError("the strategy's code changed since registration; register it again")
    for g in gate_history(conn, sid):
        # A failed OOS gate is governed by the OOS touch rule below (human override).
        if g["to_status"] == to_status and not g["passed"] and to_status != "oos_passed":
            raise GateError(
                f"{sid} already failed the {to_status} gate ({g['check_id']}); "
                "one attempt per gate, a new idea needs a new registration"
            )

    v = cfg.settings.validation
    if to_status == "paper":
        passed_oos = Check("previous gate", "oos_passed", entry.status, True)
        result = GateResult(sid, entry.status, to_status, [passed_oos], [])
    elif to_status in GATE_SPLIT:
        record = _run(cfg, store, conn, entry, GATE_SPLIT[to_status], out_root)
        m, sm = record.metrics["trades"], record.stress_metrics["trades"]
        checks = [_enforced(record)]
        if to_status == "candidate":
            checks.append(_ge("train profit factor", m["profit_factor"], v.min_pf_candidate))
        else:
            checks += [
                _gt0("validation expectancy R", m["expectancy_r"]),
                _ge("validation profit factor", m["profit_factor"], v.min_pf_validation),
                _gt0("validation cost-stress expectancy R", sm["expectancy_r"]),
            ]
        result = GateResult(sid, entry.status, to_status, checks, [record.run["run_id"]])
    else:  # validated -> oos_passed
        if not touch_oos:
            raise GateError("the OOS gate opens out-of-sample data once: pass touch_oos")
        previous = oos_touches(conn, sid)
        if previous and not (override_reason and override_reason.strip()):
            raise GateError(
                f"{sid} already touched out-of-sample ({', '.join(previous)}); a second "
                "touch needs a human override with a written reason"
            )
        eid = _log_oos_touch(conn, cfg, entry, override_reason if previous else None)
        oos = _run(cfg, store, conn, entry, "out_of_sample", out_root, allow_oos=True)
        checks, run_ids = _oos_checks(
            conn, cfg, store, entry, oos, baseline_seeds or v.baseline_seeds
        )
        result = GateResult(sid, entry.status, to_status, checks, run_ids)
        _close_oos_touch(conn, eid, result)

    _record(conn, cfg, result)
    if result.passed:
        promote_automatic(conn, sid, to_status)
    return result
