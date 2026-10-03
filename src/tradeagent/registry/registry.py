"""Strategy registry and promotion ladder (SPEC §8.1; docs/PHASE_7_TASKS.md 7.4).

`research -> candidate -> validated -> oos_passed -> paper -> approved -> production`

- One row per candidate: a strategy variant (parameters, timeframes, direction, ...)
  on one symbol, with the seed and a hash of the strategy's code at registration.
- Promotion is one step at a time. The first four steps are decided by code in
  `registry/gates.py` from recorded evidence (`gate_checks`); `paper -> approved` and
  `approved -> production` are **human only**, recorded in `approvals` with the name
  of the person and a written reason. No code path here or in gates.py takes a
  strategy past `paper` on its own.
- `retired` can be reached from any status, also only by a human with a reason.
"""

import hashlib
import inspect
import json
import sqlite3
import sys
from dataclasses import dataclass
from typing import Any

from tradeagent.config import AppConfig
from tradeagent.provenance import git_commit
from tradeagent.strategies import registry as strategy_registry
from tradeagent.strategies import variants
from tradeagent.strategies.variants import VariantSpec, build
from tradeagent.timeutil import utc_now

LADDER = ("research", "candidate", "validated", "oos_passed", "paper", "approved", "production")
AUTOMATIC = {
    ("research", "candidate"),
    ("candidate", "validated"),
    ("validated", "oos_passed"),
    ("oos_passed", "paper"),
}
HUMAN = {("paper", "approved"), ("approved", "production")}


class RegistryError(ValueError):
    """Raised for an unknown strategy or a step the ladder does not allow."""


@dataclass(frozen=True)
class Entry:
    strategy_id: str
    name: str
    version: str
    symbol: str
    spec: VariantSpec
    seed: int
    code_hash: str
    status: str


def code_hash(spec: VariantSpec) -> str:
    """sha256 of the strategy's module source (and of variants.py for a variant)."""
    strategy_registry.load_builtins()
    inner = strategy_registry.create(spec.strategy)
    sources = [inspect.getsource(sys.modules[type(inner).__module__])]
    if not spec.is_plain:
        sources.append(inspect.getsource(variants))
    return hashlib.sha256("\n".join(sources).encode()).hexdigest()


def _variant_json(spec: VariantSpec) -> str:
    return json.dumps(spec.to_dict(), sort_keys=True)


def strategy_id(name: str, version: str, symbol: str, spec: VariantSpec) -> str:
    digest = hashlib.sha256(f"{_variant_json(spec)}|{symbol}".encode()).hexdigest()[:8]
    return f"{name}@{version}:{symbol}#{digest}"


def register(
    conn: sqlite3.Connection, cfg: AppConfig, spec: VariantSpec, symbol: str, seed: int = 1
) -> Entry:
    """Add a candidate at `research` (or return the existing row for the same id)."""
    strategy = build(spec, seed)
    sid = strategy_id(strategy.name, strategy.version, symbol, spec)
    existing = conn.execute("SELECT 1 FROM strategies WHERE strategy_id = ?", (sid,)).fetchone()
    if existing is None:
        now = utc_now().isoformat()
        with conn:
            conn.execute(
                "INSERT INTO strategies (strategy_id, name, version, params_json, code_hash, "
                "status, git_commit, config_hash, created_at, symbol, variant_json, seed, "
                "updated_at) VALUES (?, ?, ?, ?, ?, 'research', ?, ?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    strategy.name,
                    strategy.version,
                    json.dumps(dict(strategy.params), sort_keys=True),
                    code_hash(spec),
                    git_commit(),
                    cfg.config_hash,
                    now,
                    symbol,
                    _variant_json(spec),
                    seed,
                    now,
                ),
            )
    return get(conn, sid)


def get(conn: sqlite3.Connection, sid: str) -> Entry:
    row = conn.execute(
        "SELECT strategy_id, name, version, symbol, variant_json, seed, code_hash, status "
        "FROM strategies WHERE strategy_id = ?",
        (sid,),
    ).fetchone()
    if row is None:
        raise RegistryError(f"no registered strategy {sid!r}")
    sid, name, version, symbol, variant_json, seed, chash, status = row
    spec = VariantSpec.from_dict(json.loads(variant_json))
    return Entry(sid, name, version, symbol, spec, int(seed), chash, status)


def list_entries(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT strategy_id, status, symbol, updated_at FROM strategies ORDER BY created_at"
    ).fetchall()
    return [
        dict(zip(("strategy_id", "status", "symbol", "updated_at"), r, strict=True)) for r in rows
    ]


def next_status(status: str) -> str | None:
    if status not in LADDER or status == LADDER[-1]:
        return None
    return LADDER[LADDER.index(status) + 1]


def _set_status(conn: sqlite3.Connection, sid: str, status: str) -> None:
    with conn:
        conn.execute(
            "UPDATE strategies SET status = ?, updated_at = ? WHERE strategy_id = ?",
            (status, utc_now().isoformat(), sid),
        )


def promote_automatic(conn: sqlite3.Connection, sid: str, to_status: str) -> None:
    """Called by gates.py only, after a passed gate check was recorded."""
    entry = get(conn, sid)
    if (entry.status, to_status) not in AUTOMATIC:
        raise RegistryError(f"{entry.status} -> {to_status} is not an automatic step")
    _set_status(conn, sid, to_status)


def _approval(
    conn: sqlite3.Connection, sid: str, from_s: str, to_s: str, by: str, reason: str, evidence: str
) -> None:
    n = conn.execute("SELECT COUNT(*) FROM approvals").fetchone()[0] + 1
    with conn:
        conn.execute(
            "INSERT INTO approvals (approval_id, strategy_id, from_status, to_status, "
            "approved_by, evidence_json, time) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"A{n:04d}",
                sid,
                from_s,
                to_s,
                by,
                json.dumps({"reason": reason, "evidence": evidence}),
                utc_now().isoformat(),
            ),
        )


def _human(by: str, reason: str) -> None:
    if not by.strip() or not reason.strip():
        raise RegistryError("a human step needs the approver's name and a written reason")


def approve(conn: sqlite3.Connection, sid: str, by: str, reason: str, evidence: str = "") -> str:
    """Human step: paper -> approved, or approved -> production. Returns the new status."""
    _human(by, reason)
    entry = get(conn, sid)
    to_status = next_status(entry.status)
    if to_status is None or (entry.status, to_status) not in HUMAN:
        raise RegistryError(
            f"{entry.status}: human approval applies only to paper -> approved and "
            "approved -> production; earlier steps are decided by the gates"
        )
    _approval(conn, sid, entry.status, to_status, by, reason, evidence)
    _set_status(conn, sid, to_status)
    return to_status


def retire(conn: sqlite3.Connection, sid: str, by: str, reason: str) -> None:
    _human(by, reason)
    entry = get(conn, sid)
    if entry.status == "retired":
        raise RegistryError(f"{sid} is already retired")
    _approval(conn, sid, entry.status, "retired", by, reason, "")
    _set_status(conn, sid, "retired")
