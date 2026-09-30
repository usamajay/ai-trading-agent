"""Split guard: the one place that decides whether an experiment may use a split.

- train: always.
- validation: only after a recorded **train pass** of the exact same variant on the
  same symbol, and only once per variant and symbol.
- out_of_sample (and anything else): refused here. OOS is opened once per candidate
  by the Phase 7 gates, never by research.
"""

import sqlite3


class SplitGuardError(ValueError):
    """Raised when an experiment may not use the split it asks for."""


def check_split(
    conn: sqlite3.Connection,
    split: str,
    variant_json: str,
    symbol: str,
    parent_experiment_id: str | None,
) -> None:
    if split == "train":
        return
    if split != "validation":
        raise SplitGuardError(f"research may use train and validation only, not {split!r}")
    if parent_experiment_id is None:
        raise SplitGuardError("validation needs the train experiment that passed (--parent)")
    parent = conn.execute(
        "SELECT dataset_split, variant_json, symbol, status, verdict FROM experiments "
        "WHERE experiment_id = ?",
        (parent_experiment_id,),
    ).fetchone()
    if parent is None:
        raise SplitGuardError(f"no experiment {parent_experiment_id!r}")
    p_split, p_variant, p_symbol, p_status, p_verdict = parent
    if p_split != "train" or p_variant != variant_json or p_symbol != symbol:
        raise SplitGuardError("the parent must be a train experiment of the same variant/symbol")
    if p_status != "done" or p_verdict != "pass":
        raise SplitGuardError("validation opens only after the train experiment passed")
    used = conn.execute(
        "SELECT experiment_id FROM experiments WHERE dataset_split = 'validation' "
        "AND variant_json = ? AND symbol = ?",
        (variant_json, symbol),
    ).fetchone()
    if used is not None:
        raise SplitGuardError(
            f"validation was already used for this variant ({used[0]}); one attempt only"
        )
