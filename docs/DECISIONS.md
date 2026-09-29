# Decision log

Newest first. Format: date — decision — why.

- 2026-09-29 — Start with SQLite + Parquet, move to PostgreSQL/TimescaleDB in Phase 8 — zero setup for a beginner; storage is behind `store.py` so the switch is contained.
- 2026-09-29 — Testing only on an Exness MT5 **demo** account; MT5 client refuses non-demo accounts unless live mode passes all SPEC §9 checks — protects real funds.
- 2026-09-29 — Risk engine is deterministic and token-gated; the AI cannot change limits — requirement §8 of the brief.
- 2026-09-29 — Min reward:risk set to 2.0 to match Usama's manual trading rule.
- 2026-09-29 — Build phases 0–10 as in SPEC §13; manual approval for every promotion to production.
