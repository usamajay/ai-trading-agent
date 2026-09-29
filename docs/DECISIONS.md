# Decision log

Newest first. Format: date — decision — why.

- 2026-09-29 — Use Usama's existing Exness MT5 account (MT5 reports trade_mode = REAL) instead of a separate demo account, at his explicit choice — safe because the MT5 client is read-only (data only) and paper trading is simulated in Python; no code path sends orders until Phase 10. Replaces the earlier "demo account only" rule.
- 2026-09-29 — Start with SQLite + Parquet, move to PostgreSQL/TimescaleDB in Phase 8 — zero setup for a beginner; storage is behind `store.py` so the switch is contained.
- 2026-09-29 — ~~Testing only on an Exness MT5 demo account~~ (superseded same day, see top entry).
- 2026-09-29 — Risk engine is deterministic and token-gated; the AI cannot change limits — requirement §8 of the brief.
- 2026-09-29 — Min reward:risk set to 2.0 to match Usama's manual trading rule.
- 2026-09-29 — Build phases 0–10 as in SPEC §13; manual approval for every promotion to production.
