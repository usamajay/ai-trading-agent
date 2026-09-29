# Decision log

Newest first. Format: date — decision — why.

- 2026-09-29 — Use Usama's existing Exness MT5 **Standard Demo** account for all testing (confirmed in the Exness app: Demo · MT5 · Standard). An external MT5 connector had mislabelled it as "real", so our own client must verify `trade_mode` itself on every connect and refuse REAL accounts outside live mode. Demo orders allowed from Phase 8 forward testing.
- 2026-09-29 — Start with SQLite + Parquet, move to PostgreSQL/TimescaleDB in Phase 8 — zero setup for a beginner; storage is behind `store.py` so the switch is contained.
- 2026-09-29 — Risk engine is deterministic and token-gated; the AI cannot change limits — requirement §8 of the brief.
- 2026-09-29 — Min reward:risk set to 2.0 to match Usama's manual trading rule.
- 2026-09-29 — Build phases 0–10 as in SPEC §13; manual approval for every promotion to production.
