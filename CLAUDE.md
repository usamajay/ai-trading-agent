# CLAUDE.md — rules for every Claude Code session in this repo

## Who you are working with
Usama is a beginner at building this kind of system. He knows basic programming (CS degree, n8n, APIs) but not trading systems, quant research, or this stack.
- Explain every step in simple English before doing it, and what it achieved after.
- When he must do something himself (install, click, paste a key), give numbered steps with exact commands for **Windows PowerShell**.
- Never assume he knows a term — explain it in brackets the first time.
- Keep each change small; end every session with: what was done, how to run/check it, what's next.

## Source of truth
- `docs/SPEC.md` is the master design. Read it at the start of every session.
- The current phase task list is `docs/PHASE_<n>_TASKS.md`. Work only on the current phase unless told otherwise.
- Record every design decision or deviation in `docs/DECISIONS.md` (date, decision, why).

## Hard safety rules (never break, even if asked casually)
1. **No real-money trading code paths are enabled** before Phase 10. Default mode is `paper`. The system uses Usama's **Exness MT5 Standard DEMO account** for all testing (Phases 1–9). The MT5 client must check `account_info().trade_mode` on connect and **refuse to run if the account is REAL** unless live mode passes every SPEC §9 check. Order functions live only in `src/tradeagent/execution/`; demo orders are allowed only in Phase 8+ forward testing.
2. The **risk engine** (`src/tradeagent/risk/`) is deterministic Python: no LLM calls, no ML. Execution must refuse any order without a risk-approval token.
3. Never edit `config/risk.yaml` or `config/live.yaml` limits without Usama explicitly asking in this session; log the change in `docs/DECISIONS.md`.
4. Never commit secrets. Credentials go in `.env` (git-ignored). Provide `.env.example` with placeholders. Assume the repo may be public — check every commit for keys, passwords and account numbers.
5. Out-of-sample data is touched once per candidate; never tune on it.
6. Strategy promotion to `approved`/`production` is always a human action.

## Engineering rules
- Python 3.11+, package in `src/tradeagent`, tests in `tests/`, run with `pytest`.
- Lint/format with `ruff`; type hints everywhere.
- All timestamps UTC in code and storage; convert to PKT (UTC+5) only for display.
- Every result/experiment row stores `git_commit` and `config_hash`.
- Signals use closed bars only; keep the look-ahead-bias test passing.
- Before finishing a task: run `pytest` and `ruff check .`, and fix failures.
- Commit in small steps with clear messages; push to GitHub at the end of each session.

## Useful commands (update as they are added)
```powershell
uv sync                      # install dependencies
uv run pytest                # run tests
uv run ruff check .          # lint
uv run tradeagent --help     # CLI
uv run tradeagent check-config  # validate config/*.yaml, print config_hash
uv run tradeagent data ping     # connect to MT5 (demo only), show account + last prices
```
