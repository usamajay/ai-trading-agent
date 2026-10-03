# ruff: noqa: F811  (pytest fixtures shared from test_runs are re-bound as arguments)
"""Claude API hypothesis loop (Phase 6.7). A fake client: no network, no spend."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from test_runs import cfg, conn, registered  # noqa: F401 (pytest fixtures)

from tradeagent.config import AppConfig, LlmSettings
from tradeagent.research.experiments import add_hypothesis
from tradeagent.research.hypothesis import (
    RESPONSE_SCHEMA,
    LlmError,
    Proposal,
    SpendCapError,
    anthropic_client,
    build_summary,
    check_budget,
    cost_usd,
    dry_run,
    hypothesis_id,
    import_answer,
    month_spend,
    parse_answer,
    propose,
)

IDEA = {
    "statement": "Gold breakout_compression only between 01:00 and 04:00 UTC beats random.",
    "rationale": "Scan's closest group was hour 02 UTC; a session filter is cheap to test.",
    "strategy_change": "none; time-of-day filter variant",
    "params": [{"name": "hours_utc", "value": "1-4"}],
    "expected_effect": "expectancy from about 0 R to +0.1 R after costs",
    "how_to_falsify": "standard criterion on train fails (CI lower bound <= 0).",
}


def _answer(*ideas: dict[str, Any]) -> str:
    return json.dumps({"hypotheses": list(ideas)})


class FakeMessages:
    def __init__(self, text: str, stop_reason: str = "end_turn", tokens: int = 10_000) -> None:
        self.text, self.stop_reason, self.tokens = text, stop_reason, tokens
        self.error: Exception | None = None
        self.count_error: Exception | None = None
        self.created: list[dict[str, Any]] = []

    def count_tokens(self, **kwargs: Any) -> Any:
        if self.count_error is not None:
            raise self.count_error
        return SimpleNamespace(input_tokens=self.tokens)

    def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self.text),
            ],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=self.tokens, output_tokens=2_000),
            _request_id="req_test",
        )


class FakeClient:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


def _calls(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
    return conn.execute(
        "SELECT call_id, outcome, cost_usd, input_tokens, output_tokens FROM llm_calls"
    ).fetchall()


def _experiment(conn: sqlite3.Connection, eid: str, split: str, verdict: str) -> None:
    metrics = {"metrics": {"trades": 120, "expectancy_r": -0.05, "secret_extra": 1}}
    with conn:
        conn.execute(
            "INSERT INTO experiments (experiment_id, hypothesis_id, dataset_split, date_range, "
            "git_commit, config_hash, created_at, symbol, variant_json, success_criterion, "
            "status, verdict, metrics_json) VALUES (?, 'H1', ?, 'x', 'g', 'c', 't', 'XAUUSD', "
            "'{\"strategy\": \"s\"}', '{\"min_trades\": 1}', 'done', ?, ?)",
            (eid, split, verdict, json.dumps(metrics)),
        )
        conn.execute(
            "INSERT INTO lessons (lesson_id, experiment_id, text, git_commit, config_hash, "
            "created_at) VALUES (?, ?, ?, 'g', 'c', 't')",
            (f"L{eid}", eid, f"lesson of {eid} on {split}"),
        )


# --- summary: what may be sent ---------------------------------------------------------


def test_summary_has_no_out_of_sample_and_no_trade_lessons(
    conn: sqlite3.Connection, cfg: AppConfig
) -> None:
    add_hypothesis(conn, cfg, "H1", "idea", "why", "human")
    _experiment(conn, "E0001", "train", "fail")
    _experiment(conn, "E0002", "validation", "fail")
    _experiment(conn, "E0003", "out_of_sample", "pass")
    conn.execute("PRAGMA foreign_keys = OFF")  # a paper/live trade lesson must never be sent
    with conn:
        conn.execute(
            "INSERT INTO lessons (lesson_id, trade_id, text, git_commit, config_hash, "
            "created_at) VALUES ('LT1', 'T1', 'trade lesson', 'g', 'c', 't')"
        )
    conn.execute("PRAGMA foreign_keys = ON")
    summary = build_summary(conn, cfg)
    assert [e["id"] for e in summary["experiments"]] == ["E0001", "E0002"]
    assert [x["experiment"] for x in summary["lessons"]] == ["E0001", "E0002"]
    assert summary["experiments"][0]["metrics"] == {"trades": 120, "expectancy_r": -0.05}
    text = json.dumps(summary)
    assert "E0003" not in text and "out_of_sample" not in text and "trade lesson" not in text
    assert "random_baseline" not in [s["name"] for s in summary["strategies"]]


def test_dry_run_sends_nothing_and_writes_nothing(conn: sqlite3.Connection, cfg: AppConfig) -> None:
    prompt, budget = dry_run(conn, cfg)
    assert "Propose the next hypotheses" in prompt.user
    assert budget.input_tokens > 0 and budget.worst_case_usd > 0
    assert _calls(conn) == []


# --- answer validation --------------------------------------------------------------------


def test_schema_matches_the_model() -> None:
    item = RESPONSE_SCHEMA["properties"]["hypotheses"]["items"]
    assert set(item["properties"]) == set(Proposal.model_fields) == set(item["required"])
    assert item["additionalProperties"] is False


def test_parse_answer() -> None:
    assert parse_answer(_answer(IDEA), 5)[0].params[0].name == "hours_utc"
    assert parse_answer(_answer(), 5) == []
    with pytest.raises(LlmError, match="at most 2"):
        parse_answer(_answer(IDEA, IDEA, IDEA), 2)
    with pytest.raises(LlmError, match="schema"):
        parse_answer(_answer({**IDEA, "run_backtest": True}), 5)  # unknown key
    with pytest.raises(LlmError, match="schema"):
        parse_answer(_answer({k: v for k, v in IDEA.items() if k != "how_to_falsify"}), 5)
    with pytest.raises(LlmError, match="schema"):
        parse_answer('{"hypotheses": [', 5)


def test_hypothesis_id_is_stable() -> None:
    assert hypothesis_id("Gold  Breakout works") == hypothesis_id("gold breakout works")
    assert hypothesis_id("a b c d e f") != hypothesis_id("a b c d e g")


# --- spend caps --------------------------------------------------------------------------

LLM = LlmSettings(
    model="m",
    effort="high",
    max_input_tokens=40_000,
    max_output_tokens=16_000,
    input_usd_per_mtok=4.0,
    output_usd_per_mtok=20.0,
    max_usd_per_run=0.50,
    max_usd_per_month=5.0,
    max_hypotheses=5,
)


def test_cost_and_budget_checks() -> None:
    assert cost_usd(LLM, 1_000_000, 0) == 4.0 and cost_usd(LLM, 0, 1_000_000) == 20.0
    ok = check_budget(LLM, 10_000, 0.0)
    assert ok.allowed and ok.worst_case_usd == pytest.approx(0.04 + 0.32)
    assert "max_input_tokens" in (check_budget(LLM, 40_001, 0.0).refusal or "")
    small = LLM.model_copy(update={"max_usd_per_run": 0.30})
    assert "max_usd_per_run" in (check_budget(small, 10_000, 0.0).refusal or "")
    assert "max_usd_per_month" in (check_budget(LLM, 10_000, 4.70).refusal or "")
    assert check_budget(LLM, 10_000, 4.60).allowed


def test_month_spend_counts_only_this_utc_month(conn: sqlite3.Connection) -> None:
    for cid, when, cost in [
        ("C1", "2026-09-30T23:59:00+00:00", 3.0),
        ("C2", "2026-10-02T01:00:00+00:00", 0.25),
    ]:
        with conn:
            conn.execute(
                "INSERT INTO llm_calls (call_id, purpose, model, prompt_sha256, cost_usd, "
                "worst_case_usd, outcome, git_commit, config_hash, created_at) "
                "VALUES (?, 'hypotheses', 'm', 'h', ?, 0.5, 'ok', 'g', 'c', ?)",
                (cid, cost, when),
            )
    assert month_spend(conn, datetime(2026, 10, 3, tzinfo=UTC)) == pytest.approx(0.25)


# --- live round (fake client) --------------------------------------------------------------


def test_propose_stores_llm_hypotheses_and_records_cost(
    conn: sqlite3.Connection, cfg: AppConfig
) -> None:
    fake = FakeMessages(_answer(IDEA))
    result = propose(conn, cfg, FakeClient(fake))
    hid = hypothesis_id(IDEA["statement"])
    assert result.stored == [hid] and result.duplicates == []
    row = conn.execute(
        "SELECT source, status, text, rationale FROM hypotheses WHERE hypothesis_id = ?", (hid,)
    ).fetchone()
    assert row[:3] == ("llm", "proposed", IDEA["statement"])
    details = json.loads(row[3])
    assert details["params"] == {"hours_utc": "1-4"} and details["llm_call"] == result.call_id
    expected = cost_usd(cfg.settings.llm, 10_000, 2_000)
    assert _calls(conn) == [(result.call_id, "ok", pytest.approx(expected), 10_000, 2_000)]
    sent = fake.created[0]
    assert sent["model"] == cfg.settings.llm.model
    assert sent["max_tokens"] == cfg.settings.llm.max_output_tokens
    assert sent["output_config"]["format"]["schema"] == RESPONSE_SCHEMA
    assert sent["thinking"] == {"type": "adaptive"}
    # The same idea again is not stored twice.
    again = propose(conn, cfg, FakeClient(FakeMessages(_answer(IDEA))))
    assert again.stored == [] and again.duplicates == [hid]


def test_propose_refused_by_cap_sends_nothing(conn: sqlite3.Connection, cfg: AppConfig) -> None:
    fake = FakeMessages(_answer(IDEA), tokens=50_000)  # above max_input_tokens
    with pytest.raises(SpendCapError, match="max_input_tokens"):
        propose(conn, cfg, FakeClient(fake))
    assert fake.created == [] and _calls(conn) == []


def test_propose_refused_when_month_is_spent(conn: sqlite3.Connection, cfg: AppConfig) -> None:
    llm = cfg.settings.llm
    with conn:
        conn.execute(
            "INSERT INTO llm_calls (call_id, purpose, model, prompt_sha256, cost_usd, "
            "worst_case_usd, outcome, git_commit, config_hash, created_at) "
            "VALUES ('C0001', 'hypotheses', 'm', 'h', ?, 0.5, 'ok', 'g', 'c', ?)",
            (llm.max_usd_per_month - 0.01, datetime.now(UTC).isoformat()),
        )
    fake = FakeMessages(_answer(IDEA))
    with pytest.raises(SpendCapError, match="max_usd_per_month"):
        propose(conn, cfg, FakeClient(fake))
    assert fake.created == []


@pytest.mark.parametrize(
    ("text", "stop", "outcome"),
    [
        ('{"hypotheses": [', "max_tokens", "max_tokens"),
        ("", "refusal", "refusal"),
        (_answer({**IDEA, "extra": 1}), "end_turn", "invalid_answer"),
    ],
)
def test_unusable_answers_are_charged_but_store_nothing(
    conn: sqlite3.Connection, cfg: AppConfig, text: str, stop: str, outcome: str
) -> None:
    with pytest.raises(LlmError, match="nothing stored"):
        propose(conn, cfg, FakeClient(FakeMessages(text, stop_reason=stop)))
    calls = _calls(conn)
    assert len(calls) == 1 and calls[0][1] == outcome and calls[0][2] > 0
    assert conn.execute("SELECT COUNT(*) FROM hypotheses WHERE source = 'llm'").fetchone()[0] == 0


def test_connection_error_is_charged_at_worst_case(
    conn: sqlite3.Connection, cfg: AppConfig
) -> None:
    fake = FakeMessages(_answer(IDEA))
    fake.error = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    with pytest.raises(LlmError, match="connection_error"):
        propose(conn, cfg, FakeClient(fake))
    (call,) = _calls(conn)
    _, budget = dry_run(conn, cfg)
    assert call[1] == "connection_error" and call[2] == pytest.approx(
        cost_usd(cfg.settings.llm, 10_000 + 1500, cfg.settings.llm.max_output_tokens)
    )
    assert budget.month_spent_usd == pytest.approx(call[2])


def test_real_client_never_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert anthropic_client().max_retries == 0  # type: ignore[attr-defined]


def test_count_error_is_clean_and_free(conn: sqlite3.Connection, cfg: AppConfig) -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens")
    fake = FakeMessages(_answer(IDEA))
    fake.count_error = anthropic.BadRequestError(
        "credit balance is too low", response=httpx2.Response(400, request=request), body=None
    )
    with pytest.raises(LlmError, match="token count failed"):
        propose(conn, cfg, FakeClient(fake))
    assert fake.created == [] and _calls(conn) == []


# --- import (ideas written outside the API) ------------------------------------------------


def test_import_stores_like_live_at_zero_cost(
    conn: sqlite3.Connection, cfg: AppConfig, tmp_path: Path
) -> None:
    path = tmp_path / "manual.json"
    path.write_text(_answer(IDEA), encoding="utf-8")
    result = import_answer(conn, cfg, path)
    hid = hypothesis_id(IDEA["statement"])
    assert result.stored == [hid] and result.cost_usd == 0.0
    source, status, rationale = conn.execute(
        "SELECT source, status, rationale FROM hypotheses WHERE hypothesis_id = ?", (hid,)
    ).fetchone()
    assert (source, status) == ("llm", "proposed")
    assert json.loads(rationale)["origin"] == "manual-claude-code"
    row = conn.execute("SELECT model, cost_usd, outcome, request_id FROM llm_calls").fetchone()
    assert row == ("manual-claude-code", 0.0, "ok", "manual.json")
    assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0  # not registered
    assert import_answer(conn, cfg, path).duplicates == [hid]


def test_import_rejects_invalid_files(
    conn: sqlite3.Connection, cfg: AppConfig, tmp_path: Path
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(_answer(IDEA, IDEA, IDEA, IDEA, IDEA, IDEA), encoding="utf-8")  # 6 > 5
    with pytest.raises(LlmError, match="nothing stored"):
        import_answer(conn, cfg, path)
    path.write_text(_answer({**IDEA, "params": {"hours_utc": "1-4"}}), encoding="utf-8")
    with pytest.raises(LlmError, match="schema"):
        import_answer(conn, cfg, path)
    assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
    assert [r[0] for r in conn.execute("SELECT outcome FROM llm_calls")] == ["invalid_answer"] * 2
