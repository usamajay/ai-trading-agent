"""Claude API hypothesis loop (SPEC §8, docs/PHASE_6_TASKS.md 6.7).

One round = one API call:
1. `build_summary()` collects what research has learned so far from the database:
   hypotheses, train/validation experiments with their verdicts and key metrics,
   experiment lessons, and the strategies and variant options that exist. It never
   reads out-of-sample experiments, trade lessons, or anything about the account.
2. The prompt (fixed instructions + the summary as JSON) asks for at most
   `max_hypotheses` testable ideas in the SPEC §8 JSON schema.
3. **Dry run is the default**: the prompt and its worst-case cost are shown and nothing
   is sent. A live call happens only when asked, and only if its worst case fits the
   per-run cap, the token caps and what is left of the monthly cap.
4. Every live call is recorded in `llm_calls` with its real cost (from the API's token
   counts) **before** the answer is used, so the monthly cap counts failed calls too.
5. The answer is validated (strict schema); valid ideas are stored as `llm` hypotheses
   with status `proposed`. A human chooses which to register as experiments; the model
   never runs backtests or edits strategy code.
"""

import hashlib
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tradeagent.config import AppConfig, LlmSettings
from tradeagent.provenance import git_commit
from tradeagent.research.experiments import add_hypothesis, research_run_total
from tradeagent.strategies import registry
from tradeagent.timeutil import utc_now

PURPOSE = "hypotheses"
RESEARCH_SPLITS = ("train", "validation")  # the only experiment splits ever summarised
HIDDEN_STRATEGIES = ("random_baseline",)
METRIC_KEYS = (
    "trades",
    "expectancy_r",
    "ci_low",
    "ci_high",
    "baseline_percentile",
    "baseline_mean",
    "stress_expectancy_r",
    "avg_cost_r",
    "long_share",
)
CHARS_PER_TOKEN_ESTIMATE = 3  # dry run, no network: a deliberately high token estimate
SCHEMA_TOKEN_MARGIN = 1500  # the response schema is added to the prompt by the API


class LlmError(RuntimeError):
    """The call was refused before sending, failed, or its answer was unusable."""


class SpendCapError(LlmError):
    """The call's worst case does not fit a token or dollar cap; nothing was sent."""


# --- answer schema (SPEC §8) ---------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParamValue(_Strict):
    name: str = Field(min_length=1, max_length=60)
    value: str = Field(min_length=1, max_length=60)


class Proposal(_Strict):
    statement: str = Field(min_length=10, max_length=400)
    rationale: str = Field(min_length=10, max_length=1500)
    strategy_change: str = Field(min_length=3, max_length=800)
    params: list[ParamValue] = Field(max_length=10)
    expected_effect: str = Field(min_length=3, max_length=600)
    how_to_falsify: str = Field(min_length=10, max_length=800)


class ProposalSet(_Strict):
    hypotheses: list[Proposal] = Field(max_length=5)


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_TEXT = {"type": "string"}
RESPONSE_SCHEMA: dict[str, Any] = _object(
    {
        "hypotheses": {
            "type": "array",
            "items": _object(
                {
                    "statement": _TEXT,
                    "rationale": _TEXT,
                    "strategy_change": _TEXT,
                    "params": {
                        "type": "array",
                        "items": _object({"name": _TEXT, "value": _TEXT}),
                    },
                    "expected_effect": _TEXT,
                    "how_to_falsify": _TEXT,
                }
            ),
        }
    }
)


def parse_answer(text: str, max_hypotheses: int) -> list[Proposal]:
    """Validated proposals from the model's JSON text; LlmError if anything is off."""
    try:
        answer = ProposalSet.model_validate_json(text)
    except ValidationError as e:
        raise LlmError(f"answer does not match the schema: {e.error_count()} error(s)") from e
    if len(answer.hypotheses) > max_hypotheses:
        raise LlmError(f"{len(answer.hypotheses)} hypotheses, at most {max_hypotheses} allowed")
    return answer.hypotheses


# --- summary -------------------------------------------------------------------------


def _strategy_cards() -> list[dict[str, Any]]:
    registry.load_builtins()
    cards = []
    for name in registry.names():
        if name in HIDDEN_STRATEGIES:
            continue
        s = registry.create(name)
        doc = (sys.modules[type(s).__module__].__doc__ or "").strip()
        cards.append(
            {
                "name": name,
                "description": doc,
                "style": s.style,
                "timeframes": list(s.timeframes),
                "suited_regimes": list(s.suited_regimes),
                "default_params": dict(s.params),
                "param_ranges": {p.name: [p.low, p.high] for p in s.param_specs},
            }
        )
    return cards


def _metrics(metrics_json: str | None) -> dict[str, Any]:
    if not metrics_json:
        return {}
    m = json.loads(metrics_json).get("metrics", {})
    return {k: round(m[k], 4) if isinstance(m[k], float) else m[k] for k in METRIC_KEYS if k in m}


def build_summary(conn: sqlite3.Connection, cfg: AppConfig) -> dict[str, Any]:
    """What research knows so far. Train/validation only; no account data."""
    placeholders = ",".join("?" for _ in RESEARCH_SPLITS)
    hypotheses = [
        {"id": hid, "source": source, "status": status, "text": text}
        for hid, source, status, text in conn.execute(
            "SELECT hypothesis_id, source, status, text FROM hypotheses ORDER BY hypothesis_id"
        )
    ]
    experiments = []
    for row in conn.execute(
        "SELECT experiment_id, hypothesis_id, dataset_split, symbol, variant_json, "
        "success_criterion, status, verdict, metrics_json FROM experiments "
        f"WHERE dataset_split IN ({placeholders}) ORDER BY experiment_id",
        RESEARCH_SPLITS,
    ):
        eid, hid, split, symbol, variant, criterion, status, verdict, metrics = row
        experiments.append(
            {
                "id": eid,
                "hypothesis": hid,
                "split": split,
                "symbol": symbol,
                "variant": json.loads(variant) if variant else None,
                "criterion": json.loads(criterion) if criterion else None,
                "status": status,
                "verdict": verdict,
                "metrics": _metrics(metrics),
            }
        )
    # Experiment lessons only (trade lessons come from paper/live trading = account data).
    lessons = [
        {"experiment": eid, "text": text}
        for eid, text in conn.execute(
            "SELECT l.experiment_id, l.text FROM lessons l JOIN experiments e "
            "ON e.experiment_id = l.experiment_id WHERE l.trade_id IS NULL "
            f"AND e.dataset_split IN ({placeholders}) ORDER BY l.lesson_id",
            RESEARCH_SPLITS,
        )
    ]
    train = cfg.splits.train if cfg.splits else None
    return {
        "symbols": sorted(cfg.settings.symbols),
        "train_period": None if train is None else [str(train.start), str(train.end)],
        "costs_and_rules": {
            "results_unit": "R = planned loss if the stop is hit; expectancy is after costs",
            "risk_engine": {
                "risk_per_trade_pct": cfg.risk.risk_per_trade_pct,
                "min_reward_risk": cfg.risk.min_reward_risk,
                "stop_atr_range": [cfg.risk.sl_atr_min, cfg.risk.sl_atr_max],
                "news_blackout_minutes": cfg.risk.news_blackout_minutes,
            },
            "standard_criterion": (
                ">= 100 trades, expectancy > 0, 95% bootstrap CI lower bound > 0, "
                ">= 95th percentile of the matching random baseline, still positive "
                "with spread and slippage x 1.5"
            ),
            "drift_rule": (
                "gold rose strongly over train; any direction-biased gold idea must beat "
                "random trades with the same long/short mix"
            ),
        },
        "variant_options": {
            "timeframes": ["M15", "H1", "H4", "D1"],
            "direction": ["both", "long", "short"],
            "style": ["scalp", "intraday", "swing"],
            "regime_filter": "trade only in the strategy's suited_regimes (trend/volatility)",
            "params": "within each strategy's param_ranges",
        },
        "strategies": _strategy_cards(),
        "hypotheses": hypotheses,
        "experiments": experiments,
        "lessons": lessons,
        "research_runs_so_far": research_run_total(conn),
    }


# --- prompt --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You help a small, careful trading-research project (gold XAUUSD and oil USOIL CFDs on a \
retail broker) decide what to test next. You propose hypotheses; you do not run tests.

How the project works:
- Every idea becomes a pre-registered experiment on the train split first. Its success \
criterion is fixed before the run. Only a train pass may be tried once on validation. \
Out-of-sample data is never used for research.
- Every experiment run adds to a multiple-testing count, so each proposal has a real \
cost. Propose fewer, better ideas rather than many small parameter tweaks.
- Costs (spread, slippage, swap) are charged on every trade; many ideas fail because \
their edge is smaller than costs. Results are compared with random trades of the same \
symbol, timeframe, style and direction mix.
- Most hypotheses fail. Falsified ideas and their lessons are in the summary: do not \
propose them again unless you name what is materially different and why it should matter.

Each hypothesis must:
- be testable with the listed strategies and variant options (timeframe, direction, \
style, regime filter, parameters inside their ranges), or describe a small, concrete \
strategy change a human could implement and review;
- give the exact parameter values to test in `params` (empty list if none);
- state in `how_to_falsify` a concrete pass/fail criterion that can be written down \
before the run (the standard criterion is the default; say if you want something else);
- say in `expected_effect` what should change and roughly by how much, in R after costs.

Answer only with JSON in the required schema. Return at most {max_hypotheses} \
hypotheses; returning fewer, or none, is fine when the evidence does not support more.\
"""

USER_PROMPT = """\
Research summary (JSON). Train and validation results only.

{summary}

Propose the next hypotheses to test.\
"""


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256((self.system + "\n\n" + self.user).encode()).hexdigest()

    def estimated_tokens(self) -> int:
        chars = len(self.system) + len(self.user)
        return math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE) + SCHEMA_TOKEN_MARGIN


def build_prompt(summary: dict[str, Any], llm: LlmSettings) -> Prompt:
    return Prompt(
        SYSTEM_PROMPT.format(max_hypotheses=llm.max_hypotheses),
        USER_PROMPT.format(summary=json.dumps(summary, indent=1, sort_keys=True)),
    )


# --- spend caps ----------------------------------------------------------------------


def cost_usd(llm: LlmSettings, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * llm.input_usd_per_mtok + output_tokens * llm.output_usd_per_mtok
    ) / 1_000_000


def month_spend(conn: sqlite3.Connection, now: datetime) -> float:
    """Recorded LLM spend in the current UTC calendar month (every call, any outcome)."""
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    row = conn.execute("SELECT SUM(cost_usd) FROM llm_calls WHERE created_at >= ?", (start,))
    return float(row.fetchone()[0] or 0.0)


@dataclass(frozen=True)
class Budget:
    input_tokens: int  # estimated (dry run) or counted by the API, plus a schema margin
    worst_case_usd: float  # input tokens + the full max_output_tokens
    month_spent_usd: float
    refusal: str | None  # why the call may not be sent; None = within every cap

    @property
    def allowed(self) -> bool:
        return self.refusal is None


def check_budget(llm: LlmSettings, input_tokens: int, month_spent: float) -> Budget:
    worst = cost_usd(llm, input_tokens, llm.max_output_tokens)
    refusal = None
    if input_tokens > llm.max_input_tokens:
        refusal = f"prompt ~{input_tokens} tokens > max_input_tokens {llm.max_input_tokens}"
    elif worst > llm.max_usd_per_run:
        refusal = f"worst case ${worst:.3f} > max_usd_per_run ${llm.max_usd_per_run:.2f}"
    elif month_spent + worst > llm.max_usd_per_month:
        refusal = (
            f"spent ${month_spent:.3f} this month + worst case ${worst:.3f} > "
            f"max_usd_per_month ${llm.max_usd_per_month:.2f}"
        )
    return Budget(input_tokens, worst, month_spent, refusal)


# --- the call ------------------------------------------------------------------------


class MessagesApi(Protocol):
    def count_tokens(self, **kwargs: Any) -> Any: ...
    def create(self, **kwargs: Any) -> Any: ...


class Client(Protocol):
    """The part of `anthropic.Anthropic` used here (tests pass a fake)."""

    @property
    def messages(self) -> MessagesApi: ...


def anthropic_client() -> Client:
    """Real client; the key comes from .env (or the environment). No retries: one call
    is one billed attempt, so the cap check above stays true."""
    import os

    import anthropic
    from dotenv import dotenv_values

    from tradeagent.config import PROJECT_ROOT

    key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or dotenv_values(PROJECT_ROOT / ".env").get("ANTHROPIC_API_KEY")
        or ""
    ).strip()
    if not key:
        raise LlmError("ANTHROPIC_API_KEY is not set in .env")
    return anthropic.Anthropic(api_key=key, max_retries=0, timeout=600.0)


def _record_call(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    prompt: Prompt,
    budget: Budget,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    cost: float,
    stop_reason: str | None,
    request_id: str | None,
    outcome: str,
    response_text: str | None,
) -> str:
    n = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] + 1
    call_id = f"C{n:04d}"
    with conn:
        conn.execute(
            "INSERT INTO llm_calls (call_id, purpose, model, prompt_sha256, input_tokens, "
            "output_tokens, cost_usd, worst_case_usd, stop_reason, request_id, outcome, "
            "response_text, git_commit, config_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                call_id,
                PURPOSE,
                cfg.settings.llm.model,
                prompt.sha256,
                input_tokens,
                output_tokens,
                cost,
                budget.worst_case_usd,
                stop_reason,
                request_id,
                outcome,
                response_text,
                git_commit(),
                cfg.config_hash,
                utc_now().isoformat(),
            ),
        )
    return call_id


def _set_outcome(conn: sqlite3.Connection, call_id: str, outcome: str) -> None:
    with conn:
        conn.execute("UPDATE llm_calls SET outcome = ? WHERE call_id = ?", (outcome, call_id))


def hypothesis_id(statement: str) -> str:
    """Stable id, so the same idea proposed twice is stored once."""
    key = " ".join(statement.lower().split())
    return "L" + hashlib.sha256(key.encode()).hexdigest()[:6]


@dataclass(frozen=True)
class RoundResult:
    call_id: str
    cost_usd: float
    proposals: list[Proposal]
    stored: list[str]  # new hypothesis ids
    duplicates: list[str]  # ids that already existed


def dry_run(conn: sqlite3.Connection, cfg: AppConfig) -> tuple[Prompt, Budget]:
    """The prompt and its cap check, with no network call and nothing written."""
    prompt = build_prompt(build_summary(conn, cfg), cfg.settings.llm)
    budget = check_budget(cfg.settings.llm, prompt.estimated_tokens(), month_spend(conn, utc_now()))
    return prompt, budget


def propose(conn: sqlite3.Connection, cfg: AppConfig, client: Client) -> RoundResult:
    """One live round: cap check -> one call -> record cost -> validate -> store."""
    import anthropic

    llm = cfg.settings.llm
    prompt = build_prompt(build_summary(conn, cfg), llm)
    request: dict[str, Any] = {
        "model": llm.model,
        "system": prompt.system,
        "messages": [{"role": "user", "content": prompt.user}],
    }
    # Exact input count from the API (free), plus a margin for the schema it adds.
    try:
        counted = client.messages.count_tokens(**request).input_tokens + SCHEMA_TOKEN_MARGIN
    except anthropic.APIError as e:  # token counting is free: nothing to record
        raise LlmError(f"not sent: token count failed: {e}") from e
    budget = check_budget(llm, counted, month_spend(conn, utc_now()))
    if not budget.allowed:
        raise SpendCapError(f"not sent: {budget.refusal}")

    def failed(outcome: str, cost: float, err: Exception) -> LlmError:
        call_id = _record_call(
            conn,
            cfg,
            prompt,
            budget,
            input_tokens=None,
            output_tokens=None,
            cost=cost,
            stop_reason=None,
            request_id=None,
            outcome=outcome,
            response_text=str(err)[:2000],
        )
        return LlmError(f"{call_id}: {outcome}: {err}")

    try:
        response = client.messages.create(
            **request,
            max_tokens=llm.max_output_tokens,
            thinking={"type": "adaptive"},
            output_config={
                "effort": llm.effort,
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            },
        )
    except anthropic.APIStatusError as e:  # rejected by the API: not billed
        raise failed("api_error", 0.0, e) from e
    except anthropic.APIConnectionError as e:  # incl. timeout: may have been billed
        raise failed("connection_error", budget.worst_case_usd, e) from e

    usage = response.usage
    in_tokens = (
        usage.input_tokens
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        + (getattr(usage, "cache_read_input_tokens", 0) or 0)
    )
    out_tokens = usage.output_tokens
    cost = cost_usd(llm, in_tokens, out_tokens)
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    call_id = _record_call(
        conn,
        cfg,
        prompt,
        budget,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        cost=cost,
        stop_reason=response.stop_reason,
        request_id=getattr(response, "_request_id", None),
        outcome="received",
        response_text=text,
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        _set_outcome(conn, call_id, response.stop_reason)
        raise LlmError(f"{call_id}: stopped with {response.stop_reason}; nothing stored")
    try:
        proposals = parse_answer(text, llm.max_hypotheses)
    except LlmError as e:
        _set_outcome(conn, call_id, "invalid_answer")
        raise LlmError(f"{call_id}: {e}; nothing stored") from e

    stored, duplicates = [], []
    for p in proposals:
        hid = hypothesis_id(p.statement)
        rationale = json.dumps(
            {
                "rationale": p.rationale,
                "strategy_change": p.strategy_change,
                "params": {pv.name: pv.value for pv in p.params},
                "expected_effect": p.expected_effect,
                "how_to_falsify": p.how_to_falsify,
                "llm_call": call_id,
                "model": llm.model,
            },
            ensure_ascii=False,
        )
        if add_hypothesis(conn, cfg, hid, p.statement, rationale, "llm"):
            stored.append(hid)
        else:
            duplicates.append(hid)
    _set_outcome(conn, call_id, "ok")
    return RoundResult(call_id, cost, proposals, stored, duplicates)
