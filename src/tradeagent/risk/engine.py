"""THE hard gate (SPEC §6): every order is checked here before execution.

Deterministic Python: no LLM, no ML, no network. Limits come from config/risk.yaml
(frozen config, read only). A decision lists every failed rule with a reason code
and text (SPEC §10), sizes the trade, and, only when everything passes, issues an
approval token for that exact order; execution refuses orders without one.

Trade-level rules: stop-loss present and on the right side, stop 0.5-3 x ATR,
reward:risk >= min, spread <= 2 x median, size >= min lot at 0.5% risk, open-position
limits, gold/oil same-direction correlation, news blackout.
Account-level rules (`account_rules`): kill switch, drawdown shutdown, daily and
weekly loss stops, consecutive-loss pause. Backtests may report these as flags
instead of enforcing them (docs/PHASE_4_TASKS.md 4.8); the kill switch always applies
when one is given.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

from tradeagent.config import RiskLimits, SymbolCosts
from tradeagent.risk.killswitch import KillSwitch
from tradeagent.risk.sizing import position_size
from tradeagent.risk.state import RiskState
from tradeagent.risk.tokens import ApprovalToken, TokenSigner

NewsStatus = Literal["clear", "blackout", "unknown"]
_EPS = 1e-9

# Reason codes, in the order they are checked.
CODES = (
    "kill_switch",
    "shutdown",
    "daily_loss",
    "weekly_loss",
    "loss_streak",
    "no_stop",
    "sl_atr",
    "rr",
    "spread",
    "news",
    "news_unknown",
    "max_positions",
    "correlation",
    "min_lot",
)


class NewsSource(Protocol):
    def status(self, now: datetime) -> tuple[NewsStatus, str]:
        """('clear' | 'blackout' | 'unknown', detail)."""
        ...


@dataclass(frozen=True)
class OrderRequest:
    order_id: str
    symbol: str
    direction: str  # "long" | "short"
    entry_ref: float  # expected entry price (ask for longs, bid for shorts)
    stop_loss: float | None
    take_profit: float
    time_utc: datetime
    atr: float
    spread: float  # current spread, points
    median_spread: float  # reference median spread, points


@dataclass(frozen=True)
class Rejection:
    code: str
    detail: str


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    lots: float
    rejections: tuple[Rejection, ...] = ()
    token: ApprovalToken | None = None

    @property
    def codes(self) -> list[str]:
        return [r.code for r in self.rejections]

    def explanation(self) -> str:
        if self.approved:
            return f"approved: {self.lots} lots"
        return "rejected: " + "; ".join(f"{r.code} ({r.detail})" for r in self.rejections)


@dataclass
class RiskEngine:
    limits: RiskLimits
    specs: dict[str, SymbolCosts]  # symbol -> contract and lot sizes
    signer: TokenSigner | None = None  # None: decide only, issue no tokens (backtests)
    news: NewsSource | None = None
    kill_switch: KillSwitch | None = None
    account_rules: bool = True
    strict_news: bool = True  # an unknown calendar blocks entries (live/paper)
    decisions: int = field(default=0, init=False)

    def evaluate(
        self,
        req: OrderRequest,
        state: RiskState,
        correlation: float | None = None,
    ) -> RiskDecision:
        """`correlation`: 20-day correlation of gold and oil daily returns, if known."""
        self.decisions += 1
        out: list[Rejection] = []
        if self.kill_switch is not None and self.kill_switch.active():
            out.append(Rejection("kill_switch", "KILL file present: all trading stopped"))
        if self.account_rules:
            out += [Rejection(code, detail) for code, detail in state.blocks(req.time_utc)]
        out += self._trade_rules(req, state, correlation)

        lots = 0.0
        spec = self.specs.get(req.symbol)
        if spec is None:
            out.append(Rejection("min_lot", f"no contract details for {req.symbol}"))
        elif not any(r.code == "no_stop" for r in out):  # size only a valid stop
            assert req.stop_loss is not None
            distance = abs(req.entry_ref - req.stop_loss)
            lots = position_size(state.equity, self.limits.risk_per_trade_pct, distance, spec)
            if lots < spec.volume_min - _EPS:
                out.append(
                    Rejection(
                        "min_lot",
                        f"{self.limits.risk_per_trade_pct}% of {state.equity:,.2f} allows {lots} lots, "
                        f"below the minimum {spec.volume_min}",
                    )
                )
                lots = 0.0

        if out:
            return RiskDecision(False, 0.0, tuple(out))
        token = None
        if self.signer is not None:
            assert req.stop_loss is not None
            token = self.signer.issue(
                req.order_id, req.symbol, req.direction, lots, req.stop_loss, req.time_utc
            )
        return RiskDecision(True, lots, (), token)

    def _trade_rules(
        self, req: OrderRequest, state: RiskState, correlation: float | None
    ) -> list[Rejection]:
        lim, out = self.limits, []
        long = req.direction == "long"
        sl = req.stop_loss
        if (
            sl is None
            or not math.isfinite(sl)
            or (sl >= req.entry_ref if long else sl <= req.entry_ref)
        ):
            out.append(
                Rejection("no_stop", "a stop-loss on the losing side of the entry is required")
            )
        else:
            risk = abs(req.entry_ref - sl)
            if not math.isfinite(req.atr) or req.atr <= 0:
                out.append(Rejection("sl_atr", "ATR not available"))
            else:
                in_atr = risk / req.atr
                if in_atr < lim.sl_atr_min - _EPS or in_atr > lim.sl_atr_max + _EPS:
                    out.append(
                        Rejection(
                            "sl_atr",
                            f"stop {in_atr:.2f} x ATR, allowed {lim.sl_atr_min}-{lim.sl_atr_max}",
                        )
                    )
            reward = (
                (req.take_profit - req.entry_ref) if long else (req.entry_ref - req.take_profit)
            )
            if reward / risk < lim.min_reward_risk - _EPS:
                out.append(
                    Rejection(
                        "rr", f"reward:risk {reward / risk:.2f}, minimum {lim.min_reward_risk}"
                    )
                )
        if (
            req.median_spread > 0
            and req.spread > lim.max_spread_multiple * req.median_spread + _EPS
        ):
            out.append(
                Rejection(
                    "spread",
                    f"spread {req.spread:g} > {lim.max_spread_multiple} x median {req.median_spread:g}",
                )
            )
        if self.news is not None:
            status, detail = self.news.status(req.time_utc)
            if status == "blackout":
                out.append(Rejection("news", detail))
            elif status == "unknown" and self.strict_news:
                out.append(Rejection("news_unknown", f"news calendar unavailable: {detail}"))
        same_symbol = [p for p in state.positions if p.symbol == req.symbol]
        if len(state.positions) >= lim.max_open_positions:
            out.append(
                Rejection(
                    "max_positions",
                    f"{len(state.positions)} open, limit {lim.max_open_positions} in total",
                )
            )
        elif len(same_symbol) >= lim.max_open_per_symbol:
            out.append(
                Rejection(
                    "max_positions",
                    f"{len(same_symbol)} open on {req.symbol}, limit {lim.max_open_per_symbol}",
                )
            )
        if correlation is not None and correlation > lim.correlation_limit:
            clash = [
                p
                for p in state.positions
                if p.symbol != req.symbol and p.direction == req.direction
            ]
            if clash:
                out.append(
                    Rejection(
                        "correlation",
                        f"{clash[0].symbol} already {req.direction} and 20-day correlation "
                        f"{correlation:.2f} > {lim.correlation_limit}",
                    )
                )
        return out
