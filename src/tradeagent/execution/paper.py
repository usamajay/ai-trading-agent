"""Paper execution (SPEC §9): simulated fills, gated by the risk engine's tokens.

`PaperBroker.submit(order, token)` refuses any order without a valid approval token
for that exact order (symbol, side, lots, stop-loss): missing, forged, expired,
reused or changed after approval. While the kill switch is on it refuses
everything, and `close_all` / `cancel_all` flatten the book. No broker calls here:
demo-account orders arrive in Phase 8 and will pass through the same gate.
"""

from dataclasses import dataclass, field
from datetime import datetime

from tradeagent.risk.killswitch import KillSwitch
from tradeagent.risk.tokens import ApprovalToken, TokenSigner


class ExecutionRefused(Exception):
    """The order was not executed; the message says why."""


@dataclass(frozen=True)
class PaperOrder:
    order_id: str
    symbol: str
    direction: str  # "long" | "short"
    lots: float
    stop_loss: float
    take_profit: float
    price: float  # simulated fill price


@dataclass(frozen=True)
class PaperFill:
    order: PaperOrder
    time_utc: datetime
    price: float
    action: str  # "open" or "close"


@dataclass
class PaperBroker:
    signer: TokenSigner
    kill_switch: KillSwitch | None = None
    positions: dict[str, PaperOrder] = field(default_factory=dict)  # order id -> order
    pending: dict[str, PaperOrder] = field(default_factory=dict)
    fills: list[PaperFill] = field(default_factory=list)
    refusals: list[tuple[str, str]] = field(default_factory=list)  # (order id, reason)

    def _refuse(self, order: PaperOrder, reason: str) -> ExecutionRefused:
        self.refusals.append((order.order_id, reason))
        return ExecutionRefused(f"order {order.order_id} refused: {reason}")

    def submit(self, order: PaperOrder, token: ApprovalToken | None, now: datetime) -> PaperFill:
        if self.kill_switch is not None and self.kill_switch.active():
            raise self._refuse(order, "kill switch is on")
        reason = self.signer.verify(
            token, order.order_id, order.symbol, order.direction, order.lots, order.stop_loss, now
        )
        if reason is not None:
            raise self._refuse(order, reason)
        fill = PaperFill(order, now, order.price, "open")
        self.positions[order.order_id] = order
        self.fills.append(fill)
        return fill

    def close_all(self, prices: dict[str, float], now: datetime) -> list[PaperFill]:
        """Close every open position at `prices[symbol]` (kill switch, shutdown)."""
        closed = [
            PaperFill(order, now, prices[order.symbol], "close")
            for order in self.positions.values()
        ]
        self.positions.clear()
        self.fills.extend(closed)
        return closed

    def cancel_all(self) -> int:
        count = len(self.pending)
        self.pending.clear()
        return count
