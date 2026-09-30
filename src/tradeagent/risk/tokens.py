"""Approval tokens (SPEC §6): execution only accepts orders the risk engine approved.

A token is an HMAC-SHA256 signature over the exact order (id, symbol, side, lots,
stop-loss) and an expiry, made with a secret created in memory when the signer is
built and never stored. Execution verifies every field, the expiry, and that the
token has not been used before. A changed size or stop-loss therefore fails.
"""

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

DEFAULT_TTL = timedelta(seconds=60)


@dataclass(frozen=True)
class ApprovalToken:
    order_id: str
    symbol: str
    direction: str
    lots: float
    stop_loss: float
    expires_utc: datetime
    nonce: str
    signature: str


def _payload(
    order_id: str,
    symbol: str,
    direction: str,
    lots: float,
    stop_loss: float,
    expires_utc: datetime,
    nonce: str,
) -> bytes:
    return "|".join(
        [
            order_id,
            symbol,
            direction,
            f"{lots:.8f}",
            f"{stop_loss:.8f}",
            expires_utc.isoformat(),
            nonce,
        ]
    ).encode()


class TokenSigner:
    """Issues and verifies tokens. Only the risk engine should call `issue`."""

    def __init__(self, ttl: timedelta = DEFAULT_TTL, secret: bytes | None = None) -> None:
        self._secret = secret or secrets.token_bytes(32)
        self._ttl = ttl
        self._used: set[str] = set()

    def _sign(self, payload: bytes) -> str:
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def issue(
        self,
        order_id: str,
        symbol: str,
        direction: str,
        lots: float,
        stop_loss: float,
        now: datetime,
    ) -> ApprovalToken:
        expires = now + self._ttl
        nonce = secrets.token_hex(8)
        signature = self._sign(
            _payload(order_id, symbol, direction, lots, stop_loss, expires, nonce)
        )
        return ApprovalToken(
            order_id, symbol, direction, lots, stop_loss, expires, nonce, signature
        )

    def verify(
        self,
        token: ApprovalToken | None,
        order_id: str,
        symbol: str,
        direction: str,
        lots: float,
        stop_loss: float,
        now: datetime,
    ) -> str | None:
        """None if the token approves exactly this order now; otherwise the reason.
        A valid token is consumed (single use)."""
        if token is None:
            return "no approval token"
        expected = self._sign(
            _payload(
                token.order_id,
                token.symbol,
                token.direction,
                token.lots,
                token.stop_loss,
                token.expires_utc,
                token.nonce,
            )
        )
        if not hmac.compare_digest(expected, token.signature):
            return "token signature is invalid (not issued by this risk engine)"
        mismatches = [
            name
            for name, a, b in (
                ("order id", token.order_id, order_id),
                ("symbol", token.symbol, symbol),
                ("direction", token.direction, direction),
                ("lots", f"{token.lots:.8f}", f"{lots:.8f}"),
                ("stop-loss", f"{token.stop_loss:.8f}", f"{stop_loss:.8f}"),
            )
            if a != b
        ]
        if mismatches:
            return f"token does not match the order: {', '.join(mismatches)}"
        if now > token.expires_utc:
            return "token has expired"
        if token.nonce in self._used:
            return "token was already used"
        self._used.add(token.nonce)
        return None
