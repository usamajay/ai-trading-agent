"""Paper execution refuses every order the risk engine did not approve exactly."""

import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tradeagent.config import load_config
from tradeagent.execution.paper import ExecutionRefused, PaperBroker, PaperOrder
from tradeagent.risk.engine import OrderRequest, RiskEngine
from tradeagent.risk.killswitch import KillSwitch
from tradeagent.risk.state import RiskState
from tradeagent.risk.tokens import TokenSigner

NOW = datetime(2026, 1, 6, 15, 0, tzinfo=UTC)
CFG = load_config()


def approved(signer: TokenSigner, order_id: str = "o1"):  # type: ignore[no-untyped-def]
    engine = RiskEngine(CFG.risk, {"XAUUSD": CFG.costs.symbols["XAUUSD"]}, signer=signer)  # type: ignore[union-attr]
    req = OrderRequest(order_id, "XAUUSD", "long", 2000.0, 1995.0, 2010.0, NOW, 5.0, 190, 190)
    decision = engine.evaluate(req, RiskState.start(10_000, NOW))
    assert decision.approved
    order = PaperOrder(order_id, "XAUUSD", "long", decision.lots, 1995.0, 2010.0, 2000.2)
    return order, decision.token


def test_risk_approved_order_fills() -> None:
    signer = TokenSigner()
    broker = PaperBroker(signer)
    order, token = approved(signer)
    fill = broker.submit(order, token, NOW)
    assert fill.action == "open" and fill.price == 2000.2
    assert broker.positions == {"o1": order}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda o, t: (o, None), "no approval token"),
        (lambda o, t: (dataclasses.replace(o, lots=o.lots * 2), t), "lots"),
        (lambda o, t: (dataclasses.replace(o, stop_loss=1990.0), t), "stop-loss"),
        (lambda o, t: (dataclasses.replace(o, direction="short"), t), "direction"),
        (
            lambda o, t: (o, TokenSigner().issue("o1", "XAUUSD", "long", o.lots, 1995.0, NOW)),
            "signature",
        ),
    ],
)
def test_orders_not_approved_exactly_are_refused(change, reason: str) -> None:  # type: ignore[no-untyped-def]
    signer = TokenSigner()
    broker = PaperBroker(signer)
    order, token = change(*approved(signer))
    with pytest.raises(ExecutionRefused, match=reason):
        broker.submit(order, token, NOW)
    assert broker.positions == {} and broker.refusals[0][0] == "o1"


def test_expired_and_reused_tokens() -> None:
    signer = TokenSigner()
    broker = PaperBroker(signer)
    order, token = approved(signer)
    with pytest.raises(ExecutionRefused, match="expired"):
        broker.submit(order, token, NOW + timedelta(minutes=2))
    order2, token2 = approved(signer, "o2")
    broker.submit(order2, token2, NOW)
    with pytest.raises(ExecutionRefused, match="already used"):
        broker.submit(order2, token2, NOW)


def test_kill_switch_refuses_and_flattens(tmp_path: Path) -> None:
    signer = TokenSigner()
    ks = KillSwitch(tmp_path / "KILL")
    broker = PaperBroker(signer, ks)
    order, token = approved(signer)
    broker.submit(order, token, NOW)
    broker.pending["p1"] = order
    ks.activate("drill", "pytest", NOW)
    order2, token2 = approved(signer, "o2")
    with pytest.raises(ExecutionRefused, match="kill switch"):
        broker.submit(order2, token2, NOW)
    closed = broker.close_all({"XAUUSD": 1999.0}, NOW)
    assert [f.action for f in closed] == ["close"] and closed[0].price == 1999.0
    assert broker.positions == {} and broker.cancel_all() == 1 and broker.pending == {}
