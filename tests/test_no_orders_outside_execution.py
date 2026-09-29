"""Safety rule (CLAUDE.md #1, SPEC §9): only execution/ may place or check orders."""

from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "tradeagent"
ORDER_FUNCTIONS = ("order_send", "order_check")


def test_order_functions_only_in_execution() -> None:
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        if relative.parts[0] == "execution":
            continue
        text = path.read_text(encoding="utf-8")
        offenders += [f"{relative}: {name}" for name in ORDER_FUNCTIONS if name in text]
    assert not offenders, f"Order functions referenced outside execution/: {offenders}"


def test_scan_covers_the_package() -> None:
    # Guard against the scan silently checking nothing (e.g. after a folder move).
    assert (PACKAGE / "data" / "mt5_client.py").is_file()
