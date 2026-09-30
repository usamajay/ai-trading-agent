"""Load the logged research leads (docs/RESEARCH_HYPOTHESES.md) as `human` hypotheses."""

import re
from dataclasses import dataclass
from pathlib import Path

_ROW = re.compile(r"^\|\s*(H\d+)\s*\|(.+)$")


@dataclass(frozen=True)
class Lead:
    hypothesis_id: str
    text: str
    rationale: str


def _plain(cell: str) -> str:
    return cell.replace("**", "").strip()


def parse_leads(markdown: str) -> list[Lead]:
    """Rows of the table: | id | hypothesis | why | how to test | source | status |."""
    leads = []
    for line in markdown.splitlines():
        m = _ROW.match(line.strip())
        if not m:
            continue
        cells = [c for c in m.group(2).split("|")]
        if len(cells) < 3:
            continue
        text, why, test = (_plain(c) for c in cells[:3])
        leads.append(Lead(m.group(1), text, f"{why} Test: {test}"))
    return leads


def read_leads(path: Path) -> list[Lead]:
    return parse_leads(path.read_text(encoding="utf-8"))
