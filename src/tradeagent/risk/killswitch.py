"""Kill switch (SPEC §6): the file `KILL` in the project root stops everything.

While it exists the risk engine rejects every order and execution closes all
positions and cancels all orders. Creating it is always allowed; removing it is a
human action with a written reason (logged by the CLI to `risk_events`).
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tradeagent.config import PROJECT_ROOT
from tradeagent.timeutil import utc_now

KILL_FILE = PROJECT_ROOT / "KILL"


@dataclass(frozen=True)
class KillSwitch:
    path: Path = KILL_FILE

    def active(self) -> bool:
        return self.path.exists()

    def activate(self, reason: str, by: str, now: datetime | None = None) -> None:
        info = {"reason": reason, "by": by, "time_utc": (now or utc_now()).isoformat()}
        self.path.write_text(json.dumps(info), encoding="utf-8")

    def info(self) -> dict[str, str] | None:
        """What was written when it was activated (None if not active or unreadable)."""
        if not self.active():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"reason": "KILL file present (unreadable)", "by": "?", "time_utc": "?"}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else None

    def clear(self, reason: str) -> None:
        """Remove the switch. Callers must log `reason` (a human decision)."""
        if not reason.strip():
            raise ValueError("clearing the kill switch needs a written reason")
        self.path.unlink(missing_ok=True)
