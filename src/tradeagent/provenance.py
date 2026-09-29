"""Where a result came from: git commit and run ids (CLAUDE.md: every row stores
git_commit and config_hash)."""

import subprocess
import uuid

from tradeagent.config import PROJECT_ROOT
from tradeagent.timeutil import utc_now


def git_commit() -> str:
    """Short commit hash, with '-dirty' if tracked files have uncommitted changes."""
    try:
        sha = _git("rev-parse", "--short=12", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def new_run_id() -> str:
    """Sortable unique id, e.g. '20260929T181500Z-3f9a1c'."""
    return f"{utc_now():%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()
