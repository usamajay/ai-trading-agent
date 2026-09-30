"""Would the risk.yaml loss limits have been breached? (informational until Phase 4)

Measured on the daily mark-to-market equity (one value per trading day, 17:00 New
York close; see metrics.py):
- daily loss:  a day's close vs the previous day's close (the starting balance
  for the first day) falls by >= max_daily_loss_pct % of that previous value;
- weekly loss: any day's close vs the close of the previous week's last day (the
  starting balance for the first week) falls by >= max_weekly_loss_pct %;
  weeks are Monday-Friday trading days (ISO weeks of the trading day);
- drawdown:    a day's close is >= max_drawdown_pct % below the running peak
  (the starting balance included).
Daily marks miss moves inside the day, so a real account could breach earlier.
The Phase 4 risk engine will enforce these limits; here they only raise a flag.
"""

from typing import Any

import pandas as pd

from tradeagent.config import RiskLimits


def _rule(limit: float, falls_pct: pd.Series, days: pd.Series) -> dict[str, Any]:
    breached = falls_pct >= limit - 1e-12
    first = days[breached].iloc[0] if breached.any() else None
    return {
        "limit_pct": limit,
        "breached": bool(breached.any()),
        "first_breach_day": str(first) if first is not None else None,
        "breaches": int(breached.sum()),
        "worst_pct": float(falls_pct.max()) if len(falls_pct) else 0.0,
    }


def risk_limit_flags(daily: pd.DataFrame, start: float, limits: RiskLimits) -> dict[str, Any]:
    """`daily`: columns trading_day, equity (mark-to-market), oldest first."""
    equity = daily["equity"].astype(float).reset_index(drop=True)
    days = daily["trading_day"].reset_index(drop=True)
    if equity.empty:
        empty = pd.Series(dtype=float)
        return {
            "daily_loss": _rule(limits.max_daily_loss_pct, empty, empty),
            "weekly_loss": _rule(limits.max_weekly_loss_pct, empty, empty),
            "max_drawdown": _rule(limits.max_drawdown_pct, empty, empty),
        }

    previous = equity.shift(fill_value=start)
    daily_fall = ((previous - equity) / previous * 100).clip(lower=0)

    iso = pd.to_datetime(days).dt.isocalendar()
    week = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    new_week = week != week.shift()
    week_open = previous.where(new_week).ffill()  # equity at the end of the week before
    weekly_fall = ((week_open - equity) / week_open * 100).clip(lower=0)

    peak = pd.concat([pd.Series([start]), equity], ignore_index=True).cummax().iloc[1:]
    peak.index = equity.index
    dd = ((peak - equity) / peak * 100).clip(lower=0)

    return {
        "daily_loss": _rule(limits.max_daily_loss_pct, daily_fall, days),
        "weekly_loss": _rule(limits.max_weekly_loss_pct, weekly_fall, days),
        "max_drawdown": _rule(limits.max_drawdown_pct, dd, days),
    }
