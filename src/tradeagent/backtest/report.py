"""Write a backtest run's files: report.md, equity.png, trades/daily Parquet, metrics.json.

The chart follows the project's chart rules: one y-axis per panel (equity on top,
drawdown in its own panel below, never a second axis), two series in fixed
categorical colours (validated for colour-blind separation and contrast), a
legend plus direct end labels, thin lines, a recessive grid.
"""

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import pandas as pd

from tradeagent.backtest.engine import BacktestResult
from tradeagent.backtest.records import to_json

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = {"mtm": "#2a78d6", "closed": "#eb6834"}  # categorical slots 1 and 2


# --- number formatting --------------------------------------------------------------------


def _num(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "–"
    if isinstance(value, str):
        return "∞" if value == "inf" else value
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value:,.{digits}f}{suffix}"


def _usd(value: Any) -> str:
    if value is None:
        return "–"
    return f"-${-value:,.2f}" if value < 0 else f"${value:,.2f}"


def _ci(pair: Any, digits: int = 2) -> str:
    if not pair:
        return "–"
    return f"{_num(pair[0], digits)} to {_num(pair[1], digits)}"


# --- chart --------------------------------------------------------------------------------


def equity_chart(result: BacktestResult, path: Path, title: str) -> None:
    daily = result.daily
    closed = result.equity.dropna(subset=["time_utc"])
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    if len(daily):
        top.plot(
            daily["time_utc"],
            daily["equity"],
            color=SERIES["mtm"],
            linewidth=3,  # wider, underneath: visible wherever the two lines differ
            label="Daily mark-to-market equity",
            zorder=2,
        )
        top.annotate(
            "mark-to-market",
            (daily["time_utc"].iloc[-1], daily["equity"].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=TEXT_SECONDARY,
            fontsize=8,
            va="center",
        )
    if len(closed) > 1:
        top.step(
            closed["time_utc"],
            closed["equity"],
            where="post",
            color=SERIES["closed"],
            linewidth=1.2,
            label="Closed-trade equity",
            zorder=3,
        )
        top.annotate(
            "closed trades",
            (closed["time_utc"].iloc[-1], closed["equity"].iloc[-1]),
            xytext=(6, -10),
            textcoords="offset points",
            color=TEXT_SECONDARY,
            fontsize=8,
            va="center",
        )
    top.axhline(result.start_balance, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
    top.set_ylabel("Equity (USD)", color=TEXT_SECONDARY, fontsize=9)
    # Legend above the plot, so it never covers the curve (which starts top-left).
    top.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT,
        borderaxespad=0.2,
    )
    top.set_title(title, color=TEXT, fontsize=11, loc="left", pad=22)

    if len(daily):
        values = pd.concat([pd.Series([result.start_balance]), daily["equity"]], ignore_index=True)
        peak = values.cummax()
        dd = ((values - peak) / peak * 100).iloc[1:].to_numpy()
        bottom.fill_between(daily["time_utc"], dd, 0, color=SERIES["mtm"], alpha=0.25, linewidth=0)
        bottom.plot(daily["time_utc"], dd, color=SERIES["mtm"], linewidth=1.5)
    bottom.set_ylabel("Drawdown (%)", color=TEXT_SECONDARY, fontsize=9)
    bottom.set_xlabel("Date (UTC)", color=TEXT_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


# --- markdown -----------------------------------------------------------------------------


def _group_table(groups: dict[str, dict[str, Any]], heading: str) -> list[str]:
    if not groups:
        return [f"No trades to split by {heading.lower()}.", ""]
    costs = "avg_cost_r" in next(iter(groups.values()))  # by_direction adds costs in R
    extra_head, extra_rule = (" Cost (R) | Swap paid (R) |", "---:|---:|") if costs else ("", "")
    lines = [
        (
            f"| {heading} | Trades | Win rate | Expectancy (R) | Net P&L | Profit factor |"
            f"{extra_head} Sample |"
        ),
        f"|---|---:|---:|---:|---:|---:|{extra_rule}---|",
    ]
    for key, s in groups.items():
        sample = "insufficient" if s["insufficient_sample"] else "ok"
        extra = f" {_num(s['avg_cost_r'], 3)} | {_num(s['avg_swap_paid_r'], 3)} |" if costs else ""
        lines.append(
            f"| {key} | {s['trades']} | {_num(s['win_rate_pct'], 1, '%')} | "
            f"{_num(s['expectancy_r'], 3)} | {_usd(s['net_pnl_usd'])} | "
            f"{_num(s['profit_factor'])} |{extra} {sample} |"
        )
    return [*lines, ""]


def report_markdown(
    run: dict[str, Any],
    metrics: dict[str, Any],
    stress: dict[str, Any],
    flags: dict[str, Any],
    counts: dict[str, int],
    created: str,
) -> str:
    t, st = metrics["trades"], stress["trades"]
    e, se = metrics["equity"], stress["equity"]
    ci = metrics["confidence_95"]
    c = metrics["costs"]
    mb = metrics["min_balance"]
    stress_ok = run["stress_pass"]
    lines = [
        (
            f"# Backtest: {run['strategy']} {run['strategy_version']} on {run['symbol']} "
            f"{run['timeframe']} ({run['split']})"
        ),
        "",
    ]
    if t["insufficient_sample"]:
        lines += [
            (
                f"> **INSUFFICIENT SAMPLE: {t['trades']} trades** (fewer than 30). Treat every "
                "number below as unreliable."
            ),
            "",
        ]
    lines += [
        f"- Run id: `{run['run_id']}`, created {created}",
        f"- Period: {run['date_start']} to {run['date_end']} (UTC, [start, end))",
        f"- Seed: {run['seed']}; parameters: `{run['params_json']}`",
        "",
        "## Provenance",
        "",
        f"- git commit: `{run['git_commit']}`",
        f"- config_hash: `{run['config_hash']}`",
        f"- data fingerprint (bars used): `{run['data_hash']}`",
        "",
        "## Multiple testing (SPEC §7.4)",
        "",
        (
            f"This is run **#{run['run_number']}** of `{run['strategy']}` on **{run['split']}**. "
            f"Runs so far (any version or parameters): **train {counts['train']}**, "
            f"**validation {counts['validation']}**. Every extra variant tried raises the "
            "chance of a lucky result; Phase 7 corrects for this count."
        ),
        "",
        (
            f"## Summary: normal costs vs cost stress (spread and slippage × "
            f"{run['cost_stress_multiple']})"
        ),
        "",
        "| | Normal | Stress |",
        "|---|---:|---:|",
        f"| Trades | {t['trades']} | {st['trades']} |",
        f"| Win rate | {_num(t['win_rate_pct'], 1, '%')} | {_num(st['win_rate_pct'], 1, '%')} |",
        f"| Profit factor | {_num(t['profit_factor'])} | {_num(st['profit_factor'])} |",
        f"| Expectancy (R) | {_num(t['expectancy_r'], 3)} | {_num(st['expectancy_r'], 3)} |",
        f"| Expectancy ($) | {_usd(t['expectancy_usd'])} | {_usd(st['expectancy_usd'])} |",
        f"| Net profit | {_usd(e['net_profit_usd'])} | {_usd(se['net_profit_usd'])} |",
        (
            f"| Max drawdown (mark-to-market) | {_num(e['max_dd_pct'], 2, '%')} | "
            f"{_num(se['max_dd_pct'], 2, '%')} |"
        ),
        f"| Sharpe | {_num(e['sharpe'])} | {_num(se['sharpe'])} |",
        "",
        (
            f"**Cost stress: {'PASS' if stress_ok else 'FAIL'}** (expectancy after costs must stay "
            "above 0 with stressed costs)."
        ),
        "",
        "## 95% confidence intervals (trade bootstrap)",
        "",
        "| Metric | Estimate | 95% interval |",
        "|---|---:|---:|",
        f"| Win rate | {_num(t['win_rate_pct'], 1, '%')} | {_ci(ci['win_rate_pct'], 1)} % |",
        f"| Expectancy (R) | {_num(t['expectancy_r'], 3)} | {_ci(ci['expectancy_r'], 3)} |",
        f"| Profit factor | {_num(t['profit_factor'])} | {_ci(ci['profit_factor'])} |",
        "",
        "## Equity and drawdown",
        "",
        "![Equity curve](equity.png)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Start → final balance | {_usd(e['start_balance'])} → {_usd(e['final_balance'])} |",
        f"| Return / CAGR | {_num(e['return_pct'], 2, '%')} / {_num(e['cagr_pct'], 2, '%')} |",
        (
            f"| Max drawdown, daily mark-to-market | {_num(e['max_dd_pct'], 2, '%')} "
            f"({_usd(e['max_dd_usd'])}), {e['max_dd_duration_days']} trading days |"
        ),
        (
            f"| Max drawdown, closed trades only | {_num(e['closed_trade_max_dd_pct'], 2, '%')} "
            f"({_usd(e['closed_trade_max_dd_usd'])}) |"
        ),
        f"| Sharpe / Sortino | {_num(e['sharpe'])} / {_num(e['sortino'])} |",
        f"| Calmar / recovery factor | {_num(e['calmar'])} / {_num(e['recovery_factor'])} |",
        f"| Avg win / avg loss (R) | {_num(t['avg_win_r'], 3)} / {_num(t['avg_loss_r'], 3)} |",
        f"| Longest losing streak | {t['longest_losing_streak']} |",
        f"| Exposure | {_num(e['exposure_pct'], 1, '%')} of bars |",
        f"| Trading days | {e['trading_days']} |",
        "",
        "## Account risk limits",
        "",
        (
            "**Enforced** in this run: trading stopped when a limit was hit."
            if '"__account_limits": "enforced"' in run["params_json"]
            else "Reported as flags in this run (not enforced), so the whole period is "
            "evaluated; paper and live trading always enforce them."
        ),
        "",
        (
            "Checked on daily mark-to-market equity (moves inside a day are not seen, so a real "
            "account could breach earlier)."
        ),
        "",
        "| Limit (risk.yaml) | Breached? | First breach | Days breached | Worst |",
        "|---|---|---|---:|---:|",
    ]
    names = {
        "daily_loss": "Max daily loss",
        "weekly_loss": "Max weekly loss",
        "max_drawdown": "Max drawdown",
    }
    for key, label in names.items():
        f = flags[key]
        lines.append(
            f"| {label} {_num(f['limit_pct'], 0, '%')} | {'**YES**' if f['breached'] else 'no'} | "
            f"{f['first_breach_day'] or '–'} | {f['breaches']} | {_num(f['worst_pct'], 2, '%')} |"
        )
    lines += [
        "",
        "## Costs (% of gross profit)",
        "",
        (
            f"Gross profit (sum of positive pre-cost trade P&L): {_usd(c['gross_profit_usd'])}. "
            f"Average cost per trade: {_num(c['avg_cost_r'], 3)} R."
        ),
        "",
        "| Cost | USD | % of gross profit |",
        "|---|---:|---:|",
    ]
    for name in ("spread", "slippage", "swap", "commission", "total"):
        lines.append(
            f"| {name} | {_usd(c[f'{name}_usd'])} | {_num(c[f'{name}_pct_of_gross_profit'], 1, '%')} |"
        )
    lines += [
        "",
        "## Minimum balance (0.01 lot, risk per trade from risk.yaml)",
        "",
        (
            f"- Largest: {_usd(mb['max'])}; 95th percentile: {_usd(mb['p95'])}; median: "
            f"{_usd(mb['median'])}"
        ),
        (
            f"- Signals skipped because the size was below the minimum lot: "
            f"{mb['signals_too_small_for_min_lot']}"
        ),
        "",
        "## Breakdowns",
        "",
        *_group_table(metrics.get("by_direction", {}), "Direction"),
        *_group_table(metrics["by_session"], "Session"),
        *_group_table(metrics["by_year"], "Year"),
        *_group_table(metrics["by_weekend_hold"], "Weekend"),
        "Per regime: not available until the regime detector (Phase 8).",
        "",
        "## Exits, ties and engine counts",
        "",
        "Exit reasons: "
        + (", ".join(f"{k} {v}" for k, v in metrics["by_exit_reason"].items()) or "none"),
        "",
        "Counts: " + ", ".join(f"{k} {v}" for k, v in metrics["counts"].items()),
        "",
        "## Conventions",
        "",
        *[f"- **{k}**: {v}" for k, v in metrics["conventions"].items()],
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    out_dir: Path,
    run: dict[str, Any],
    result: BacktestResult,
    metrics: dict[str, Any],
    stress_metrics: dict[str, Any],
    flags: dict[str, Any],
    counts: dict[str, int],
    created: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.trades.to_parquet(out_dir / "trades.parquet", index=False)
    daily = result.daily.assign(trading_day=result.daily["trading_day"].astype(str))
    daily.to_parquet(out_dir / "daily.parquet", index=False)
    (out_dir / "metrics.json").write_text(
        to_json(
            {"run": run, "metrics": metrics, "stress_metrics": stress_metrics, "risk_flags": flags}
        ),
        encoding="utf-8",
    )
    title = f"{run['strategy']} on {run['symbol']} {run['timeframe']}, {run['split']}"
    equity_chart(result, out_dir / "equity.png", title)
    (out_dir / "report.md").write_text(
        report_markdown(run, metrics, stress_metrics, flags, counts, created), encoding="utf-8"
    )
