"""Command-line entry point. Commands are added phase by phase."""

from typing import Annotated

import typer

from tradeagent.registry.cli import gate_app
from tradeagent.research.cli import research_app

app = typer.Typer(help="AI Trading Agent (paper mode by default).", no_args_is_help=True)
data_app = typer.Typer(help="Market data commands (read-only).", no_args_is_help=True)
app.add_typer(data_app, name="data")
backtest_app = typer.Typer(
    help="Backtesting (research only; never places orders).", no_args_is_help=True
)
app.add_typer(backtest_app, name="backtest")
risk_app = typer.Typer(
    help="Risk engine (read-only view of limits and state).", no_args_is_help=True
)
app.add_typer(risk_app, name="risk")
news_app = typer.Typer(help="High-impact USD news calendar (news blackout).", no_args_is_help=True)
app.add_typer(news_app, name="news")
prob_app = typer.Typer(help="Probability / EV engine (SPEC §5).", no_args_is_help=True)
app.add_typer(prob_app, name="prob")
app.add_typer(research_app, name="research")
app.add_typer(gate_app, name="gate")


@app.callback()
def main() -> None:
    """AI Trading Agent (paper mode by default). See docs/SPEC.md."""
    # A callback keeps Typer in "group" mode, so `tradeagent --help` lists all
    # commands even while there is only one.


@app.command()
def version() -> None:
    """Print the version."""
    from tradeagent import __version__

    typer.echo(__version__)


@app.command("check-config")
def check_config() -> None:
    """Load and validate config/*.yaml, then print a short summary."""
    from tradeagent.config import ConfigError, load_config

    try:
        cfg = load_config()
    except ConfigError as exc:
        typer.secho(f"Config problem: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho("Config OK", fg=typer.colors.GREEN)
    typer.echo(f"  mode:          {cfg.settings.mode}")
    typer.echo(f"  live enabled:  {cfg.live.enabled}")
    typer.echo(f"  symbols:       {', '.join(cfg.settings.symbols.values())}")
    typer.echo(f"  risk/trade:    {cfg.risk.risk_per_trade_pct}%")
    typer.echo(f"  min RR:        {cfg.risk.min_reward_risk}")
    typer.echo(f"  config_hash:   {cfg.config_hash[:12]}")


@data_app.command("ping")
def data_ping() -> None:
    """Connect to MT5 (demo only), show the account and the latest price per symbol."""
    from datetime import timedelta

    from tradeagent.config import load_config
    from tradeagent.data.mt5_client import MT5Client, MT5Credentials, MT5Error
    from tradeagent.timeutil import fmt_utc_pkt, utc_now

    cfg = load_config()
    try:
        with MT5Client(mode=cfg.settings.mode, credentials=MT5Credentials.from_env()) as client:
            account = client.account
            typer.secho(f"Account type: {account.trade_mode}", fg=typer.colors.GREEN)
            typer.echo(f"Login:        {account.login} ({account.server})")
            typer.echo(f"Balance:      {account.balance:,.2f} {account.currency}")
            typer.echo(f"Equity:       {account.equity:,.2f} {account.currency}")
            for symbol in cfg.settings.symbols.values():
                tick = client.last_tick(symbol)
                if tick is not None:
                    typer.echo(
                        f"{symbol:<9} bid {tick.bid}  ask {tick.ask}  at {fmt_utc_pkt(tick.time_utc)}"
                    )
                    continue
                now = utc_now()
                bars = client.get_bars(symbol, "M1", now - timedelta(days=7), now)
                if bars.empty:
                    typer.echo(f"{symbol:<9} no price available")
                else:
                    last = bars.iloc[-1]
                    typer.echo(
                        f"{symbol:<9} last M1 close {last['close']}  "
                        f"at {fmt_utc_pkt(last['time_utc'].to_pydatetime())} (no live tick)"
                    )
    except MT5Error as exc:
        typer.secho(f"MT5 problem: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@data_app.command("fetch")
def data_fetch(
    symbol: Annotated[
        list[str] | None,
        typer.Option("--symbol", "-s", help="Internal symbol, e.g. XAUUSD (default: all)."),
    ] = None,
    timeframe: Annotated[
        list[str] | None,
        typer.Option("--timeframe", "-t", help="e.g. M5 (default: all in settings)."),
    ] = None,
) -> None:
    """Download missing history from MT5 into data/bars (incremental)."""
    from tradeagent.config import load_config, project_path
    from tradeagent.data.historical import fetch_history, history_start
    from tradeagent.data.mt5_client import MT5Client, MT5Credentials, MT5Error
    from tradeagent.data.store import BarStore, connect_db
    from tradeagent.timeutil import utc_now

    cfg = load_config()
    symbols = cfg.settings.symbols
    wanted_symbols = symbol or list(symbols)
    wanted_tfs = timeframe or list(cfg.settings.timeframes)
    unknown = [s for s in wanted_symbols if s not in symbols] + [
        t for t in wanted_tfs if t not in cfg.settings.timeframes
    ]
    if unknown:
        typer.secho(f"Unknown symbol/timeframe: {unknown}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    connect_db(project_path(cfg.settings.storage.sqlite_path)).close()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    try:
        with MT5Client(mode=cfg.settings.mode, credentials=MT5Credentials.from_env()) as client:
            now = utc_now()
            for name in wanted_symbols:
                for tf in wanted_tfs:
                    start = history_start(tf, cfg.settings, now)
                    r = fetch_history(client, store, name, symbols[name], tf, start, now)
                    note = ""
                    if r.first is not None and r.first > start + (now - start) * 0.02:
                        note = "  (broker history starts later than wanted)"
                    typer.echo(
                        f"{name:<7} {tf:<4} +{r.new_bars:>7} new  "
                        f"{_fmt_day(r.first)} -> {_fmt_day(r.last)}{note}"
                    )
    except MT5Error as exc:
        typer.secho(f"MT5 problem: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@data_app.command("summary")
def data_summary() -> None:
    """Show stored bar counts and date ranges (UTC)."""
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import BarStore

    cfg = load_config()
    table = BarStore(project_path(cfg.settings.storage.bars_dir)).summary()
    if table.empty:
        typer.echo("No bars stored yet. Run: uv run tradeagent data fetch")
        return
    typer.echo(f"{'symbol':<7} {'tf':<4} {'bars':>9}  first (UTC)        last (UTC)")
    for row in table.itertuples():
        typer.echo(
            f"{row.symbol:<7} {row.timeframe:<4} {row.bars:>9,}  "
            f"{row.first_utc:%Y-%m-%d %H:%M}   {row.last_utc:%Y-%m-%d %H:%M}"
        )


@data_app.command("validate")
def data_validate(
    symbol: Annotated[
        list[str] | None,
        typer.Option("--symbol", "-s", help="Internal symbol, e.g. XAUUSD (default: all)."),
    ] = None,
    timeframe: Annotated[
        list[str] | None,
        typer.Option("--timeframe", "-t", help="e.g. M5 (default: all in settings)."),
    ] = None,
    examples: Annotated[
        int, typer.Option("--examples", "-n", help="Worst examples to show per issue type.")
    ] = 3,
    save: Annotated[
        bool, typer.Option("--save/--no-save", help="Write issues to data_quality_log.")
    ] = True,
) -> None:
    """Check stored bars for gaps, spikes, duplicates, bad prices and spread outliers."""
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import BarStore, connect_db
    from tradeagent.data.validation import ISSUE_TYPES, log_issues, validate_bars
    from tradeagent.provenance import git_commit, new_run_id
    from tradeagent.timeutil import fmt_utc_pkt

    cfg = load_config()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    run_id, commit = new_run_id(), git_commit()
    short = {
        "duplicate": "dup",
        "not_monotonic": "order",
        "ohlc_invalid": "ohlc",
        "misaligned": "align",
        "weekend_bar": "wkndbar",
        "gap_intraday": "gap!",
        "gap_holiday": "holiday",
        "spike": "spike",
        "spread_zero": "spr=0",
        "spread_outlier": "spr>10x",
    }

    results = []
    conn = connect_db(project_path(cfg.settings.storage.sqlite_path)) if save else None
    for name in symbol or list(cfg.settings.symbols):
        for tf in timeframe or list(cfg.settings.timeframes):
            bars = store.read(name, tf)
            if bars.empty:
                continue
            result = validate_bars(bars, name, tf)
            results.append(result)
            if conn is not None:
                log_issues(conn, result, run_id, commit, cfg.config_hash)
    if conn is not None:
        conn.close()

    typer.echo(f"Validation run {run_id}  (commit {commit})")
    typer.echo("\nISSUES (need review; '.' = none)")
    typer.echo(
        f"{'symbol':<7}{'tf':<5}{'bars':>9} " + "".join(f"{short[t]:>8}" for t in ISSUE_TYPES)
    )
    for r in results:
        counts = r.counts()
        typer.echo(
            f"{r.symbol:<7}{r.timeframe:<5}{r.bars:>9,} "
            + "".join(f"{counts[t] or '.':>8}" for t in ISSUE_TYPES)
        )

    typer.echo("\nEXPECTED market behaviour (counted, not logged)")
    typer.echo(
        f"{'symbol':<7}{'tf':<5}{'weekend gaps':>14}{'daily breaks':>14}{'Sunday stubs':>14}"
    )
    for r in results:
        e = r.expected
        typer.echo(
            f"{r.symbol:<7}{r.timeframe:<5}{e['gap_weekend']:>14}"
            f"{e['gap_daily_break']:>14}{e['sunday_stub']:>14}"
        )

    if examples > 0:
        typer.echo(f"\nWORST EXAMPLES (up to {examples} per type)")
        for r in results:
            for issue_type, group in r.issues.groupby("issue_type"):
                typer.echo(f"  {r.symbol} {r.timeframe} {issue_type} ({len(group)}):")
                worst = group.nlargest(examples, "severity")
                for time_utc, details in zip(worst["time_utc"], worst["details"], strict=True):
                    typer.echo(f"      {fmt_utc_pkt(time_utc.to_pydatetime())}  {details}")
    if save:
        typer.echo(f"\nSaved to data_quality_log with run_id {run_id}")


@data_app.command("watch")
def data_watch(
    timeframe: Annotated[
        list[str] | None,
        typer.Option("--timeframe", "-t", help="Timeframes to follow (default: M1 and M5)."),
    ] = None,
    interval: Annotated[
        float, typer.Option("--interval", help="Seconds between checks.", min=1)
    ] = 10.0,
    minutes: Annotated[
        float | None, typer.Option("--minutes", help="Stop after this many minutes.")
    ] = None,
) -> None:
    """Append each newly closed bar (polling), validate and log it. Stop with Ctrl+C."""
    from datetime import timedelta

    from tradeagent.config import load_config, project_path
    from tradeagent.data.live import LiveUpdater, run_watch
    from tradeagent.data.mt5_client import MT5Client, MT5Credentials, MT5Error
    from tradeagent.data.store import BarStore, connect_db
    from tradeagent.logs import add_daily_log
    from tradeagent.provenance import git_commit, new_run_id

    cfg = load_config()
    tfs = timeframe or ["M1", "M5"]
    unknown = [t for t in tfs if t not in cfg.settings.timeframes]
    if unknown:
        typer.secho(f"Unknown timeframe: {unknown}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    log_dir = project_path(cfg.settings.storage.bars_dir).parent / "logs"
    add_daily_log(log_dir, "watch")
    run_id = new_run_id()
    conn = connect_db(project_path(cfg.settings.storage.sqlite_path))
    updater = LiveUpdater(
        store=BarStore(project_path(cfg.settings.storage.bars_dir)),
        conn=conn,
        symbols=dict(cfg.settings.symbols),
        timeframes=tfs,
        run_id=run_id,
        git_commit=git_commit(),
        config_hash=cfg.config_hash,
    )
    credentials = MT5Credentials.from_env()
    typer.echo(
        f"Watching {', '.join(cfg.settings.symbols)} {', '.join(tfs)} every {interval:g}s "
        f"(run {run_id}). Press Ctrl+C to stop."
    )
    try:
        stats = run_watch(
            lambda: MT5Client(mode=cfg.settings.mode, credentials=credentials),
            updater,
            interval_seconds=interval,
            stop_after=timedelta(minutes=minutes) if minutes else None,
        )
    except MT5Error as exc:  # refused account: never retried
        typer.secho(f"MT5 problem: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()
    typer.echo(
        f"Stopped. polls {stats.polls}, new bars {stats.new_bars}, data issues {stats.issues}, "
        f"MT5 errors {len(stats.errors)}, reconnects {stats.reconnects}. Log: {log_dir}"
    )


@data_app.command("resample-check")
def data_resample_check(
    symbol: Annotated[
        list[str] | None,
        typer.Option("--symbol", "-s", help="Internal symbol, e.g. XAUUSD (default: all)."),
    ] = None,
    examples: Annotated[
        int, typer.Option("--examples", "-n", help="Partial days to list per symbol.")
    ] = 10,
) -> None:
    """Build New York-close D1/H4 bars from H1 and compare them with the broker's bars."""
    import pandas as pd

    from tradeagent.config import load_config, project_path
    from tradeagent.data.resample import resample_ny_close
    from tradeagent.data.store import BarStore
    from tradeagent.features.indicators import atr

    cfg = load_config()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for sym in symbol or list(cfg.settings.symbols):
        h1 = store.read(sym, "H1")
        if h1.empty:
            typer.secho(f"{sym}: no H1 bars stored", fg=typer.colors.YELLOW)
            continue
        d1 = resample_ny_close(h1, "D1")
        h4 = resample_ny_close(h1, "H4")
        typer.secho(
            f"\n{sym}: {len(h1):,} H1 bars, {h1['time_utc'].min():%Y-%m-%d} to "
            f"{h1['time_utc'].max():%Y-%m-%d} (UTC)",
            bold=True,
        )

        per_day = pd.to_datetime(d1["trading_day"]).dt.dayofweek.value_counts()
        counts = "  ".join(f"{days[d]} {int(per_day.get(d, 0))}" for d in range(7))
        typer.echo(f"  NY-close D1 bars per weekday: {counts}")
        typer.echo(
            f"  D1: {len(d1):,} bars, {int(d1['partial'].sum())} partial | "
            f"H4: {len(h4):,} bars, {int(h4['partial'].sum())} partial, "
            f"{int(h4.groupby('trading_day').size().eq(6).sum())} days with all 6 blocks"
        )

        for tf, ours in (("D1", d1), ("H4", h4)):
            broker = store.read(sym, tf)
            if broker.empty:
                continue
            stub = broker["time_utc"].dt.dayofweek == 6  # Sunday candles (00:00 UTC cut)
            if tf == "H4":
                # The Friday 20:00 UTC H4 bar is also partial (about 1 hour of trading).
                stub |= (broker["time_utc"].dt.dayofweek == 4) & (broker["time_utc"].dt.hour == 20)
            full = ours[~ours["partial"]].reset_index(drop=True)
            typer.echo(
                f"  {tf} ATR(14), median: NY-close {atr(full).median():.3f} | "
                f"broker {atr(broker).median():.3f} (with {int(stub.sum())} stub bars) | "
                f"broker without stubs {atr(broker[~stub].reset_index(drop=True)).median():.3f}"
            )

        partial = d1[d1["partial"]]
        if len(partial) and examples > 0:
            typer.echo("  Partial D1 days (holiday, early close, data hole or data edge):")
            for row in partial.head(examples).itertuples():
                typer.echo(f"    {row.trading_day}  {row.h1_bars:>2}/23 H1 bars")
            if len(partial) > examples:
                typer.echo(f"    ... and {len(partial) - examples} more")


@backtest_app.command("splits")
def backtest_splits(
    freeze: Annotated[
        bool, typer.Option("--freeze", help="Save the proposed dates to config/splits.yaml.")
    ] = False,
    approved_by: Annotated[
        str | None, typer.Option("--approved-by", help="Who approved the dates (with --freeze).")
    ] = None,
) -> None:
    """Show the train / validation / out-of-sample dates and what backtests exclude."""
    from tradeagent.backtest.dataset import exclusion_windows, flag_bars
    from tradeagent.backtest.splits import (
        SPLIT_NAMES,
        SplitError,
        freeze_splits,
        label_times,
        propose_splits,
    )
    from tradeagent.config import DEFAULT_CONFIG_DIR, SPLITS_FILE, load_config, project_path
    from tradeagent.data.store import BarStore
    from tradeagent.data.validation import find_gaps
    from tradeagent.timeutil import utc_now

    cfg = load_config()
    bt = cfg.settings.backtest
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    source_tf = bt.exclusions_source_timeframe
    ranges = [store.time_range(sym, source_tf) for sym in cfg.settings.symbols]
    if any(r is None for r in ranges):
        typer.secho(f"Missing {source_tf} bars; run `tradeagent data fetch` first.", fg="red")
        raise typer.Exit(code=1)
    first = max(r[0] for r in ranges if r is not None)
    last = min(r[1] for r in ranges if r is not None)

    if cfg.splits is not None:
        periods = {name: cfg.splits.period(name) for name in SPLIT_NAMES}
        typer.secho(
            f"Split dates FROZEN (approved by {cfg.splits.approved_by} on "
            f"{cfg.splits.approved_on}), config/{SPLITS_FILE}",
            fg=typer.colors.GREEN,
            bold=True,
        )
    else:
        try:
            periods = propose_splits(first, last, cfg.settings.data_splits, bt.embargo_weeks)
        except SplitError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        typer.secho(
            "Split dates PROPOSED (not frozen; nothing can run on them yet)",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    typer.echo(
        f"  {source_tf} data on disk (both symbols): {first:%Y-%m-%d %H:%M} to "
        f"{last:%Y-%m-%d %H:%M} UTC. Periods are [start, end), Sunday 00:00 UTC."
    )
    typer.echo("")

    total_days = sum((p.end - p.start).days for p in periods.values())
    for i, name in enumerate(SPLIT_NAMES):
        p = periods[name]
        days = (p.end - p.start).days
        typer.echo(
            f"  {name:<14} {p.start:%a %Y-%m-%d} -> {p.end:%a %Y-%m-%d}   "
            f"{days // 7} weeks ({days / total_days:.0%})"
        )
        if i < len(SPLIT_NAMES) - 1:
            nxt = periods[SPLIT_NAMES[i + 1]]
            typer.echo(
                f"  {'embargo':<14} {p.end:%a %Y-%m-%d} -> {nxt.start:%a %Y-%m-%d}   "
                f"{(nxt.start - p.end).days // 7} weeks, never used"
            )
    oos_end = periods["out_of_sample"].end
    typer.echo(f"  {'forward':<14} {oos_end:%a %Y-%m-%d} -> ...              for paper trading")
    typer.echo("  out_of_sample is LOCKED in Phase 2 (opened once per candidate in Phase 7).")

    typer.echo("")
    typer.secho("Bars per split (XAUUSD):", bold=True)
    for tf in cfg.settings.timeframes:
        bars = store.read("XAUUSD", tf)
        if bars.empty:
            continue
        counts = label_times(bars["time_utc"], periods).value_counts()
        shares = "  ".join(f"{n} {int(counts.get(n, 0)):>7,}" for n in SPLIT_NAMES)
        typer.echo(f"  {tf:<4} from {bars['time_utc'].min():%Y-%m-%d}   {shares}")

    for sym in cfg.settings.symbols:
        source = store.read(sym, source_tf)
        windows = exclusion_windows(
            find_gaps(source, source_tf), cfg.exclusions, sym, bt.max_hole_minutes
        )
        typer.echo("")
        typer.secho(f"{sym}: excluded windows ({len(windows)})", bold=True)
        labels = label_times(windows["start_utc"], periods)
        for start, end, src, reason, label in zip(
            windows["start_utc"],
            windows["end_utc"],
            windows["source"],
            windows["reason"],
            labels,
            strict=True,
        ):
            typer.echo(
                f"  [{label:<13}] {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC  "
                f"{src}: {reason}"
            )
        flagged = flag_bars(
            source, source_tf, sym, windows, cfg.exclusions, bt, flat_before_weekend=True
        )
        split_of = label_times(flagged["time_utc"], periods)
        for name in SPLIT_NAMES:
            part = flagged[split_of == name]
            typer.echo(
                f"  {name:<14} {len(part):>7,} {source_tf} bars: "
                f"{int(part['excluded'].sum()):>3} excluded, "
                f"{int(part['no_new_entries'].sum()):>6,} no-new-entry (scalp/intraday rules), "
                f"{int(part['gap_before'].sum()):>4} with a gap before"
            )

    typer.echo("")
    if freeze:
        if cfg.splits is not None:
            typer.secho("Already frozen; nothing changed.", fg=typer.colors.YELLOW)
            return
        if not approved_by:
            typer.secho("--freeze needs --approved-by NAME", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        path = DEFAULT_CONFIG_DIR / SPLITS_FILE
        freeze_splits(periods, approved_by, path, today=utc_now().date())
        typer.secho(f"Saved {path}. Record the approval in docs/DECISIONS.md.", fg="green")
    elif cfg.splits is None:
        typer.echo("After approval: uv run tradeagent backtest splits --freeze --approved-by Usama")


@backtest_app.command("costs")
def backtest_costs(
    snapshot: Annotated[
        bool,
        typer.Option("--snapshot", help="Read costs from MT5 (demo) and save config/costs.yaml."),
    ] = False,
) -> None:
    """Show the broker cost snapshot backtests use (contract size, $ per point, swaps)."""
    from tradeagent.backtest.costs import CostModel, symbol_costs, write_snapshot
    from tradeagent.config import (
        COSTS_FILE,
        DEFAULT_CONFIG_DIR,
        ConfigError,
        CostSnapshot,
        load_config,
    )
    from tradeagent.data.mt5_client import MT5Client, MT5Credentials, MT5Error
    from tradeagent.timeutil import utc_now

    cfg = load_config()
    if snapshot:
        try:
            with MT5Client(mode=cfg.settings.mode, credentials=MT5Credentials.from_env()) as c:
                symbols = {
                    internal: symbol_costs(c.symbol_info(broker))
                    for internal, broker in cfg.settings.symbols.items()
                }
        except (MT5Error, ValueError) as exc:
            typer.secho(f"Could not take a snapshot: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        path = DEFAULT_CONFIG_DIR / COSTS_FILE
        write_snapshot(
            CostSnapshot(
                taken_utc=utc_now().replace(microsecond=0),
                source="MT5 symbol_info, Exness demo account",
                symbols=symbols,
            ),
            path,
        )
        typer.secho(f"Saved {path}", fg=typer.colors.GREEN)
        try:
            cfg = load_config()
        except ConfigError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    if cfg.costs is None:
        typer.echo("No cost snapshot yet. Run: uv run tradeagent backtest costs --snapshot")
        return
    bt = cfg.settings.backtest
    typer.secho(f"Cost snapshot taken {cfg.costs.taken_utc:%Y-%m-%d %H:%M} UTC", bold=True)
    typer.echo(
        f"  spread used = stored bar spread x {bt.spread_margin_multiple} + "
        f"{bt.spread_margin_points} pts; slippage {bt.slippage_spread_multiple} x spread; "
        f"commission ${bt.commission_per_lot_usd}/lot round turn"
    )
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for internal, spec in cfg.costs.symbols.items():
        CostModel.from_config(cfg, internal)  # proves the snapshot is usable
        vpp = spec.value_per_point_per_lot
        triple = spec.triple_swap_weekday
        typer.echo("")
        typer.secho(f"{internal} ({spec.broker_symbol})", bold=True)
        typer.echo(
            f"  contract {spec.contract_size:g}, point {spec.point}, ${vpp:g} per point per lot, "
            f"lots {spec.volume_min}-{spec.volume_max} step {spec.volume_step}"
        )
        for side, points in (("long", spec.swap_long), ("short", spec.swap_short)):
            typer.echo(
                f"  swap {side:<5} {points:>8g} pts = ${points * vpp:>9.2f} per lot per night "
                f"(${points * vpp * 0.01:.2f} at 0.01 lot)"
            )
        typer.echo(
            "  triple swap: "
            + (f"{days[triple]} (x3)" if triple is not None else "none (every weekday x1)")
            + f" [MT5 swap_rollover3days = {spec.swap_rollover3days}]"
        )


@backtest_app.command("spread-check")
def backtest_spread_check(
    weeks: Annotated[
        int, typer.Option("--weeks", "-w", help="Complete weeks of ticks to measure.")
    ] = 4,
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="M1 or M5.")] = "M5",
) -> None:
    """Measure real tick spreads vs the (minimum) spread MT5 stores per bar."""
    from datetime import UTC, datetime, time, timedelta

    from tradeagent.backtest.spread_check import bar_tick_spreads, propose_multiple, summarize
    from tradeagent.config import load_config, project_path
    from tradeagent.data.mt5_client import TIMEFRAMES, MT5Client, MT5Credentials, MT5Error
    from tradeagent.data.store import BarStore
    from tradeagent.timeutil import utc_now

    if timeframe not in ("M1", "M5"):
        typer.secho("timeframe must be M1 or M5", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    cfg = load_config()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    today = utc_now().date()
    end_day = today - timedelta(days=(today.weekday() + 1) % 7)  # last Sunday
    start_day = end_day - timedelta(weeks=weeks)
    start = datetime.combine(start_day, time(), UTC)
    end = datetime.combine(end_day, time(), UTC)
    minutes = int(TIMEFRAMES[timeframe][1].total_seconds() // 60)
    typer.echo(f"Ticks {start:%Y-%m-%d} -> {end:%Y-%m-%d} UTC ({weeks} weeks), {timeframe} bars")

    summaries = []
    try:
        with MT5Client(mode=cfg.settings.mode, credentials=MT5Credentials.from_env()) as c:
            for internal, broker in cfg.settings.symbols.items():
                point = c.symbol_info(broker).point
                ticks = c.get_ticks(broker, start, end)
                bars = store.read(internal, timeframe, start, end)
                s = summarize(bar_tick_spreads(ticks, bars, point, minutes))
                summaries.append(s)
                typer.echo("")
                typer.secho(f"{internal}: {len(ticks):,} ticks, {int(s['bars']):,} bars", bold=True)
                if not s["bars"]:
                    typer.echo("  no bars with ticks in this range")
                    continue
                typer.echo(
                    f"  stored spread = the minimum tick spread in {s['stored_is_min']:.1%} of bars"
                )
                typer.echo(
                    "  time-weighted average / stored: "
                    f"median {s['twavg_median']:.3f}, mean {s['twavg_mean']:.3f}, "
                    f"p90 {s['twavg_p90']:.3f}, p99 {s['twavg_p99']:.3f}, max {s['twavg_max']:.2f}"
                )
                typer.echo(
                    f"  spread at the bar's open / stored: p99 {s['open_p99']:.3f}, "
                    f"max {s['open_max']:.2f}"
                )
    except MT5Error as exc:
        typer.secho(f"MT5 problem: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("")
    typer.secho(
        f"Proposed spread_margin_multiple: {propose_multiple(summaries)} "
        f"(covers the 99th percentile; current setting "
        f"{cfg.settings.backtest.spread_margin_multiple})",
        bold=True,
    )


def _money(value: float) -> str:
    return f"-${-value:,.2f}" if value < 0 else f"${value:,.2f}"


def _parse_params(items: list[str] | None) -> dict[str, float]:
    params: dict[str, float] = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--param must look like name=value, got {item!r}")
        params[name.strip()] = float(value)
    return params


@backtest_app.command("run")
def backtest_run(
    strategy: Annotated[str, typer.Option("--strategy", help="Registered strategy name.")],
    symbol: Annotated[str, typer.Option("--symbol", "-s")] = "XAUUSD",
    split: Annotated[str, typer.Option("--split", help="train or validation.")] = "train",
    seed: Annotated[int, typer.Option("--seed")] = 1,
    param: Annotated[
        list[str] | None, typer.Option("--param", help="Parameter override, name=value.")
    ] = None,
    enforce_account_limits: Annotated[
        bool,
        typer.Option(
            "--enforce-account-limits",
            help="Stop trading when daily/weekly/drawdown/loss-streak limits hit.",
        ),
    ] = False,
) -> None:
    """Backtest a strategy on one split, save it to backtest_runs and write a report."""
    from pathlib import Path

    from tradeagent.backtest.costs import CostError
    from tradeagent.backtest.runs import execute_run
    from tradeagent.backtest.splits import SplitError
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import BarStore, connect_db
    from tradeagent.strategies import registry
    from tradeagent.strategies.base import InvalidStrategy

    if split not in ("train", "validation", "out_of_sample"):
        typer.secho("split must be train or validation", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    registry.load_builtins()
    cfg = load_config()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    conn = connect_db(project_path(cfg.settings.storage.sqlite_path))
    try:
        rec = execute_run(
            cfg,
            store,
            conn,
            strategy,
            symbol,
            split,  # type: ignore[arg-type]
            seed,
            _parse_params(param),
            project_path(Path("data/backtests")),
            enforce_account_limits=enforce_account_limits,
        )
    except (SplitError, CostError, InvalidStrategy, KeyError, ValueError) as exc:
        typer.secho(f"Cannot run: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    t, e, st = rec.metrics["trades"], rec.metrics["equity"], rec.stress_metrics["trades"]
    ci = rec.metrics["confidence_95"]["expectancy_r"]

    def fmt(v: object, d: int = 3) -> str:
        return "-" if v is None else f"{v:.{d}f}" if isinstance(v, float) else str(v)

    typer.secho(
        f"{strategy} on {symbol} {rec.run['timeframe']} ({split}), run #{rec.run['run_number']}",
        bold=True,
    )
    if t["insufficient_sample"]:
        typer.secho(f"  INSUFFICIENT SAMPLE: {t['trades']} trades (< 30)", fg=typer.colors.YELLOW)
    ci_text = f"{fmt(ci[0])} to {fmt(ci[1])}" if ci else "-"
    typer.echo(
        f"  trades {t['trades']}, win rate {fmt(t['win_rate_pct'], 1)}%, "
        f"profit factor {fmt(t['profit_factor'], 2)}"
    )
    typer.echo(
        f"  expectancy {fmt(t['expectancy_r'])} R (95% CI {ci_text}), "
        f"net {_money(e['net_profit_usd'])}"
    )
    typer.echo(
        f"  max drawdown {e['max_dd_pct']:.2f}% (mark-to-market), "
        f"{e['closed_trade_max_dd_pct']:.2f}% (closed trades); Sharpe {fmt(e['sharpe'], 2)}"
    )
    verdict = "PASS" if rec.run["stress_pass"] else "FAIL"
    typer.echo(
        f"  cost stress x{rec.run['cost_stress_multiple']}: expectancy "
        f"{fmt(st['expectancy_r'])} R -> {verdict}"
    )
    breached = [k for k, v in rec.risk_flags.items() if v["breached"]]
    typer.echo(
        "  risk-limit flags: "
        + (
            ", ".join(f"{k} (first {rec.risk_flags[k]['first_breach_day']})" for k in breached)
            or "none"
        )
    )
    typer.echo(
        f"  runs of {strategy}: train {rec.counts['train']}, validation {rec.counts['validation']}"
    )
    typer.echo(
        f"  data fingerprint {rec.run['data_hash'][:12]}, config {cfg.config_hash[:12]}, "
        f"git {rec.run['git_commit']}"
    )
    typer.echo(f"  report: {rec.output_dir / 'report.md'}")


@backtest_app.command("list")
def backtest_list(
    strategy: Annotated[str | None, typer.Option("--strategy")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Show recent backtest runs (newest first)."""
    import pandas as pd

    from tradeagent.backtest.records import list_runs
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import connect_db

    cfg = load_config()
    conn = connect_db(project_path(cfg.settings.storage.sqlite_path))
    try:
        runs = list_runs(conn, strategy, limit)
    finally:
        conn.close()
    if runs.empty:
        typer.echo("No backtest runs yet. Run: uv run tradeagent backtest run --strategy NAME")
        return
    typer.echo(
        f"{'run_id':<24} {'strategy':<18} {'sym':<7} {'tf':<4} {'split':<10} "
        f"{'#':>3} {'trades':>6} {'exp R':>7} {'stress':>6}"
    )
    for r in runs.to_dict("records"):
        # A missing expectancy comes back from SQLite as NaN, not None.
        value = r["expectancy_r"]
        exp = "-" if pd.isna(value) else f"{float(value):.3f}"
        flag = "*" if r["insufficient_sample"] else " "
        typer.echo(
            f"{r['run_id']!s:<24} {r['strategy']!s:<18} {r['symbol']!s:<7} "
            f"{r['timeframe']!s:<4} {r['split']!s:<10} {int(r['run_number']):>3} "
            f"{int(r['trades']):>5}{flag} {exp:>7} {'PASS' if r['stress_pass'] else 'FAIL':>6}"
        )
    typer.echo("* = insufficient sample (< 30 trades)")


@backtest_app.command("lookahead-check")
def backtest_lookahead_check(
    strategy: Annotated[str, typer.Option("--strategy")],
    symbol: Annotated[str, typer.Option("--symbol", "-s")] = "XAUUSD",
    split: Annotated[str, typer.Option("--split")] = "train",
    seed: Annotated[int, typer.Option("--seed")] = 1,
    bars: Annotated[
        int, typer.Option("--bars", help="Last N decision bars of the split to check.")
    ] = 3000,
    cuts: Annotated[int, typer.Option("--cuts", help="Truncation points.")] = 20,
    param: Annotated[list[str] | None, typer.Option("--param")] = None,
) -> None:
    """Truncation test on real data: does the strategy use future bars? (Task 2.6)"""
    import pandas as pd

    from tradeagent.backtest.dataset import load_bars, load_dataset
    from tradeagent.backtest.lookahead import check_lookahead
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import BarStore
    from tradeagent.strategies import registry

    registry.load_builtins()
    params = _parse_params(param)
    cfg = load_config()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    first = registry.create(strategy, seed, params)
    decision_tf = first.timeframes[0]
    dataset = load_dataset(cfg, store, symbol, decision_tf, split, first.style, warmup_bars=0)  # type: ignore[arg-type]
    decision = dataset.bars.tail(bars).reset_index(drop=True)
    start, end = decision["time_utc"].iloc[0], dataset.end_utc
    frames = {decision_tf: decision}
    for tf in first.timeframes[1:]:
        other = load_bars(store, symbol, tf)
        history = start - pd.Timedelta(days=60)  # context history before the first bar
        frames[tf] = other[(other["time_utc"] >= history) & (other["time_utc"] < end)]
    typer.echo(
        f"Checking {strategy} on {symbol} {decision_tf}: last {len(decision):,} bars of "
        f"{split}, {cuts} cuts ..."
    )
    problems = check_lookahead(
        lambda: registry.create(strategy, seed, params), symbol, frames, cuts
    )
    if not problems:
        typer.secho(
            "PASS: no look-ahead found (signals and indicators unchanged by future data)",
            fg=typer.colors.GREEN,
        )
        return
    typer.secho(
        f"FAIL: {len(problems)} problem(s) - the strategy uses future data", fg=typer.colors.RED
    )
    for p in problems[:10]:
        typer.echo(f"  cut {p.cut_utc}: {p.what} at {p.where}: {p.detail}")
    raise typer.Exit(code=1)


@backtest_app.command("baseline")
def backtest_baseline(
    symbol: Annotated[str, typer.Option("--symbol", "-s")] = "XAUUSD",
    seeds: Annotated[int, typer.Option("--seeds", help="Seeds 1..N.")] = 100,
    timeframe: Annotated[str, typer.Option("--timeframe", "-t")] = "M15",
    split: Annotated[str, typer.Option("--split")] = "train",
    style: Annotated[str, typer.Option("--style", help="intraday or swing")] = "intraday",
    direction: Annotated[
        str, typer.Option("--direction", help="both, long or short (one-direction baseline)")
    ] = "both",
) -> None:
    """Random baseline over many seeds; saves the distribution to data/baselines/."""
    from pathlib import Path

    from tradeagent.backtest.baseline import baseline_distribution, save_distribution
    from tradeagent.config import SplitName, load_config, project_path
    from tradeagent.data.store import BarStore
    from tradeagent.strategies.base import Style

    if (
        split not in ("train", "validation")
        or style not in ("intraday", "swing")
        or direction not in ("both", "long", "short")
    ):
        typer.secho(
            "split: train/validation; style: intraday/swing; direction: both/long/short",
            fg="red",
            err=True,
        )
        raise typer.Exit(code=1)
    split_name: SplitName = "train" if split == "train" else "validation"
    style_name: Style = "intraday" if style == "intraday" else "swing"
    cfg = load_config()
    store = BarStore(project_path(cfg.settings.storage.bars_dir))
    df, data_hash = baseline_distribution(
        cfg,
        store,
        symbol,
        range(1, seeds + 1),
        split_name,
        timeframe,
        style=style_name,
        direction=direction,
    )
    path = save_distribution(
        df, project_path(Path("data/baselines")), symbol, timeframe, split, style, direction
    )
    e = df["expectancy_r"]
    typer.echo(
        f"{symbol} {timeframe} {style} {split} {direction}, {seeds} seeds: trades/run median "
        f"{df['trades'].median():.0f}; expectancy mean {e.mean():.3f} R "
        f"(sd {e.std():.3f}, 5-95% {e.quantile(0.05):.3f} to {e.quantile(0.95):.3f}); "
        f"cost/trade {df['avg_cost_r'].mean():.3f} R; data {data_hash[:12]}"
    )
    typer.echo(f"saved {path}")


@backtest_app.command("compare")
def backtest_compare(
    split: Annotated[str, typer.Option("--split")] = "train",
) -> None:
    """Latest run per strategy x symbol vs the random baseline (same timeframe and style)."""
    from pathlib import Path

    from tradeagent.backtest.compare import compare_rows, verdict
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import connect_db
    from tradeagent.strategies import registry

    registry.load_builtins()
    styles = {name: registry.create(name).style for name in registry.names()}
    cfg = load_config()
    conn = connect_db(project_path(cfg.settings.storage.sqlite_path))
    try:
        rows = compare_rows(conn, project_path(Path("data/baselines")), split, styles)
    finally:
        conn.close()
    if not rows:
        typer.echo(f"No strategy runs on {split} yet.")
        return
    typer.echo(
        f"{'strategy':<22} {'sym':<7} {'tf':<4} {'trades':>6} {'exp R':>7} "
        f"{'stress':>7} {'vs random':>9}  verdict"
    )
    for r in rows:
        pct = r["baseline_percentile"]
        exp = "-" if r["expectancy_r"] is None else f"{r['expectancy_r']:.3f}"
        stress = "-" if r["stress_expectancy_r"] is None else f"{r['stress_expectancy_r']:.3f}"
        typer.echo(
            f"{r['strategy']:<22} {r['symbol']:<7} {r['timeframe']:<4} {r['trades']:>6} "
            f"{exp:>7} {stress:>7} {'-' if pct is None else f'{pct:.0f}%':>9}  "
            f"{verdict(r)}"
        )
    typer.echo("vs random = % of 100 random-baseline runs with a lower expectancy")


@app.command("kill")
def kill(
    reason: Annotated[
        str, typer.Option("--reason", help="Why (required to activate or clear).")
    ] = "",
    status: Annotated[
        bool, typer.Option("--status", help="Only show whether it is active.")
    ] = False,
    clear: Annotated[
        bool, typer.Option("--clear", help="Remove the kill switch (human decision).")
    ] = False,
) -> None:
    """Kill switch: stop all trading now (creates the KILL file). SPEC §6."""
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import connect_db
    from tradeagent.provenance import git_commit
    from tradeagent.risk.killswitch import KillSwitch
    from tradeagent.risk.state import log_risk_event
    from tradeagent.timeutil import utc_now

    ks = KillSwitch()
    if status:
        info = ks.info()
        if info is None and not ks.active():
            typer.echo("Kill switch: off")
        else:
            typer.secho(f"Kill switch: ON ({info})", fg=typer.colors.RED, bold=True)
        return
    if not reason.strip():
        typer.secho("--reason is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    cfg = load_config()
    now = utc_now()
    conn = connect_db(project_path(cfg.settings.storage.sqlite_path))
    try:
        if clear:
            ks.clear(reason)
            log_risk_event(
                conn,
                now,
                f"kill_switch_cleared: {reason}",
                None,
                None,
                "kill",
                git_commit(),
                cfg.config_hash,
            )
            typer.secho("Kill switch cleared (logged).", fg=typer.colors.YELLOW)
        else:
            ks.activate(reason, "cli", now)
            log_risk_event(
                conn,
                now,
                f"kill_switch: {reason}",
                None,
                None,
                "kill",
                git_commit(),
                cfg.config_hash,
            )
            typer.secho(
                "KILL SWITCH ON: all trading stopped (logged).", fg=typer.colors.RED, bold=True
            )
    finally:
        conn.close()


@risk_app.command("status")
def risk_status() -> None:
    """Limits in force (config/risk.yaml), kill switch, and the saved paper state."""
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import connect_db
    from tradeagent.risk.killswitch import KillSwitch
    from tradeagent.risk.state import load_state
    from tradeagent.timeutil import utc_now

    cfg = load_config()
    r = cfg.risk
    typer.secho("Limits (config/risk.yaml, read only)", bold=True)
    typer.echo(
        f"  risk per trade {r.risk_per_trade_pct}% | daily loss {r.max_daily_loss_pct}% | "
        f"weekly loss {r.max_weekly_loss_pct}% | max drawdown {r.max_drawdown_pct}%"
    )
    typer.echo(
        f"  positions {r.max_open_positions} total, {r.max_open_per_symbol} per symbol | "
        f"loss streak {r.max_consecutive_losses} -> pause {r.consecutive_loss_pause_hours} h"
    )
    typer.echo(
        f"  min RR {r.min_reward_risk} | stop {r.sl_atr_min}-{r.sl_atr_max} x ATR | "
        f"spread <= {r.max_spread_multiple} x median | news blackout "
        f"{r.news_blackout_minutes} min | correlation limit {r.correlation_limit}"
    )
    ks = KillSwitch()
    typer.echo("Kill switch: " + ("ON " + str(ks.info()) if ks.active() else "off"))
    conn = connect_db(project_path(cfg.settings.storage.sqlite_path))
    try:
        state = load_state(conn)
    finally:
        conn.close()
    if state is None:
        typer.echo("Account state: none saved yet (paper trading starts in Phase 8)")
        return
    now = utc_now()
    typer.echo(
        f"Account state: equity {state.equity:,.2f}, peak {state.peak_equity:,.2f} "
        f"(drawdown {state.drawdown_pct():.2f}%), open positions {len(state.positions)}"
    )
    blocks = state.blocks(now)
    typer.echo("Stops in force: " + ("; ".join(d for _, d in blocks) if blocks else "none"))


NEWS_HISTORY = "config/news/usd_high_impact_history.csv"
NEWS_CACHE = "data/news"


@news_app.command("fetch")
def news_fetch() -> None:
    """Download this week's high-impact USD events and cache them in data/news/."""
    from pathlib import Path
    from urllib.error import URLError

    from tradeagent.config import project_path
    from tradeagent.data.news import fetch_live_feed, parse_live_feed, save_live_feed
    from tradeagent.timeutil import utc_now

    now = utc_now()
    try:
        raw = fetch_live_feed()
        path = save_live_feed(project_path(Path(NEWS_CACHE)), now, raw)
    except (URLError, TimeoutError, ValueError, KeyError) as exc:
        typer.secho(
            f"Could not fetch the news feed: {exc}. Without it, live entries are "
            "blocked this week (news_unknown).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(f"{len(parse_live_feed(raw))} high-impact USD events this week; saved {path}")


@news_app.command("show")
def news_show(days: Annotated[int, typer.Option("--days")] = 7) -> None:
    """Upcoming high-impact USD events and the blackout status right now."""
    from pathlib import Path

    from tradeagent.config import load_config, project_path
    from tradeagent.data.news import build_calendar, cached_live_feed
    from tradeagent.timeutil import PKT, utc_now

    cfg = load_config()
    now = utc_now()
    raw = cached_live_feed(project_path(Path(NEWS_CACHE)), now)
    calendar = build_calendar(
        project_path(Path(NEWS_HISTORY)), raw, now, cfg.risk.news_blackout_minutes
    )
    status, detail = calendar.status(now)
    typer.secho(f"Now: {status} ({detail})", bold=True)
    if raw is None:
        typer.echo("This week's live feed is not cached: run `uv run tradeagent news fetch`.")
    for event in calendar.upcoming(now, days):
        typer.echo(
            f"  {event.time_utc:%a %Y-%m-%d %H:%M} UTC "
            f"({event.time_utc.tz_convert(PKT):%H:%M} PKT)  {event.title}"
        )


def _fmt_day(ts: object) -> str:
    return "-" if ts is None else f"{ts:%Y-%m-%d}"


if __name__ == "__main__":
    app()


@prob_app.command("calibrate")
def prob_calibrate(
    run: Annotated[str | None, typer.Option("--run", help="One backtest run id.")] = None,
    latest: Annotated[
        bool, typer.Option("--latest", help="Latest train run of each strategy x symbol.")
    ] = False,
) -> None:
    """Walk-forward calibration of p_win / ev_r on backtest trades (never out-of-sample)."""
    from pathlib import Path

    import pandas as pd

    from tradeagent.backtest.compare import latest_runs
    from tradeagent.config import load_config, project_path
    from tradeagent.data.store import connect_db
    from tradeagent.prob.calibration import (
        check_split,
        markdown_section,
        reliability,
        summarise,
        walk_forward,
    )
    from tradeagent.provenance import git_commit

    if (run is None) == (not latest):
        raise typer.BadParameter("give either --run RUN_ID or --latest")
    cfg = load_config()
    rules = cfg.settings.entry_rules
    conn = connect_db(project_path(cfg.settings.storage.sqlite_path))
    try:
        if latest:
            runs = latest_runs(conn, "train")
        else:
            runs = pd.read_sql_query(
                "SELECT * FROM backtest_runs WHERE run_id = ?", conn, params=(run,)
            )
    finally:
        conn.close()
    if runs.empty:
        typer.echo("No matching runs.")
        raise typer.Exit(1)
    sections = []
    for r in runs.to_dict("records"):
        check_split(r["split"])
        trades = pd.read_parquet(
            project_path(Path("data/backtests")) / r["run_id"] / "trades.parquet"
        )
        if trades.empty:
            continue
        walk = walk_forward(trades, rules)
        s = summarise(walk)
        title = f"{r['strategy']} {r['symbol']} {r['timeframe']} {r['split']} ({r['run_id']})"
        sections.append(markdown_section(title, s, reliability(walk)))
        typer.echo(
            f"{r['strategy']:<22} {r['symbol']:<7} trades {s.trades:>4}  hit {s.hit_rate:6.1%}  "
            f"p_win {s.mean_p_win:6.1%}  Brier {s.brier:.4f} (50%: {s.brier_half:.4f})  "
            f"ev {s.mean_ev_r:+.3f} vs real {s.realised_r:+.3f}  gate passed {s.gate_passed}"
        )
    header = (
        "# Phase 5 calibration report\n\n"
        f"Generated by `tradeagent prob calibrate`, git {git_commit()}, "
        f"config {cfg.config_hash}. Each trade is predicted only from trades of the same run "
        "that closed before it entered "
        f"(Beta prior strength {rules.prior_strength:g}, {rules.credible_level:.0%} interval). "
        f"Entry gate: n >= {rules.min_sample_trades}, p_win lower bound >= "
        f"{rules.min_p_win_lower:.2f}, ev_r >= {rules.min_ev_r:.2f} R.\n\n"
    )
    if latest:
        out = project_path(Path("docs/reports/PHASE_5_CALIBRATION.md"))
    else:
        out = project_path(Path(f"data/prob/{run}_calibration.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n".join(sections), encoding="utf-8")
    typer.echo(f"saved {out}")
