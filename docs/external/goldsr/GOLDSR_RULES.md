# GoldSR EA v2.4: rules for hypothesis H5

Source: `GoldSR_EA_v2_4.mq5` (copied from Usama's MT5 Experts folder, 2026-09-30). The EA's own code is the reference. This file is a plain-English summary for porting it into `strategies/`.

## Defaults that are ON
- **Signal timeframe:** M15. The signal is checked once per closed bar (`r[1]`); entry is a market order on the next bar, which matches our t+1 open rule.
- **Levels:** the most recent confirmed swing low is the "floor" (support) and the most recent confirmed swing high is the "ceiling" (resistance). A pivot needs 5 bars on each side (`InpPivLen=5`), and the scan uses 600 bars, closed bars only. So a pivot is only known 5 bars after it forms, with no look-ahead.
- **M15 trend:** EMA20 and EMA50 on close. UP = EMA20 > EMA50 and close > EMA50. DOWN = the mirror. Anything else is SIDEWAYS.
- **SELL (breakdown):** trend is DOWN, bar1 closes below the floor, and bar2 closed at or above the floor.
- **BUY (breakout):** trend is UP, bar1 closes above the ceiling, and bar2 closed at or below the ceiling.
- **v2.3 filters** (all must pass):
  - H4 trend agrees: H4 EMA20 vs EMA50 plus H4 close vs EMA50, using the last closed H4 bar. A neutral H4 trend blocks the trade.
  - H1 ADX(14) on the last closed H1 bar is between 20 and 50.
  - The breakout candle is strong: body ≥ 50% of its range, it closes in the trade direction, and it closes beyond the level by at least 0.1×ATR(14, M15).
- **Stop loss:**
  - SELL: max(bar1 high, floor) + 1.0×ATR.
  - BUY: min(bar1 low, ceiling) − 1.0×ATR.
  - If risk < $2, the SL is widened to $2. If risk > $20, the trade is skipped.
- **Take profit:** R-multiples. TP1 = 1.0R, TP2 = 1.5R, TP3 = 2.0R. **TP3 is the broker TP.**
- **Management** (checked every tick):
  - When TP1 is hit, move the SL to entry ± $0.30. For a buy that's entry + 0.3; for a sell, entry − 0.3.
  - When TP2 is hit, move the SL to TP1.
- **Re-entry after a WIN only:** up to 2 re-entries per signal, within 10 M15 bars after the close.
  - Price must come back within $0.50 of the first entry.
  - The setup must still be valid: same trend, close still beyond the level, and H4 still agrees.
  - It reuses the original SL price.
- **Safety:**
  - Spread ≤ $0.80.
  - Server hours 07:00–24:00.
  - At most 8 entries per day.
  - Daily loss limit is 5% of balance.
  - High-impact USD news blackout of 30 minutes each side. This is disabled inside the MT5 tester.
  - Close any open trade on Friday at 21:00 server time.
- **Lot size:** 1% risk per trade (hard cap 5%), with at most 50% of free margin used.
- **One position at a time**, identified by the magic number 240927.

## Defaults that are OFF
BOUNCE (buying off the floor), RANGE mode, trailing profit lock, and partial closes.

## Differences from our system, to settle when porting
1. **Risk per trade:** EA 1%, ours 0.5% (`risk.yaml`). **Daily loss:** EA 5%, ours 2%. Use ours. Our risk engine is the authority.
2. **Reward-to-risk:** the final target is 2.0R, which just meets `min_reward_risk 2.0`. But breakeven after 1R means many trades end at about 0R. Report the R distribution: SL / BE / TP1-lock / TP3.
3. **Re-entry after a win** breaks the "each signal is independent" assumption. Test with re-entry both ON and OFF, and register both as experiments.
4. **SL moves inside a bar** (TP1 → BE, TP2 → TP1) depend on the path price took, so use the M5 inside-bar resolution the engine already has for M15 strategies.
5. **The $2 minimum SL** is only about 7 spreads on XAUUSDm. Check that cost as a % of gross stays reasonable.
6. **The news filter is off in MT5 tests**, so any MT5 Strategy Tester results Usama saw had no news blackout. Our backtest applies the historical news list.
7. **Server time:** Exness server time is UTC+0. Convert the 07:00–24:00 window and the Friday 21:00 close using the actual server offset, and record it in DECISIONS.md.
8. **The optimizer score (`OnTester`)** means the EA's defaults may already be tuned on past data. So H5 must be judged on validation only, with the defaults frozen as-is (no re-tuning), and it counts as 1 strategy run (or 2 with the re-entry variant).

## Source file used for H5 (reproducibility)
The `.mq5` source is kept local only (git-ignored). The H5 port and experiments E0021/E0022 were built from this exact file:

| File | sha256 |
|---|---|
| `GoldSR_EA_v2_4.mq5` | `df41775f989848701c56b9880ebfadd54f51a06f7af6e2455c9dc8209a6086f0` |

Check a local copy with PowerShell: `Get-FileHash docs\external\goldsr\GoldSR_EA_v2_4.mq5 -Algorithm SHA256`.

**H5 result (2026-10-03): falsified** on train for both variants; see `docs/reports/PHASE_6_LEADS.md`.
