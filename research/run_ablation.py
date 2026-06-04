"""
run_ablation.py — CWT vs direct strategy removal ablation study.

Configs:
  A  BASELINE              All strategies, CWT off
  B  DISABLE_MR_ONLY       MR off, PB on, CWT off
  C  DISABLE_PB_ONLY       PB off, MR on, CWT off
  D  DISABLE_MR_AND_PB     MR off, PB off, CWT off
  E  CWT_FILTER_MR_PB      MR + PB gated by CWT

Answers:
  1. Does disabling MR alone explain CWT's Sharpe gain?
  2. Does disabling PB alone explain it?
  3. Does D == E in Sharpe? (CWT = fancy kill-switch?)
  4. Does CWT outperform simple removal?
  5. If not, should CWT be diagnostics-only?
  6. Which config becomes the new baseline?

Usage:
    python run_ablation.py
"""

import logging
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-8s %(message)s",
)

import config
import main as pipeline

CAPITAL = 25_000

# ── Config definitions ──────────────────────────────────────────────────────────
CONFIGS = {
    "A_BASELINE": {
        "USE_CWT_CHOP_FILTER": False,
        "DISABLE_MEAN_REVERT":  False,
        "DISABLE_PULLBACK":     False,
        "CWT_APPLY_TO_MEAN_REVERT": False,
        "CWT_APPLY_TO_PULLBACK":    False,
    },
    "B_DISABLE_MR_ONLY": {
        "USE_CWT_CHOP_FILTER": False,
        "DISABLE_MEAN_REVERT":  True,
        "DISABLE_PULLBACK":     False,
        "CWT_APPLY_TO_MEAN_REVERT": False,
        "CWT_APPLY_TO_PULLBACK":    False,
    },
    "C_DISABLE_PB_ONLY": {
        "USE_CWT_CHOP_FILTER": False,
        "DISABLE_MEAN_REVERT":  False,
        "DISABLE_PULLBACK":     True,
        "CWT_APPLY_TO_MEAN_REVERT": False,
        "CWT_APPLY_TO_PULLBACK":    False,
    },
    "D_DISABLE_MR_AND_PB": {
        "USE_CWT_CHOP_FILTER": False,
        "DISABLE_MEAN_REVERT":  True,
        "DISABLE_PULLBACK":     True,
        "CWT_APPLY_TO_MEAN_REVERT": False,
        "CWT_APPLY_TO_PULLBACK":    False,
    },
    "E_CWT_FILTER_MR_PB": {
        "USE_CWT_CHOP_FILTER": True,
        "DISABLE_MEAN_REVERT":  False,
        "DISABLE_PULLBACK":     False,
        "CWT_APPLY_TO_MEAN_REVERT": True,
        "CWT_APPLY_TO_PULLBACK":    True,
        "CWT_APPLY_TO_STAT_ARB":    False,
        "CWT_APPLY_TO_TREND":       False,
    },
}


# ── Config patch helpers ────────────────────────────────────────────────────────
def _patch(overrides):
    original = {}
    for k, v in overrides.items():
        original[k] = getattr(config, k, None)
        setattr(config, k, v)
    return original


def _restore(original):
    for k, v in original.items():
        setattr(config, k, v)


# ── Metrics extraction ──────────────────────────────────────────────────────────
def _extract(name, result):
    stats = result.get("stats", {})
    tl    = result.get("trade_log", pd.DataFrame())
    pr    = result.get("portfolio_returns", None)
    if tl is None:
        tl = pd.DataFrame()

    # Dollar P&L: underlying_ret × entry_weight × capital (weight-adjusted approximation)
    if len(tl):
        tl = tl.copy()
        tl["dollar_pnl"] = tl["pnl"] * tl["entry_weight"] * CAPITAL
        tl["win"]        = tl["pnl"] > 0

    # Portfolio-level totals — stats uses *_pct keys (already %) and *_ratio for Sharpe
    total_ret_pct        = stats.get("total_return_pct",     np.nan)  # e.g. 15.75
    dollar_pnl_portfolio = (total_ret_pct / 100) * CAPITAL if not np.isnan(float(total_ret_pct or np.nan)) else np.nan

    # Weekly P&L
    worst_week_dollar   = np.nan
    weekly_win_rate     = np.nan
    if len(tl) and "entry_date" in tl.columns:
        tl["entry_week"] = pd.to_datetime(tl["entry_date"]).dt.to_period("W")
        wk = tl.groupby("entry_week")["dollar_pnl"].sum()
        worst_week_dollar = wk.min()
        weekly_win_rate   = (wk > 0).mean()

    # By-strategy P&L (trade-level, weight-adjusted)
    strat_pnl = {}
    for s in ["TREND", "PULLBACK", "MEAN_REVERT", "STAT_ARB"]:
        sub = tl[tl["strategy"] == s] if len(tl) and "strategy" in tl.columns else pd.DataFrame()
        strat_pnl[f"dollar_{s}"]  = sub["dollar_pnl"].sum() if len(sub) else 0.0
        strat_pnl[f"trades_{s}"]  = len(sub)
        strat_pnl[f"winpct_{s}"]  = sub["win"].mean() * 100 if len(sub) else np.nan

    # By-exit reason
    exit_counts = {}
    for er in ["STOP_LOSS", "SHOCK_EXIT", "TRAIL_STOP", "FAST_EXIT",
               "MAX_HOLD_EXIT_TREND", "SIGNAL_FUNDAMENTAL_EXIT"]:
        sub = tl[tl["exit_reason"] == er] if len(tl) and "exit_reason" in tl.columns else pd.DataFrame()
        exit_counts[f"n_{er}"]      = len(sub)
        exit_counts[f"dollar_{er}"] = sub["dollar_pnl"].sum() if len(sub) else 0.0

    # By-regime P&L
    regime_pnl = {}
    for r in ["TREND", "CHOP", "TRANSITION", "SHOCK"]:
        sub = tl[tl["entry_regime"] == r] if len(tl) and "entry_regime" in tl.columns else pd.DataFrame()
        regime_pnl[f"dollar_regime_{r}"] = sub["dollar_pnl"].sum() if len(sub) else 0.0

    # Sortino (annualised from daily returns)
    sortino = np.nan
    if pr is not None and len(pr) > 30:
        neg = pr[pr < 0]
        dd  = neg.std() * np.sqrt(252) if len(neg) > 5 else np.nan
        ann = pr.mean() * 252
        sortino = ann / dd if (dd and dd > 0) else np.nan

    row = {
        "config":            name,
        "sharpe":            float(stats.get("sharpe_ratio") or np.nan),
        "sortino":           sortino,
        "total_return_pct":  float(stats.get("total_return_pct") or np.nan),
        "ann_return_pct":    float(stats.get("annualized_return_pct") or np.nan),
        "max_drawdown_pct":  float(stats.get("max_drawdown_pct") or np.nan),
        "dollar_pnl":        dollar_pnl_portfolio,
        "n_trades":          len(tl),
        "win_rate_pct":      tl["win"].mean() * 100 if len(tl) else np.nan,
        "profit_factor":     float(stats.get("profit_factor") or np.nan),
        "avg_trade_dollar":  tl["dollar_pnl"].mean() if len(tl) else np.nan,
        "worst_week_dollar": worst_week_dollar,
        "weekly_win_rate_pct": weekly_win_rate * 100 if not np.isnan(float(weekly_win_rate or np.nan)) else np.nan,
    }
    row.update(strat_pnl)
    row.update(exit_counts)
    row.update(regime_pnl)
    return row, tl


# ── Blocked signal forensics ────────────────────────────────────────────────────
def _blocked_signal_analysis(tl_A, tl_E, df_signals_E):
    """
    Compares trade log A (baseline) vs E (CWT) to find:
      - How many baseline trades were entirely prevented by CWT
      - Their P&L in baseline (hypothetical)
      - The 659 blocked SIGNAL rows vs actual TRADE prevention
    """
    print("\n" + "=" * 72)
    print("  BLOCKED SIGNAL FORENSICS")
    print("=" * 72)

    # ── 1. Raw signal-level block count ────────────────────────────────────────
    n_raw_blocked = 0
    if df_signals_E is not None and "signal_reason" in df_signals_E.columns:
        n_raw_blocked = df_signals_E["signal_reason"].str.contains("BLOCKED_CWT", na=False).sum()
    print(f"\n  Raw 'BLOCKED_CWT' signal rows:    {n_raw_blocked:>6}")
    print(f"  (These are (date, symbol) rows where signal_engine output was")
    print(f"   changed to FLAT by the CWT gate — includes HOLD days, not just entries)")

    # ── 2. Decompose: entry-day blocks vs hold-day blocks ──────────────────────
    if df_signals_E is not None and "signal_reason" in df_signals_E.columns:
        blocked_rows = df_signals_E[df_signals_E["signal_reason"].str.contains("BLOCKED_CWT", na=False)]
        mr_blocks  = blocked_rows["signal_reason"].str.contains("BLOCKED_CWT_MR",  na=False).sum()
        pb_blocks  = blocked_rows["signal_reason"].str.contains("BLOCKED_CWT_PB",  na=False).sum()
        print(f"\n  By strategy:")
        print(f"    MEAN_REVERT blocked: {mr_blocks:>4} signal-rows")
        print(f"    PULLBACK blocked:    {pb_blocks:>4} signal-rows")
        print(f"\n  IMPORTANT: signal_engine fires on every (date, symbol) pair.")
        print(f"  A single 3-day MR trade = 3 blocked signal rows (entry + 2 holds).")
        print(f"  The backtest uses signal_engine output for NEW ENTRIES only")
        print(f"  (when its own position is FLAT). Hold-day blocks drive")
        print(f"  SIGNAL_FUNDAMENTAL_EXIT in the backtest for open non-TREND positions.")

    # ── 3. Actual trades prevented ─────────────────────────────────────────────
    if len(tl_A) == 0 or len(tl_E) == 0:
        print("\n  Cannot compare — one trade log is empty.")
        return

    # Baseline MR + PB trades
    mr_pb_A = tl_A[tl_A["strategy"].isin(["MEAN_REVERT", "PULLBACK"])].copy()
    mr_pb_E = tl_E[tl_E["strategy"].isin(["MEAN_REVERT", "PULLBACK"])].copy()

    n_prevented = len(mr_pb_A) - len(mr_pb_E)
    print(f"\n  Baseline (A) MR+PB trades:        {len(mr_pb_A):>6}")
    print(f"  CWT config (E) MR+PB trades:      {len(mr_pb_E):>6}")
    print(f"  Trades actually prevented by CWT: {n_prevented:>6}")

    # Estimate blocked signal rows per prevented trade
    if n_prevented > 0 and n_raw_blocked > 0:
        print(f"  → ~{n_raw_blocked/n_prevented:.0f} blocked signal-rows per prevented trade")
        print(f"    (multi-day holds + repeated daily re-blocking inflates the count)")

    # ── 4. P&L of prevented trades (from baseline A) ──────────────────────────
    if len(mr_pb_A) > 0:
        mr_pb_A["dollar_pnl"] = mr_pb_A["pnl"] * mr_pb_A["entry_weight"] * CAPITAL
        mr_pb_A["win"]        = mr_pb_A["pnl"] > 0

        print(f"\n  P&L of prevented MR+PB trades (from baseline A):")
        print(f"  {'Strategy':<16} {'Trades':>6} {'Win%':>7} {'Total $':>10} {'Avg $':>9}")
        print(f"  {'-'*52}")
        for strat, grp in mr_pb_A.groupby("strategy"):
            wr  = grp["win"].mean() * 100
            tot = grp["dollar_pnl"].sum()
            avg = grp["dollar_pnl"].mean()
            print(f"  {strat:<16} {len(grp):>6}  {wr:>6.1f}%  {tot:>+10,.2f}  {avg:>+8.2f}")
        print(f"  {'TOTAL':<16} {len(mr_pb_A):>6}  "
              f"{mr_pb_A['win'].mean()*100:>6.1f}%  "
              f"{mr_pb_A['dollar_pnl'].sum():>+10,.2f}  "
              f"{mr_pb_A['dollar_pnl'].mean():>+8.2f}")

        tot_prevented = mr_pb_A["dollar_pnl"].sum()
        wins_prevented = mr_pb_A[mr_pb_A["win"]]["dollar_pnl"].sum()
        loss_prevented = mr_pb_A[~mr_pb_A["win"]]["dollar_pnl"].sum()
        print(f"\n  Prevented winners: ${wins_prevented:+,.2f}")
        print(f"  Prevented losers:  ${loss_prevented:+,.2f}")
        print(f"  Net avoided loss:  ${-tot_prevented:+,.2f}  "
              f"({'gain' if tot_prevented < 0 else 'cost'} from blocking)")

    # ── 5. Exit reason shift ───────────────────────────────────────────────────
    if "exit_reason" in tl_A.columns and "exit_reason" in tl_E.columns:
        print(f"\n  Exit reason shift (A → E):")
        reasons = sorted(set(tl_A["exit_reason"].unique()) | set(tl_E["exit_reason"].unique()))
        print(f"  {'Exit reason':<30} {'A count':>8} {'E count':>8} {'delta':>7}")
        print(f"  {'-'*57}")
        for r in reasons:
            na = (tl_A["exit_reason"] == r).sum()
            ne = (tl_E["exit_reason"] == r).sum()
            print(f"  {r:<30} {na:>8}  {ne:>8}  {ne-na:>+7}")


# ── Main ────────────────────────────────────────────────────────────────────────
def run():
    results  = []
    trade_logs = {}
    signals_E  = None

    print("=" * 72)
    print("  CWT ABLATION STUDY — A vs B vs C vs D vs E")
    print("=" * 72)

    for name, overrides in CONFIGS.items():
        print(f"\n  ▶  Running {name} ...", flush=True)
        orig = _patch(overrides)
        try:
            result = pipeline.run()
            row, tl = _extract(name, result)
            results.append(row)
            trade_logs[name] = tl
            if name == "E_CWT_FILTER_MR_PB":
                signals_E = result.get("df_signals", None)
            print(f"  ✓  Sharpe={row['sharpe']:.3f}  "
                  f"Return={row['total_return_pct']:.2f}%  "
                  f"Trades={row['n_trades']}  "
                  f"SL={row.get('n_STOP_LOSS',0)}")
        except Exception as ex:
            import traceback
            print(f"  ✗  ERROR: {ex}")
            traceback.print_exc()
            results.append({"config": name, "error": str(ex)})
        finally:
            _restore(orig)

    # ── Summary table ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out = Path(config.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation_results.csv", index=False)
    print(f"\n  Saved ablation_results.csv")

    _print_summary(df)
    _print_strategy_breakdown(df)
    _print_exit_reason_table(df)
    _print_regime_table(df)

    # Blocked signal analysis uses A and E trade logs
    tl_A = trade_logs.get("A_BASELINE", pd.DataFrame())
    tl_E = trade_logs.get("E_CWT_FILTER_MR_PB", pd.DataFrame())

    _blocked_signal_analysis(tl_A, tl_E, signals_E)
    _print_verdict(df, tl_A, tl_E)

    return df


def _print_summary(df):
    print("\n\n" + "=" * 72)
    print("  ABLATION RESULTS SUMMARY — $25,000 BASE")
    print("=" * 72)
    cols = ["config", "sharpe", "sortino", "total_return_pct", "ann_return_pct",
            "max_drawdown_pct", "dollar_pnl", "n_trades", "win_rate_pct",
            "profit_factor", "n_STOP_LOSS", "n_SHOCK_EXIT",
            "worst_week_dollar", "weekly_win_rate_pct"]
    cols = [c for c in cols if c in df.columns]
    avail = df[cols].copy()

    fmt = {
        "sharpe":            "{:>7.3f}",
        "sortino":           "{:>7.3f}",
        "total_return_pct":  "{:>+7.2f}%",
        "ann_return_pct":    "{:>+7.2f}%",
        "max_drawdown_pct":  "{:>7.2f}%",
        "dollar_pnl":        "${:>+8,.0f}",
        "n_trades":          "{:>6.0f}",
        "win_rate_pct":      "{:>6.1f}%",
        "profit_factor":     "{:>7.3f}",
        "n_STOP_LOSS":       "{:>5.0f}",
        "n_SHOCK_EXIT":      "{:>5.0f}",
        "worst_week_dollar": "${:>+8,.0f}",
        "weekly_win_rate_pct": "{:>6.1f}%",
    }
    labels = {
        "config":            "Config",
        "sharpe":            "Sharpe",
        "sortino":           "Sortino",
        "total_return_pct":  "Total%",
        "ann_return_pct":    "Ann%",
        "max_drawdown_pct":  "MaxDD",
        "dollar_pnl":        "$ P&L",
        "n_trades":          "Trades",
        "win_rate_pct":      "Win%",
        "profit_factor":     "PF",
        "n_STOP_LOSS":       "SL",
        "n_SHOCK_EXIT":      "SHOCK",
        "worst_week_dollar": "WorstWk",
        "weekly_win_rate_pct": "WkWin%",
    }

    # Print transposed (one metric per row, one config per column)
    cfg_names = df["config"].tolist()
    width = 24
    print(f"\n  {'Metric':<26}" + "".join(f"{c:<22}" for c in cfg_names))
    print("  " + "-" * (26 + 22 * len(cfg_names)))
    for col in cols:
        if col == "config":
            continue
        label = labels.get(col, col)
        row_str = f"  {label:<26}"
        for _, r in df.iterrows():
            val = r.get(col, np.nan)
            try:
                if np.isnan(val):
                    cell = "     N/A"
                else:
                    cell = fmt.get(col, "{:>8.3f}").format(val)
            except Exception:
                cell = f"{val!s:>8}"
            row_str += f"{cell:<22}"
        print(row_str)


def _print_strategy_breakdown(df):
    print("\n\n  ── Strategy $ P&L by config ────────────────────────────────────────")
    strats = ["TREND", "PULLBACK", "MEAN_REVERT", "STAT_ARB"]
    cfg_names = df["config"].tolist()
    print(f"  {'Strategy':<16}" + "".join(f"{c:<22}" for c in cfg_names))
    print("  " + "-" * (16 + 22 * len(cfg_names)))
    for s in strats:
        col  = f"dollar_{s}"
        wc   = f"winpct_{s}"
        tc   = f"trades_{s}"
        line = f"  {s:<16}"
        for _, r in df.iterrows():
            d  = r.get(col, 0.0)
            w  = r.get(wc,  np.nan)
            t  = int(r.get(tc, 0))
            if t == 0:
                line += f"{'—':>22}"
            else:
                line += f"${d:>+8,.0f} ({w:.0f}%,{t:d}){'':<1}"
        print(line)


def _print_exit_reason_table(df):
    print("\n\n  ── Exit reason $ P&L by config ─────────────────────────────────────")
    reasons = ["STOP_LOSS", "SHOCK_EXIT", "TRAIL_STOP", "FAST_EXIT",
               "MAX_HOLD_EXIT_TREND", "SIGNAL_FUNDAMENTAL_EXIT"]
    cfg_names = df["config"].tolist()
    print(f"  {'Exit reason':<30}" + "".join(f"{c:<22}" for c in cfg_names))
    print("  " + "-" * (30 + 22 * len(cfg_names)))
    for er in reasons:
        dc = f"dollar_{er}"
        nc = f"n_{er}"
        line = f"  {er:<30}"
        for _, r in df.iterrows():
            d = r.get(dc, 0.0)
            n = int(r.get(nc, 0))
            if n == 0:
                line += f"{'$0 (0)':>22}"
            else:
                line += f"${d:>+8,.0f} ({n:d}){'':<5}"
        print(line)


def _print_regime_table(df):
    print("\n\n  ── Regime $ P&L by config ──────────────────────────────────────────")
    regimes   = ["TREND", "CHOP", "TRANSITION", "SHOCK"]
    cfg_names = df["config"].tolist()
    print(f"  {'Regime':<14}" + "".join(f"{c:<22}" for c in cfg_names))
    print("  " + "-" * (14 + 22 * len(cfg_names)))
    for reg in regimes:
        col  = f"dollar_regime_{reg}"
        line = f"  {reg:<14}"
        for _, r in df.iterrows():
            d = r.get(col, 0.0)
            line += f"${d:>+8,.0f}{'':<12}"
        print(line)


def _print_verdict(df, tl_A, tl_E):
    print("\n\n" + "=" * 72)
    print("  ABLATION VERDICT")
    print("=" * 72)

    def g(name, col):
        rows = df[df["config"] == name]
        return rows.iloc[0][col] if len(rows) else np.nan

    sharpe_A = g("A_BASELINE",         "sharpe")
    sharpe_B = g("B_DISABLE_MR_ONLY",  "sharpe")
    sharpe_C = g("C_DISABLE_PB_ONLY",  "sharpe")
    sharpe_D = g("D_DISABLE_MR_AND_PB","sharpe")
    sharpe_E = g("E_CWT_FILTER_MR_PB", "sharpe")

    def delta(s, ref=sharpe_A):
        if np.isnan(s) or np.isnan(ref):
            return "N/A"
        d = s - ref
        return f"{d:+.3f}"

    print(f"\n  Q1. Does disabling MR alone explain CWT's Sharpe gain?")
    print(f"      A={sharpe_A:.3f}  B(MR off)={sharpe_B:.3f} (Δ={delta(sharpe_B)})  "
          f"E(CWT)={sharpe_E:.3f} (Δ={delta(sharpe_E)})")
    if not np.isnan(sharpe_B) and not np.isnan(sharpe_E):
        frac = (sharpe_B - sharpe_A) / (sharpe_E - sharpe_A + 1e-9) * 100
        print(f"      MR-only explains {frac:.0f}% of CWT's gain.")
        print(f"      → {'YES — MR removal drives most of the improvement.' if frac > 70 else 'NO — MR alone is not enough.'}")

    print(f"\n  Q2. Does disabling PB alone explain CWT's Sharpe gain?")
    print(f"      A={sharpe_A:.3f}  C(PB off)={sharpe_C:.3f} (Δ={delta(sharpe_C)})  "
          f"E(CWT)={sharpe_E:.3f}")
    if not np.isnan(sharpe_C) and not np.isnan(sharpe_E):
        frac = (sharpe_C - sharpe_A) / (sharpe_E - sharpe_A + 1e-9) * 100
        print(f"      PB-only explains {frac:.0f}% of CWT's gain.")
        print(f"      → {'YES — PB removal drives most of the improvement.' if frac > 70 else 'NO — PB alone is not enough.'}")

    print(f"\n  Q3. Does D (MR+PB off) == E (CWT) in Sharpe?")
    print(f"      D={sharpe_D:.3f}  E={sharpe_E:.3f}  |D-E|={abs(sharpe_D - sharpe_E):.4f}")
    if not np.isnan(sharpe_D) and not np.isnan(sharpe_E):
        if abs(sharpe_D - sharpe_E) < 0.005:
            print(f"      → YES — CWT and direct removal produce identical Sharpe.")
            print(f"        CWT adds NO incremental value beyond simply turning off MR+PB.")
        elif sharpe_E > sharpe_D:
            print(f"      → CWT is BETTER than simple removal by {sharpe_E - sharpe_D:.3f}.")
            print(f"        CWT selectively allows some MR/PB trades that add value.")
        else:
            print(f"      → Direct removal is BETTER than CWT by {sharpe_D - sharpe_E:.3f}.")
            print(f"        CWT is noisier than a clean disable.")

    print(f"\n  Q4. Does CWT outperform simple removal (D)?")
    if not np.isnan(sharpe_D) and not np.isnan(sharpe_E):
        if sharpe_E > sharpe_D + 0.005:
            print(f"      YES — E ({sharpe_E:.3f}) > D ({sharpe_D:.3f}). CWT adds selective alpha.")
        elif abs(sharpe_E - sharpe_D) <= 0.005:
            print(f"      NO — tied within noise. CWT is a complex kill-switch.")
        else:
            print(f"      NO — E ({sharpe_E:.3f}) < D ({sharpe_D:.3f}). CWT is worse than a hard off.")

    print(f"\n  Q5. If CWT does not outperform simple removal — diagnostics only?")
    if not np.isnan(sharpe_D) and not np.isnan(sharpe_E):
        if abs(sharpe_E - sharpe_D) <= 0.005:
            print(f"      YES. CWT should be DIAGNOSTIC ONLY.")
            print(f"      Use it to understand regime structure, not to filter entries.")
        else:
            print(f"      CWT does outperform — consider keeping it live.")

    # Best config
    valid = df[df["sharpe"].notna()].copy()
    best  = valid.loc[valid["sharpe"].idxmax(), "config"] if len(valid) else "UNKNOWN"
    print(f"\n  Q6. Which config is the new baseline before stop-loss testing?")
    print(f"      Best Sharpe: {best} ({valid['sharpe'].max():.3f})")
    print(f"      Recommendation: use {best} as the foundation.")
    print(f"      Stop-loss threshold testing should be layered on top of this config.")

    # Final recommendation
    print("\n" + "-" * 72)
    if not np.isnan(sharpe_D) and not np.isnan(sharpe_E):
        if abs(sharpe_E - sharpe_D) <= 0.005:
            verdict    = "DISABLE_MR_PB_WITHOUT_CWT"
            rationale  = (
                f"CWT (E={sharpe_E:.3f}) matches direct removal (D={sharpe_D:.3f}) "
                f"within {abs(sharpe_E-sharpe_D):.4f} Sharpe. "
                f"Adding CWT complexity gains nothing. Disable MR and PB cleanly."
            )
        elif sharpe_E > sharpe_D + 0.010:
            verdict   = "MAKE_CWT_LIVE"
            rationale = (
                f"CWT ({sharpe_E:.3f}) outperforms direct removal ({sharpe_D:.3f}) "
                f"by {sharpe_E-sharpe_D:.3f}. CWT is adding real selective filtering value."
            )
        elif sharpe_E > sharpe_D:
            verdict   = "KEEP_CWT_DIAGNOSTIC_ONLY"
            rationale = (
                f"CWT marginally better ({sharpe_E:.3f} vs {sharpe_D:.3f}) but gain "
                f"is too small to justify complexity. Keep for regime analysis only."
            )
        else:
            verdict   = "REJECT_CWT"
            rationale = (
                f"Direct removal ({sharpe_D:.3f}) beats CWT ({sharpe_E:.3f}). "
                f"CWT adds noise and complexity without benefit."
            )
    else:
        verdict   = "CONTINUE_TESTING"
        rationale = "Incomplete results — one or more configs failed."

    print(f"\n  FINAL RECOMMENDATION: {verdict}")
    print(f"  {rationale}")
    print("=" * 72)


if __name__ == "__main__":
    run()
