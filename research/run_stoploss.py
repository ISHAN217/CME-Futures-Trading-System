"""
run_stoploss.py — Stop-loss threshold sweep on Config D baseline.

Baseline: DISABLE_MEAN_REVERT=True, DISABLE_PULLBACK=True, CWT off.
Current stop: -1.5% (STOP_LOSS_THRESHOLD = -0.015).

Tests 10 thresholds from tight (-0.010) to disabled (None).
Also runs a secondary sweep: stop + paired trail-arm/drawdown adjustments.

Questions answered:
  1. What is the optimal stop-loss threshold for Sharpe?
  2. What is the optimal for max-drawdown?
  3. At what threshold do stops help vs hurt?
  4. How many stop-outs recover vs continue falling?
  5. What is the P&L cost/benefit vs no-stop baseline?
  6. Should the trail be tightened when the stop is loosened?

Usage:
    python run_stoploss.py
"""

import logging
import sys
import itertools
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

CAPITAL  = 25_000

# ── Config D baseline (always applied) ─────────────────────────────────────────
D_BASE = {
    "USE_CWT_CHOP_FILTER":  False,
    "DISABLE_MEAN_REVERT":  True,
    "DISABLE_PULLBACK":     True,
    "CWT_APPLY_TO_MEAN_REVERT": False,
    "CWT_APPLY_TO_PULLBACK":    False,
}

# ── Stop-loss sweep thresholds ──────────────────────────────────────────────────
# None = disabled (no stop loss at all)
STOP_THRESHOLDS = [
    -0.010,   # -1.0%  very tight
    -0.012,   # -1.2%  tight
    -0.015,   # -1.5%  current
    -0.018,   # -1.8%
    -0.020,   # -2.0%
    -0.025,   # -2.5%
    -0.030,   # -3.0%
    -0.040,   # -4.0%
    -0.050,   # -5.0%
    None,     # disabled
]

# ── Secondary: paired trail adjustments with best stop ─────────────────────────
# (run after primary sweep to find if trail tightening complements a looser stop)
TRAIL_PAIRS = [
    # (trail_arm, trail_drawdown) — tested against the best stop from sweep
    (0.008, 0.005),   # tighter arm + tighter drawdown
    (0.008, 0.007),   # tighter arm, same drawdown
    (0.010, 0.005),   # same arm, tighter drawdown
    (0.010, 0.007),   # current (baseline)
    (0.010, 0.010),   # same arm, looser drawdown
    (0.012, 0.007),   # looser arm, same drawdown
    (0.015, 0.007),   # much looser arm
    (0.015, 0.010),   # much looser arm + looser drawdown
]


# ── Patch / restore helpers ─────────────────────────────────────────────────────
def _patch(overrides):
    orig = {}
    for k, v in overrides.items():
        orig[k] = getattr(config, k, None)
        setattr(config, k, v)
    return orig

def _restore(orig):
    for k, v in orig.items():
        setattr(config, k, v)


# ── Metrics extraction ──────────────────────────────────────────────────────────
def _extract(label, result, stop_thresh):
    stats = result.get("stats", {})
    tl    = result.get("trade_log", pd.DataFrame())
    pr    = result.get("portfolio_returns", None)

    if tl is None:
        tl = pd.DataFrame()
    if len(tl):
        tl = tl.copy()
        tl["dollar_pnl"] = tl["pnl"] * tl["entry_weight"] * CAPITAL
        tl["win"]        = tl["pnl"] > 0

    # Portfolio P&L from authoritative stats
    total_ret_pct = float(stats.get("total_return_pct") or np.nan)
    dollar_portfolio = (total_ret_pct / 100.0) * CAPITAL

    # Sortino
    sortino = np.nan
    if pr is not None and len(pr) > 30:
        neg = pr[pr < 0]
        dd  = neg.std() * np.sqrt(252) if len(neg) > 5 else np.nan
        ann = pr.mean() * 252
        sortino = ann / dd if dd and dd > 0 else np.nan

    # Weekly
    worst_wk = np.nan
    wk_wr    = np.nan
    if len(tl) and "entry_date" in tl.columns:
        tl["entry_week"] = pd.to_datetime(tl["entry_date"]).dt.to_period("W")
        wk = tl.groupby("entry_week")["dollar_pnl"].sum()
        worst_wk = wk.min()
        wk_wr    = (wk > 0).mean() * 100

    # Stop-loss specific
    sl_sub = tl[tl["exit_reason"] == "STOP_LOSS"] if len(tl) and "exit_reason" in tl.columns else pd.DataFrame()
    sl_n      = len(sl_sub)
    sl_dollar = sl_sub["dollar_pnl"].sum() if len(sl_sub) else 0.0
    sl_avg    = sl_sub["dollar_pnl"].mean() if len(sl_sub) else np.nan

    # SHOCK exits
    shock_sub = tl[tl["exit_reason"] == "SHOCK_EXIT"] if len(tl) and "exit_reason" in tl.columns else pd.DataFrame()
    shock_n   = len(shock_sub)

    # By strategy
    strat = {}
    for s in ["TREND", "STAT_ARB"]:
        sub = tl[tl["strategy"] == s] if len(tl) and "strategy" in tl.columns else pd.DataFrame()
        strat[f"n_{s}"]       = len(sub)
        strat[f"dollar_{s}"]  = sub["dollar_pnl"].sum() if len(sub) else 0.0
        strat[f"winpct_{s}"]  = sub["win"].mean() * 100 if len(sub) else np.nan

    row = {
        "label":            label,
        "stop_thresh":      stop_thresh if stop_thresh is not None else "NONE",
        "sharpe":           float(stats.get("sharpe_ratio") or np.nan),
        "sortino":          sortino,
        "total_return_pct": total_ret_pct,
        "ann_return_pct":   float(stats.get("annualized_return_pct") or np.nan),
        "max_drawdown_pct": float(stats.get("max_drawdown_pct") or np.nan),
        "ann_vol_pct":      float(stats.get("annualized_vol_pct") or np.nan),
        "dollar_pnl":       dollar_portfolio,
        "n_trades":         len(tl),
        "win_rate_pct":     tl["win"].mean() * 100 if len(tl) else np.nan,
        "profit_factor":    float(stats.get("profit_factor") or np.nan),
        "avg_trade_dollar": tl["dollar_pnl"].mean() if len(tl) else np.nan,
        "worst_week_dollar":worst_wk,
        "weekly_win_rate_pct": wk_wr,
        "n_stop_loss":      sl_n,
        "dollar_stop_loss": sl_dollar,
        "avg_stop_loss":    sl_avg,
        "n_shock_exit":     shock_n,
        "trail_stop_n":     int(stats.get("trail_stop_count") or 0),
        "fast_exit_n":      int(stats.get("fast_exit_count")  or 0),
        "extension_n":      int(stats.get("extension_count")  or 0),
        "extension_wr":     float(stats.get("extended_win_rate") or np.nan) * 100,
    }
    row.update(strat)
    return row, tl


# ── Primary sweep ───────────────────────────────────────────────────────────────
def run_primary_sweep():
    print("=" * 74)
    print("  STOP-LOSS THRESHOLD SWEEP  (Config D baseline: MR+PB disabled)")
    print("=" * 74)

    results = []
    trade_logs = {}

    for thresh in STOP_THRESHOLDS:
        label = f"SL_{abs(thresh)*100:.1f}pct" if thresh is not None else "SL_NONE"
        pct_str = f"{abs(thresh)*100:.1f}%" if thresh is not None else "DISABLED"
        print(f"\n  ▶  Stop={pct_str:>10}  ({label})", flush=True)

        overrides = {**D_BASE, "STOP_LOSS_THRESHOLD": thresh if thresh is not None else -99.0}
        orig = _patch(overrides)
        try:
            result  = pipeline.run()
            row, tl = _extract(label, result, thresh)
            results.append(row)
            trade_logs[label] = tl
            print(f"     Sharpe={row['sharpe']:.3f}  "
                  f"DD={row['max_drawdown_pct']:.2f}%  "
                  f"Return={row['total_return_pct']:.2f}%  "
                  f"SL_n={row['n_stop_loss']}  "
                  f"SL_$=${row['dollar_stop_loss']:+,.0f}")
        except Exception as ex:
            import traceback
            print(f"  ERROR: {ex}")
            traceback.print_exc()
            results.append({"label": label, "stop_thresh": thresh, "error": str(ex)})
        finally:
            _restore(orig)

    df = pd.DataFrame(results)
    out = Path(config.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stoploss_sweep_results.csv", index=False)
    return df, trade_logs


# ── Trail sweep (around best stop) ─────────────────────────────────────────────
def run_trail_sweep(best_stop):
    print("\n\n" + "=" * 74)
    print(f"  TRAIL PARAMETER SWEEP  (best stop={abs(best_stop)*100:.1f}%)")
    print("=" * 74)

    results = []
    for arm, draw in TRAIL_PAIRS:
        label = f"arm{arm*100:.1f}_draw{draw*100:.1f}"
        print(f"\n  ▶  trail_arm={arm*100:.1f}%  trail_drawdown={draw*100:.1f}%", flush=True)

        overrides = {
            **D_BASE,
            "STOP_LOSS_THRESHOLD": best_stop,
            "TRAIL_ARM_THRESHOLD":  arm,
            "TRAIL_STOP_DRAWDOWN":  draw,
        }
        orig = _patch(overrides)
        try:
            result  = pipeline.run()
            row, _  = _extract(label, result, best_stop)
            row["trail_arm"]  = arm
            row["trail_draw"] = draw
            results.append(row)
            print(f"     Sharpe={row['sharpe']:.3f}  DD={row['max_drawdown_pct']:.2f}%  "
                  f"Return={row['total_return_pct']:.2f}%  Trail_n={row['trail_stop_n']}")
        except Exception as ex:
            print(f"  ERROR: {ex}")
            results.append({"label": label, "trail_arm": arm, "trail_draw": draw, "error": str(ex)})
        finally:
            _restore(orig)

    df = pd.DataFrame(results)
    df.to_csv(Path(config.OUTPUT_DIR) / "trail_sweep_results.csv", index=False)
    return df


# ── Recovery analysis ───────────────────────────────────────────────────────────
def _recovery_analysis(trade_logs, primary_df):
    """
    For each stop threshold, look at trades that were stopped out.
    Compare their stopping cum_ret with what the trade WOULD have returned
    if held to max hold. Since we don't have counterfactuals, we use the
    no-stop config's trade log as a proxy.
    """
    print("\n\n" + "=" * 74)
    print("  STOP-LOSS RECOVERY ANALYSIS")
    print("=" * 74)

    # Use the no-stop trade log as the "what would have happened" reference
    nostp_tl = trade_logs.get("SL_NONE")
    curr_tl  = trade_logs.get("SL_1.5pct")

    if nostp_tl is None or curr_tl is None or len(nostp_tl) == 0:
        print("  Cannot run: missing trade log for SL_NONE or SL_1.5pct")
        return

    # Trades that exist in no-stop but not in current (because current stopped them)
    # Match by entry_date + symbol
    nostp_key = nostp_tl.set_index(["entry_date", "symbol"])["pnl"].to_dict() if "entry_date" in nostp_tl.columns else {}
    curr_key  = curr_tl.set_index(["entry_date", "symbol"])["pnl"].to_dict()  if "entry_date" in curr_tl.columns else {}

    stopped_keys = set(curr_tl[curr_tl.get("exit_reason","") == "STOP_LOSS"]
                       .set_index(["entry_date","symbol"]).index
                      ) if "exit_reason" in curr_tl.columns else set()

    print(f"\n  Trades stopped out at -1.5%: {len(stopped_keys)}")
    print(f"  Checking what happened after stop in the no-stop reference run:")

    recovered, continued_down, same_exit = 0, 0, 0
    recovered_pnl, fell_pnl = [], []

    for key in stopped_keys:
        nostp_pnl = nostp_key.get(key, None)
        if nostp_pnl is None:
            continue  # trade didn't exist in no-stop (different paths due to state)
        stop_pnl  = -0.015   # approximate — the stop level
        if nostp_pnl > stop_pnl + 0.005:    # recovered meaningfully
            recovered += 1
            recovered_pnl.append(nostp_pnl)
        elif nostp_pnl < stop_pnl - 0.005:  # continued falling
            continued_down += 1
            fell_pnl.append(nostp_pnl)
        else:
            same_exit += 1

    total_matched = recovered + continued_down + same_exit
    if total_matched > 0:
        print(f"\n  Of {total_matched} matched stop-outs (no-stop reference):")
        print(f"    Recovered past stop level: {recovered:3d} ({recovered/total_matched*100:.0f}%)")
        print(f"    Continued falling:         {continued_down:3d} ({continued_down/total_matched*100:.0f}%)")
        print(f"    Ended near stop:           {same_exit:3d} ({same_exit/total_matched*100:.0f}%)")
        if recovered_pnl:
            print(f"\n  Avg final return if NOT stopped (recoveries): "
                  f"{np.mean(recovered_pnl)*100:+.2f}%  "
                  f"(${np.mean(recovered_pnl)*0.15*CAPITAL:+.0f} on avg weight)")
        if fell_pnl:
            print(f"  Avg final return if NOT stopped (continued falls): "
                  f"{np.mean(fell_pnl)*100:+.2f}%  "
                  f"(${np.mean(fell_pnl)*0.15*CAPITAL:+.0f} on avg weight)")
    else:
        print("  Could not match stop-outs to no-stop reference (state divergence).")
        print("  This is expected — once a trade is stopped early, subsequent state diverges.")
        print(f"\n  Alternative view: P&L difference between SL_NONE and SL_1.5pct configurations:")
        none_total = primary_df[primary_df["label"]=="SL_NONE"]["dollar_pnl"].values
        curr_total = primary_df[primary_df["label"]=="SL_1.5pct"]["dollar_pnl"].values
        if len(none_total) and len(curr_total):
            diff = float(none_total[0]) - float(curr_total[0])
            print(f"    SL_NONE:   ${float(none_total[0]):+,.2f}")
            print(f"    SL_1.5pct: ${float(curr_total[0]):+,.2f}")
            print(f"    Net cost of -1.5% stop vs no stop: ${diff:+,.2f}")


# ── Print full summary ──────────────────────────────────────────────────────────
def _print_summary(df, trail_df):
    print("\n\n" + "=" * 74)
    print("  STOP-LOSS SWEEP — FULL RESULTS  ($25,000 base)")
    print("=" * 74)

    fmt_cols = [
        ("Stop level",      "stop_thresh",       "{:>10}"),
        ("Sharpe",          "sharpe",            "{:>7.3f}"),
        ("Sortino",         "sortino",           "{:>7.3f}"),
        ("Return%",         "total_return_pct",  "{:>+7.2f}%"),
        ("MaxDD%",          "max_drawdown_pct",  "{:>7.2f}%"),
        ("$ P&L",           "dollar_pnl",        "${:>+8,.0f}"),
        ("Trades",          "n_trades",          "{:>6.0f}"),
        ("Win%",            "win_rate_pct",       "{:>6.1f}%"),
        ("PF",              "profit_factor",     "{:>6.3f}"),
        ("SL_n",            "n_stop_loss",       "{:>5.0f}"),
        ("SL_$",            "dollar_stop_loss",  "${:>+7,.0f}"),
        ("WkWin%",          "weekly_win_rate_pct","{:>6.1f}%"),
        ("WorstWk$",        "worst_week_dollar",  "${:>+7,.0f}"),
    ]

    header = ""
    for lbl, _, _ in fmt_cols:
        header += f"  {lbl:>10}"
    print(header)
    print("  " + "-" * (12 * len(fmt_cols)))

    best_sharpe_row = df.loc[df["sharpe"].idxmax()] if "sharpe" in df.columns else None
    best_dd_row     = df.loc[df["max_drawdown_pct"].idxmax()] if "max_drawdown_pct" in df.columns else None

    for _, row in df.iterrows():
        line = ""
        for _, col, fmt in fmt_cols:
            val = row.get(col, np.nan)
            try:
                if col == "stop_thresh":
                    s = f"{abs(float(val))*100:.1f}%" if val != "NONE" and val is not None else "DISABLED"
                    line += f"  {s:>10}"
                elif pd.isna(float(val)):
                    line += f"  {'N/A':>10}"
                else:
                    line += f"  {fmt.format(float(val)):>10}"
            except Exception:
                line += f"  {str(val)[:10]:>10}"

        # Mark best
        markers = ""
        if best_sharpe_row is not None and row.get("label") == best_sharpe_row.get("label"):
            markers += " ← best Sharpe"
        if best_dd_row is not None and row.get("label") == best_dd_row.get("label") and row.get("label") != best_sharpe_row.get("label"):
            markers += " ← best DD"
        print(line + markers)

    # Trail sweep summary
    if trail_df is not None and len(trail_df) and "sharpe" in trail_df.columns:
        print(f"\n\n  ── Trail parameter sweep (at best stop) ───────────────────────")
        print(f"  {'arm':>6} {'draw':>6}  {'Sharpe':>8} {'Sortino':>8} {'Return%':>8} {'MaxDD%':>8} {'Trail_n':>8}")
        print(f"  {'-'*58}")
        for _, r in trail_df.sort_values("sharpe", ascending=False).iterrows():
            marker = " ← best" if r["sharpe"] == trail_df["sharpe"].max() else ""
            print(f"  {r['trail_arm']*100:>5.1f}% {r['trail_draw']*100:>5.1f}%  "
                  f"{r['sharpe']:>8.3f} {r.get('sortino',np.nan):>8.3f} "
                  f"{r['total_return_pct']:>+7.2f}% {r['max_drawdown_pct']:>7.2f}%  "
                  f"{int(r.get('trail_stop_n',0)):>7}{marker}")


def _print_verdict(primary_df, trail_df):
    print("\n\n" + "=" * 74)
    print("  STOP-LOSS VERDICT")
    print("=" * 74)

    if "sharpe" not in primary_df.columns or primary_df["sharpe"].isna().all():
        print("  No valid results.")
        return

    valid = primary_df[primary_df["sharpe"].notna()].copy()
    base  = valid[valid["stop_thresh"] == -0.015]
    if len(base) == 0:
        base = valid.iloc[:1]
    base_row = base.iloc[0]

    best_sharpe = valid.loc[valid["sharpe"].idxmax()]
    best_dd     = valid.loc[valid["max_drawdown_pct"].idxmax()]  # least negative = best

    print(f"\n  Current stop (-1.5%):   Sharpe={base_row['sharpe']:.3f}  "
          f"DD={base_row['max_drawdown_pct']:.2f}%  "
          f"$ P&L=${base_row['dollar_pnl']:+,.0f}  "
          f"SL_n={int(base_row['n_stop_loss'])}  "
          f"SL_$=${base_row['dollar_stop_loss']:+,.0f}")

    print(f"\n  Best Sharpe stop:       {best_sharpe['stop_thresh']}  "
          f"Sharpe={best_sharpe['sharpe']:.3f}  "
          f"DD={best_sharpe['max_drawdown_pct']:.2f}%  "
          f"$ P&L=${best_sharpe['dollar_pnl']:+,.0f}")

    print(f"\n  Best DD stop:           {best_dd['stop_thresh']}  "
          f"Sharpe={best_dd['sharpe']:.3f}  "
          f"DD={best_dd['max_drawdown_pct']:.2f}%")

    # Direction of improvement
    print(f"\n  ── Sharpe by threshold (trend) ─────────────────────────────────")
    for _, r in valid.sort_values("stop_thresh", key=lambda x: x.map(lambda v: float(v) if v != "NONE" else 0)).iterrows():
        bar = "█" * int(max(0, (r["sharpe"] - 0.60) * 200))
        thresh_str = f"{abs(float(r['stop_thresh']))*100:.1f}%" if r["stop_thresh"] != "NONE" else "disabled"
        print(f"  {thresh_str:>10}  {r['sharpe']:.3f}  {bar}")

    # Key insight
    print(f"\n  ── Key findings ────────────────────────────────────────────────")
    tighter = valid[valid["stop_thresh"].apply(lambda x: float(x) > -0.015 if x != "NONE" else False)]
    looser  = valid[valid["stop_thresh"].apply(lambda x: float(x) < -0.015 if x != "NONE" else False)]

    if len(tighter) and tighter["sharpe"].max() > base_row["sharpe"]:
        print(f"  • Tighter stops IMPROVE Sharpe (best tight: {tighter.loc[tighter['sharpe'].idxmax(), 'stop_thresh']})")
    else:
        print(f"  • Tighter stops do NOT improve Sharpe vs -1.5%")

    if len(looser) and looser["sharpe"].max() > base_row["sharpe"]:
        best_loose = looser.loc[looser["sharpe"].idxmax()]
        print(f"  • Looser stops IMPROVE Sharpe (best loose: {best_loose['stop_thresh']}  "
              f"Sharpe={best_loose['sharpe']:.3f})")
    else:
        print(f"  • Looser stops do NOT improve Sharpe")

    no_stop = valid[valid["stop_thresh"] == "NONE"]
    if len(no_stop):
        print(f"  • No-stop Sharpe: {no_stop.iloc[0]['sharpe']:.3f}  "
              f"DD={no_stop.iloc[0]['max_drawdown_pct']:.2f}%")
        if no_stop.iloc[0]["sharpe"] > base_row["sharpe"]:
            print(f"    → Removing stops entirely IMPROVES Sharpe but worsens DD")
        else:
            print(f"    → Stops are net-positive even if not at optimal threshold")

    # Best trail pair
    if trail_df is not None and len(trail_df) and "sharpe" in trail_df.columns:
        bt = trail_df.loc[trail_df["sharpe"].idxmax()]
        print(f"\n  • Best trail pair: arm={bt['trail_arm']*100:.1f}%  "
              f"draw={bt['trail_draw']*100:.1f}%  "
              f"Sharpe={bt['sharpe']:.3f}")

    # Recommendation
    print(f"\n  ── Recommendation ──────────────────────────────────────────────")
    recommended_stop = best_sharpe["stop_thresh"]
    recommended_sharpe = best_sharpe["sharpe"]
    delta_sharpe = recommended_sharpe - base_row["sharpe"]
    delta_dollar = best_sharpe["dollar_pnl"] - base_row["dollar_pnl"]

    print(f"\n  OPTIMAL STOP LOSS: {recommended_stop}")
    print(f"  Sharpe: {base_row['sharpe']:.3f} → {recommended_sharpe:.3f}  ({delta_sharpe:+.3f})")
    print(f"  $ P&L:  ${base_row['dollar_pnl']:+,.0f} → ${best_sharpe['dollar_pnl']:+,.0f}  "
          f"(${delta_dollar:+,.0f})")
    print(f"  Stop-loss count: {int(base_row['n_stop_loss'])} → {int(best_sharpe['n_stop_loss'])}")
    print(f"  Stop-loss cost:  ${base_row['dollar_stop_loss']:+,.0f} → "
          f"${best_sharpe['dollar_stop_loss']:+,.0f}")
    print("=" * 74)


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Primary sweep
    primary_df, trade_logs = run_primary_sweep()

    # Find best stop for trail sweep
    valid = primary_df[primary_df["sharpe"].notna()] if "sharpe" in primary_df.columns else pd.DataFrame()
    # Exclude "NONE" from best stop determination (we want a real threshold)
    valid_real = valid[valid["stop_thresh"] != "NONE"]
    best_stop = float(valid_real.loc[valid_real["sharpe"].idxmax(), "stop_thresh"]) if len(valid_real) else -0.020

    # Trail sweep at best stop
    trail_df = run_trail_sweep(best_stop)

    # Recovery analysis
    _recovery_analysis(trade_logs, primary_df)

    # Full summary
    _print_summary(primary_df, trail_df)
    _print_verdict(primary_df, trail_df)
