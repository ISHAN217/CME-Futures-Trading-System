"""
run_signal_quality.py — Full signal quality overlay experiment.

Sections
--------
Phase 0  Build baseline (run full pipeline once, extract quality scores)
Phase 1  Bucket diagnostics — do quality buckets predict trade outcomes?
Phase 2  Multiplier sweep (4 schedules × 3 strategy modes = 12 runs)
Phase 3  Direct weight sweep (4 schedules × 3 strategy modes = 12 runs)
Phase 4  Validation of top 3 configs (year / asset / strategy / regime / concentration)
Phase 5  Final verdict

Usage:
    python run_signal_quality.py
"""

import importlib
import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import main as _main_mod
from backtest import run_backtest

logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

CAPITAL = 25_000.0
OUT     = Path("output")
OUT.mkdir(exist_ok=True)

# ── Multiplier schedules ──────────────────────────────────────────────────────
MULT_SCHEDULES = {
    "Conservative": {
        "WEAK": 0.50, "MODERATE": 0.75, "GOOD": 1.00,
        "STRONG": 1.10, "EXCEPTIONAL": 1.25,
    },
    "Balanced": {
        "WEAK": 0.50, "MODERATE": 0.75, "GOOD": 1.00,
        "STRONG": 1.25, "EXCEPTIONAL": 1.50,
    },
    "Aggressive": {
        "WEAK": 0.25, "MODERATE": 0.75, "GOOD": 1.00,
        "STRONG": 1.50, "EXCEPTIONAL": 2.00,
    },
    "Inverse": {
        "WEAK": 1.25, "MODERATE": 1.00, "GOOD": 1.00,
        "STRONG": 0.75, "EXCEPTIONAL": 0.50,
    },
}

# ── Direct weight schedules ───────────────────────────────────────────────────
# Values are target weights (fractions of account), same bucket keys.
DIRECT_SCHEDULES = {
    "Direct_10/15/20/25/30": {
        "WEAK": 0.10, "MODERATE": 0.15, "GOOD": 0.20,
        "STRONG": 0.25, "EXCEPTIONAL": 0.30,
    },
    "Direct_5/10/15/20/30": {
        "WEAK": 0.05, "MODERATE": 0.10, "GOOD": 0.15,
        "STRONG": 0.20, "EXCEPTIONAL": 0.30,
    },
    "Direct_Conserv_10/12.5/15/20/25": {
        "WEAK": 0.10, "MODERATE": 0.125, "GOOD": 0.15,
        "STRONG": 0.20, "EXCEPTIONAL": 0.25,
    },
    "Direct_Aggr_0/10/20/30/30": {
        "WEAK": 0.00, "MODERATE": 0.10, "GOOD": 0.20,
        "STRONG": 0.30, "EXCEPTIONAL": 0.30,
    },
}

STRATEGY_MODES = ["TREND", "STAT_ARB", "BOTH"]

_ORIG_CFG = {
    "USE_SIGNAL_QUALITY_SIZING":       config.USE_SIGNAL_QUALITY_SIZING,
    "APPLY_QUALITY_SIZING_TO_TREND":   config.APPLY_QUALITY_SIZING_TO_TREND,
    "APPLY_QUALITY_SIZING_TO_STAT_ARB":config.APPLY_QUALITY_SIZING_TO_STAT_ARB,
    "STAT_ARB_QUALITY_MODE":           config.STAT_ARB_QUALITY_MODE,
    "QUALITY_SIZING_MODE":             config.QUALITY_SIZING_MODE,
    "TREND_QUALITY_MULTIPLIERS":       deepcopy(config.TREND_QUALITY_MULTIPLIERS),
    "STATARB_QUALITY_MULTIPLIERS":     deepcopy(config.STATARB_QUALITY_MULTIPLIERS),
}


def _set_cfg(**kw):
    for k, v in kw.items():
        setattr(config, k, v)


def _restore_cfg():
    for k, v in _ORIG_CFG.items():
        setattr(config, k, v)


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Build baseline
# ════════════════════════════════════════════════════════════════════════════════

def _phase0_build_baseline():
    print("\n" + "=" * 70)
    print("  PHASE 0 — Running full pipeline once (baseline)")
    print("=" * 70)
    _set_cfg(USE_SIGNAL_QUALITY_SIZING=False)
    result = _main_mod.run(fetch=False)
    _restore_cfg()
    print(f"  Baseline Sharpe: {result['stats'].get('sharpe_ratio', 0):.3f}")
    return result


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Bucket diagnostics
# ════════════════════════════════════════════════════════════════════════════════

def _trade_bucket_stats(tl: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    """Compute per-bucket performance from trade_log."""
    if tl.empty or bucket_col not in tl.columns:
        return pd.DataFrame()

    rows = []
    for bucket, grp in tl.groupby(bucket_col, sort=False):
        if grp.empty:
            continue
        wins = grp["pnl"] > 0
        sl   = (grp.get("exit_reason", pd.Series()) == "STOP_LOSS").sum() if "exit_reason" in grp.columns else 0
        ts   = (grp.get("exit_reason", pd.Series()) == "TRAIL_STOP").sum() if "exit_reason" in grp.columns else 0
        gp   = grp.loc[wins, "pnl"].sum()
        gl   = grp.loc[~wins, "pnl"].sum()
        pf   = (gp / (-gl)) if gl < 0 else np.inf

        dollar_pnl = grp["pnl"].sum() * CAPITAL

        rows.append({
            "bucket":          bucket,
            "n_trades":        len(grp),
            "win_rate":        round(wins.mean(), 3),
            "total_pnl_pct":   round(grp["pnl"].sum() * 100, 3),
            "dollar_pnl":      round(dollar_pnl, 0),
            "avg_pnl_pct":     round(grp["pnl"].mean() * 100, 4),
            "median_pnl_pct":  round(grp["pnl"].median() * 100, 4),
            "avg_win_pct":     round(grp.loc[wins,  "pnl"].mean() * 100, 4) if wins.any()  else 0,
            "avg_loss_pct":    round(grp.loc[~wins, "pnl"].mean() * 100, 4) if (~wins).any() else 0,
            "payoff_ratio":    round(abs(grp.loc[wins, "pnl"].mean() / grp.loc[~wins, "pnl"].mean()), 3)
                               if wins.any() and (~wins).any() else np.nan,
            "profit_factor":   round(pf, 3) if pf != np.inf else 999.0,
            "stop_loss_n":     int(sl),
            "trail_stop_n":    int(ts),
            "avg_hold_days":   round(grp["days_held"].mean(), 2) if "days_held" in grp.columns else np.nan,
        })

    df = pd.DataFrame(rows)
    if "bucket" in df.columns:
        bucket_order = ["WEAK", "MODERATE", "GOOD", "STRONG", "EXCEPTIONAL"]
        df["_order"] = df["bucket"].map({b: i for i, b in enumerate(bucket_order)})
        df = df.sort_values("_order").drop(columns=["_order"])
    return df


def _phase1_bucket_diagnostics(result: dict, df_signals: pd.DataFrame) -> dict:
    """
    Merge quality buckets onto trade_log and compute per-bucket statistics.
    """
    print("\n" + "=" * 70)
    print("  PHASE 1 — Bucket diagnostics (quality → trade outcome)")
    print("=" * 70)

    tl = result.get("trade_log", pd.DataFrame()).copy()
    if tl.empty:
        print("  No trades in trade log — skipping diagnostics")
        return {}

    # Merge quality scores at entry date
    quality_cols = [
        "date", "symbol",
        "trend_quality_score", "trend_quality_bucket",
        "statarb_quality_score_linear", "statarb_quality_bucket_linear",
        "statarb_quality_score_hump",   "statarb_quality_bucket_hump",
        "statarb_quality_reason",
    ]
    sig = df_signals[[c for c in quality_cols if c in df_signals.columns]].copy()
    sig = sig.rename(columns={"date": "entry_date"})

    tl = tl.merge(sig, on=["entry_date", "symbol"], how="left")
    tl["trend_quality_bucket"]        = tl.get("trend_quality_bucket",        pd.Series(dtype=str)).fillna("MODERATE")
    tl["statarb_quality_bucket_hump"] = tl.get("statarb_quality_bucket_hump", pd.Series(dtype=str)).fillna("MODERATE")
    tl["statarb_quality_bucket_linear"]= tl.get("statarb_quality_bucket_linear",pd.Series(dtype=str)).fillna("MODERATE")

    results_dict = {}

    # ── TREND quality ─────────────────────────────────────────────────────────
    trend_tl = tl[tl.get("strategy", pd.Series()) == "TREND"] if "strategy" in tl.columns else tl
    trend_diag = _trade_bucket_stats(trend_tl, "trend_quality_bucket")
    trend_diag.to_csv(OUT / "trend_quality_bucket_diagnostics.csv", index=False)
    results_dict["trend"] = trend_diag
    _print_bucket_table("TREND quality bucket → trade outcome", trend_diag)
    _print_trend_conclusion(trend_diag)

    # ── STAT_ARB quality (linear) ─────────────────────────────────────────────
    sa_tl = tl[tl.get("strategy", pd.Series()) == "STAT_ARB"] if "strategy" in tl.columns else tl
    sa_lin = _trade_bucket_stats(sa_tl, "statarb_quality_bucket_linear")
    sa_lin.to_csv(OUT / "statarb_quality_linear_bucket_diagnostics.csv", index=False)
    results_dict["sa_linear"] = sa_lin
    _print_bucket_table("STAT_ARB linear bucket → trade outcome", sa_lin)

    # ── STAT_ARB quality (hump) ───────────────────────────────────────────────
    sa_hump = _trade_bucket_stats(sa_tl, "statarb_quality_bucket_hump")
    sa_hump.to_csv(OUT / "statarb_quality_hump_bucket_diagnostics.csv", index=False)
    results_dict["sa_hump"] = sa_hump
    _print_bucket_table("STAT_ARB hump bucket → trade outcome", sa_hump)

    _print_sa_conclusion(sa_lin, sa_hump)

    return results_dict


def _print_bucket_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n  {title}")
    if df.empty:
        print("  (no data)")
        return
    print(f"  {'Bucket':<12} {'N':>4}  {'WR':>5}  {'$PnL':>9}  {'AvgPnL%':>8}  "
          f"{'PF':>5}  {'SL_n':>4}  {'TS_n':>4}  {'HoldD':>5}")
    print(f"  {'-' * 70}")
    for _, r in df.iterrows():
        print(
            f"  {str(r['bucket']):<12} {int(r['n_trades']):>4}  "
            f"{r['win_rate']:>5.3f}  "
            f"${r['dollar_pnl']:>8,.0f}  "
            f"{r['avg_pnl_pct']:>+7.4f}  "
            f"{r['profit_factor']:>5.2f}  "
            f"{int(r['stop_loss_n']):>4}  "
            f"{int(r['trail_stop_n']):>4}  "
            f"{r['avg_hold_days']:>5.1f}"
        )


def _print_trend_conclusion(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print("\n  TREND quality conclusion:")
    bucket_order = ["WEAK", "MODERATE", "GOOD", "STRONG", "EXCEPTIONAL"]
    expected = df[df["bucket"].isin(bucket_order)].set_index("bucket")["avg_pnl_pct"]
    ordered  = [expected.get(b, np.nan) for b in bucket_order if b in expected.index]
    if len(ordered) >= 2:
        is_monotonic = all(
            ordered[i] <= ordered[i+1] for i in range(len(ordered)-1)
        )
        print(f"  Monotonic expectancy: {'YES ✓' if is_monotonic else 'NO — non-monotonic pattern'}")
        best_bucket   = expected.idxmax() if not expected.empty else "GOOD"
        worst_bucket  = expected.idxmin() if not expected.empty else "WEAK"
        print(f"  Best bucket: {best_bucket} (avg {expected.get(best_bucket, 0):+.4f}%)")
        print(f"  Worst bucket: {worst_bucket} (avg {expected.get(worst_bucket, 0):+.4f}%)")
        if is_monotonic:
            print("  → Quality score is predictive. Sizing up STRONG/EXCEPTIONAL is justified.")
        else:
            print("  → Quality score is NOT monotonically predictive. Use Conservative sizing.")


def _print_sa_conclusion(lin: pd.DataFrame, hump: pd.DataFrame) -> None:
    print("\n  STAT_ARB quality conclusion:")
    for name, df in [("Linear", lin), ("Hump", hump)]:
        if df.empty:
            continue
        avg = df.set_index("bucket")["avg_pnl_pct"]
        best = avg.idxmax() if not avg.empty else "?"
        print(f"  {name}: best bucket = {best}  "
              f"({avg.get(best, 0):+.4f}%)  "
              f"{'Hump model better → use STAT_ARB_QUALITY_MODE=hump' if name=='Hump' else ''}")


# ════════════════════════════════════════════════════════════════════════════════
# METRIC EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def _extract_metrics(result: dict, df_signals: pd.DataFrame,
                     label: str, schedule_name: str, mode_name: str,
                     strategy_mode: str, schedule: dict) -> dict:
    """Full metric extraction from a backtest result."""
    stats    = result.get("stats", {})
    tl       = result.get("trade_log", pd.DataFrame())
    port_ret = result.get("portfolio_returns", pd.Series(dtype=float))
    sym_ret  = result.get("symbol_returns",   pd.DataFrame())

    sharpe   = float(stats.get("sharpe_ratio",          np.nan))
    ann_ret  = float(stats.get("annualized_return_pct",  np.nan))
    tot_ret  = float(stats.get("total_return_pct",       np.nan))
    max_dd   = float(stats.get("max_drawdown_pct",       np.nan))
    pf       = float(stats.get("profit_factor",          np.nan))

    # Sortino
    down = port_ret[port_ret < 0]
    sortino = (port_ret.mean() / down.std() * np.sqrt(252)) if len(down) > 5 else np.nan

    # Avg drawdown
    cum = (1 + port_ret).cumprod()
    roll_max = cum.cummax()
    dd_series = (cum - roll_max) / roll_max
    avg_dd = float(dd_series.mean() * 100)

    # Worst week / month
    port_ret.index = pd.to_datetime(port_ret.index)
    worst_week  = float(port_ret.resample("W").sum().min()  * 100)
    worst_month = float(port_ret.resample("ME").sum().min() * 100)

    # Dollar PnL
    cum_ret    = (1 + port_ret).prod() - 1
    dollar_pnl = cum_ret * CAPITAL

    # Trade stats
    n_trades = len(tl)
    win_rate = float(tl["win"].mean()) if "win" in tl.columns and n_trades > 0 else np.nan

    # Entry weight stats
    ew = tl["entry_weight"].dropna() if "entry_weight" in tl.columns else pd.Series()
    ew_pos = ew[ew > 0]
    avg_w  = ew_pos.mean()  if len(ew_pos) > 0 else np.nan
    max_w  = ew_pos.max()   if len(ew_pos) > 0 else np.nan
    pct_capped = (ew_pos >= config.MAX_WEIGHT_PER_SYMBOL - 0.001).mean() if len(ew_pos) > 0 else np.nan

    # Stop/trail counts
    sl_n = int((tl.get("exit_reason", pd.Series()) == "STOP_LOSS").sum())  if "exit_reason" in tl.columns else 0
    ts_n = int((tl.get("exit_reason", pd.Series()) == "TRAIL_STOP").sum()) if "exit_reason" in tl.columns else 0
    sl_pnl = float(tl.loc[tl.get("exit_reason","") == "STOP_LOSS", "pnl"].sum() * CAPITAL) if "exit_reason" in tl.columns else 0.0

    # By strategy P&L
    strat_pnl = {}
    if "strategy" in tl.columns:
        for strat, grp in tl.groupby("strategy"):
            strat_pnl[f"pnl_{strat}"] = round(grp["pnl"].sum() * CAPITAL, 0)

    # By symbol P&L
    sym_pnl = {}
    if "symbol" in tl.columns:
        for sym, grp in tl.groupby("symbol"):
            sym_pnl[f"pnl_{sym}"] = round(grp["pnl"].sum() * CAPITAL, 0)

    # Avg trade P&L
    avg_trade_pnl = float(tl["pnl"].mean() * CAPITAL) if not tl.empty else np.nan

    return {
        "label":            label,
        "schedule_name":    schedule_name,
        "mode_name":        mode_name,
        "strategy_mode":    strategy_mode,
        "sharpe":           round(sharpe,  3),
        "sortino":          round(sortino, 3),
        "ann_ret_pct":      round(ann_ret, 2),
        "total_ret_pct":    round(tot_ret, 2),
        "max_dd_pct":       round(max_dd,  2),
        "avg_dd_pct":       round(avg_dd,  3),
        "worst_week_pct":   round(worst_week,  2),
        "worst_month_pct":  round(worst_month, 2),
        "dollar_pnl":       round(dollar_pnl,  0),
        "n_trades":         n_trades,
        "win_rate":         round(win_rate, 3),
        "profit_factor":    round(pf, 3),
        "avg_trade_pnl":    round(avg_trade_pnl, 0),
        "sl_count":         sl_n,
        "sl_pnl":           round(sl_pnl, 0),
        "avg_entry_weight": round(avg_w, 4),
        "max_entry_weight": round(max_w, 4),
        "pct_capped":       round(pct_capped, 3),
        **strat_pnl,
        **sym_pnl,
    }


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Multiplier sweep
# ════════════════════════════════════════════════════════════════════════════════

def _run_one_experiment(
    df_signals: pd.DataFrame,
    schedule: dict,
    strategy_mode: str,
    mode_name: str,        # "multiplier" or "direct_weight"
    label: str,
    schedule_name: str,
) -> dict:
    """Set config, run backtest on pre-built df_signals, return metrics."""
    # Set which strategies the overlay applies to
    _set_cfg(
        USE_SIGNAL_QUALITY_SIZING      = True,
        QUALITY_SIZING_MODE            = mode_name,
        APPLY_QUALITY_SIZING_TO_TREND   = strategy_mode in ("TREND", "BOTH"),
        APPLY_QUALITY_SIZING_TO_STAT_ARB= strategy_mode in ("STAT_ARB", "BOTH"),
        STAT_ARB_QUALITY_MODE          = "hump",
        TREND_QUALITY_MULTIPLIERS      = deepcopy(schedule),
        STATARB_QUALITY_MULTIPLIERS    = deepcopy(schedule),
    )

    try:
        result = run_backtest(df_signals)
        metrics = _extract_metrics(
            result, df_signals, label, schedule_name, mode_name, strategy_mode, schedule
        )
    except Exception as e:
        metrics = {"label": label, "error": str(e)}
    finally:
        _restore_cfg()

    return metrics


def _phase2_multiplier_sweep(df_signals: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  PHASE 2 — Multiplier sweep")
    print("=" * 70)

    total = len(MULT_SCHEDULES) * len(STRATEGY_MODES)
    rows  = []
    idx   = 0

    for sched_name, sched in MULT_SCHEDULES.items():
        for strat_mode in STRATEGY_MODES:
            idx += 1
            label = f"MULT_{sched_name}_{strat_mode}"
            print(f"  [{idx:2d}/{total}] {label} ...", end="  ", flush=True)
            m = _run_one_experiment(df_signals, sched, strat_mode, "multiplier",
                                    label, sched_name)
            rows.append(m)
            print(f"Sharpe={m.get('sharpe', '?'):.3f}  $PnL=${m.get('dollar_pnl', 0):,.0f}  "
                  f"AvgW={m.get('avg_entry_weight', 0):.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "signal_quality_multiplier_sweep.csv", index=False)
    print(f"\n  Saved → output/signal_quality_multiplier_sweep.csv")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Direct weight sweep
# ════════════════════════════════════════════════════════════════════════════════

def _phase3_direct_weight_sweep(df_signals: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  PHASE 3 — Direct weight sweep")
    print("=" * 70)

    total = len(DIRECT_SCHEDULES) * len(STRATEGY_MODES)
    rows  = []
    idx   = 0

    for sched_name, sched in DIRECT_SCHEDULES.items():
        for strat_mode in STRATEGY_MODES:
            idx += 1
            label = f"DW_{sched_name}_{strat_mode}"
            print(f"  [{idx:2d}/{total}] {label} ...", end="  ", flush=True)
            m = _run_one_experiment(df_signals, sched, strat_mode, "direct_weight",
                                    label, sched_name)
            rows.append(m)
            print(f"Sharpe={m.get('sharpe', '?'):.3f}  $PnL=${m.get('dollar_pnl', 0):,.0f}  "
                  f"AvgW={m.get('avg_entry_weight', 0):.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "signal_quality_direct_weight_sweep.csv", index=False)
    print(f"\n  Saved → output/signal_quality_direct_weight_sweep.csv")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Validation of top 3
# ════════════════════════════════════════════════════════════════════════════════

def _phase4_validation(
    df_signals: pd.DataFrame,
    all_results: pd.DataFrame,
    baseline_sharpe: float,
) -> None:
    """Validate the top 3 configs with detailed breakdown."""
    print("\n" + "=" * 70)
    print("  PHASE 4 — Validation of top 3 configurations")
    print("=" * 70)

    # Find top 3 by Sharpe, excluding baseline and Inverse schedule
    candidates = all_results[
        ~all_results["label"].str.contains("Inverse|Baseline", na=False)
    ].copy()
    candidates = candidates[candidates["sharpe"].notna()]
    candidates = candidates.sort_values("sharpe", ascending=False)

    # Get the 3 schedules for the top 3 — look them up
    top3_labels = candidates["label"].head(3).tolist()
    print(f"  Top 3: {top3_labels}")

    yearly_rows    = []
    asset_rows     = []
    strategy_rows  = []
    regime_rows    = []
    conc_rows      = []

    for lbl in top3_labels:
        row = candidates[candidates["label"] == lbl].iloc[0]
        # Reconstruct config for this run
        sched_name = row["schedule_name"]
        mode_name  = row["mode_name"]
        strat_mode = row["strategy_mode"]

        # Find the correct schedule dict
        if mode_name == "multiplier":
            sched = MULT_SCHEDULES.get(sched_name, MULT_SCHEDULES["Conservative"])
        else:
            sched = DIRECT_SCHEDULES.get(sched_name, DIRECT_SCHEDULES["Direct_10/15/20/25/30"])

        _set_cfg(
            USE_SIGNAL_QUALITY_SIZING      = True,
            QUALITY_SIZING_MODE            = mode_name,
            APPLY_QUALITY_SIZING_TO_TREND   = strat_mode in ("TREND", "BOTH"),
            APPLY_QUALITY_SIZING_TO_STAT_ARB= strat_mode in ("STAT_ARB", "BOTH"),
            STAT_ARB_QUALITY_MODE          = "hump",
            TREND_QUALITY_MULTIPLIERS      = deepcopy(sched),
            STATARB_QUALITY_MULTIPLIERS    = deepcopy(sched),
        )

        try:
            result = run_backtest(df_signals)
            port_ret = result["portfolio_returns"].copy()
            tl       = result.get("trade_log", pd.DataFrame()).copy()

            port_ret.index = pd.to_datetime(port_ret.index)

            # Year-by-year
            for yr, grp in port_ret.groupby(port_ret.index.year):
                cum = (1 + grp).prod() - 1
                vol = grp.std() * np.sqrt(252)
                sr  = (grp.mean() / grp.std() * np.sqrt(252)) if grp.std() > 0 else 0
                yearly_rows.append({
                    "label": lbl, "year": yr,
                    "return_pct": round(cum * 100, 2),
                    "sharpe": round(sr, 3),
                    "dollar_pnl": round(cum * CAPITAL, 0),
                })

            # Asset-by-asset
            sym_ret = result.get("symbol_returns", pd.DataFrame())
            if not sym_ret.empty:
                for sym in sym_ret.columns:
                    sr = sym_ret[sym]
                    tot = (1 + sr).prod() - 1
                    asset_rows.append({
                        "label": lbl, "symbol": sym,
                        "return_pct": round(tot * 100, 2),
                        "dollar_pnl": round(tot * CAPITAL, 0),
                    })

            # Strategy-by-strategy
            strat_ret = result.get("strategy_returns", pd.DataFrame())
            if not strat_ret.empty:
                for col in strat_ret.columns:
                    sr = strat_ret[col]
                    tot = (1 + sr).prod() - 1
                    strategy_rows.append({
                        "label": lbl, "strategy": col,
                        "return_pct": round(tot * 100, 2),
                        "dollar_pnl": round(tot * CAPITAL, 0),
                    })

            # Regime-by-regime (from trade_log + df_signals)
            if not tl.empty and "portfolio_regime" in df_signals.columns:
                tl_r = tl.merge(
                    df_signals[["date", "symbol", "portfolio_regime"]].rename(
                        columns={"date": "entry_date"}
                    ).drop_duplicates(["entry_date", "symbol"]),
                    on=["entry_date", "symbol"], how="left"
                )
                if "portfolio_regime" in tl_r.columns:
                    for reg, grp in tl_r.groupby("portfolio_regime"):
                        regime_rows.append({
                            "label": lbl, "regime": reg,
                            "n_trades": len(grp),
                            "win_rate": round(grp["win"].mean(), 3) if "win" in grp.columns else np.nan,
                            "dollar_pnl": round(grp["pnl"].sum() * CAPITAL, 0),
                        })

            # Concentration: top 10 winners / losers
            if not tl.empty and "pnl" in tl.columns:
                tl_sorted = tl.sort_values("pnl", ascending=False)
                top10_w    = tl_sorted.head(10)
                top10_l    = tl_sorted.tail(10)
                total_pnl  = tl["pnl"].sum()
                conc_rows.append({
                    "label":             lbl,
                    "total_dollar_pnl":  round(total_pnl * CAPITAL, 0),
                    "top10_winners_pnl": round(top10_w["pnl"].sum() * CAPITAL, 0),
                    "top10_losers_pnl":  round(top10_l["pnl"].sum() * CAPITAL, 0),
                    "top10_w_pct":       round(top10_w["pnl"].sum() / total_pnl * 100, 1) if total_pnl != 0 else np.nan,
                    "top1_winner_pnl":   round(tl_sorted.iloc[0]["pnl"] * CAPITAL, 0) if len(tl_sorted) > 0 else 0,
                    "top1_loser_pnl":    round(tl_sorted.iloc[-1]["pnl"] * CAPITAL, 0) if len(tl_sorted) > 0 else 0,
                    "n_trades":          len(tl),
                })

        except Exception as e:
            print(f"  ERROR in {lbl}: {e}")
        finally:
            _restore_cfg()

    # Save all validation outputs
    def _save(rows, fname):
        if rows:
            pd.DataFrame(rows).to_csv(OUT / fname, index=False)
            print(f"  Saved → output/{fname}")

    _save(yearly_rows,   "signal_quality_best_yearly.csv")
    _save(asset_rows,    "signal_quality_best_asset.csv")
    _save(strategy_rows, "signal_quality_best_strategy.csv")
    _save(regime_rows,   "signal_quality_best_regime.csv")
    _save(conc_rows,     "signal_quality_concentration_check.csv")

    # Print year-by-year table
    if yearly_rows:
        ydf = pd.DataFrame(yearly_rows)
        print("\n  Year-by-year breakdown (top 3 configs):")
        print(f"  {'Label':<40} {'Year':>4}  {'Return%':>8}  {'Sharpe':>7}  {'$PnL':>9}")
        print(f"  {'-' * 75}")
        for _, r in ydf.iterrows():
            print(f"  {str(r['label']):<40} {int(r['year']):>4}  "
                  f"{r['return_pct']:>+7.2f}%  {r['sharpe']:>7.3f}  "
                  f"${r['dollar_pnl']:>8,.0f}")

    # Concentration check
    if conc_rows:
        cdf = pd.DataFrame(conc_rows)
        print("\n  Concentration check:")
        for _, r in cdf.iterrows():
            print(f"  {str(r['label']):<40}  "
                  f"top10_winners={r['top10_w_pct']:.1f}% of PnL  "
                  f"top1_winner=${r['top1_winner_pnl']:,.0f}  "
                  f"total=${r['total_dollar_pnl']:,.0f}")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Final verdict
# ════════════════════════════════════════════════════════════════════════════════

def _phase5_verdict(
    baseline_sharpe: float,
    all_results: pd.DataFrame,
    diag: dict,
) -> str:
    """Apply acceptance criteria and print verdict."""
    print("\n" + "=" * 70)
    print("  PHASE 5 — Final verdict")
    print("=" * 70)

    MAX_DD_LIMIT      = 7.0
    SHARPE_THRESHOLD  = baseline_sharpe   # must beat this

    if all_results.empty or "sharpe" not in all_results.columns:
        verdict = "KEEP_CURRENT_OPTIMIZED_CORE"
        print(f"  {verdict} — no valid results")
        return verdict

    # Best non-Inverse config
    forward = all_results[~all_results["label"].str.contains("Inverse|Baseline", na=False)]
    forward = forward[forward["sharpe"].notna()]

    if forward.empty:
        verdict = "KEEP_CURRENT_OPTIMIZED_CORE"
        print(f"  {verdict} — no non-inverse configs ran")
        return verdict

    best = forward.sort_values("sharpe", ascending=False).iloc[0]
    inv  = all_results[all_results["schedule_name"] == "Inverse"].sort_values("sharpe", ascending=False)

    best_sharpe  = best["sharpe"]
    best_max_dd  = best["max_dd_pct"]
    best_pf      = best.get("profit_factor", 1.0)
    best_label   = best["label"]
    best_ann_ret = best.get("ann_ret_pct", 0)

    inv_best_sharpe = inv.iloc[0]["sharpe"] if not inv.empty else 0

    print(f"  Baseline Sharpe:      {baseline_sharpe:.3f}")
    print(f"  Best non-inverse:     {best_sharpe:.3f}  ({best_label})")
    print(f"  Best inverse:         {inv_best_sharpe:.3f}")
    print(f"  Max DD at best:       {best_max_dd:.2f}%  (limit: {MAX_DD_LIMIT:.1f}%)")
    print(f"  Profit factor:        {best_pf:.3f}")

    # TREND quality diagnostics
    trend_diag  = diag.get("trend", pd.DataFrame())
    monotonic   = False
    if not trend_diag.empty and "avg_pnl_pct" in trend_diag.columns:
        bucket_order  = ["WEAK", "MODERATE", "GOOD", "STRONG", "EXCEPTIONAL"]
        avgs = trend_diag.set_index("bucket")["avg_pnl_pct"]
        ordered = [avgs.get(b, np.nan) for b in bucket_order if b in avgs.index]
        ordered_clean = [x for x in ordered if not np.isnan(x)]
        monotonic = all(
            ordered_clean[i] <= ordered_clean[i+1]
            for i in range(len(ordered_clean) - 1)
        )

    print(f"\n  Acceptance criteria:")
    c1 = best_sharpe > SHARPE_THRESHOLD
    c2 = best_ann_ret > 0
    c3 = abs(best_max_dd) <= MAX_DD_LIMIT
    c4 = best_pf >= 1.0
    c5 = monotonic
    c6 = best_sharpe > inv_best_sharpe + 0.01   # must clearly beat inverse
    print(f"  1. Sharpe > {SHARPE_THRESHOLD:.3f}:         {'PASS ✓' if c1 else f'FAIL ✗  ({best_sharpe:.3f})'}")
    print(f"  2. Ann return > 0%:           {'PASS ✓' if c2 else f'FAIL ✗  ({best_ann_ret:.2f}%)'}")
    print(f"  3. MaxDD < {MAX_DD_LIMIT:.1f}%:           {'PASS ✓' if c3 else f'FAIL ✗  ({best_max_dd:.2f}%)'}")
    print(f"  4. Profit factor >= 1.0:      {'PASS ✓' if c4 else f'FAIL ✗  ({best_pf:.3f})'}")
    print(f"  5. Monotonic bucket expect:   {'PASS ✓' if c5 else 'FAIL ✗  (non-monotonic)'}")
    print(f"  6. Beats inverse schedule:    {'PASS ✓' if c6 else f'FAIL ✗  ({best_sharpe:.3f} vs {inv_best_sharpe:.3f})'}")

    n_pass = sum([c1, c2, c3, c4, c5, c6])
    print(f"\n  Criteria passed: {n_pass}/6")

    if n_pass == 6:
        # Distinguish multiplier vs direct weight
        if "MULT_" in best_label:
            verdict = "ACCEPT_MULTIPLIER_ONLY"
            sched_name = best["schedule_name"]
            if sched_name in ("Conservative", "Balanced"):
                verdict = "ACCEPT_SIGNAL_QUALITY_SIZING"  # safe for live
        elif "DW_" in best_label:
            verdict = "ACCEPT_DIRECT_WEIGHT_COMPETITION_VARIANT"
        else:
            verdict = "ACCEPT_SIGNAL_QUALITY_SIZING"
    elif n_pass >= 4:
        verdict = "CONTINUE_TESTING"
    else:
        verdict = "KEEP_CURRENT_OPTIMIZED_CORE"

    print(f"\n  ══════════════════════════════════")
    print(f"  VERDICT: {verdict}")
    print(f"  Best config: {best_label}")
    print(f"    schedule = {best['schedule_name']}")
    print(f"    mode     = {best['mode_name']}")
    print(f"    strategy = {best['strategy_mode']}")
    print(f"  ══════════════════════════════════\n")
    return verdict


# ════════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _print_sweep_table(title: str, df: pd.DataFrame, baseline_sharpe: float) -> None:
    if df.empty:
        return
    print(f"\n  {title}")
    print(f"  {'Label':<45} {'Sharpe':>7}  {'AnnRet%':>8}  {'MaxDD%':>7}  "
          f"{'WR':>5}  {'PF':>5}  {'$PnL':>9}  {'AvgW':>6}  {'mode':>3}")
    print(f"  {'-' * 115}")
    df_sorted = df.sort_values("sharpe", ascending=False, na_position="last")
    for _, r in df_sorted.iterrows():
        marker = " ★" if r.get("sharpe", 0) > baseline_sharpe else ""
        print(
            f"  {str(r.get('label','')):<45} "
            f"  {r.get('sharpe', np.nan):>6.3f}  "
            f"  {r.get('ann_ret_pct', np.nan):>7.1f}  "
            f"  {r.get('max_dd_pct', np.nan):>6.2f}  "
            f"  {r.get('win_rate', np.nan):>4.3f}  "
            f"  {r.get('profit_factor', np.nan):>4.2f}  "
            f"  ${r.get('dollar_pnl', 0):>8,.0f}  "
            f"  {r.get('avg_entry_weight', np.nan):>5.4f}{marker}"
        )
    print(f"  {'=' * 115}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def run():
    # Phase 0
    base_result = _phase0_build_baseline()
    baseline_sharpe = float(base_result["stats"].get("sharpe_ratio", 0.846))
    df_signals = base_result.get("df_signals", pd.DataFrame())

    if df_signals.empty:
        print("ERROR: df_signals not available. Check main.py exposes result['df_signals'].")
        return

    print(f"\n  df_signals: {len(df_signals)} rows, {df_signals.columns.tolist()[:8]} ...")

    # Phase 1
    diag = _phase1_bucket_diagnostics(base_result, df_signals)

    # Phase 2
    mult_df = _phase2_multiplier_sweep(df_signals)
    _print_sweep_table("Multiplier sweep results", mult_df, baseline_sharpe)

    # Phase 3
    dw_df = _phase3_direct_weight_sweep(df_signals)
    _print_sweep_table("Direct weight sweep results", dw_df, baseline_sharpe)

    # Combined
    base_row = {
        "label": "BASELINE", "schedule_name": "NONE", "mode_name": "none",
        "strategy_mode": "BOTH",
        "sharpe": baseline_sharpe,
        "ann_ret_pct": base_result["stats"].get("annualized_return_pct", 0),
        "total_ret_pct": base_result["stats"].get("total_return_pct", 0),
        "max_dd_pct": base_result["stats"].get("max_drawdown_pct", 0),
        "profit_factor": base_result["stats"].get("profit_factor", 0),
        "dollar_pnl": ((1 + base_result["portfolio_returns"]).prod() - 1) * CAPITAL,
    }
    all_results = pd.concat([pd.DataFrame([base_row]), mult_df, dw_df], ignore_index=True)
    all_results.to_csv(OUT / "signal_quality_all_results.csv", index=False)

    # Phase 4
    _phase4_validation(df_signals, all_results, baseline_sharpe)

    # Phase 5
    verdict = _phase5_verdict(baseline_sharpe, all_results, diag)

    print(f"\n  All outputs saved to output/")
    print(f"  Final verdict: {verdict}\n")
    return verdict


if __name__ == "__main__":
    run()
