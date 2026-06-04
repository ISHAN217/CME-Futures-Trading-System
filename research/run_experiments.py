"""
run_experiments.py — CWT chop filter experiment runner.

Runs experiments A, B, C, D and a sensitivity grid.
Each experiment temporarily patches config, runs the full pipeline,
and collects key metrics.

Usage:
    python run_experiments.py

Output:
    output/cwt_experiment_results.csv
    output/cwt_sensitivity_results.csv
    Console: final acceptance recommendation
"""

import logging
import sys
import itertools
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

# Silence pipeline logging during experiments
logging.basicConfig(stream=sys.stdout, level=logging.WARNING,
                    format="%(asctime)s %(levelname)-8s %(message)s")

import config
import main as pipeline

CAPITAL = 25_000


EXPERIMENTS = {
    "A_Baseline": {
        "USE_CWT_CHOP_FILTER": False,
        "CWT_APPLY_TO_MEAN_REVERT": False,
        "CWT_APPLY_TO_PULLBACK": False,
        "CWT_APPLY_TO_STAT_ARB": False,
        "CWT_APPLY_TO_TREND": False,
    },
    "B_CWT_MR_Only": {
        "USE_CWT_CHOP_FILTER": True,
        "CWT_APPLY_TO_MEAN_REVERT": True,
        "CWT_APPLY_TO_PULLBACK": False,
        "CWT_APPLY_TO_STAT_ARB": False,
        "CWT_APPLY_TO_TREND": False,
    },
    "C_CWT_MR_PB": {
        "USE_CWT_CHOP_FILTER": True,
        "CWT_APPLY_TO_MEAN_REVERT": True,
        "CWT_APPLY_TO_PULLBACK": True,
        "CWT_APPLY_TO_STAT_ARB": False,
        "CWT_APPLY_TO_TREND": False,
    },
    "D_CWT_MR_PB_SA": {
        "USE_CWT_CHOP_FILTER": True,
        "CWT_APPLY_TO_MEAN_REVERT": True,
        "CWT_APPLY_TO_PULLBACK": True,
        "CWT_APPLY_TO_STAT_ARB": True,
        "CWT_APPLY_TO_TREND": False,
    },
}


def _patch_config(overrides: dict):
    """Apply overrides to config module. Returns dict of original values."""
    original = {}
    for k, v in overrides.items():
        original[k] = getattr(config, k, None)
        setattr(config, k, v)
    return original


def _restore_config(original: dict):
    for k, v in original.items():
        setattr(config, k, v)


def _extract_metrics(result: dict, name: str) -> dict:
    stats = result.get("stats", {})
    tl = result.get("trade_log", pd.DataFrame())

    if tl is None or len(tl) == 0:
        return {"experiment": name}

    # Weekly P&L
    tl2 = tl.copy()
    tl2["entry_week"] = pd.to_datetime(tl2["entry_date"]).dt.to_period("W")
    weekly = tl2.groupby("entry_week")["pnl"].sum()
    weekly_win_rate = (weekly > 0).mean()
    worst_week_dollar = weekly.min() * CAPITAL

    # By strategy
    strat_pnl = {}
    for s in ["TREND", "PULLBACK", "MEAN_REVERT", "STAT_ARB"]:
        sub = tl[tl["strategy"] == s]
        strat_pnl[f"pnl_{s}"] = sub["pnl"].sum() * CAPITAL if len(sub) else 0.0

    # By exit reason
    er_pnl = {}
    for er in ["STOP_LOSS", "SHOCK_EXIT", "TRAIL_STOP", "FAST_EXIT", "MAX_HOLD_EXIT_TREND"]:
        sub = tl[tl["exit_reason"] == er] if "exit_reason" in tl.columns else pd.DataFrame()
        er_pnl[f"n_{er}"] = len(sub)

    # By regime
    reg_pnl = {}
    if "entry_regime" in tl.columns:
        for r in ["TREND", "CHOP", "TRANSITION", "SHOCK"]:
            sub = tl[tl["entry_regime"] == r]
            reg_pnl[f"pnl_regime_{r}"] = sub["pnl"].sum() * CAPITAL if len(sub) else 0.0

    # CWT label P&L
    cwt_pnl = {}
    if "cwt_chop_label" in tl.columns:
        for lbl in tl["cwt_chop_label"].unique():
            sub = tl[tl["cwt_chop_label"] == lbl]
            cwt_pnl[f"pnl_cwt_{lbl}"] = sub["pnl"].sum() * CAPITAL

    row = {
        "experiment":        name,
        "sharpe":            stats.get("sharpe_ratio",        float("nan")),
        "sortino":           stats.get("sortino",             float("nan")),
        "total_return_pct":  stats.get("total_return",        float("nan")) * 100 if stats.get("total_return") is not None else float("nan"),
        "annual_return_pct": stats.get("annualized_return_pct", float("nan")),
        "max_drawdown_pct":  stats.get("max_drawdown_pct",    float("nan")),
        "dollar_pnl":        stats.get("total_return",        0) * CAPITAL if stats.get("total_return") is not None else 0.0,
        "n_trades":          len(tl),
        "win_rate":          tl["win"].mean() if "win" in tl.columns else float("nan"),
        "profit_factor":     stats.get("profit_factor",       float("nan")),
        "avg_trade_pnl":     tl["pnl"].mean() * CAPITAL,
        "weekly_win_rate":   weekly_win_rate,
        "worst_week_dollar": worst_week_dollar,
    }
    row.update(strat_pnl)
    row.update(er_pnl)
    row.update(reg_pnl)
    row.update(cwt_pnl)
    return row


def run_all_experiments():
    print("\n" + "="*70)
    print("  CWT CHOP FILTER — EXPERIMENT RUNNER")
    print("="*70)
    results = []

    for name, overrides in EXPERIMENTS.items():
        print(f"\n  Running experiment: {name} ...")
        original = _patch_config(overrides)
        try:
            result = pipeline.run()
            metrics = _extract_metrics(result, name)
            results.append(metrics)
            sharpe  = metrics.get("sharpe",           "?")
            ret     = metrics.get("total_return_pct", "?")
            dd      = metrics.get("max_drawdown_pct", "?")
            trades  = metrics.get("n_trades",         "?")
            if isinstance(sharpe, float):
                print(f"  -> Sharpe={sharpe:.3f}  Return={ret:.2f}%  MaxDD={dd:.2f}%  Trades={trades}")
            else:
                print(f"  -> Sharpe={sharpe}  Return={ret}  MaxDD={dd}  Trades={trades}")
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            results.append({"experiment": name, "error": str(e)})
        finally:
            _restore_config(original)

    df_results = pd.DataFrame(results)
    out = Path(config.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out / "cwt_experiment_results.csv", index=False)
    print(f"\n  Saved cwt_experiment_results.csv")
    return df_results


def run_sensitivity():
    print("\n" + "="*70)
    print("  CWT SENSITIVITY TEST (entropy_high x compression_z)")
    print("="*70)

    entropy_vals     = [0.75, 0.80, 0.85]
    compression_vals = [-0.50, -0.75, -1.00]

    results = []
    base_overrides = {
        "USE_CWT_CHOP_FILTER": True,
        "CWT_APPLY_TO_MEAN_REVERT": True,
        "CWT_APPLY_TO_PULLBACK": True,
        "CWT_APPLY_TO_STAT_ARB": False,
        "CWT_APPLY_TO_TREND": False,
    }

    for entropy, compression in itertools.product(entropy_vals, compression_vals):
        overrides = {**base_overrides,
                     "CWT_ENTROPY_HIGH": entropy,
                     "CWT_ENERGY_COMPRESSION_Z": compression}
        name = f"entropy={entropy}_compZ={compression}"
        print(f"  Running: {name} ...")
        original = _patch_config(overrides)
        try:
            result = pipeline.run()
            m = _extract_metrics(result, name)
            m["CWT_ENTROPY_HIGH"] = entropy
            m["CWT_ENERGY_COMPRESSION_Z"] = compression
            results.append(m)
            sharpe = m.get("sharpe", "?")
            ret    = m.get("total_return_pct", "?")
            if isinstance(sharpe, float):
                print(f"  -> Sharpe={sharpe:.3f}  Return={ret:.2f}%")
            else:
                print(f"  -> Sharpe={sharpe}  Return={ret}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"experiment": name, "error": str(e),
                            "CWT_ENTROPY_HIGH": entropy, "CWT_ENERGY_COMPRESSION_Z": compression})
        finally:
            _restore_config(original)

    df_sens = pd.DataFrame(results)
    out = Path(config.OUTPUT_DIR)
    df_sens.to_csv(out / "cwt_sensitivity_results.csv", index=False)
    print(f"\n  Saved cwt_sensitivity_results.csv")
    return df_sens


def print_final_verdict(exp_results: pd.DataFrame, sens_results: pd.DataFrame):
    print("\n" + "="*70)
    print("  FINAL CWT ACCEPTANCE / REJECTION ANALYSIS")
    print("="*70)

    base_rows = exp_results[exp_results["experiment"] == "A_Baseline"]
    base = base_rows.iloc[0] if len(base_rows) else {}

    cwt_rows = exp_results[exp_results["experiment"] != "A_Baseline"]
    best_cwt_idx = cwt_rows["sharpe"].idxmax() if len(cwt_rows) and "sharpe" in cwt_rows.columns else None
    best_cwt = exp_results.loc[best_cwt_idx] if best_cwt_idx is not None else {}

    def cmp(q, baseline_val, cwt_val, higher_better=True):
        try:
            baseline_val = float(baseline_val)
            cwt_val      = float(cwt_val)
        except (TypeError, ValueError):
            return f"  {q}: DATA MISSING"
        if pd.isna(baseline_val) or pd.isna(cwt_val):
            return f"  {q}: DATA MISSING"
        delta = cwt_val - baseline_val
        improved = delta > 0 if higher_better else delta < 0
        symbol = "+" if improved else "-"
        return f"  [{symbol}] {q}: baseline={baseline_val:.4f}  best_cwt={cwt_val:.4f}  delta={delta:+.4f}"

    print(f"\n  Best CWT config: {best_cwt.get('experiment', '?')}")
    print()
    print(cmp("Q1. Improved overall Sharpe?",      base.get("sharpe", float("nan")),          best_cwt.get("sharpe", float("nan"))))
    print(cmp("Q2. Reduced max drawdown?",          base.get("max_drawdown_pct", float("nan")), best_cwt.get("max_drawdown_pct", float("nan")), higher_better=False))
    print(cmp("Q3. Reduced STOP_LOSS exits?",       base.get("n_STOP_LOSS", float("nan")),     best_cwt.get("n_STOP_LOSS", float("nan")),       higher_better=False))
    print(cmp("Q7. Improved PULLBACK P&L?",         base.get("pnl_PULLBACK", float("nan")),    best_cwt.get("pnl_PULLBACK", float("nan"))))
    print(cmp("Q6. Improved MEAN_REVERT P&L?",      base.get("pnl_MEAN_REVERT", float("nan")), best_cwt.get("pnl_MEAN_REVERT", float("nan"))))
    print(cmp("Q8. Did CWT hurt STAT_ARB?",         base.get("pnl_STAT_ARB", float("nan")),    best_cwt.get("pnl_STAT_ARB", float("nan")),      higher_better=True))

    # Sensitivity stability
    if sens_results is not None and "sharpe" in sens_results.columns:
        sharpe_vals  = sens_results["sharpe"].dropna()
        sharpe_std   = sharpe_vals.std()
        sharpe_range = sharpe_vals.max() - sharpe_vals.min()
        stable = sharpe_std < 0.05
        print(f"\n  Q11. Robust across parameters? sharpe_std={sharpe_std:.4f}  range={sharpe_range:.4f}  {'STABLE' if stable else 'UNSTABLE'}")

    # Trade frequency check
    base_trades = base.get("n_trades", float("nan"))
    cwt_trades  = best_cwt.get("n_trades", float("nan"))
    try:
        base_trades = float(base_trades)
        cwt_trades  = float(cwt_trades)
        if not pd.isna(base_trades) and not pd.isna(cwt_trades) and base_trades > 0:
            trade_reduction = (base_trades - cwt_trades) / base_trades * 100
            sharpe_delta = float(best_cwt.get("sharpe", 0)) - float(base.get("sharpe", 0))
            if trade_reduction > 20 and sharpe_delta < 0.02:
                print(f"\n  Q10. [-] CWT is mainly reducing frequency ({trade_reduction:.1f}% fewer trades) without clear Sharpe gain.")
            else:
                print(f"\n  Q10. Trade reduction: {trade_reduction:.1f}%  Sharpe delta: {sharpe_delta:+.4f}")
    except (TypeError, ValueError):
        pass

    # Final verdict
    try:
        sharpe_base = float(base.get("sharpe", 0.0) or 0.0)
        sharpe_best = float(best_cwt.get("sharpe", 0.0) or 0.0)
    except (TypeError, ValueError):
        sharpe_base = sharpe_best = 0.0
    sharpe_gain = sharpe_best - sharpe_base

    stable = False
    if sens_results is not None and "sharpe" in sens_results.columns:
        sharpe_vals = sens_results["sharpe"].dropna()
        stable = sharpe_vals.std() < 0.05 if len(sharpe_vals) > 1 else False

    print("\n" + "-"*70)
    if sharpe_gain > 0.03 and stable:
        verdict = "KEEP_CWT_FILTER"
        reason  = f"Sharpe improved by {sharpe_gain:.3f} and result is stable across parameters."
    elif sharpe_gain > 0.01:
        verdict = "CONTINUE_TESTING"
        reason  = f"Small Sharpe gain ({sharpe_gain:.3f}). Needs more regime-specific validation."
    elif sharpe_gain > -0.01:
        verdict = "KEEP_FOR_DIAGNOSTICS_ONLY"
        reason  = "No clear Sharpe improvement. CWT labels useful for research, not trading decisions."
    else:
        verdict = "REJECT_CWT_FILTER"
        reason  = f"CWT reduces Sharpe by {abs(sharpe_gain):.3f}. Adds complexity without benefit."

    print(f"\n  FINAL RECOMMENDATION: {verdict}")
    print(f"  Reason: {reason}")
    print("="*70)


if __name__ == "__main__":
    exp_results  = run_all_experiments()
    sens_results = run_sensitivity()
    print_final_verdict(exp_results, sens_results)
