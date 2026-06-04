"""
run_trend_state_diagnostics.py — Phases 11-12: Feature diagnostics for trend_state_engine.
===========================================================================================

Computes:
  - Forward return buckets for every key feature (quintile analysis)
  - Per-state performance summary (count, fwd returns, MFE, MAE, hit rate)
  - Saves diagnostic CSVs to output/trend_state/

Outputs:
  trend_state_feature_diagnostics.csv
  trend_state_forward_returns.csv
  trend_state_by_symbol.csv
  trend_state_by_year.csv
  trend_state_performance_diagnostics.csv
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import config
from data_loader        import load_daily, load_4h
from features           import compute_features
from momentum_engine    import compute_momentum
from weekly_bias        import compute_weekly_bias
from regime_engine      import compute_regime
from volatility_engine  import compute_volatility_features
from intraday_4h_engine import compute_4h_features, aggregate_4h_to_daily, merge_4h_into_daily
from mean_reversion     import compute_mean_reversion
from stat_arb           import compute_stat_arb
from sentiment_layer    import compute_sentiment
from signal_engine      import compute_signals
from signal_quality_engine          import compute_signal_quality
from trend_continuation_edge_engine import compute_trend_continuation_edge
from trend_state_engine import compute_trend_state_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = "/Users/ishanbhardwaj/cme_competition_system/output/trend_state"
os.makedirs(OUT_DIR, exist_ok=True)
def out(name): return os.path.join(OUT_DIR, name)

FEATURE_BUCKETS = [
    "buyer_pressure_score",
    "seller_pressure_score",
    "trend_efficiency_20",
    "healthy_pullback_score",
    "trend_breaking_pullback_score",
    "breakout_quality_score",
    "failed_breakout_score",
    "bull_exhaustion_score",
    "bear_exhaustion_score",
    "cross_market_divergence",
    "trend_opportunity_score",
    "trend_risk_score",
    "structure_progress_score",
    "trend_cleanliness_score",
    "volume_divergence_score",
    "up_volume_ratio",
    "pullback_depth_pct",
    "pullback_holds_prior_swing_flag",
    "consecutive_up_bars",
    "bars_since_last_pullback",
]

TREND_STATES = [
    "BULL_IMPULSE", "BULL_PULLBACK", "BULL_RESET_CONFIRMED",
    "BULL_EXHAUSTION", "BULL_BREAKDOWN_WARNING",
    "BEAR_IMPULSE", "BEAR_RALLY", "BEAR_RESET_CONFIRMED",
    "BEAR_EXHAUSTION", "BEAR_BREAKDOWN_WARNING",
    "RANGE", "CHAOTIC", "UNCLEAR",
]


def build_full_df() -> pd.DataFrame:
    """Full accepted pipeline + trend state features."""
    logger.info("Loading full pipeline ...")
    df       = load_daily()
    df_4h_raw = load_4h()

    df = compute_features(df)
    df = compute_momentum(df)
    df = compute_weekly_bias(df)
    df = compute_regime(df)
    df = compute_volatility_features(df)

    if config.USE_CWT_CHOP_FILTER:
        from cwt_chop_filter import compute_cwt_chop_features
        df = compute_cwt_chop_features(df)
    else:
        for col, val in [("cwt_chop_label","CWT_UNCLEAR"),("cwt_trade_allowed",1),
                         ("cwt_size_multiplier",1.0),("cwt_confidence",0.0),
                         ("cwt_total_energy",0.0),("cwt_entropy",0.5),("cwt_energy_z",0.0),
                         ("cwt_compression_flag",0),("cwt_expansion_flag",0),
                         ("cwt_high_energy_ratio",1/3),("cwt_mid_energy_ratio",1/3),
                         ("cwt_slow_energy_ratio",1/3)]:
            if col not in df.columns:
                df[col] = val

    if df_4h_raw is not None:
        df_4h_feat  = compute_4h_features(df_4h_raw)
        df_4h_daily = aggregate_4h_to_daily(df_4h_feat)
    else:
        df_4h_daily = None
    df = merge_4h_into_daily(df, df_4h_daily)

    df = compute_mean_reversion(df)
    df = compute_stat_arb(df)
    df = compute_sentiment(df)
    df = compute_signals(df)
    df = compute_signal_quality(df)
    df = compute_trend_continuation_edge(df)

    logger.info("Computing trend state features ...")
    df = compute_trend_state_features(df)

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Forward return proxies (no lookahead — shift signal, not return)
    for sym, g in df.groupby("symbol"):
        mask = df["symbol"] == sym
        ret  = df.loc[mask, "ret_1d"]
        df.loc[mask, "fwd_ret_1d"]  = ret.shift(-1).values
        df.loc[mask, "fwd_ret_5d"]  = ret.rolling(5).sum().shift(-5).values
        df.loc[mask, "fwd_ret_20d"] = ret.rolling(20).sum().shift(-20).values

    logger.info("Pipeline complete: %d rows", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def _quintile_stats(df: pd.DataFrame, feat: str, fwd: str) -> pd.DataFrame:
    """Quintile breakdown of forward returns by feature value."""
    if feat not in df.columns or fwd not in df.columns:
        return pd.DataFrame()

    sub = df[[feat, fwd, "symbol", "year", "trend_state", "final_direction"]].dropna()
    if len(sub) < 50:
        return pd.DataFrame()

    sub = sub.copy()
    try:
        sub["quintile"] = pd.qcut(sub[feat], q=5, labels=["Q1","Q2","Q3","Q4","Q5"],
                                  duplicates="drop")
    except Exception:
        return pd.DataFrame()

    rows = []
    for q, g in sub.groupby("quintile", observed=True):
        fwd_vals = g[fwd].values
        rows.append({
            "feature":          feat,
            "fwd_horizon":      fwd,
            "quintile":         str(q),
            "n":                len(g),
            "feat_mean":        g[feat].mean(),
            "avg_fwd_ret":      fwd_vals.mean(),
            "med_fwd_ret":      np.median(fwd_vals),
            "hit_rate":         (fwd_vals > 0).mean(),
            "fwd_ret_std":      fwd_vals.std(),
            "sharpe_proxy":     fwd_vals.mean() / (fwd_vals.std() + 1e-9) * np.sqrt(252),
            "bull_trend_pct":   (g["final_direction"] == "LONG").mean(),
        })
    return pd.DataFrame(rows)


def run_feature_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Running feature diagnostics ...")
    all_rows = []
    for feat in FEATURE_BUCKETS:
        for fwd in ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d"]:
            rows = _quintile_stats(df, feat, fwd)
            if not rows.empty:
                all_rows.append(rows)
    if not all_rows:
        return pd.DataFrame()
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(out("trend_state_feature_diagnostics.csv"), index=False)
    logger.info("  Saved trend_state_feature_diagnostics.csv (%d rows)", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD RETURN BY STATE
# ─────────────────────────────────────────────────────────────────────────────

def run_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing forward returns by trend_state ...")
    rows = []
    for state in TREND_STATES:
        g = df[df["trend_state"] == state]
        if len(g) < 5:
            continue
        for fwd in ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d"]:
            v = g[fwd].dropna().values
            if len(v) < 5:
                continue
            rows.append({
                "trend_state":  state,
                "fwd_horizon":  fwd,
                "n":            len(v),
                "avg_ret":      v.mean(),
                "med_ret":      np.median(v),
                "std_ret":      v.std(),
                "hit_rate":     (v > 0).mean(),
                "sharpe":       v.mean() / (v.std() + 1e-9) * np.sqrt(252),
                "mfe_approx":   np.percentile(v, 75),
                "mae_approx":   np.percentile(v, 25),
            })
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result.to_csv(out("trend_state_forward_returns.csv"), index=False)
    logger.info("  Saved trend_state_forward_returns.csv")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BY SYMBOL / YEAR
# ─────────────────────────────────────────────────────────────────────────────

def run_by_symbol_year(df: pd.DataFrame) -> None:
    logger.info("Computing by-symbol and by-year breakdowns ...")

    # By symbol
    sym_rows = []
    for sym, g in df.groupby("symbol"):
        for state in TREND_STATES:
            s = g[g["trend_state"] == state]
            if len(s) < 3:
                continue
            v1 = s["fwd_ret_1d"].dropna().values
            v5 = s["fwd_ret_5d"].dropna().values
            sym_rows.append({
                "symbol":     sym,
                "trend_state":state,
                "n":          len(s),
                "avg_1d":     v1.mean() if len(v1) else np.nan,
                "avg_5d":     v5.mean() if len(v5) else np.nan,
                "hit_1d":     (v1 > 0).mean() if len(v1) else np.nan,
                "bull_pct":   (s["final_direction"] == "LONG").mean(),
            })
    pd.DataFrame(sym_rows).to_csv(out("trend_state_by_symbol.csv"), index=False)

    # By year
    yr_rows = []
    for yr, g in df.groupby("year"):
        state_dist = g["trend_state"].value_counts().to_dict()
        trend_long = (g["final_direction"] == "LONG").sum()
        v1 = g["fwd_ret_1d"].dropna().values
        yr_rows.append({
            "year":        yr,
            "n_rows":      len(g),
            "trend_long":  trend_long,
            "avg_fwd_1d":  v1.mean() if len(v1) else np.nan,
            "hit_1d":      (v1 > 0).mean() if len(v1) else np.nan,
            "bull_impulse_pct": state_dist.get("BULL_IMPULSE", 0) / max(len(g), 1),
            "bull_exh_pct":     state_dist.get("BULL_EXHAUSTION", 0) / max(len(g), 1),
            "range_pct":        state_dist.get("RANGE", 0) / max(len(g), 1),
            "chaotic_pct":      state_dist.get("CHAOTIC", 0) / max(len(g), 1),
        })
    pd.DataFrame(yr_rows).to_csv(out("trend_state_by_year.csv"), index=False)
    logger.info("  Saved by-symbol and by-year CSVs")


# ─────────────────────────────────────────────────────────────────────────────
# PER-STATE PERFORMANCE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def run_state_performance(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing per-state performance diagnostics ...")
    rows = []

    for state in TREND_STATES:
        g = df[df["trend_state"] == state]
        n = len(g)
        if n < 3:
            continue

        v1  = g["fwd_ret_1d"].dropna().values
        v5  = g["fwd_ret_5d"].dropna().values
        v20 = g["fwd_ret_20d"].dropna().values

        # Baseline trade activity in this state
        long_rows  = g[g["final_direction"] == "LONG"]
        trend_pct  = (g["final_direction"] == "LONG").mean()
        high_conv  = g.get("high_conviction", pd.Series(False, index=g.index)).sum()

        rows.append({
            "trend_state":      state,
            "n_obs":            n,
            "pct_of_total":     n / max(len(df), 1),
            # Forward returns
            "avg_ret_1d":       v1.mean()  if len(v1)  else np.nan,
            "avg_ret_5d":       v5.mean()  if len(v5)  else np.nan,
            "avg_ret_20d":      v20.mean() if len(v20) else np.nan,
            "hit_rate_1d":      (v1 > 0).mean() if len(v1) else np.nan,
            "hit_rate_5d":      (v5 > 0).mean() if len(v5) else np.nan,
            "sharpe_1d":        (v1.mean() / (v1.std() + 1e-9) * np.sqrt(252)) if len(v1) > 5 else np.nan,
            # MFE / MAE proxies
            "fwd_mfe_1d":       np.percentile(v1, 75) if len(v1) else np.nan,
            "fwd_mae_1d":       np.percentile(v1, 25) if len(v1) else np.nan,
            "fwd_ret_std_1d":   v1.std() if len(v1) else np.nan,
            # Baseline system activity in this state
            "baseline_long_pct": trend_pct,
            "high_conviction_pct": high_conv / max(n, 1),
            # Feature medians for this state
            "med_bull_exh":     g.get("bull_exhaustion_score", pd.Series(dtype=float)).median(),
            "med_bp":           g.get("buyer_pressure_score",  pd.Series(dtype=float)).median(),
            "med_efficiency":   g.get("trend_efficiency_20",   pd.Series(dtype=float)).median(),
            "med_opportunity":  g.get("trend_opportunity_score", pd.Series(dtype=float)).median(),
            "med_risk":         g.get("trend_risk_score",      pd.Series(dtype=float)).median(),
            # State stability (how often does it transition?)
            "state_confidence_mean": g.get("trend_state_confidence", pd.Series(dtype=float)).mean(),
        })

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("n_obs", ascending=False)
    result.to_csv(out("trend_state_performance_diagnostics.csv"), index=False)
    logger.info("  Saved trend_state_performance_diagnostics.csv")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTOUT
# ─────────────────────────────────────────────────────────────────────────────

def _print_feature_summary(diag_df: pd.DataFrame) -> dict:
    """Print top/bottom quintile comparison and return predictive feature dict."""
    if diag_df.empty:
        return {}

    print("\n  TOP PREDICTIVE FEATURES (Q5 vs Q1 avg_fwd_ret, horizon=1d):")
    sub = diag_df[diag_df["fwd_horizon"] == "fwd_ret_1d"]

    diffs = {}
    for feat, g in sub.groupby("feature"):
        q1 = g[g["quintile"] == "Q1"]["avg_fwd_ret"].values
        q5 = g[g["quintile"] == "Q5"]["avg_fwd_ret"].values
        if len(q1) > 0 and len(q5) > 0:
            diffs[feat] = float(q5[0]) - float(q1[0])

    for feat, diff in sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        direction = "MONOTONE_UP" if diff > 0 else "INVERTED"
        print(f"    {feat:45s}  Δ(Q5-Q1)={diff*100:+.3f}%  [{direction}]")

    return diffs


def _print_state_summary(perf_df: pd.DataFrame) -> None:
    if perf_df.empty:
        return

    print("\n  TREND STATE PERFORMANCE SUMMARY:")
    print(f"  {'State':<30}  {'N':>5}  {'%':>5}  {'AvgRet1d':>9}  {'Hit%':>6}  "
          f"{'Sharpe1d':>9}  {'BLong%':>7}  {'BullExh':>8}")
    print("  " + "-" * 90)

    for _, row in perf_df.sort_values("n_obs", ascending=False).iterrows():
        print(f"  {row['trend_state']:<30}  "
              f"{int(row['n_obs']):>5}  "
              f"{row['pct_of_total']*100:>4.1f}%  "
              f"{row.get('avg_ret_1d', 0)*100:>+8.3f}%  "
              f"{row.get('hit_rate_1d', 0)*100:>5.1f}%  "
              f"{row.get('sharpe_1d', 0):>9.3f}  "
              f"{row.get('baseline_long_pct', 0)*100:>6.1f}%  "
              f"{row.get('med_bull_exh', 0):>8.3f}")

    print()
    print("  KEY QUESTIONS:")

    # Q1: Good states for new entries?
    good_entry = perf_df[perf_df["avg_ret_1d"] > 0].sort_values("sharpe_1d", ascending=False)
    print(f"  Q1  Good for new entries?  {', '.join(good_entry['trend_state'].head(3).tolist())}")

    # Q2: States to avoid
    bad_entry = perf_df[perf_df["avg_ret_1d"] <= 0].sort_values("avg_ret_1d")
    print(f"  Q2  Avoid new entries?     {', '.join(bad_entry['trend_state'].head(3).tolist())}")

    # Q7: Does exhaustion predict worse fwd returns?
    exh_states = perf_df[perf_df["trend_state"].str.contains("EXHAUSTION")]
    non_exh    = perf_df[perf_df["trend_state"].isin(["BULL_IMPULSE","BULL_RESET_CONFIRMED"])]
    if not exh_states.empty and not non_exh.empty:
        exh_ret   = exh_states["avg_ret_1d"].mean()
        non_ret   = non_exh["avg_ret_1d"].mean()
        supported = exh_ret < non_ret
        print(f"  Q7  Exhaustion worse than impulse?  exh={exh_ret*100:+.3f}%  "
              f"impulse={non_ret*100:+.3f}%  SUPPORTED={'YES' if supported else 'NO'}")

    # Q6: BULL_RESET vs BULL_IMPULSE
    reset_row   = perf_df[perf_df["trend_state"] == "BULL_RESET_CONFIRMED"]
    impulse_row = perf_df[perf_df["trend_state"] == "BULL_IMPULSE"]
    if not reset_row.empty and not impulse_row.empty:
        reset_ret   = float(reset_row.iloc[0]["avg_ret_1d"])
        impulse_ret = float(impulse_row.iloc[0]["avg_ret_1d"])
        print(f"  Q6  BULL_RESET_CONFIRMED vs BULL_IMPULSE:  "
              f"reset={reset_ret*100:+.3f}%  impulse={impulse_ret*100:+.3f}%  "
              f"{'RESET_BETTER' if reset_ret > impulse_ret else 'IMPULSE_BETTER'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  TREND STATE DIAGNOSTICS  —  Phases 11-12")
    print("=" * 80)

    df = build_full_df()

    # Save full enriched df for experiments
    df.to_parquet(out("trend_state_full_df.parquet"), index=False)
    logger.info("Saved trend_state_full_df.parquet")

    # State distribution
    print("\n  TREND STATE DISTRIBUTION:")
    dist = df.groupby(["symbol","trend_state"]).size().unstack(fill_value=0)
    print(dist.to_string())

    # Feature diagnostics
    diag_df = run_feature_diagnostics(df)
    feat_diffs = _print_feature_summary(diag_df)

    # Forward returns by state
    fwd_df = run_forward_returns(df)

    # By symbol/year
    run_by_symbol_year(df)

    # Per-state performance
    perf_df = run_state_performance(df)
    _print_state_summary(perf_df)

    # Save feature ranking
    if feat_diffs:
        rank_df = pd.DataFrame([
            {"feature": k, "q5_minus_q1_1d": v, "abs_diff": abs(v),
             "direction": "MONOTONE_UP" if v > 0 else "INVERTED"}
            for k, v in sorted(feat_diffs.items(), key=lambda x: abs(x[1]), reverse=True)
        ])
        rank_df.to_csv(out("trend_state_feature_ranking.csv"), index=False)

    print(f"\n  All diagnostic outputs saved to: {OUT_DIR}")
    print("  PHASE 11-12 COMPLETE")

    return df, diag_df, perf_df


if __name__ == "__main__":
    main()
