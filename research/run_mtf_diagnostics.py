"""
run_mtf_diagnostics.py — Phases 9-10: MTF Structure Diagnostics + Inverse/Random Tests
========================================================================================

Phase 9: Bucket forward returns by every MTF label combination.
Phase 10: Inverse direction and randomized signal tests.

Input:
  - output/mtf/ES_MTF_features.parquet
  - output/mtf/NQ_MTF_features.parquet
  - Daily data from main system (via data_loader)

Output:
  - mtf_structure_forward_return_diagnostics.csv
  - mtf_structure_by_symbol.csv
  - mtf_structure_by_year.csv
  - mtf_structure_by_regime.csv
  - mtf_structure_baseline_state.csv
  - mtf_inverse_random_tests.csv
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_daily
from features    import compute_features
from regime_engine import compute_regime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = "/Users/ishanbhardwaj/cme_competition_system/output/mtf"
def out(name): return os.path.join(OUT_DIR, name)

N_RANDOM_SHUFFLES = 30
FORWARD_HORIZONS  = [1, 2, 4, 12, 24]   # in daily bars (approx 1d, 2d, 4d, 2wk, 4wk)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _load_mtf_features() -> pd.DataFrame:
    parts = []
    for sym in ["ES", "NQ"]:
        fp = out(f"{sym}_MTF_features.parquet")
        if os.path.exists(fp):
            df = pd.read_parquet(fp)
            parts.append(df)
        else:
            logger.warning("Missing %s_MTF_features.parquet", sym)
    if not parts:
        raise FileNotFoundError("No MTF feature files found. Run run_mtf_structure.py first.")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_daily_with_features() -> pd.DataFrame:
    """Load daily data with features and regime for both symbols."""
    df = load_daily()   # returns all symbols combined
    df = compute_features(df)
    df = compute_regime(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward return columns for each horizon."""
    df = df.copy().sort_values(["symbol","date"])
    for h in FORWARD_HORIZONS:
        df[f"fwd_ret_{h}d"] = (
            df.groupby("symbol")["ret_1d"]
            .transform(lambda s: s.shift(-h).rolling(h, min_periods=h).sum())
        )
    # Forward MFE and MAE over 4-bar horizon
    df["fwd_mfe_4d"] = df.groupby("symbol")["ret_1d"].transform(
        lambda s: s.shift(-1).rolling(4, min_periods=1).max()
    )
    df["fwd_mae_4d"] = df.groupby("symbol")["ret_1d"].transform(
        lambda s: s.shift(-1).rolling(4, min_periods=1).min()
    )
    return df


def _bucket_stats(grp: pd.DataFrame, label: str, bucket_val) -> dict:
    """Compute statistics for a bucket of rows."""
    n = len(grp)
    if n < 5:
        return None
    stats = {
        "bucket_col":   label,
        "bucket_val":   str(bucket_val),
        "n":            n,
    }
    for h in FORWARD_HORIZONS:
        col = f"fwd_ret_{h}d"
        if col not in grp.columns:
            continue
        vals = grp[col].dropna()
        if len(vals) < 3:
            continue
        stats[f"avg_ret_{h}d"]   = round(vals.mean(), 6)
        stats[f"hit_rate_{h}d"]  = round((vals > 0).mean(), 4)
        stats[f"std_ret_{h}d"]   = round(vals.std(), 6)
        stats[f"sharpe_{h}d"]    = round(vals.mean() / (vals.std() + 1e-9), 4)
    # MFE/MAE ratio
    mfe = grp["fwd_mfe_4d"].dropna().mean()
    mae = grp["fwd_mae_4d"].dropna().mean()
    stats["avg_mfe_4d"]    = round(mfe, 6)
    stats["avg_mae_4d"]    = round(mae, 6)
    stats["mfe_mae_ratio"] = round(abs(mfe / (mae + 1e-9)), 4)
    return stats


def bucket_by_column(merged: pd.DataFrame, col: str) -> list:
    rows = []
    for val, grp in merged.groupby(col):
        s = _bucket_stats(grp, col, val)
        if s:
            rows.append(s)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 9 — FORWARD RETURN DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────────────

def phase9_diagnostics(merged: pd.DataFrame):
    logger.info("=== PHASE 9: FORWARD RETURN DIAGNOSTICS ===")
    all_rows = []

    # Single label buckets
    for col in [
        "mtf_final_bias",
        "structure_1h_label", "structure_4h_label",
        "structure_12h_label","structure_24h_label",
    ]:
        if col in merged.columns:
            all_rows.extend(bucket_by_column(merged, col))

    # Quintile buckets
    for col in ["mtf_alignment_score","mtf_conflict_score",
                "mtf_bias_confidence","mtf_entry_timing_score"]:
        if col in merged.columns:
            merged[f"{col}_q"] = pd.qcut(merged[col], q=5, labels=False, duplicates="drop")
            all_rows.extend(bucket_by_column(merged, f"{col}_q"))

    # Combined label buckets
    combo_cols = [
        ("structure_4h_label","structure_12h_label"),
        ("structure_4h_label","structure_24h_label"),
        ("structure_12h_label","structure_24h_label"),
        ("structure_1h_label","structure_4h_label"),
        ("structure_1h_label","mtf_final_bias"),
    ]
    for c1, c2 in combo_cols:
        if c1 in merged.columns and c2 in merged.columns:
            combo_col = f"{c1}+{c2}"
            merged[combo_col] = merged[c1].astype(str) + " | " + merged[c2].astype(str)
            all_rows.extend(bucket_by_column(merged, combo_col))

    fwd_diag = pd.DataFrame([r for r in all_rows if r])
    fwd_diag.to_csv(out("mtf_structure_forward_return_diagnostics.csv"), index=False)
    logger.info("  Saved forward return diagnostics  (%d rows)", len(fwd_diag))

    # By symbol
    sym_rows = []
    for sym, g in merged.groupby("symbol"):
        for val, grp in g.groupby("mtf_final_bias"):
            s = _bucket_stats(grp, "mtf_final_bias", val)
            if s:
                s["symbol"] = sym
                sym_rows.append(s)
    pd.DataFrame(sym_rows).to_csv(out("mtf_structure_by_symbol.csv"), index=False)

    # By year
    merged["year"] = merged["date"].dt.year
    yr_rows = []
    for (yr, sym), g in merged.groupby(["year","symbol"]):
        for val, grp in g.groupby("mtf_final_bias"):
            s = _bucket_stats(grp, "mtf_final_bias", val)
            if s:
                s["year"] = yr; s["symbol"] = sym
                yr_rows.append(s)
    pd.DataFrame(yr_rows).to_csv(out("mtf_structure_by_year.csv"), index=False)

    # By regime
    if "regime" in merged.columns:
        reg_rows = []
        for (reg, sym), g in merged.groupby(["regime","symbol"]):
            for val, grp in g.groupby("mtf_final_bias"):
                s = _bucket_stats(grp, "mtf_final_bias", val)
                if s:
                    s["regime"] = reg; s["symbol"] = sym
                    reg_rows.append(s)
        pd.DataFrame(reg_rows).to_csv(out("mtf_structure_by_regime.csv"), index=False)
    logger.info("  Saved by_symbol, by_year, by_regime")

    # Print key diagnostics
    print("\n  KEY FORWARD RETURN DIAGNOSTICS (mtf_final_bias, 1-day horizon):")
    _print_key_stats(fwd_diag, "mtf_final_bias", "avg_ret_1d", "hit_rate_1d", "sharpe_1d")

    return fwd_diag


def _print_key_stats(df, bucket_col, *stat_cols):
    sub = df[df["bucket_col"]==bucket_col].sort_values("bucket_val")
    cols_to_show = ["bucket_val","n"] + [c for c in stat_cols if c in sub.columns]
    if len(sub):
        print(sub[cols_to_show].to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 10 — INVERSE AND RANDOM TESTS
# ──────────────────────────────────────────────────────────────────────────────

def _expectancy(merged: pd.DataFrame, bias_col: str, direction_map: dict) -> dict:
    """
    Compute average forward 1-day return when trading in `direction_map[bias]` direction.
    direction_map: {"BULLISH": +1, "BEARISH": -1, "NEUTRAL": 0, "NO_TRADE": 0}
    """
    merged = merged.copy()
    merged["signal"] = merged[bias_col].map(direction_map).fillna(0)
    merged["trade_ret"] = merged["signal"] * merged["fwd_ret_1d"]
    active = merged[merged["signal"] != 0]
    if len(active) < 5:
        return {"n": 0, "avg_ret": np.nan, "hit_rate": np.nan, "sharpe": np.nan}
    rets = active["trade_ret"].dropna()
    return {
        "n":        len(rets),
        "avg_ret":  round(rets.mean(), 6),
        "hit_rate": round((rets > 0).mean(), 4),
        "sharpe":   round(rets.mean() / (rets.std() + 1e-9), 4),
    }


def phase10_inverse_random(merged: pd.DataFrame):
    logger.info("=== PHASE 10: INVERSE AND RANDOM TESTS ===")
    bias_col  = "mtf_final_bias"
    rows      = []

    NORMAL_MAP  = {"BULLISH": +1, "BEARISH": -1, "NEUTRAL": 0, "NO_TRADE": 0}
    INVERSE_MAP = {"BULLISH": -1, "BEARISH": +1, "NEUTRAL": 0, "NO_TRADE": 0}

    # Normal
    normal = _expectancy(merged, bias_col, NORMAL_MAP)
    normal["test_type"] = "NORMAL"
    normal["shuffle_id"]= 0
    rows.append(normal)

    # Inverse
    inverse = _expectancy(merged, bias_col, INVERSE_MAP)
    inverse["test_type"] = "INVERSE"
    inverse["shuffle_id"]= 0
    rows.append(inverse)

    # Random shuffles — within symbol+year groups to preserve regime distribution
    rng = np.random.default_rng(42)
    rand_results = []
    for i in range(N_RANDOM_SHUFFLES):
        shuffled = merged.copy()
        for (sym, yr), g in shuffled.groupby(["symbol", "year"]):
            idx = g.index
            perm = rng.permutation(g[bias_col].values)
            shuffled.loc[idx, "bias_shuffled"] = perm
        r = _expectancy(shuffled, "bias_shuffled", NORMAL_MAP)
        r["test_type"]  = "RANDOM"
        r["shuffle_id"] = i + 1
        rand_results.append(r)
        rows.append(r)

    rand_df      = pd.DataFrame(rand_results)
    rand_avg_ret = rand_df["avg_ret"].dropna().mean()
    rand_p5_ret  = rand_df["avg_ret"].dropna().quantile(0.05)
    rand_p95_ret = rand_df["avg_ret"].dropna().quantile(0.95)

    inv_test    = pd.DataFrame(rows)
    inv_test.to_csv(out("mtf_inverse_random_tests.csv"), index=False)
    logger.info("  Saved mtf_inverse_random_tests.csv")

    # Print verdict
    n_ret  = normal.get("avg_ret", np.nan)
    i_ret  = inverse.get("avg_ret", np.nan)
    beats_random  = not np.isnan(n_ret) and (n_ret > rand_p95_ret)
    inverse_fails = not np.isnan(i_ret) and (i_ret < rand_avg_ret)

    print(f"\n  Normal     avg_ret={n_ret:.6f}  hit_rate={normal.get('hit_rate',np.nan):.4f}  sharpe={normal.get('sharpe',np.nan):.4f}")
    print(f"  Inverse    avg_ret={i_ret:.6f}  hit_rate={inverse.get('hit_rate',np.nan):.4f}  sharpe={inverse.get('sharpe',np.nan):.4f}")
    print(f"  Random p5={rand_p5_ret:.6f}  avg={rand_avg_ret:.6f}  p95={rand_p95_ret:.6f}")
    print(f"  Beats random:   {'YES' if beats_random else 'NO'}")
    print(f"  Inverse fails:  {'YES' if inverse_fails else 'NO'}")

    if beats_random and inverse_fails:
        signal_verdict = "MTF_SIGNAL_HAS_EDGE"
    elif beats_random:
        signal_verdict = "MTF_SIGNAL_MARGINAL_EDGE"
    elif not inverse_fails:
        signal_verdict = "MTF_SIGNAL_SYMMETRIC_NO_EDGE"
    else:
        signal_verdict = "MTF_SIGNAL_NO_EDGE"

    print(f"  Signal verdict: {signal_verdict}")
    return inv_test, signal_verdict


# ──────────────────────────────────────────────────────────────────────────────
# ANSWER THE 8 DIAGNOSTIC QUESTIONS
# ──────────────────────────────────────────────────────────────────────────────

def answer_diagnostic_questions(merged: pd.DataFrame, fwd_diag: pd.DataFrame,
                                 inv_test: pd.DataFrame, signal_verdict: str):
    print("\n" + "═"*70)
    print("  MTF DIAGNOSTIC QUESTIONS")
    print("═"*70)

    # Q1: Does MTF bullish bias predict positive forward returns?
    bull_rows = fwd_diag[
        (fwd_diag["bucket_col"]=="mtf_final_bias") &
        (fwd_diag["bucket_val"]=="BULLISH")
    ]
    bull_ret = bull_rows["avg_ret_1d"].values[0] if len(bull_rows) > 0 else np.nan
    print(f"\n  Q1. MTF bullish predicts positive returns?")
    print(f"      avg_ret_1d={bull_ret:.6f}  {'YES' if bull_ret > 0 else 'NO'}")

    # Q2: Does MTF bearish predict negative?
    bear_rows = fwd_diag[
        (fwd_diag["bucket_col"]=="mtf_final_bias") &
        (fwd_diag["bucket_val"]=="BEARISH")
    ]
    bear_ret = bear_rows["avg_ret_1d"].values[0] if len(bear_rows) > 0 else np.nan
    print(f"\n  Q2. MTF bearish predicts negative returns?")
    print(f"      avg_ret_1d={bear_ret:.6f}  {'YES' if bear_ret < 0 else 'NO'}")

    # Q3: Does high alignment outperform?
    hi_align = fwd_diag[
        (fwd_diag["bucket_col"]=="mtf_alignment_score_q") &
        (fwd_diag["bucket_val"]=="4")
    ]
    lo_align = fwd_diag[
        (fwd_diag["bucket_col"]=="mtf_alignment_score_q") &
        (fwd_diag["bucket_val"]=="0")
    ]
    hi_ret = hi_align["avg_ret_1d"].values[0] if len(hi_align) > 0 else np.nan
    lo_ret = lo_align["avg_ret_1d"].values[0] if len(lo_align) > 0 else np.nan
    print(f"\n  Q3. High alignment outperforms low alignment?")
    print(f"      high_align_ret={hi_ret:.6f}  low_align_ret={lo_ret:.6f}  "
          f"{'YES' if (not np.isnan(hi_ret) and not np.isnan(lo_ret) and hi_ret > lo_ret) else 'NO'}")

    # Q4: Does conflict predict poor returns?
    hi_conflict = fwd_diag[
        (fwd_diag["bucket_col"]=="mtf_conflict_score_q") &
        (fwd_diag["bucket_val"]=="4")
    ]
    lo_conflict = fwd_diag[
        (fwd_diag["bucket_col"]=="mtf_conflict_score_q") &
        (fwd_diag["bucket_val"]=="0")
    ]
    hi_c_ret = hi_conflict["sharpe_1d"].values[0] if len(hi_conflict) > 0 else np.nan
    lo_c_ret = lo_conflict["sharpe_1d"].values[0] if len(lo_conflict) > 0 else np.nan
    print(f"\n  Q4. High conflict = poor/noisy returns?")
    print(f"      high_conflict_sharpe={hi_c_ret:.4f}  low_conflict_sharpe={lo_c_ret:.4f}  "
          f"{'YES' if (not np.isnan(hi_c_ret) and not np.isnan(lo_c_ret) and abs(hi_c_ret) < abs(lo_c_ret)) else 'UNCLEAR'}")

    # Q5: Does 1H work only when HTF agrees?
    align_1h = fwd_diag[
        (fwd_diag["bucket_col"].str.contains("structure_1h_label")) &
        (fwd_diag["bucket_col"].str.contains("mtf_final_bias"))
    ]
    print(f"\n  Q5. 1H trigger works only when higher TF agrees?")
    print(f"      Combined label rows: {len(align_1h)} — examine mtf_structure_forward_return_diagnostics.csv")

    # Q6: Which TF contributes most?
    tf_sharpes = {}
    for tf in ["1h","4h","12h","24h"]:
        sub = fwd_diag[fwd_diag["bucket_col"]==f"structure_{tf}_label"]
        if len(sub) > 0:
            tf_sharpes[tf] = sub["sharpe_1d"].abs().max() if "sharpe_1d" in sub.columns else 0
    best_tf = max(tf_sharpes, key=tf_sharpes.get) if tf_sharpes else "unknown"
    print(f"\n  Q6. Which TF contributes most?  → {best_tf.upper()}")
    for tf, s in sorted(tf_sharpes.items(), key=lambda x: -x[1]):
        print(f"      {tf}: max_abs_sharpe={s:.4f}")

    # Q7: Does MTF add beyond daily TREND?
    print(f"\n  Q7. MTF adds beyond daily TREND? → examine correlation with TREND signal")
    print(f"      Signal verdict: {signal_verdict}")

    # Q8: Better when baseline flat?
    if "trend_signal" in merged.columns:
        flat  = merged[merged["trend_signal"]==0]
        trend = merged[merged["trend_signal"]!=0]
        flat_ret  = flat["fwd_ret_1d"].mean() if len(flat)>5 else np.nan
        trend_ret = trend["fwd_ret_1d"].mean() if len(trend)>5 else np.nan
        print(f"\n  Q8. Better when baseline flat?  flat_ret={flat_ret:.6f}  trend_ret={trend_ret:.6f}")
    else:
        print(f"\n  Q8. Better when baseline flat? → trend_signal column not in merged data")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  MTF DIAGNOSTICS  —  Phases 9-10")
    print("=" * 80)

    # Load MTF features
    logger.info("Loading MTF features ...")
    mtf = _load_mtf_features()
    logger.info("  %s rows  %s symbols", f"{len(mtf):,}", mtf["symbol"].nunique())

    # Load daily data with features
    logger.info("Loading daily features ...")
    try:
        daily = _load_daily_with_features()
        daily["date"] = pd.to_datetime(daily["date"])
    except Exception as e:
        logger.error("Could not load daily features: %s", e)
        daily = pd.DataFrame(columns=["date","symbol","ret_1d"])

    # Merge MTF with daily
    merged = mtf.merge(
        daily[["date","symbol","ret_1d"] +
               [c for c in ["regime","slope_20d","r2_20d","vol_regime","ewma_vol"]
                if c in daily.columns]],
        on=["date","symbol"],
        how="inner",
    )
    logger.info("  Merged: %s rows", f"{len(merged):,}")

    if len(merged) < 50:
        print("  ERROR: Insufficient merged data for diagnostics")
        sys.exit(1)

    # Add forward returns
    merged = _compute_forward_returns(merged)
    merged["year"] = merged["date"].dt.year

    # Phase 9
    fwd_diag = phase9_diagnostics(merged)

    # Phase 10
    inv_test, signal_verdict = phase10_inverse_random(merged)

    # Answer the 8 questions
    answer_diagnostic_questions(merged, fwd_diag, inv_test, signal_verdict)

    # Save baseline state analysis
    if "regime" in merged.columns:
        base_rows = []
        for (regime, bias), g in merged.groupby(["regime","mtf_final_bias"]):
            s = _bucket_stats(g, f"regime={regime}+bias={bias}", bias)
            if s:
                s["regime"] = regime
                s["mtf_final_bias"] = bias
                base_rows.append(s)
        pd.DataFrame(base_rows).to_csv(out("mtf_structure_baseline_state.csv"), index=False)

    print(f"\n  Signal verdict: {signal_verdict}")
    return signal_verdict


if __name__ == "__main__":
    verdict = main()
    # Exit 0 even if signal is weak — let experiments decide
    sys.exit(0)
