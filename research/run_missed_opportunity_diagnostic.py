"""
run_missed_opportunity_diagnostic.py

Missed Opportunity Diagnostic for the CME Competition TREND Engine.
Diagnoses whether the TREND engine is too selective and missing profitable
opportunities. Pure diagnostics — no production config changes.

Sections:
    1. Candidate Signal Inventory
    2. Forward Return Audit of Rejected Signals
    3. Weak Trend Bucket Diagnostic
    4. Missed Trend Leg Audit
    5. Simulated Weak-Trend Entry Tests
    6. Rejection Rule Attribution
    8. Final Report (10 diagnostic questions + recommendation)

Usage:
    python run_missed_opportunity_diagnostic.py
"""

import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import config
from backtest import run_backtest
from data_loader import load_daily, load_4h
from features import compute_features
from intraday_4h_engine import compute_4h_features, aggregate_4h_to_daily, merge_4h_into_daily
from mean_reversion import compute_mean_reversion
from momentum_engine import compute_momentum
from regime_engine import compute_regime
from sentiment_layer import compute_sentiment
from signal_engine import compute_signals
from signal_quality_engine import compute_signal_quality
from stat_arb import compute_stat_arb
from trend_continuation_edge_engine import compute_trend_continuation_edge
from volatility_engine import compute_volatility_features
from weekly_bias import compute_weekly_bias

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BASELINE_SHARPE = 0.882
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "opportunity_diag")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helper
# ─────────────────────────────────────────────────────────────────────────────

def build_base_df() -> pd.DataFrame:
    """Run the full pipeline and return the enriched daily DataFrame."""
    logger.info("Building base DataFrame — running full pipeline …")
    df        = load_daily()
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
        for col, val in [
            ("cwt_chop_label",       "CWT_UNCLEAR"),
            ("cwt_trade_allowed",    1),
            ("cwt_size_multiplier",  1.0),
            ("cwt_confidence",       0.0),
            ("cwt_total_energy",     0.0),
            ("cwt_entropy",          0.5),
            ("cwt_energy_z",         0.0),
            ("cwt_compression_flag", 0),
            ("cwt_expansion_flag",   0),
            ("cwt_high_energy_ratio",1/3),
            ("cwt_mid_energy_ratio", 1/3),
            ("cwt_slow_energy_ratio",1/3),
        ]:
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
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Pipeline complete — %d rows, %d symbols", len(df), df["symbol"].nunique())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Candidate Signal Inventory
# ─────────────────────────────────────────────────────────────────────────────

def classify_candidate_class(row) -> str:
    dir_   = str(row.get("final_direction", "FLAT") or "FLAT")
    strat  = str(row.get("strategy_used",   "NONE") or "NONE")
    reason = str(row.get("signal_reason",   "")     or "")
    regime = str(row.get("portfolio_regime","CHOP") or "CHOP")
    bias   = str(row.get("weekly_bias",     "NEUTRAL") or "NEUTRAL")
    days   = int(row.get("days_held",       0) or 0)
    sent   = str(row.get("portfolio_sentiment_flag", "NORMAL") or "NORMAL")

    if dir_ == "LONG":
        if strat == "TREND":
            return "TAKEN_TREND" if days <= 1 else "TAKEN_TREND_HOLD"
        if strat == "STAT_ARB":
            return "TAKEN_STAT_ARB"
        return f"TAKEN_{strat}"

    if "SHOCK" in reason:                                         return "REJECTED_SHOCK"
    if "TRANSITION" in reason:                                    return "REJECTED_TRANSITION"
    if "HIGH_RISK" in reason:                                     return "REJECTED_SENTIMENT"
    if "LOW_QUALITY" in reason:                                   return "REJECTED_QUALITY_GATE"
    if "H4_WAIT" in reason and "HOLD" not in reason:             return "REJECTED_H4_WAIT"
    if "NO_BIAS" in reason:                                       return "REJECTED_NO_WEEKLY_BIAS"
    if "PULLBACK_DISABLED" in reason:                             return "REJECTED_PULLBACK_DISABLED"
    if "MR_DISABLED" in reason or "MR_FLAT" in reason or "MR_BLOCKED" in reason:
        return "FLAT_MR_CHOP"
    if "SA_ES_HEDGE" in reason:                                   return "FLAT_SA_ES_DISABLED"
    if regime == "CHOP":                                          return "FLAT_CHOP_REGIME"
    if regime == "TREND" and bias == "LONG":                      return "POTENTIAL_TREND_UNEXPLAINED"
    return "BASELINE_FLAT"


def compute_late_entry_pct_20d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-symbol rolling 20-day price location.
    late_entry_pct_20d = (close - min_20d) / (max_20d - min_20d).
    No lookahead: uses only past data in the rolling window.
    """
    df = df.copy()
    df["late_entry_pct_20d"] = np.nan
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date")
        roll_min = g["close"].rolling(20, min_periods=5).min()
        roll_max = g["close"].rolling(20, min_periods=5).max()
        rng = roll_max - roll_min
        pct = np.where(rng > 1e-6, (g["close"].values - roll_min.values) / rng.values, 0.5)
        df.loc[g.index, "late_entry_pct_20d"] = pct
    return df


def compute_nq_confirms(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ES row, nq_confirms = True if NQ has final_direction == 'LONG' on the same date.
    """
    df = df.copy()
    df["nq_confirms"] = False
    nq_long_dates = set(
        df[(df["symbol"] == "NQ") & (df["final_direction"] == "LONG")]["date"].unique()
    )
    mask_es = df["symbol"] == "ES"
    df.loc[mask_es, "nq_confirms"] = df.loc[mask_es, "date"].isin(nq_long_dates)
    return df


def section1_candidate_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candidate_class, rejection_reason, is_potential_trend_raw,
    late_entry_pct_20d, nq_confirms. Save inventory CSV and print counts.
    """
    logger.info("=== Section 1: Candidate Signal Inventory ===")

    df = compute_late_entry_pct_20d(df)
    df = compute_nq_confirms(df)

    # Candidate class
    df["candidate_class"] = df.apply(classify_candidate_class, axis=1)

    # Raw potential trend flag
    df["is_potential_trend_raw"] = (
        (df["portfolio_regime"] == "TREND") &
        (df["weekly_bias"] == "LONG")
    )

    # rejection_reason convenience column (same as signal_reason for rejects)
    df["rejection_reason"] = df.apply(
        lambda r: r["signal_reason"] if r["candidate_class"].startswith("REJECTED_") else "",
        axis=1,
    )

    # Print count table
    count_tbl = (
        df.groupby(["symbol", "candidate_class"])
        .size()
        .reset_index(name="count")
        .sort_values(["symbol", "count"], ascending=[True, False])
    )
    print("\n--- Candidate Class Counts by Symbol ---")
    print(count_tbl.to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, "trend_candidate_inventory.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved candidate inventory → %s", out_path)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Forward Return Audit of Rejected Signals
# ─────────────────────────────────────────────────────────────────────────────

def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fwd_Nd, fwd_mfe_Nd, fwd_mae_Nd columns. Per symbol. Diagnostic only.
    No lookahead used in production — these columns are clearly labeled as
    diagnostic-only and are never fed back into signal computation.
    """
    df = df.copy()
    fwd_cols = [
        "fwd_1d","fwd_2d","fwd_3d","fwd_5d","fwd_10d","fwd_20d",
        "fwd_mfe_5d","fwd_mfe_10d","fwd_mfe_20d",
        "fwd_mae_5d","fwd_mae_10d","fwd_mae_20d",
    ]
    for col in fwd_cols:
        df[col] = np.nan

    for sym, g in df.groupby("symbol"):
        g   = g.sort_values("date")
        cl  = g["close"].values
        N   = len(cl)
        idx = g.index

        for n, col in [
            (1,  "fwd_1d"),
            (2,  "fwd_2d"),
            (3,  "fwd_3d"),
            (5,  "fwd_5d"),
            (10, "fwd_10d"),
            (20, "fwd_20d"),
        ]:
            fwd = np.full(N, np.nan)
            for i in range(N - n):
                fwd[i] = cl[i + n] / cl[i] - 1
            df.loc[idx, col] = fwd

        for n, mfe_col, mae_col in [
            (5,  "fwd_mfe_5d",  "fwd_mae_5d"),
            (10, "fwd_mfe_10d", "fwd_mae_10d"),
            (20, "fwd_mfe_20d", "fwd_mae_20d"),
        ]:
            mfe = np.full(N, np.nan)
            mae = np.full(N, np.nan)
            for i in range(N - 1):
                fut = cl[i+1 : min(i+1+n, N)]
                if len(fut) > 0:
                    mfe[i] = np.max(fut) / cl[i] - 1
                    mae[i] = np.min(fut) / cl[i] - 1
            df.loc[idx, "fwd_mfe_" + str(n) + "d"] = mfe
            df.loc[idx, "fwd_mae_" + str(n) + "d"] = mae

    return df


def _fwd_summary(sub: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate forward return stats by a grouping column."""
    rows = []
    for key, g in sub.groupby(group_col):
        n = len(g)
        if n == 0:
            continue
        hr3  = (g["fwd_3d"]  > 0).mean()
        mfe_mae = (g["fwd_mfe_10d"].mean() / abs(g["fwd_mae_10d"].mean())
                   if g["fwd_mae_10d"].mean() < -1e-9 else np.nan)
        rows.append({
            group_col:         key,
            "count":           n,
            "hit_rate_3d":     round(hr3, 4),
            "avg_fwd_3d":      round(g["fwd_3d"].mean(),    4),
            "avg_fwd_5d":      round(g["fwd_5d"].mean(),    4),
            "avg_fwd_10d":     round(g["fwd_10d"].mean(),   4),
            "avg_mfe_10d":     round(g["fwd_mfe_10d"].mean(),4),
            "avg_mae_10d":     round(g["fwd_mae_10d"].mean(),4),
            "mfe_mae_ratio":   round(mfe_mae, 3) if not np.isnan(mfe_mae) else np.nan,
        })
    return pd.DataFrame(rows)


def section2_forward_return_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    For fresh rejection rows (days_held == 0) compute forward returns and
    produce summary tables. Also computes baseline for TAKEN_TREND rows.
    """
    logger.info("=== Section 2: Forward Return Audit of Rejected Signals ===")

    df = compute_forward_returns(df)

    rejected_classes = {c for c in df["candidate_class"].unique()
                        if c.startswith("REJECTED_")
                        or c.startswith("FLAT_CHOP")
                        or c.startswith("FLAT_SA")
                        or c == "POTENTIAL_TREND_UNEXPLAINED"}

    fresh = df[(df["candidate_class"].isin(rejected_classes)) & (df["days_held"] == 0)].copy()
    taken = df[df["candidate_class"].isin({"TAKEN_TREND", "TAKEN_TREND_HOLD"})].copy()

    # Summaries
    by_class  = _fwd_summary(fresh, "candidate_class")
    by_symbol = _fwd_summary(fresh, "symbol")

    fresh["entry_year"] = fresh["date"].dt.year
    by_year   = _fwd_summary(fresh, "entry_year")

    fresh["tq_bucket"] = fresh["trend_quality_bucket"].fillna("UNKNOWN")
    by_tq     = _fwd_summary(fresh, "tq_bucket")

    fresh["tce_bucket"] = fresh["trend_continuation_edge_bucket"].fillna("UNKNOWN")
    by_tce    = _fwd_summary(fresh, "tce_bucket")

    # TAKEN_TREND baseline
    taken["entry_year"] = taken["date"].dt.year
    taken_by_class = _fwd_summary(taken, "candidate_class")

    # Combine all into one output with a section tag
    def _tag(df_in, section):
        df_in = df_in.copy()
        df_in.insert(0, "section", section)
        return df_in

    combined = pd.concat([
        _tag(by_class,      "by_candidate_class"),
        _tag(by_symbol,     "by_symbol"),
        _tag(by_year,       "by_year"),
        _tag(by_tq,         "by_trend_quality_bucket"),
        _tag(by_tce,        "by_trend_continuation_edge_bucket"),
        _tag(taken_by_class,"taken_trend_baseline"),
    ], ignore_index=True)

    out_path = os.path.join(OUTPUT_DIR, "rejected_trend_forward_returns.csv")
    combined.to_csv(out_path, index=False)
    logger.info("Saved forward return audit → %s", out_path)

    # Quick print
    print("\n--- Forward Returns by Candidate Class (fresh rejections) ---")
    print(by_class.to_string(index=False))
    print("\n--- TAKEN_TREND Baseline ---")
    print(taken_by_class.to_string(index=False))

    return df  # df now has fwd cols attached


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Weak Trend Bucket Diagnostic
# ─────────────────────────────────────────────────────────────────────────────

_STRONG_BUCKETS = {"STRONG", "EXCEPTIONAL"}


def _sub_bucket(row) -> str:
    cclass  = str(row.get("candidate_class", "") or "")
    tq      = str(row.get("trend_quality_bucket", "WEAK") or "WEAK")
    regime  = str(row.get("portfolio_regime", "CHOP") or "CHOP")
    sym     = str(row.get("symbol", "") or "")
    pct     = float(row.get("late_entry_pct_20d", 0.5) or 0.5)

    taken   = cclass.startswith("TAKEN_TREND")
    rejected= cclass.startswith("REJECTED_") or cclass.startswith("FLAT_")

    if taken:
        if tq in _STRONG_BUCKETS: return "TAKEN_STRONG"
        if tq == "MODERATE":      return "TAKEN_MODERATE"
        return "TAKEN_WEAK"

    if rejected:
        if regime == "TRANSITION" and tq in _STRONG_BUCKETS: return "TRANSITION_STRONG"
        if tq in _STRONG_BUCKETS: return "REJECTED_STRONG"
        if tq == "MODERATE":      return "REJECTED_MODERATE"
        return "REJECTED_WEAK"

    return "OTHER"


def section3_weak_trend_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Focus on TREND regime + weekly_bias LONG universe. Sub-bucket by quality
    and compute forward return stats plus simulated PnL.
    """
    logger.info("=== Section 3: Weak Trend Bucket Diagnostic ===")

    universe = df[df["is_potential_trend_raw"]].copy()
    universe["sub_bucket"] = universe.apply(_sub_bucket, axis=1)

    # Early location bucket
    early_mask = universe["late_entry_pct_20d"] < 0.40
    universe.loc[early_mask & ~universe["candidate_class"].str.startswith("TAKEN_TREND"),
                 "sub_bucket"] = "EARLY_LOCATION"

    # ES/NQ blocked (any rejection reason)
    rejected_mask = (
        universe["candidate_class"].str.startswith("REJECTED_") |
        universe["candidate_class"].str.startswith("FLAT_")
    )
    universe.loc[rejected_mask & (universe["symbol"] == "ES"), "sub_bucket"] = "ES_BLOCKED"
    universe.loc[rejected_mask & (universe["symbol"] == "NQ"), "sub_bucket"] = "NQ_BLOCKED"

    rows = []
    for bucket, g in universe.groupby("sub_bucket"):
        n    = len(g)
        hr3  = (g["fwd_3d"]  > 0).mean() if n > 0 else np.nan
        hr5  = (g["fwd_5d"]  > 0).mean() if n > 0 else np.nan
        sim5  = (g["fwd_5d"].fillna(0) * 0.05).sum()
        sim10 = (g["fwd_5d"].fillna(0) * 0.10).sum()
        rows.append({
            "sub_bucket":                bucket,
            "count":                     n,
            "hit_rate_3d":               round(hr3,  4),
            "hit_rate_5d":               round(hr5,  4),
            "avg_fwd_3d":                round(g["fwd_3d"].mean(),    4),
            "avg_fwd_5d":                round(g["fwd_5d"].mean(),    4),
            "avg_fwd_10d":               round(g["fwd_10d"].mean(),   4),
            "avg_mfe_10d":               round(g["fwd_mfe_10d"].mean(),4),
            "avg_mae_10d":               round(g["fwd_mae_10d"].mean(),4),
            "simulated_pnl_5pct_size":   round(sim5,  4),
            "simulated_pnl_10pct_size":  round(sim10, 4),
        })
    result = pd.DataFrame(rows).sort_values("count", ascending=False)

    out_path = os.path.join(OUTPUT_DIR, "weak_trend_bucket_diagnostic.csv")
    result.to_csv(out_path, index=False)
    logger.info("Saved weak trend bucket diagnostic → %s", out_path)

    print("\n--- Weak Trend Bucket Diagnostic ---")
    print(result.to_string(index=False))

    return universe  # return universe subset with sub_bucket column


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Missed Trend Leg Audit
# ─────────────────────────────────────────────────────────────────────────────

def find_trend_legs(price_series: pd.Series, min_move_pct: float = 0.03) -> list:
    """
    Identify up-legs: from rolling 20d low to subsequent 20d high.
    Returns list of dicts with leg metadata.
    """
    prices = price_series.sort_index()
    dates  = prices.index
    N      = len(prices)

    roll_min = prices.rolling(20, min_periods=5).min()
    at_20d_low = prices == roll_min

    legs = []
    i = 0
    while i < N - 5:
        if at_20d_low.iloc[i]:
            leg_start       = dates[i]
            leg_start_price = prices.iloc[i]
            end             = min(i + 60, N)
            future_prices   = prices.iloc[i+1:end]
            if len(future_prices) == 0:
                i += 1
                continue
            peak_idx   = future_prices.idxmax()
            peak_price = future_prices[peak_idx]
            pct        = (peak_price - leg_start_price) / leg_start_price
            if pct >= min_move_pct:
                duration = (peak_idx - leg_start).days
                legs.append({
                    "leg_start":       leg_start,
                    "leg_peak":        peak_idx,
                    "start_price":     leg_start_price,
                    "peak_price":      peak_price,
                    "leg_return_pct":  round(pct * 100, 2),
                    "duration_days":   duration,
                })
                i = prices.index.get_loc(peak_idx) + 1
            else:
                i += 1
        else:
            i += 1
    return legs


def _classify_leg_coverage(
    leg: dict,
    trade_log: pd.DataFrame,
    sym: str,
) -> str:
    """
    Given a price leg and the backtest trade log, classify coverage:
    TRADED / TRADED_LATE / PARTLY_MISSED / FULLY_MISSED
    """
    if trade_log is None or trade_log.empty:
        return "FULLY_MISSED"

    sym_trades = trade_log[
        (trade_log["symbol"]    == sym) &
        (trade_log["strategy"]  == "TREND") &
        (trade_log["direction"] == "LONG")
    ].copy()

    leg_start    = leg["leg_start"]
    leg_peak     = leg["leg_peak"]
    leg_duration = (leg_peak - leg_start).days
    if leg_duration == 0:
        return "FULLY_MISSED"

    leg_move = leg["leg_return_pct"] / 100.0  # fractional

    # Trades that overlap with this leg
    overlapping = sym_trades[
        (sym_trades["entry_date"] <= leg_peak) &
        (sym_trades["exit_date"]  >= leg_start)
    ]

    if overlapping.empty:
        return "FULLY_MISSED"

    # Find the earliest entry relative to the leg
    earliest_entry = overlapping["entry_date"].min()
    entry_offset   = (earliest_entry - leg_start).days
    leg_30pct_mark = leg_duration * 0.30
    leg_70pct_mark = leg_duration * 0.70

    if entry_offset <= leg_30pct_mark:
        entry_timing = "EARLY"
    elif entry_offset <= leg_70pct_mark:
        entry_timing = "LATE"
    else:
        return "FULLY_MISSED"  # entered after 70% of the leg elapsed

    # Estimate captured move: use the best trade's pnl as proxy
    best_pnl = overlapping["pnl"].max()
    captured_fraction = best_pnl / leg_move if leg_move > 1e-6 else 0.0

    if entry_timing == "EARLY":
        return "TRADED" if captured_fraction >= 0.5 else "PARTLY_MISSED"
    else:
        return "TRADED_LATE"


def section4_missed_trend_leg_audit(df: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
    """
    Identify meaningful price legs per symbol and classify coverage.
    """
    logger.info("=== Section 4: Missed Trend Leg Audit ===")

    all_leg_rows = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date").set_index("date")
        legs = find_trend_legs(g["close"], min_move_pct=0.03)
        for leg in legs:
            coverage = _classify_leg_coverage(leg, trade_log, sym)
            all_leg_rows.append({
                "symbol":          sym,
                "leg_start":       leg["leg_start"],
                "leg_peak":        leg["leg_peak"],
                "start_price":     round(leg["start_price"],   2),
                "peak_price":      round(leg["peak_price"],    2),
                "leg_return_pct":  leg["leg_return_pct"],
                "duration_days":   leg["duration_days"],
                "coverage":        coverage,
                "year":            leg["leg_start"].year,
            })

    leg_df = pd.DataFrame(all_leg_rows)
    if leg_df.empty:
        logger.warning("No trend legs found — skipping leg audit")
        return leg_df

    # Summary stats
    total_legs   = len(leg_df)
    total_move   = leg_df["leg_return_pct"].sum()
    traded       = leg_df[leg_df["coverage"].isin({"TRADED","TRADED_LATE"})]
    missed       = leg_df[leg_df["coverage"].isin({"PARTLY_MISSED","FULLY_MISSED"})]
    fully_missed = leg_df[leg_df["coverage"] == "FULLY_MISSED"]

    print(f"\n--- Missed Trend Leg Audit ---")
    print(f"Total legs found:     {total_legs}   Total move: {total_move:.1f}%")
    print(f"Traded legs:          {len(traded)}   Move: {traded['leg_return_pct'].sum():.1f}%")
    print(f"Missed/partly missed: {len(missed)}   Move: {missed['leg_return_pct'].sum():.1f}%")
    print(f"Fully missed:         {len(fully_missed)}   Move: {fully_missed['leg_return_pct'].sum():.1f}%")

    print("\n--- By Coverage Type ---")
    print(leg_df.groupby("coverage")["leg_return_pct"].agg(["count","sum","mean"]).round(2))

    print("\n--- By Year ---")
    print(leg_df.groupby(["year","coverage"])["leg_return_pct"].agg(["count","sum"]).round(2))

    print("\n--- By Symbol ---")
    print(leg_df.groupby(["symbol","coverage"])["leg_return_pct"].agg(["count","sum"]).round(2))

    print("\n--- Top 20 Fully Missed Legs ---")
    top20 = (fully_missed.sort_values("leg_return_pct", ascending=False).head(20)
             [["symbol","leg_start","leg_peak","leg_return_pct","duration_days"]])
    print(top20.to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, "missed_trend_leg_audit.csv")
    leg_df.to_csv(out_path, index=False)
    logger.info("Saved missed trend leg audit → %s", out_path)

    return leg_df


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Simulated Weak-Trend Entry Tests
# ─────────────────────────────────────────────────────────────────────────────

def _identify_weak_candidates(df: pd.DataFrame) -> pd.Index:
    """
    Rows where system was flat for a controllable weak reason while in
    TREND regime with LONG weekly bias.
    """
    controllable_reasons = {
        "H4_WAIT", "LOW_QUALITY", "NO_BIAS", "PULLBACK_DISABLED",
        "TREND_BLOCKED_H4_WAIT", "TREND_FLAT_NO_BIAS", "TREND_FLAT_LOW_QUALITY",
    }
    mask = (
        (df["final_direction"]          == "FLAT") &
        (df["strategy_used"]            == "NONE") &
        (df["portfolio_regime"]         == "TREND") &
        (df["weekly_bias"]              == "LONG") &
        (df["portfolio_sentiment_flag"] != "HIGH_RISK") &
        (df["portfolio_regime"]         != "SHOCK")
    )
    reason_mask = df["signal_reason"].apply(
        lambda r: any(kw in str(r) for kw in controllable_reasons)
    )
    return df[mask & reason_mask].index


def _apply_sim_modifications(df: pd.DataFrame, mod_index: pd.Index,
                              strategy: str, size: float,
                              entry_quality: int = 3) -> pd.DataFrame:
    """
    Apply simulated entries to df copy. Sets strategy_used, final_direction,
    mtf_entry_size and entry_quality on the specified rows.
    """
    df = df.copy()
    if "mtf_entry_size" not in df.columns:
        df["mtf_entry_size"] = float("nan")

    df.loc[mod_index, "strategy_used"]   = strategy
    df.loc[mod_index, "final_direction"] = "LONG"
    df.loc[mod_index, "entry_quality"]   = entry_quality
    if strategy == "MTF_RESET":
        df.loc[mod_index, "mtf_entry_size"] = size
    return df


def _run_experiment(
    df_base: pd.DataFrame,
    label: str,
    mod_index: pd.Index,
    strategy: str,
    size: float,
    entry_quality: int = 3,
) -> dict:
    """Run a single backtest experiment and return result dict."""
    logger.info("Running experiment %s — %d added rows", label, len(mod_index))
    if len(mod_index) > 0:
        df_exp = _apply_sim_modifications(df_base, mod_index, strategy, size, entry_quality)
    else:
        df_exp = df_base.copy()
        if "mtf_entry_size" not in df_exp.columns:
            df_exp["mtf_entry_size"] = float("nan")

    result = run_backtest(df_exp)
    stats  = result["stats"]
    sym_contrib = stats.get("symbol_contribution", {})
    tl     = result.get("trade_log", pd.DataFrame())

    added_tl = pd.DataFrame()
    if len(mod_index) > 0 and not tl.empty and "entry_date" in tl.columns:
        added_dates = set(df_base.loc[mod_index, "date"].unique())
        added_tl = tl[tl["entry_date"].isin(added_dates)].copy()

    added_wr = added_tl["win"].mean() if not added_tl.empty and "win" in added_tl.columns else np.nan

    return {
        "experiment":       label,
        "sharpe":           stats["sharpe_ratio"],
        "annual_ret_pct":   stats["annualized_return_pct"],
        "total_pnl_pct":    stats["total_return_pct"],
        "max_dd_pct":       stats["max_drawdown_pct"],
        "trade_count":      stats.get("total_trades", 0),
        "win_rate":         stats.get("overall_win_rate", 0.0),
        "profit_factor":    stats.get("profit_factor", 0.0),
        "added_trade_count":len(added_tl),
        "added_win_rate":   round(added_wr, 4) if not np.isnan(added_wr) else 0.0,
        "pnl_ES":           round(sym_contrib.get("ES", 0.0), 4),
        "pnl_NQ":           round(sym_contrib.get("NQ", 0.0), 4),
        "delta_sharpe":     round(stats["sharpe_ratio"] - BASELINE_SHARPE, 3),
        "_added_tl":        added_tl,
    }


def section5_simulated_weak_trend_tests(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Run 9 simulation experiments. Return results DataFrame.
    Also saves added trade log.
    """
    logger.info("=== Section 5: Simulated Weak-Trend Entry Tests ===")

    # Identify candidate pools
    weak_idx   = _identify_weak_candidates(df_base)

    # H4_WAIT specifically
    h4_wait_idx = df_base[
        df_base["signal_reason"].str.contains("H4_WAIT|TREND_BLOCKED_H4_WAIT",
                                               na=False, regex=True)
        & (df_base["final_direction"] == "FLAT")
        & (df_base["portfolio_regime"] == "TREND")
    ].index

    # PULLBACK_DISABLED specifically
    pullback_idx = df_base[
        df_base["signal_reason"].str.contains("PULLBACK_DISABLED", na=False)
        & (df_base["final_direction"] == "FLAT")
    ].index

    # TRANSITION with strong quality
    transition_strong_idx = df_base[
        (df_base["portfolio_regime"] == "TRANSITION") &
        (df_base["final_direction"]  == "FLAT") &
        (df_base["trend_quality_bucket"].isin(_STRONG_BUCKETS))
    ].index

    # Weak + early in range (< 50%)
    early_weak_idx = weak_idx[
        df_base.loc[weak_idx, "late_entry_pct_20d"].fillna(1.0) < 0.50
    ]

    # NQ confirms for ES weak candidates
    nq_confirm_weak_es_idx = weak_idx[
        (df_base.loc[weak_idx, "symbol"] == "ES") &
        (df_base.loc[weak_idx, "nq_confirms"] == True)
    ]
    # NQ weak only
    nq_weak_idx = weak_idx[df_base.loc[weak_idx, "symbol"] == "NQ"]

    experiments_spec = [
        ("A_BASELINE",               pd.Index([]),        "NONE",      0.00, 3),
        ("B_TAKE_WEAK_TREND_5PCT",   weak_idx,            "MTF_RESET", 0.05, 3),
        ("C_TAKE_WEAK_TREND_10PCT",  weak_idx,            "MTF_RESET", 0.10, 3),
        ("D_TAKE_EARLY_WEAK_ONLY",   early_weak_idx,      "MTF_RESET", 0.075,3),
        ("E_TAKE_WEAK_WITH_NQ_CONFIRM",nq_confirm_weak_es_idx,"MTF_RESET",0.075,3),
        ("F_TAKE_WEAK_NQ_ONLY",      nq_weak_idx,         "MTF_RESET", 0.10, 3),
        ("G_TAKE_H4_WAIT_REJECTIONS",h4_wait_idx,         "TREND",     0.00, 3),
        ("H_TAKE_PULLBACK_RESETS",   pullback_idx,        "MTF_RESET", 0.075,3),
        ("I_TAKE_TRANSITION_STRONG", transition_strong_idx,"MTF_RESET",0.05, 3),
    ]

    results = []
    all_added_logs = []
    for label, mod_idx, strat, sz, eq in experiments_spec:
        res = _run_experiment(df_base, label, mod_idx, strat, sz, eq)
        added_tl = res.pop("_added_tl")
        if not added_tl.empty:
            added_tl = added_tl.copy()
            added_tl["experiment"] = label
        results.append(res)
        all_added_logs.append(added_tl)

    result_df = pd.DataFrame(results)

    print("\n--- Simulated Weak-Trend Entry Tests ---")
    display_cols = ["experiment","sharpe","delta_sharpe","annual_ret_pct",
                    "max_dd_pct","trade_count","win_rate","added_trade_count","added_win_rate"]
    print(result_df[display_cols].to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, "weak_trend_simulation_tests.csv")
    result_df.to_csv(out_path, index=False)
    logger.info("Saved simulation results → %s", out_path)

    # Combined added trade log
    combined_log = pd.concat([t for t in all_added_logs if not t.empty], ignore_index=True)
    if not combined_log.empty:
        log_cols = [c for c in ["symbol","entry_date","exit_date","exit_reason",
                                 "pnl","strategy","experiment"] if c in combined_log.columns]
        combined_log = combined_log[log_cols]

    log_path = os.path.join(OUTPUT_DIR, "weak_trend_added_trade_log.csv")
    combined_log.to_csv(log_path, index=False)
    logger.info("Saved added trade log → %s", log_path)

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Rejection Rule Attribution
# ─────────────────────────────────────────────────────────────────────────────

_REJECTION_RULES = {
    "H4_QUALITY_LOW_GATE": lambda r: "LOW_QUALITY" in str(r.get("signal_reason","") or ""),
    "H4_WAIT_NEW_ENTRY":   lambda r: str(r.get("signal_reason","") or "") == "TREND_BLOCKED_H4_WAIT",
    "NO_WEEKLY_BIAS":      lambda r: "NO_BIAS" in str(r.get("signal_reason","") or ""),
    "HIGH_RISK_SENTIMENT": lambda r: "HIGH_RISK" in str(r.get("signal_reason","") or ""),
    "SHOCK_FILTER":        lambda r: str(r.get("signal_reason","") or "") == "SHOCK_EXIT",
    "TRANSITION_BLOCK":    lambda r: "TRANSITION" in str(r.get("signal_reason","") or ""),
    "PULLBACK_DISABLED":   lambda r: str(r.get("signal_reason","") or "") == "PULLBACK_DISABLED",
    "SA_OVERRIDE_ES":      lambda r: str(r.get("signal_reason","") or "") == "SA_ES_HEDGE_DISABLED",
    "CHOP_REGIME_BLOCK":   lambda r: (
                               str(r.get("portfolio_regime","") or "") == "CHOP" and
                               str(r.get("final_direction","") or "") == "FLAT"
                           ),
    "MR_DISABLED":         lambda r: "MR_DISABLED" in str(r.get("signal_reason","") or ""),
    "ES_LATE_FILTER":      lambda r: (
                               str(r.get("symbol","") or "") == "ES" and
                               float(r.get("late_entry_pct_20d", 0) or 0) > 0.70 and
                               str(r.get("final_direction","") or "") == "FLAT"
                           ),
}


def section6_rejection_rule_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each rejection rule, compute blocked count and forward return stats.
    fresh_only: days_held == 0.
    """
    logger.info("=== Section 6: Rejection Rule Attribution ===")

    fresh = df[df["days_held"] == 0].copy()

    rows = []
    for rule_name, pred in _REJECTION_RULES.items():
        matched = fresh[fresh.apply(pred, axis=1)]
        n = len(matched)
        if n == 0:
            rows.append({
                "rule": rule_name, "count_blocked": 0,
                "avg_fwd_5d": np.nan, "hit_rate_5d": np.nan,
                "avg_fwd_10d": np.nan, "avg_mfe_10d": np.nan,
                "avg_mae_10d": np.nan, "mfe_mae_ratio": np.nan,
                "simulated_pnl_5pct": 0.0, "net_rule_value": np.nan,
            })
            continue

        avg5   = matched["fwd_5d"].mean()
        hr5    = (matched["fwd_5d"] > 0).mean()
        avg10  = matched["fwd_10d"].mean()
        mfe10  = matched["fwd_mfe_10d"].mean()
        mae10  = matched["fwd_mae_10d"].mean()
        mfe_mae= mfe10 / abs(mae10) if mae10 < -1e-9 else np.nan
        sim5   = (matched["fwd_5d"].fillna(0) * 0.05).sum()
        # net_rule_value: negative avg fwd means the rule SAVED money
        # Positive avg fwd means the rule COST opportunity
        net_val = -avg5  # if avg5 > 0, rule cost opportunity (negative net_val)

        rows.append({
            "rule":                rule_name,
            "count_blocked":       n,
            "avg_fwd_5d":          round(avg5,   4),
            "hit_rate_5d":         round(hr5,    4),
            "avg_fwd_10d":         round(avg10,  4),
            "avg_mfe_10d":         round(mfe10,  4),
            "avg_mae_10d":         round(mae10,  4),
            "mfe_mae_ratio":       round(mfe_mae,3) if not np.isnan(mfe_mae) else np.nan,
            "simulated_pnl_5pct":  round(sim5,   4),
            "net_rule_value":      round(net_val, 4),
        })

    result = pd.DataFrame(rows).sort_values("count_blocked", ascending=False)

    print("\n--- Rejection Rule Attribution ---")
    print(result.to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, "trend_rejection_rule_attribution.csv")
    result.to_csv(out_path, index=False)
    logger.info("Saved rejection rule attribution → %s", out_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Final Report
# ─────────────────────────────────────────────────────────────────────────────

def section8_final_report(
    df: pd.DataFrame,
    fwd_audit_df: pd.DataFrame,
    leg_df: pd.DataFrame,
    sim_results: pd.DataFrame,
    rule_attr: pd.DataFrame,
    universe: pd.DataFrame,
) -> None:
    """
    Print answers to 10 diagnostic questions and give a final recommendation.
    """
    logger.info("=== Section 8: Final Report ===")

    print("\n" + "="*70)
    print("MISSED OPPORTUNITY DIAGNOSTIC — FINAL REPORT")
    print("="*70)

    # ── Q1: Are weak trend labels actually weak? ───────────────────────────
    print("\nQ1: Are weak trend labels actually weak?")
    taken_bucket_by_class = fwd_audit_df[
        fwd_audit_df["section"] == "taken_trend_baseline"
    ].copy() if not fwd_audit_df.empty else pd.DataFrame()

    taken_strong = df[
        df["candidate_class"].isin({"TAKEN_TREND","TAKEN_TREND_HOLD"}) &
        df["trend_quality_bucket"].isin(_STRONG_BUCKETS)
    ]
    taken_weak = df[
        df["candidate_class"].isin({"TAKEN_TREND","TAKEN_TREND_HOLD"}) &
        (df["trend_quality_bucket"] == "WEAK")
    ]

    ts_fwd5 = taken_strong["fwd_5d"].mean() if len(taken_strong) > 0 else np.nan
    tw_fwd5 = taken_weak["fwd_5d"].mean()   if len(taken_weak)   > 0 else np.nan
    print(f"  TAKEN_STRONG avg_fwd_5d: {ts_fwd5:.4f}  (n={len(taken_strong)})")
    print(f"  TAKEN_WEAK   avg_fwd_5d: {tw_fwd5:.4f}  (n={len(taken_weak)})")
    labels_weak_verdict = (
        "YES — weak labels have lower fwd returns than strong labels"
        if (not np.isnan(ts_fwd5) and not np.isnan(tw_fwd5) and ts_fwd5 > tw_fwd5)
        else "NO — weak labels perform comparably or better than strong labels"
    )
    print(f"  VERDICT: {labels_weak_verdict}")

    # ── Q2: Which rejected buckets had positive expectancy? ────────────────
    print("\nQ2: Which rejected/flat buckets had positive forward expectancy?")
    rejected_classes = {c for c in df["candidate_class"].unique()
                        if c.startswith("REJECTED_") or c.startswith("FLAT_")
                        or c == "POTENTIAL_TREND_UNEXPLAINED"}
    fresh_rejected = df[(df["candidate_class"].isin(rejected_classes)) & (df["days_held"] == 0)]
    pos_exp = (
        fresh_rejected.groupby("candidate_class")["fwd_5d"]
        .agg(["mean","count"])
        .rename(columns={"mean":"avg_fwd_5d"})
    )
    pos_classes = pos_exp[pos_exp["avg_fwd_5d"] > 0.0]
    if pos_classes.empty:
        print("  None — all rejected buckets had negative forward expectancy on 5d horizon.")
    else:
        print("  Positive 5d expectancy classes:")
        print(pos_classes.to_string())

    # ── Q3: Is the system too selective? ──────────────────────────────────
    print("\nQ3: Is the system too selective?")
    taken_count = len(df[df["candidate_class"].isin({"TAKEN_TREND","TAKEN_TREND_HOLD"})])
    pot_count   = df["is_potential_trend_raw"].sum()
    selectivity = taken_count / pot_count if pot_count > 0 else 0.0
    avg_rej_fwd5 = fresh_rejected["fwd_5d"].mean() if len(fresh_rejected) > 0 else np.nan
    avg_tak_fwd5 = df[df["candidate_class"].isin({"TAKEN_TREND","TAKEN_TREND_HOLD"})]["fwd_5d"].mean()
    print(f"  Potential TREND universe (TREND+LONG):   {int(pot_count)} rows")
    print(f"  Actually taken (TREND entries+holds):    {taken_count} rows")
    print(f"  Selectivity ratio:                       {selectivity:.1%}")
    print(f"  Avg fwd_5d of rejected (fresh):          {avg_rej_fwd5:.4f}")
    print(f"  Avg fwd_5d of taken TREND:               {avg_tak_fwd5:.4f}")
    if not np.isnan(avg_rej_fwd5) and avg_rej_fwd5 > 0.002:
        print("  VERDICT: YES — meaningful positive expectancy being rejected")
    else:
        print("  VERDICT: NOT CLEARLY — rejected rows show low forward expectancy")

    # ── Q4 & Q5: Missed trend legs ─────────────────────────────────────────
    print("\nQ4: How many profitable trend legs were completely missed?")
    print("Q5: What was the total missed move?")
    if not leg_df.empty:
        fully = leg_df[leg_df["coverage"] == "FULLY_MISSED"]
        partly = leg_df[leg_df["coverage"] == "PARTLY_MISSED"]
        print(f"  Fully missed legs:   {len(fully)}")
        print(f"  Partly missed legs:  {len(partly)}")
        print(f"  Total missed move (fully):   {fully['leg_return_pct'].sum():.1f}%")
        print(f"  Total missed move (partly):  {partly['leg_return_pct'].sum():.1f}%")
    else:
        print("  No leg data available.")

    # ── Q6: Which rejection rule costs the most opportunity? ──────────────
    print("\nQ6: Which rejection rule costs the most opportunity?")
    if not rule_attr.empty:
        pos_cost = rule_attr[rule_attr["avg_fwd_5d"] > 0].sort_values(
            "simulated_pnl_5pct", ascending=False
        )
        if not pos_cost.empty:
            top_rule = pos_cost.iloc[0]
            print(f"  Most costly rule: {top_rule['rule']}")
            print(f"    count_blocked={int(top_rule['count_blocked'])}, "
                  f"avg_fwd_5d={top_rule['avg_fwd_5d']:.4f}, "
                  f"sim_pnl_5pct={top_rule['simulated_pnl_5pct']:.4f}")
        else:
            print("  No rules with positive forward expectancy found — all rules appear protective.")

    # ── Q7: ES vs NQ missed ────────────────────────────────────────────────
    print("\nQ7: Are missed opportunities mostly ES or NQ?")
    if not leg_df.empty:
        missed_by_sym = leg_df[
            leg_df["coverage"].isin({"FULLY_MISSED","PARTLY_MISSED"})
        ].groupby("symbol")["leg_return_pct"].agg(["count","sum"])
        print(missed_by_sym.to_string())
    sym_rejected = (
        fresh_rejected.groupby("symbol")["fwd_5d"]
        .agg(count="count", avg_fwd_5d="mean")
    )
    print("  Rejected row fwd_5d by symbol:")
    print(sym_rejected.to_string())

    # ── Q8: Type of missed opportunity ────────────────────────────────────
    print("\nQ8: Are missed opportunities early trend, pullback reset, or breakout?")
    early_miss = fresh_rejected[fresh_rejected["late_entry_pct_20d"] < 0.30]
    mid_miss   = fresh_rejected[(fresh_rejected["late_entry_pct_20d"] >= 0.30) &
                                 (fresh_rejected["late_entry_pct_20d"] < 0.70)]
    late_miss  = fresh_rejected[fresh_rejected["late_entry_pct_20d"] >= 0.70]
    print(f"  Early in range (pct<0.30):  {len(early_miss)}  avg_fwd_5d={early_miss['fwd_5d'].mean():.4f}")
    print(f"  Mid range (0.30-0.70):       {len(mid_miss)}   avg_fwd_5d={mid_miss['fwd_5d'].mean():.4f}")
    print(f"  Late in range (pct>0.70):   {len(late_miss)}  avg_fwd_5d={late_miss['fwd_5d'].mean():.4f}")

    pullback_miss = fresh_rejected[
        fresh_rejected["signal_reason"].str.contains("PULLBACK", na=False)
    ]
    print(f"  Pullback-related rejections: {len(pullback_miss)}  avg_fwd_5d={pullback_miss['fwd_5d'].mean():.4f}")

    # ── Q9: Can we add more trades safely? ────────────────────────────────
    print("\nQ9: Can we add more trades safely? (Sharpe deltas from simulations)")
    if not sim_results.empty:
        sim_display = sim_results[["experiment","sharpe","delta_sharpe","max_dd_pct",
                                    "added_trade_count","added_win_rate"]].copy()
        print(sim_display.to_string(index=False))
        positive_delta = sim_results[sim_results["delta_sharpe"] > 0.03]
        if not positive_delta.empty:
            best = positive_delta.sort_values("delta_sharpe", ascending=False).iloc[0]
            print(f"  Best experiment: {best['experiment']} — Sharpe delta: +{best['delta_sharpe']:.3f}")
        else:
            print("  No experiment improved Sharpe by >0.03 — system appears correctly calibrated.")

    # ── Q10: Next strategy sleeve ──────────────────────────────────────────
    print("\nQ10: What exact next strategy sleeve should be tested?")
    # Logic: pick based on best simulation delta and rule attribution
    recommendation = _pick_recommendation(df, sim_results, rule_attr, leg_df)
    print(f"  RECOMMENDATION: {recommendation}")

    # Final banner
    print("\n" + "="*70)
    print(f"FINAL RECOMMENDATION: {recommendation}")
    print("="*70 + "\n")


def _pick_recommendation(
    df: pd.DataFrame,
    sim_results: pd.DataFrame,
    rule_attr: pd.DataFrame,
    leg_df: pd.DataFrame,
) -> str:
    """
    Derive a single recommendation label from the diagnostic data.
    """
    # Check if any sim improved Sharpe significantly
    if not sim_results.empty:
        best_delta = sim_results["delta_sharpe"].max()
        best_exp   = sim_results.loc[sim_results["delta_sharpe"].idxmax(), "experiment"]

        if best_delta > 0.05:
            if "NQ_ONLY" in best_exp or "NQ" in best_exp:
                return "NQ_WEAK_TREND_OPPORTUNITY_FOUND"
            if "NQ_CONFIRM" in best_exp or "CONFIRM" in best_exp:
                return "ES_WEAK_TREND_NEEDS_NQ_CONFIRMATION"
            if "EARLY" in best_exp:
                return "EARLY_WEAK_TREND_OPPORTUNITY_FOUND"

    # Check if any rule is clearly too strict (high count + positive avg fwd)
    if not rule_attr.empty:
        costly = rule_attr[
            (rule_attr["avg_fwd_5d"] > 0.003) &
            (rule_attr["count_blocked"] > 20)
        ]
        if not costly.empty:
            worst = costly.sort_values("simulated_pnl_5pct", ascending=False).iloc[0]
            return f"REJECTION_RULE_TOO_STRICT ({worst['rule']})"

    # Check taken_weak vs taken_strong
    taken_strong = df[
        df["candidate_class"].isin({"TAKEN_TREND","TAKEN_TREND_HOLD"}) &
        df["trend_quality_bucket"].isin(_STRONG_BUCKETS)
    ]
    taken_weak = df[
        df["candidate_class"].isin({"TAKEN_TREND","TAKEN_TREND_HOLD"}) &
        (df["trend_quality_bucket"] == "WEAK")
    ]
    ts_fwd5 = taken_strong["fwd_5d"].mean() if len(taken_strong) > 0 else 0
    tw_fwd5 = taken_weak["fwd_5d"].mean()   if len(taken_weak)   > 0 else 0
    if ts_fwd5 > tw_fwd5 + 0.005:
        return "WEAK_LABELS_ARE_ACTUALLY_WEAK"

    # Check if many missed legs
    if not leg_df.empty:
        missed = leg_df[leg_df["coverage"] == "FULLY_MISSED"]
        if len(missed) / max(len(leg_df), 1) > 0.40:
            return "MODEL_TOO_SELECTIVE"

    return "CONTINUE_TESTING"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Output directory: %s", OUTPUT_DIR)

    # ── Build base DataFrame ───────────────────────────────────────────────
    df_base = build_base_df()

    # ── Section 1: Candidate Inventory ────────────────────────────────────
    df = section1_candidate_inventory(df_base)

    # ── Section 2: Forward Return Audit ────────────────────────────────────
    # df now has fwd cols; returns same df enriched
    df = section2_forward_return_audit(df)

    # Load fwd audit output for section 8 reference
    fwd_audit_path = os.path.join(OUTPUT_DIR, "rejected_trend_forward_returns.csv")
    fwd_audit_df   = pd.read_csv(fwd_audit_path) if os.path.exists(fwd_audit_path) else pd.DataFrame()

    # ── Section 3: Weak Trend Bucket Diagnostic ────────────────────────────
    universe = section3_weak_trend_bucket(df)

    # ── Baseline backtest for trade log ────────────────────────────────────
    logger.info("Running baseline backtest to get trade log …")
    df_for_bt = df_base.copy()
    if "mtf_entry_size" not in df_for_bt.columns:
        df_for_bt["mtf_entry_size"] = float("nan")
    baseline_result = run_backtest(df_for_bt)
    trade_log       = baseline_result.get("trade_log", pd.DataFrame())
    logger.info("Baseline Sharpe: %.3f", baseline_result["stats"]["sharpe_ratio"])

    # ── Section 4: Missed Trend Leg Audit ─────────────────────────────────
    leg_df = section4_missed_trend_leg_audit(df, trade_log)

    # ── Section 5: Simulated Weak-Trend Tests ─────────────────────────────
    # We need nq_confirms and late_entry_pct_20d in df_base for experiment logic.
    # Merge them back.
    needed_cols = ["nq_confirms", "late_entry_pct_20d", "candidate_class"]
    for col in needed_cols:
        if col not in df_base.columns and col in df.columns:
            df_base = df_base.merge(
                df[["date","symbol",col]].drop_duplicates(["date","symbol"]),
                on=["date","symbol"], how="left",
            )
    sim_results = section5_simulated_weak_trend_tests(df_base)

    # ── Section 6: Rejection Rule Attribution ─────────────────────────────
    rule_attr = section6_rejection_rule_attribution(df)

    # ── Section 8: Final Report ────────────────────────────────────────────
    section8_final_report(df, fwd_audit_df, leg_df, sim_results, rule_attr, universe)

    logger.info("All diagnostics complete. Outputs in: %s", OUTPUT_DIR)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
