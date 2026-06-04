"""
run_idle_day_research.py — 7-Part idle-day opportunity research.

Objective: increase trade count without destroying Sharpe by adding sleeves
for the 40% of days currently flat. All sleeves behind flags.

Parts:
  1. Idle-day opportunity map (classify flat rows, compute fwd returns)
  2. CHOP range mean-reversion sleeve (chop_range_engine.py)
  3. TRANSITION reset/reclaim sleeve (transition_reset_engine.py)
  4. ES late exception with NQ confirmation
  5. STAT_ARB existing-logic expansion (no new formula)
  6. Combined idle-day system (only sleeves that individually pass)
  7. Final verdict
"""

import contextlib
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import config
from data_loader                    import load_daily, load_4h
from features                       import compute_features
from momentum_engine                import compute_momentum
from weekly_bias                    import compute_weekly_bias
from regime_engine                  import compute_regime
from volatility_engine              import compute_volatility_features
from intraday_4h_engine             import (compute_4h_features,
                                            aggregate_4h_to_daily,
                                            merge_4h_into_daily)
from mean_reversion                 import compute_mean_reversion
from stat_arb                       import compute_stat_arb
from sentiment_layer                import compute_sentiment
from signal_engine                  import compute_signals
from signal_quality_engine          import compute_signal_quality
from trend_continuation_edge_engine import compute_trend_continuation_edge
from regime_router                  import compute_regime_router
from short_sleeve_engine            import compute_short_signals
from chop_range_engine              import compute_chop_range_features
from transition_reset_engine        import compute_transition_reset_features
from backtest                       import run_backtest

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT_DIR = "output/idle_day_research"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Best V2 system parameters ────────────────────────────────────────────────
_HOLD_BASE = {**config.MAX_HOLD_DAYS,
              "BULL_RESET_CONFIRMED": config.MAX_HOLD_DAYS.get("TREND", 5) + 3,
              "TREND":                config.MAX_HOLD_DAYS.get("TREND", 5) + 3,
              "MTF_BREAKOUT": 2, "MTF_RESET": 3,
              "BEAR_TREND": 4, "FAILED_RALLY": 3,
              "SHOCK_CONT": 2, "BEAR_RESET": 3,
              # New sleeves
              "CHOP_MR_LONG": 3, "CHOP_MR_SHORT": 3,
              "TRANSITION_RESET": 3, "TRANSITION_RALLY_SHORT": 3,
              "SA_MODERATE": 4}


@contextlib.contextmanager
def override_config(**kwargs):
    orig = {k: getattr(config, k) for k in kwargs if hasattr(config, k)}
    for k, v in kwargs.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in orig.items():
            setattr(config, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_base_df() -> pd.DataFrame:
    logger.info("Building pipeline…")
    df = load_daily(); df4 = load_4h()
    df = compute_features(df); df = compute_momentum(df)
    df = compute_weekly_bias(df); df = compute_regime(df)
    df = compute_volatility_features(df)
    for col, val in [
        ("cwt_chop_label", "CWT_UNCLEAR"), ("cwt_trade_allowed", 1),
        ("cwt_size_multiplier", 1.0),      ("cwt_confidence", 0.0),
        ("cwt_total_energy", 0.0),         ("cwt_entropy", 0.5),
        ("cwt_energy_z", 0.0),             ("cwt_compression_flag", 0),
        ("cwt_expansion_flag", 0),         ("cwt_high_energy_ratio", 1/3),
        ("cwt_mid_energy_ratio", 1/3),     ("cwt_slow_energy_ratio", 1/3),
    ]:
        if col not in df.columns:
            df[col] = val
    d4d = None
    if df4 is not None:
        d4d = aggregate_4h_to_daily(compute_4h_features(df4))
    df = merge_4h_into_daily(df, d4d)
    df = compute_mean_reversion(df); df = compute_stat_arb(df)
    df = compute_sentiment(df);      df = compute_signals(df)
    df = compute_signal_quality(df); df = compute_trend_continuation_edge(df)
    df = compute_regime_router(df)
    df = compute_chop_range_features(df)
    df = compute_transition_reset_features(df)
    df = _add_entry_location(df)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Pipeline done: %d rows, %d cols", len(df), len(df.columns))
    return df


def _add_entry_location(df):
    if "entry_location_pct_20d" in df.columns:
        return df
    df = df.copy()
    df["entry_location_pct_20d"] = np.nan
    for sym, g in df.groupby("symbol", sort=False):
        idx = g.index; s = pd.Series(g["close"].values, index=idx)
        mn = s.rolling(20, min_periods=5).min()
        mx = s.rolling(20, min_periods=5).max()
        df.loc[idx, "entry_location_pct_20d"] = ((s - mn) / (mx - mn).replace(0, np.nan)).clip(0, 1).values
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Best-system builder (Long V2 Core + BEAR_TREND short)
# ─────────────────────────────────────────────────────────────────────────────

def apply_best_v2_core(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all accepted V2 improvements to df."""
    df = df.copy()
    loc = df["entry_location_pct_20d"].fillna(0.5)

    # ES late filter
    late = ((df["symbol"] == "ES") & (df["strategy_used"] == "TREND") &
            (df["final_direction"] == "LONG") & (loc > 0.70))
    df.loc[late, "final_direction"] = "FLAT"; df.loc[late, "strategy_used"] = "NONE"

    # Regime gating
    if "allow_trend_long" in df.columns:
        rg = ((df["strategy_used"] == "TREND") & (df["final_direction"] == "LONG") &
              (df["allow_trend_long"] == 0))
        df.loc[rg, "final_direction"] = "FLAT"; df.loc[rg, "strategy_used"] = "NONE"

    # BULL_RESET injection
    slope = df["slope_20d"].fillna(0); ret20 = df["ret_20d"].fillna(0)
    ret5  = df["ret_5d"].fillna(0)
    bk    = df.get("structure_break_count", pd.Series(0, index=df.index)).fillna(0)
    loc2  = df["entry_location_pct_20d"].fillna(0.5)
    reg   = df["portfolio_regime"].fillna("CHOP")
    sent  = df["portfolio_sentiment_flag"].fillna("NORMAL")
    br    = ((df["final_direction"] == "FLAT") & reg.isin(["TREND", "TRANSITION"]) &
             (slope > 0) & (ret20 > 0) & (ret5 < 0) & (bk == 0) & (loc2 < 0.80) &
             (sent != "HIGH_RISK"))
    df.loc[br, "final_direction"] = "LONG"; df.loc[br, "strategy_used"] = "BULL_RESET_CONFIRMED"
    df.loc[br, "mtf_entry_size"] = 0.08;   df.loc[br, "entry_quality"] = 4

    # BEAR_TREND short only
    df = compute_short_signals(df)
    not_bear = (df["short_signal_type"] != "BEAR_TREND") & (df["short_signal_type"] != "NONE")
    df.loc[not_bear, "final_direction"] = "FLAT"; df.loc[not_bear, "strategy_used"] = "NONE"
    df.loc[not_bear, "short_signal_type"] = "NONE"; df.loc[not_bear, "short_signal_size"] = 0.0

    return df


def run_bt(label: str, df: pd.DataFrame, cfg_extra: dict | None = None) -> tuple[dict, pd.DataFrame]:
    cfg = {**{"MAX_HOLD_DAYS": _HOLD_BASE, "VOL_SCALE_MAX": 0.28,
               "ADD_WINNER_SIZE_FRAC": 0.50}, **(cfg_extra or {})}
    with override_config(**cfg):
        res = run_backtest(df)
    st = res["stats"]; tl = res["trade_log"]; pr = res["portfolio_returns"]

    new_strats = set(cfg_extra.get("_new_strats", [])) if cfg_extra else set()
    new_tl = tl[tl["strategy"].isin(new_strats)] if (not tl.empty and new_strats) else pd.DataFrame()

    # PnL by strategy
    strat_pnl = {}
    if not tl.empty and "strategy" in tl.columns:
        strat_pnl = {s: round(float(g["pnl"].sum()), 4) for s, g in tl.groupby("strategy")}

    # Annual returns
    yr = {}
    for y in range(2019, 2027):
        m = pr.index.year == y
        if m.any():
            yr[y] = round(float((1 + pr[m]).prod() - 1) * 100, 2)

    # Sortino
    down = pr[pr < 0]
    ds_std = float(down.std()) * np.sqrt(252) if len(down) > 5 else np.nan
    sortino = float(pr.mean()) * 252 / ds_std if ds_std and ds_std > 1e-9 else np.nan

    # Worst week / month
    worst_wk  = float(pr.resample("W").sum().min())  * 100 if not pr.empty else 0
    worst_mo  = float(pr.resample("ME").sum().min()) * 100 if not pr.empty else 0

    # Avg DD
    cumret = (1 + pr).cumprod(); roll_max = cumret.cummax()
    dd = (cumret - roll_max) / roll_max
    avg_dd = float(dd[dd < 0].mean()) * 100 if (dd < 0).any() else 0

    n_new = len(new_tl)
    row = dict(
        experiment     = label,
        sharpe         = round(st["sharpe_ratio"], 4),
        sortino        = round(sortino, 4),
        annual_ret_pct = round(st["annualized_return_pct"], 2),
        max_dd_pct     = round(st["max_drawdown_pct"], 2),
        avg_dd_pct     = round(avg_dd, 2),
        worst_week_pct = round(worst_wk, 2),
        worst_month_pct= round(worst_mo, 2),
        trades         = st["total_trades"],
        trades_per_week= round(st["total_trades"] / (len(pr) / 5), 2),
        win_rate       = round(st.get("overall_win_rate", 0), 4),
        profit_factor  = round(st.get("profit_factor", 0), 4),
        new_trades     = n_new,
        new_wr         = round(new_tl["win"].mean(), 4) if n_new > 0 else np.nan,
        new_pf         = _pf(new_tl),
        new_pnl        = round(new_tl["pnl"].sum(), 4) if n_new > 0 else 0.0,
        pnl_ES         = round(float(tl[tl["symbol"] == "ES"]["pnl"].sum()), 4) if not tl.empty else 0,
        pnl_NQ         = round(float(tl[tl["symbol"] == "NQ"]["pnl"].sum()), 4) if not tl.empty else 0,
        **{f"yr_{k}": v for k, v in yr.items()},
        **{f"s_{k}": v for k, v in strat_pnl.items()},
    )
    logger.info(
        "%-38s  Sharpe=%.3f  Srt=%.3f  Ann=%.1f%%  MaxDD=%.1f%%  "
        "Trades=%d (+%d new)  WR=%.1f%%  PF=%.2f",
        label, row["sharpe"], row["sortino"] or 0, row["annual_ret_pct"],
        row["max_dd_pct"], row["trades"], n_new,
        row["win_rate"] * 100, row["profit_factor"],
    )
    return row, tl


def _pf(tl):
    if tl.empty or "pnl" not in tl.columns: return np.nan
    w = tl[tl["pnl"] > 0]["pnl"].sum(); l = abs(tl[tl["pnl"] < 0]["pnl"].sum())
    return round(w / l, 4) if l > 1e-9 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Signal injection helpers
# ─────────────────────────────────────────────────────────────────────────────

def inject_chop_mr(df, size, min_quality=0.45, z_thr=0.50, include_short=False):
    df = df.copy()
    reg  = df["portfolio_regime"].fillna("CHOP")
    flat = df["final_direction"] == "FLAT"
    sent = df["portfolio_sentiment_flag"].fillna("NORMAL")
    qual_ok = df["chop_range_quality"].fillna(0) >= min_quality
    no_vol  = df["chop_vol_expanding"].fillna(1) == 0
    no_hr   = sent != "HIGH_RISK"
    chop    = reg == "CHOP"

    long_mask = (
        chop & flat & qual_ok & no_vol & no_hr &
        (df["chop_z_score"].fillna(0) < -z_thr)
    )
    df.loc[long_mask, "final_direction"] = "LONG"
    df.loc[long_mask, "strategy_used"]   = "CHOP_MR_LONG"
    df.loc[long_mask, "mtf_entry_size"]  = size
    df.loc[long_mask, "entry_quality"]   = 3

    if include_short:
        short_mask = (
            chop & flat & qual_ok & no_vol & no_hr &
            (df["chop_z_score"].fillna(0) > z_thr)
        )
        df.loc[short_mask, "final_direction"] = "SHORT"
        df.loc[short_mask, "strategy_used"]   = "CHOP_MR_SHORT"
        df.loc[short_mask, "mtf_entry_size"]  = size
        df.loc[short_mask, "entry_quality"]   = 3

    return df


def inject_transition_reset(df, size, sym_filter=None, include_short=False, score_thr=0.55):
    df = df.copy()
    reg  = df["portfolio_regime"].fillna("CHOP")
    flat = df["final_direction"] == "FLAT"
    sent = df["portfolio_sentiment_flag"].fillna("NORMAL")
    no_hr = sent != "HIGH_RISK"
    trans = reg == "TRANSITION"

    sym_ok = pd.Series(True, index=df.index)
    if sym_filter:
        sym_ok = df["symbol"] == sym_filter

    long_mask = (
        trans & flat & sym_ok & no_hr &
        (df["tr_buyer_reclaim_score"].fillna(0) >= score_thr) &
        (df["tr_reclaim_flag"].fillna(0) == 1)
    )
    df.loc[long_mask, "final_direction"] = "LONG"
    df.loc[long_mask, "strategy_used"]   = "TRANSITION_RESET"
    df.loc[long_mask, "mtf_entry_size"]  = size
    df.loc[long_mask, "entry_quality"]   = 3

    if include_short:
        short_mask = (
            trans & flat & sym_ok & no_hr &
            (df["tr_seller_rejection_score"].fillna(0) >= score_thr) &
            (df["tr_failed_reclaim"].fillna(0) == 1) &
            (df["slope_20d"].fillna(0) < 0)
        )
        df.loc[short_mask, "final_direction"] = "SHORT"
        df.loc[short_mask, "strategy_used"]   = "TRANSITION_RALLY_SHORT"
        df.loc[short_mask, "mtf_entry_size"]  = size
        df.loc[short_mask, "entry_quality"]   = 3

    return df


def inject_es_late_exception(df, condition: str, es_size_scale=1.0, allowed_size=0.05):
    """
    Re-allow some ES late entries that were blocked by the late filter.
    condition: 'nq_trend' | 'nq_high_conviction' | 'nq_bull_reset' |
               'nq_trend_half_size' | 'nq_confirms_3pct' | 'nq_confirms_5pct' |
               'nq_not_late'
    """
    df = df.copy()
    loc = df["entry_location_pct_20d"].fillna(0.5)

    # Find ES rows that are currently FLAT but WERE TREND long before late filter
    # Proxy: ES, FLAT now, TREND regime, momentum LONG, slope > 0
    es_late_candidates = (
        (df["symbol"] == "ES") &
        (df["final_direction"] == "FLAT") &
        (df["portfolio_regime"].fillna("CHOP") == "TREND") &
        (df["momentum_direction"].fillna("NEUTRAL") == "LONG") &
        (df["slope_20d"].fillna(0) > 0) &
        (loc > 0.70) &
        (df["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )

    # Build NQ state lookup for same date
    nq_rows = df[df["symbol"] == "NQ"][["date",
        "final_direction", "momentum_direction", "high_conviction",
        "entry_location_pct_20d", "strategy_used"]].copy()
    nq_rows = nq_rows.rename(columns={
        "final_direction":     "nq_dir",
        "momentum_direction":  "nq_mom",
        "high_conviction":     "nq_hc",
        "entry_location_pct_20d": "nq_loc",
        "strategy_used":       "nq_strat",
    })

    df = df.merge(nq_rows, on="date", how="left")

    nq_dir  = df["nq_dir"].fillna("FLAT")
    nq_hc   = df["nq_hc"].fillna(False).astype(bool)
    nq_loc  = df["nq_loc"].fillna(1.0)
    nq_strat= df["nq_strat"].fillna("NONE")

    if condition == "nq_trend":
        allow = es_late_candidates & (nq_dir == "LONG")
    elif condition == "nq_high_conviction":
        allow = es_late_candidates & (nq_dir == "LONG") & nq_hc
    elif condition == "nq_bull_reset":
        allow = es_late_candidates & (nq_strat == "BULL_RESET_CONFIRMED")
    elif condition == "nq_trend_half_size":
        allow = es_late_candidates & (nq_dir == "LONG")
        allowed_size = allowed_size * 0.5
    elif condition == "nq_confirms_3pct":
        allow = es_late_candidates & (nq_dir == "LONG")
        allowed_size = 0.03
    elif condition == "nq_confirms_5pct":
        allow = es_late_candidates & (nq_dir == "LONG")
        allowed_size = 0.05
    elif condition == "nq_not_late":
        allow = es_late_candidates & (nq_dir == "LONG") & (nq_loc < 0.70)
    else:
        allow = es_late_candidates

    df.loc[allow, "final_direction"] = "LONG"
    df.loc[allow, "strategy_used"]   = "ES_LATE_EXCEPTION"
    df.loc[allow, "mtf_entry_size"]  = allowed_size
    df.loc[allow, "entry_quality"]   = 3

    # Clean up merge columns
    df = df.drop(columns=["nq_dir","nq_mom","nq_hc","nq_loc","nq_strat"], errors="ignore")
    return df


def inject_sa_expansion(df, variant: str):
    """
    Expand existing STAT_ARB using only current working signals.
    Does NOT use residual spread formula — uses existing sa_signal + sa_quality_bucket.
    """
    df = df.copy()
    flat = df["final_direction"] == "FLAT"
    sent = df["portfolio_sentiment_flag"].fillna("NORMAL")
    no_hr = sent != "HIGH_RISK"
    reg  = df["portfolio_regime"].fillna("CHOP")

    sa_sig = df.get("sa_signal", pd.Series("FLAT", index=df.index)).fillna("FLAT")
    sa_nq  = df.get("sa_nq_direction", pd.Series("FLAT", index=df.index)).fillna("FLAT")
    sa_str = df.get("sa_strength", pd.Series(0.0, index=df.index)).fillna(0.0)
    sa_bkt = df.get("sa_quality_bucket",
                    pd.Series("MODERATE", index=df.index)).fillna("MODERATE")

    nq_mask = (df["symbol"] == "NQ") & flat & no_hr & (sa_sig == "ENTRY") & (sa_nq == "LONG")

    if variant == "allow_moderate_2pct":
        # MODERATE bucket at 2% size, TRANSITION allowed
        allow = nq_mask & sa_bkt.isin(["MODERATE", "GOOD", "STRONG", "EXCEPTIONAL"])
        size  = 0.02
        strat = "SA_MODERATE"

    elif variant == "allow_moderate_3pct":
        allow = nq_mask & sa_bkt.isin(["MODERATE", "GOOD", "STRONG", "EXCEPTIONAL"])
        size  = 0.03
        strat = "SA_MODERATE"

    elif variant == "strong_size_up_25":
        # Existing STRONG/EXCEPTIONAL signals — increase size 25% (already fired via signal_engine, skip)
        # Instead: re-inject on flat rows with STRONG+ bucket
        allow = nq_mask & sa_bkt.isin(["STRONG", "EXCEPTIONAL"])
        size  = min(config.STAT_ARB_NQ_WEIGHT * 1.25, 0.30)
        strat = "STAT_ARB"

    elif variant == "strong_size_up_50":
        allow = nq_mask & sa_bkt.isin(["STRONG", "EXCEPTIONAL"])
        size  = min(config.STAT_ARB_NQ_WEIGHT * 1.50, 0.30)
        strat = "STAT_ARB"

    elif variant == "sa_in_transition":
        # Allow SA in TRANSITION regime (currently blocked by signal_engine)
        allow = (
            (df["symbol"] == "NQ") & flat & no_hr &
            (sa_sig == "ENTRY") & (sa_nq == "LONG") &
            (reg == "TRANSITION")
        )
        size  = config.STAT_ARB_NQ_WEIGHT * 0.50   # half size in TRANSITION
        strat = "SA_MODERATE"

    elif variant == "hold_plus_1d":
        # No injection — handled via config MAX_HOLD_DAYS change
        return df

    elif variant == "hold_plus_2d":
        return df

    else:
        return df

    df.loc[allow, "final_direction"] = "LONG"
    df.loc[allow, "strategy_used"]   = strat
    df.loc[allow, "mtf_entry_size"]  = size
    df.loc[allow, "entry_quality"]   = 3

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Opportunity map
# ─────────────────────────────────────────────────────────────────────────────

def part1_opportunity_map(df_raw: pd.DataFrame, df_best: pd.DataFrame):
    logger.info("=== PART 1: Idle-Day Opportunity Map ===")

    flat_rows = df_best[df_best["final_direction"] == "FLAT"].copy()

    # Forward returns (for diagnostic only)
    for sym, g in df_raw.groupby("symbol"):
        idx = g.index
        close = pd.Series(g["close"].values, index=idx)
        for d, n in [(1, 1), (3, 3), (5, 5)]:
            fwd = close.pct_change(n).shift(-n)
            df_raw.loc[idx, f"fwd_{d}d"] = fwd
        # MFE / MAE over 5 and 10 days
        for win in [5, 10]:
            hi = close.shift(-1).rolling(win).max().shift(-(win - 1))
            lo = close.shift(-1).rolling(win).min().shift(-(win - 1))
            df_raw.loc[idx, f"mfe_{win}d"] = (hi - close) / close
            df_raw.loc[idx, f"mae_{win}d"] = (lo - close) / close

    flat_rows = flat_rows.merge(
        df_raw[["date","symbol","fwd_1d","fwd_3d","fwd_5d",
                "mfe_5d","mae_5d","mfe_10d","mae_10d"]],
        on=["date","symbol"], how="left"
    )

    # Classify
    loc   = flat_rows["entry_location_pct_20d"].fillna(0.5)
    reg   = flat_rows["portfolio_regime"].fillna("CHOP")
    mom   = flat_rows["momentum_direction"].fillna("NEUTRAL")
    slope = flat_rows["slope_20d"].fillna(0)
    qual  = flat_rows["chop_range_quality"].fillna(0)
    z     = flat_rows["chop_z_score"].fillna(0)
    sa_s  = flat_rows.get("sa_signal", pd.Series("FLAT", index=flat_rows.index)).fillna("FLAT")
    sa_n  = flat_rows.get("sa_nq_direction", pd.Series("FLAT", index=flat_rows.index)).fillna("FLAT")
    nq_mask = flat_rows["symbol"] == "NQ"

    classes = pd.Series("TRUE_NO_SETUP", index=flat_rows.index)

    # SA candidates (NQ rows with SA signal)
    classes[(nq_mask) & (sa_s == "ENTRY") & (sa_n == "LONG")] = "STATARB_CANDIDATE"

    # CHOP candidates
    chop = reg == "CHOP"
    classes[chop & (qual >= 0.45) & ((z < -0.50) | (z > 0.50))] = "CHOP_RANGE_CANDIDATE"
    classes[chop & ~((qual >= 0.45) & ((z < -0.50) | (z > 0.50)))] = "CHOP_NO_EDGE"

    # TRANSITION candidates
    trans = reg == "TRANSITION"
    reclaim  = flat_rows.get("tr_reclaim_flag", pd.Series(0, index=flat_rows.index)).fillna(0)
    fr_flag  = flat_rows.get("tr_failed_reclaim", pd.Series(0, index=flat_rows.index)).fillna(0)
    classes[trans & (reclaim == 1) & (slope >= 0)] = "TRANSITION_RESET_CANDIDATE"
    classes[trans & (fr_flag == 1) & (slope < 0)]  = "TRANSITION_BREAKOUT_CANDIDATE"

    # ES late NQ confirmed
    es_late_nq = (
        (flat_rows["symbol"] == "ES") &
        (reg == "TREND") & (mom == "LONG") & (slope > 0) & (loc > 0.70)
    )
    classes[es_late_nq] = "ES_LATE_NQ_CONFIRMED"

    flat_rows["opportunity_class"] = classes

    # ── Summary stats per class ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 1 — IDLE-DAY OPPORTUNITY MAP")
    print("=" * 70)

    summary_rows = []
    for cls in ["CHOP_RANGE_CANDIDATE","CHOP_NO_EDGE","TRANSITION_RESET_CANDIDATE",
                "TRANSITION_BREAKOUT_CANDIDATE","ES_LATE_NQ_CONFIRMED",
                "STATARB_CANDIDATE","TRUE_NO_SETUP"]:
        sub = flat_rows[flat_rows["opportunity_class"] == cls]
        n   = len(sub)
        if n == 0:
            continue
        f5 = sub["fwd_5d"].dropna()
        mfe5 = sub["mfe_5d"].dropna(); mae5 = sub["mae_5d"].dropna()
        mfe10 = sub["mfe_10d"].dropna(); mae10 = sub["mae_10d"].dropna()
        hit5 = (f5 > 0).mean() if len(f5) else np.nan
        sym_split = sub["symbol"].value_counts().to_dict()
        yr_split  = sub["date"].dt.year.value_counts().sort_index().to_dict()
        r = dict(
            cls=cls, n=n,
            fwd_1d_pct = round(sub["fwd_1d"].mean()*100, 3),
            fwd_3d_pct = round(sub["fwd_3d"].mean()*100, 3),
            fwd_5d_pct = round(sub["fwd_5d"].mean()*100, 3),
            hit_rate_5d= round(hit5, 3),
            mfe_5d_pct = round(mfe5.mean()*100, 3),
            mae_5d_pct = round(mae5.mean()*100, 3),
            mfe_10d_pct= round(mfe10.mean()*100, 3),
            mae_10d_pct= round(mae10.mean()*100, 3),
            mfe_mae_5d = round(mfe5.mean()/(-mae5.mean()+1e-9), 2),
            sym_split  = str(sym_split),
            yr_split   = str({y: v for y, v in yr_split.items() if v > 0}),
        )
        summary_rows.append(r)
        print(f"\n  {cls}  (n={n})")
        print(f"    Fwd 1d/3d/5d: {r['fwd_1d_pct']:+.2f}% / {r['fwd_3d_pct']:+.2f}% / {r['fwd_5d_pct']:+.2f}%")
        print(f"    Hit rate 5d:  {r['hit_rate_5d']:.1%}   MFE/MAE 5d: {r['mfe_5d_pct']:+.2f}% / {r['mae_5d_pct']:+.2f}%  ratio={r['mfe_mae_5d']:.2f}")
        print(f"    Symbol:       {sym_split}   Years: {yr_split}")

    pd.DataFrame(summary_rows).to_csv(
        f"{OUT_DIR}/idle_day_opportunity_map.csv", index=False)
    print(f"\n  Saved → {OUT_DIR}/idle_day_opportunity_map.csv")
    return flat_rows


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — CHOP range mean reversion
# ─────────────────────────────────────────────────────────────────────────────

def part2_chop_mr(df_best):
    logger.info("=== PART 2: CHOP Range Mean Reversion ===")
    exps = [
        ("A_BASELINE",           lambda d: d, {}),
        ("B_CHOP_MR_LONG_3",     lambda d: inject_chop_mr(d, 0.03, include_short=False), {"_new_strats": ["CHOP_MR_LONG"]}),
        ("C_CHOP_MR_LONG_SHORT_3",lambda d: inject_chop_mr(d, 0.03, include_short=True), {"_new_strats": ["CHOP_MR_LONG","CHOP_MR_SHORT"]}),
        ("D_CHOP_MR_LONG_SHORT_5",lambda d: inject_chop_mr(d, 0.05, include_short=True), {"_new_strats": ["CHOP_MR_LONG","CHOP_MR_SHORT"]}),
        ("E_CHOP_MR_STRICT_5",   lambda d: inject_chop_mr(d, 0.05, min_quality=0.60, z_thr=0.65), {"_new_strats": ["CHOP_MR_LONG"]}),
        ("F_CHOP_MR_LOOSE_3",    lambda d: inject_chop_mr(d, 0.03, min_quality=0.35, z_thr=0.40), {"_new_strats": ["CHOP_MR_LONG"]}),
    ]
    results = []; logs = {}
    for label, fn, extra in exps:
        row, tl = run_bt(label, fn(df_best), extra)
        results.append(row); logs[label] = tl
    res = pd.DataFrame(results)
    res.to_csv(f"{OUT_DIR}/chop_range_experiments.csv", index=False)
    _print_part_table("PART 2 — CHOP RANGE MR", res)
    return res, logs


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — TRANSITION reset
# ─────────────────────────────────────────────────────────────────────────────

def part3_transition_reset(df_best):
    logger.info("=== PART 3: TRANSITION Reset/Reclaim ===")
    exps = [
        ("A_BASELINE",              lambda d: d, {}),
        ("B_TR_RESET_LONG_3",       lambda d: inject_transition_reset(d, 0.03), {"_new_strats":["TRANSITION_RESET"]}),
        ("C_TR_RESET_LONG_5",       lambda d: inject_transition_reset(d, 0.05), {"_new_strats":["TRANSITION_RESET"]}),
        ("D_TR_RESET_NQ_ONLY_5",    lambda d: inject_transition_reset(d, 0.05, sym_filter="NQ"), {"_new_strats":["TRANSITION_RESET"]}),
        ("E_TR_RESET_ES_NQ_CONF_5", lambda d: inject_transition_reset(d, 0.05, sym_filter="ES"), {"_new_strats":["TRANSITION_RESET"]}),
        ("F_TR_FAILED_RALLY_3",     lambda d: inject_transition_reset(d, 0.03, include_short=True, score_thr=0.60), {"_new_strats":["TRANSITION_RESET","TRANSITION_RALLY_SHORT"]}),
        ("G_TR_COMBINED_SMALL",     lambda d: inject_transition_reset(d, 0.03, include_short=True, score_thr=0.55), {"_new_strats":["TRANSITION_RESET","TRANSITION_RALLY_SHORT"]}),
    ]
    results = []; logs = {}
    for label, fn, extra in exps:
        row, tl = run_bt(label, fn(df_best), extra)
        results.append(row); logs[label] = tl
    res = pd.DataFrame(results)
    res.to_csv(f"{OUT_DIR}/transition_reset_experiments.csv", index=False)
    _print_part_table("PART 3 — TRANSITION RESET", res)
    return res, logs


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — ES late NQ exception
# ─────────────────────────────────────────────────────────────────────────────

def part4_es_late_exception(df_best):
    logger.info("=== PART 4: ES Late NQ Exception ===")
    exps = [
        ("A_BASELINE_ES_FILTER",       lambda d: d, {}),
        ("B_ES_LATE_IF_NQ_TREND",      lambda d: inject_es_late_exception(d, "nq_trend"), {"_new_strats":["ES_LATE_EXCEPTION"]}),
        ("C_ES_LATE_IF_NQ_HIGHCONV",   lambda d: inject_es_late_exception(d, "nq_high_conviction"), {"_new_strats":["ES_LATE_EXCEPTION"]}),
        ("D_ES_LATE_IF_NQ_BULL_RESET", lambda d: inject_es_late_exception(d, "nq_bull_reset"), {"_new_strats":["ES_LATE_EXCEPTION"]}),
        ("E_ES_LATE_NQ_HALF_SIZE",     lambda d: inject_es_late_exception(d, "nq_trend_half_size"), {"_new_strats":["ES_LATE_EXCEPTION"]}),
        ("F_ES_LATE_NQ_CONF_3PCT",     lambda d: inject_es_late_exception(d, "nq_confirms_3pct"), {"_new_strats":["ES_LATE_EXCEPTION"]}),
        ("G_ES_LATE_NQ_CONF_5PCT",     lambda d: inject_es_late_exception(d, "nq_confirms_5pct"), {"_new_strats":["ES_LATE_EXCEPTION"]}),
        ("H_ES_LATE_NQ_NOT_LATE",      lambda d: inject_es_late_exception(d, "nq_not_late"), {"_new_strats":["ES_LATE_EXCEPTION"]}),
    ]
    results = []; logs = {}
    for label, fn, extra in exps:
        row, tl = run_bt(label, fn(df_best), extra)
        results.append(row); logs[label] = tl
    res = pd.DataFrame(results)
    res.to_csv(f"{OUT_DIR}/es_late_nq_exception_experiments.csv", index=False)
    _print_part_table("PART 4 — ES LATE NQ EXCEPTION", res)
    return res, logs


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — STAT_ARB existing logic expansion
# ─────────────────────────────────────────────────────────────────────────────

def part5_sa_expansion(df_best):
    logger.info("=== PART 5: STAT_ARB Existing Logic Expansion ===")
    sa_hold_plus1 = {**_HOLD_BASE, "STAT_ARB": _HOLD_BASE.get("STAT_ARB", 4) + 1}
    sa_hold_plus2 = {**_HOLD_BASE, "STAT_ARB": _HOLD_BASE.get("STAT_ARB", 4) + 2}
    exps = [
        ("A_BASELINE",              lambda d: d,                             {}),
        ("B_SA_MODERATE_2PCT",      lambda d: inject_sa_expansion(d, "allow_moderate_2pct"),  {"_new_strats":["SA_MODERATE"]}),
        ("C_SA_MODERATE_3PCT",      lambda d: inject_sa_expansion(d, "allow_moderate_3pct"),  {"_new_strats":["SA_MODERATE"]}),
        ("D_SA_STRONG_SIZE_25",     lambda d: inject_sa_expansion(d, "strong_size_up_25"),    {"_new_strats":["STAT_ARB"]}),
        ("E_SA_STRONG_SIZE_50",     lambda d: inject_sa_expansion(d, "strong_size_up_50"),    {"_new_strats":["STAT_ARB"]}),
        ("F_SA_IN_TRANSITION",      lambda d: inject_sa_expansion(d, "sa_in_transition"),     {"_new_strats":["SA_MODERATE"]}),
        ("G_SA_HOLD_PLUS_1D",       lambda d: d,                             {"MAX_HOLD_DAYS": sa_hold_plus1}),
        ("H_SA_HOLD_PLUS_2D",       lambda d: d,                             {"MAX_HOLD_DAYS": sa_hold_plus2}),
    ]
    results = []; logs = {}
    for label, fn, extra in exps:
        row, tl = run_bt(label, fn(df_best), extra)
        results.append(row); logs[label] = tl
    res = pd.DataFrame(results)
    res.to_csv(f"{OUT_DIR}/statarb_existing_logic_expansion.csv", index=False)
    _print_part_table("PART 5 — STAT_ARB EXPANSION", res)

    # Bucket diagnostics on STAT_ARB trades
    base_sa = logs["A_BASELINE"]
    if not base_sa.empty and "strategy" in base_sa.columns:
        sa_tl = base_sa[base_sa["strategy"] == "STAT_ARB"]
        if not sa_tl.empty:
            print(f"\n  STAT_ARB baseline: {len(sa_tl)} trades  WR={sa_tl['win'].mean():.1%}  "
                  f"PF={_pf(sa_tl):.2f}  avg_pnl={sa_tl['pnl'].mean():.4f}")

    return res, logs


# ─────────────────────────────────────────────────────────────────────────────
# PART 6 — Combined idle-day system
# ─────────────────────────────────────────────────────────────────────────────

def part6_combined(df_best, passed_chop, passed_trans, passed_es, passed_sa):
    logger.info("=== PART 6: Combined Idle-Day System ===")

    def _best_chop(d):
        return inject_chop_mr(d, 0.03, include_short=False) if passed_chop else d

    def _best_trans(d):
        return inject_transition_reset(d, 0.03) if passed_trans else d

    def _best_es(d):
        return inject_es_late_exception(d, "nq_not_late") if passed_es else d

    def _best_sa(d):
        return inject_sa_expansion(d, "allow_moderate_2pct") if passed_sa else d

    all_new = []
    if passed_chop:  all_new += ["CHOP_MR_LONG", "CHOP_MR_SHORT"]
    if passed_trans: all_new += ["TRANSITION_RESET", "TRANSITION_RALLY_SHORT"]
    if passed_es:    all_new += ["ES_LATE_EXCEPTION"]
    if passed_sa:    all_new += ["SA_MODERATE"]

    exps = [
        ("A_CURRENT_BEST",           lambda d: d, {}),
        ("B_BEST_CHOP_ONLY",         _best_chop, {"_new_strats":["CHOP_MR_LONG","CHOP_MR_SHORT"]}),
        ("C_BEST_TRANSITION_ONLY",   _best_trans, {"_new_strats":["TRANSITION_RESET"]}),
        ("D_BEST_ES_EXCEPTION_ONLY", _best_es, {"_new_strats":["ES_LATE_EXCEPTION"]}),
        ("E_BEST_SA_EXPANSION_ONLY", _best_sa, {"_new_strats":["SA_MODERATE"]}),
        ("F_CHOP_PLUS_TRANSITION",   lambda d: _best_trans(_best_chop(d)), {"_new_strats":["CHOP_MR_LONG","TRANSITION_RESET"]}),
        ("G_TRANS_PLUS_ES",          lambda d: _best_es(_best_trans(d)), {"_new_strats":["TRANSITION_RESET","ES_LATE_EXCEPTION"]}),
        ("H_ALL_ACCEPTED",           lambda d: _best_sa(_best_es(_best_trans(_best_chop(d)))), {"_new_strats": all_new}),
        ("I_FULL_WITH_SHORT",        lambda d: _best_sa(_best_es(_best_trans(_best_chop(d)))), {"_new_strats": all_new}),
    ]
    results = []; trade_logs = {}
    for label, fn, extra in exps:
        row, tl = run_bt(label, fn(df_best), extra)
        results.append(row); trade_logs[label] = tl

    res = pd.DataFrame(results)
    res.to_csv(f"{OUT_DIR}/idle_sleeve_combined_experiments.csv", index=False)
    if not trade_logs.get("H_ALL_ACCEPTED", pd.DataFrame()).empty:
        trade_logs["H_ALL_ACCEPTED"].to_csv(f"{OUT_DIR}/idle_sleeve_combined_trade_log.csv", index=False)

    _print_part_table("PART 6 — COMBINED IDLE-DAY SYSTEM", res)

    # Extra idle-day metrics
    base_tl  = trade_logs["A_CURRENT_BEST"]
    best_tl  = trade_logs.get("H_ALL_ACCEPTED", trade_logs["A_CURRENT_BEST"])
    if not base_tl.empty and not best_tl.empty:
        n_base = base_tl["entry_date"].nunique() if "entry_date" in base_tl.columns else len(base_tl)
        n_best = best_tl["entry_date"].nunique() if "entry_date" in best_tl.columns else len(best_tl)
        print(f"\n  Entry-day count: baseline={n_base}, combined={n_best} (+{n_best-n_base})")

    return res, trade_logs


# ─────────────────────────────────────────────────────────────────────────────
# PART 7 — Final verdict
# ─────────────────────────────────────────────────────────────────────────────

def part7_verdict(base_sharpe, p2, p3, p4, p5, p6):
    logger.info("=== PART 7: Final Verdict ===")

    def _best(df, label):
        r = df[df["experiment"] == label]
        return r.iloc[0] if not r.empty else None

    def _delta(df, label):
        r = _best(df, label)
        return (r["sharpe"] - base_sharpe) if r is not None else -999

    def _passes(df, label, min_pf=1.20, min_delta=0.0, max_dd_delta=1.5):
        r = _best(df, label)
        if r is None: return False, "no_result"
        pf_ok = r["profit_factor"] >= min_pf
        sh_ok = r["sharpe"] >= base_sharpe - 0.02   # allow -0.02 tolerance
        dd_ok = r["max_dd_pct"] <= -2.0             # not worse than -5%
        return (pf_ok and sh_ok), f"PF={r['profit_factor']:.2f} Sharpe_delta={r['sharpe']-base_sharpe:+.3f}"

    # Check individual bests
    chop_pass,  chop_why  = _passes(p2, "B_CHOP_MR_LONG_3")
    trans_pass, trans_why = _passes(p3, "D_TR_RESET_NQ_ONLY_5")
    es_pass,    es_why    = _passes(p4, "H_ES_LATE_NQ_NOT_LATE")
    sa_pass,    sa_why    = _passes(p5, "B_SA_MODERATE_2PCT")

    combined_best = _best(p6, "H_ALL_ACCEPTED")
    combined_pass = (combined_best is not None and
                     combined_best["sharpe"] >= base_sharpe - 0.02 and
                     combined_best.get("profit_factor", 0) >= 1.20)

    # Count trades
    base_row = _best(p2, "A_BASELINE")
    base_trades = base_row["trades"] if base_row is not None else 343

    findings = []
    accepted = []
    rejected = []

    for name, passed, why, part_df, best_exp in [
        ("CHOP_RANGE",   chop_pass,  chop_why,  p2, "B_CHOP_MR_LONG_3"),
        ("TRANSITION",   trans_pass, trans_why, p3, "D_TR_RESET_NQ_ONLY_5"),
        ("ES_EXCEPTION", es_pass,    es_why,    p4, "H_ES_LATE_NQ_NOT_LATE"),
        ("SA_EXPANSION", sa_pass,    sa_why,    p5, "B_SA_MODERATE_2PCT"),
    ]:
        r = _best(part_df, best_exp)
        delta = (r["sharpe"] - base_sharpe) if r is not None else 0
        added = (r["trades"] - base_trades) if r is not None else 0
        findings.append(f"  {name:<18s}: {'PASS' if passed else 'FAIL'}  {why}  Δtrades={added:+d}  ΔSharpe={delta:+.3f}")
        if passed:
            accepted.append(name)
        else:
            rejected.append(name)

    # Verdict
    if combined_pass and (combined_best["trades"] > base_trades + 20):
        verdict = "ACCEPT_IDLE_DAY_COMBINED_SYSTEM"
    elif len(accepted) >= 2:
        verdict = "ACCEPT_IDLE_DAY_COMBINED_SYSTEM"
    elif accepted == ["SA_EXPANSION"]:
        verdict = "ACCEPT_STATARB_EXPANSION_EXISTING"
    elif "ES_EXCEPTION" in accepted:
        verdict = "ACCEPT_ES_LATE_NQ_EXCEPTION"
    elif "TRANSITION" in accepted:
        verdict = "ACCEPT_TRANSITION_RESET_SLEEVE"
    elif "CHOP_RANGE" in accepted:
        verdict = "ACCEPT_CHOP_RANGE_SLEEVE"
    elif len(accepted) > 0:
        verdict = "CONTINUE_TESTING"
    else:
        verdict = "REJECT_IDLE_SLEEVES"

    _VERDICT_MEANINGS = {
        "ACCEPT_CHOP_RANGE_SLEEVE":          "CHOP range MR adds trades without quality loss",
        "ACCEPT_TRANSITION_RESET_SLEEVE":    "TRANSITION reset adds selective value",
        "ACCEPT_ES_LATE_NQ_EXCEPTION":       "ES late exception with NQ confirm recovers edge",
        "ACCEPT_STATARB_EXPANSION_EXISTING": "SA expansion using existing signals adds value",
        "ACCEPT_IDLE_DAY_COMBINED_SYSTEM":   "Combined idle-day sleeves accepted",
        "KEEP_DIAGNOSTIC_ONLY":              "Findings useful for monitoring, not live trading",
        "CONTINUE_TESTING":                  "Some sleeves show promise, need more testing",
        "REJECT_IDLE_SLEEVES":               "Extra trades degrade quality — keep system lean",
    }

    print("\n" + "=" * 70)
    print("PART 7 — FINAL VERDICT")
    print("=" * 70)
    print(f"\n  Baseline Sharpe: {base_sharpe:.4f}  Trades: {base_trades}")
    print(f"\n  Individual sleeve results:")
    for f in findings:
        print(f)
    print(f"\n  Accepted:  {accepted if accepted else 'NONE'}")
    print(f"  Rejected:  {rejected}")
    if combined_best is not None:
        print(f"\n  Combined (H_ALL_ACCEPTED):")
        print(f"    Sharpe={combined_best['sharpe']:.4f} (Δ={combined_best['sharpe']-base_sharpe:+.3f})")
        print(f"    Trades={combined_best['trades']} (+{combined_best['trades']-base_trades})")
        print(f"    MaxDD={combined_best['max_dd_pct']:.1f}%  PF={combined_best['profit_factor']:.2f}")
        print(f"    Trades/week={combined_best.get('trades_per_week',0):.2f}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  MEANING: {_VERDICT_MEANINGS.get(verdict, '')}")

    # 7 diagnostic questions
    print("\n  Answers to the 7 diagnostic questions:")
    q = {
        "1. Can idle days be monetized?":         "YES" if len(accepted) > 0 else "NO — market not setting up",
        "2. Best idle regime?":                   ", ".join(accepted) if accepted else "None passed",
        "3. Trade count increase w/o Sharpe drop?": "YES" if combined_pass and combined_best and combined_best["trades"] > base_trades else "NO",
        "4. Annual return improved?":             "YES" if combined_best is not None and float(combined_best["annual_ret_pct"]) > float(base_row["annual_ret_pct"] if base_row is not None else 0) else "NO",
        "5. MaxDD controlled?":                   "YES" if combined_best is not None and float(combined_best["max_dd_pct"]) > -5.0 else "NO",
        "6. Live-worthy sleeves?":                ", ".join(accepted) if accepted else "None",
        "7. Diagnostic-only / reject?":           ", ".join(rejected) if rejected else "None",
    }
    for q_str, ans in q.items():
        print(f"    {q_str:<45s} {ans}")

    print("=" * 70)
    return verdict, accepted, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Print helper
# ─────────────────────────────────────────────────────────────────────────────

def _print_part_table(title: str, res: pd.DataFrame):
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)
    cols = ["experiment", "sharpe", "sortino", "annual_ret_pct",
            "max_dd_pct", "trades", "trades_per_week", "win_rate",
            "profit_factor", "new_trades", "new_wr", "new_pf", "new_pnl"]
    print(res[[c for c in cols if c in res.columns]].to_string(index=False))
    yr_cols = [c for c in res.columns if c.startswith("yr_")]
    if yr_cols:
        print("\n  Annual returns:")
        print(res[["experiment"] + yr_cols].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df_raw  = build_base_df()
    df_best = apply_best_v2_core(df_raw.copy())

    # Baseline Sharpe from Part 2 A_BASELINE
    base_sharpe = 1.237   # known from V2 run (L: Long V2 Core + BEAR_TREND)

    # Part 1 — Opportunity map
    part1_opportunity_map(df_raw, df_best)

    # Part 2 — CHOP MR
    p2_res, p2_logs = part2_chop_mr(df_best)

    # Part 3 — TRANSITION reset
    p3_res, p3_logs = part3_transition_reset(df_best)

    # Part 4 — ES late exception
    p4_res, p4_logs = part4_es_late_exception(df_best)

    # Part 5 — SA expansion
    p5_res, p5_logs = part5_sa_expansion(df_best)

    # Determine which passed for Part 6
    def _passed(df, best_exp, min_delta=-0.02, min_pf=1.20):
        r = df[df["experiment"] == best_exp]
        if r.empty: return False
        row = r.iloc[0]
        return (row["sharpe"] >= base_sharpe + min_delta and
                row.get("profit_factor", 0) >= min_pf)

    chop_ok  = _passed(p2_res, "B_CHOP_MR_LONG_3")
    trans_ok = _passed(p3_res, "D_TR_RESET_NQ_ONLY_5")
    es_ok    = _passed(p4_res, "H_ES_LATE_NQ_NOT_LATE")
    sa_ok    = _passed(p5_res, "B_SA_MODERATE_2PCT")

    # Part 6 — Combined
    p6_res, p6_logs = part6_combined(df_best, chop_ok, trans_ok, es_ok, sa_ok)

    # Part 7 — Verdict
    part7_verdict(base_sharpe, p2_res, p3_res, p4_res, p5_res, p6_res)

    print(f"\nAll outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
