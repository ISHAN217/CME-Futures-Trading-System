"""
run_early_pullback_experiments.py — Early Trend + Pullback Continuation Engine
===============================================================================
DIAGNOSTIC CONTEXT:
  Baseline: Sharpe 0.882 | 533 trades | 61.5% WR | MaxDD -5.7%

  Key findings from missed-opportunity diagnostic:
    - System takes 91.5% of TREND signals — NOT too selective overall
    - REJECTED_NO_WEEKLY_BIAS:     avg_fwd_5d=+1.01%, MFE/MAE=3.73x (n=116)
    - REJECTED_PULLBACK_DISABLED:  avg_fwd_5d=+0.91%, MFE/MAE=2.82x (n=79)
    - NQ_BLOCKED sub-bucket:       avg_fwd_5d=+0.93%, hit_rate=69.8%
    - 23 fully missed trend legs (+145.5% total move, mostly in 2022)
    - Avg entry at 86% of 20d swing — late momentum chaser
    - Add-to-winner trades: 88% WR — drive most alpha

  Root cause: system lacks EARLY ENTRY and PULLBACK ENTRY.
  This script tests three new signal sleeves WITHOUT modifying baseline logic.

SIGNAL SLEEVES:
  1. EARLY_TREND_NQ      — NQ only; momentum early but not late; 2-day hold
  2. PULLBACK_CONTINUATION — ES+NQ; dip-buy inside TREND regime; 3-day hold
  3. RELAXED_WEEKLY_BIAS  — NQ only; allow NEUTRAL weekly bias if h4/quality confirms

INJECTION METHOD:
  Modified df rows (FLAT→LONG, strategy_used→"MTF_BREAKOUT" or "MTF_RESET")
  only on rows that were FLAT in baseline. Baseline signal rows are untouched.
  Uses existing compute_entry_weight() via "MTF_RESET"/"MTF_BREAKOUT" path.

EXPERIMENTS:
  A_BASELINE
  B_EARLY_TREND_NQ_ONLY
  C_PULLBACK_CONTINUATION_ONLY
  D_RELAXED_WEEKLY_BIAS_NQ
  E_EARLY + PULLBACK
  F_EARLY + RELAXED
  G_PULLBACK + RELAXED
  H_ALL_COMBINED

OUTPUTS (output/early_pullback/):
  early_pullback_experiments.csv
  early_pullback_trade_log.csv
"""

import contextlib
import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
from backtest           import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = "/Users/ishanbhardwaj/cme_competition_system/output/early_pullback"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Sizing constants ──────────────────────────────────────────────────────────
EARLY_TREND_SIZE   = 0.07   # 7%  — midpoint of 5-8% range specified
PULLBACK_SIZE      = 0.075  # 7.5% — midpoint of 5-10% range specified
RELAXED_BIAS_SIZE  = 0.05   # 5%  — as specified

# ── Hold day overrides ─────────────────────────────────────────────────────────
# "MTF_BREAKOUT" → 2-day hold (early trend: short leash)
# "MTF_RESET"    → default 3 days (config doesn't define it; backtest default=3)
_HOLD_WITH_EARLY = {**config.MAX_HOLD_DAYS, "MTF_BREAKOUT": 2, "MTF_RESET": 3}
_HOLD_WITHOUT    = {**config.MAX_HOLD_DAYS, "MTF_BREAKOUT": 3, "MTF_RESET": 3}


# ─────────────────────────────────────────────────────────────────────────────
# 1. FULL PIPELINE  (identical to run_trend_capture_experiments.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_base_df() -> pd.DataFrame:
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
            ("cwt_chop_label",        "CWT_UNCLEAR"),
            ("cwt_trade_allowed",     1),
            ("cwt_size_multiplier",   1.0),
            ("cwt_confidence",        0.0),
            ("cwt_total_energy",      0.0),
            ("cwt_entropy",           0.5),
            ("cwt_energy_z",          0.0),
            ("cwt_compression_flag",  0),
            ("cwt_expansion_flag",    0),
            ("cwt_high_energy_ratio", 1/3),
            ("cwt_mid_energy_ratio",  1/3),
            ("cwt_slow_energy_ratio", 1/3),
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
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. ENTRY-LOCATION + BREAKOUT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_entry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds backward-looking entry features (no lookahead):
      entry_location_pct_20d — position in 20d rolling range [0=low, 1=high]
      above_5d_high          — close > max of prior 5 closes (structure breakout)
      above_10d_high         — close > max of prior 10 closes
      ret_5d_neg             — True when ret_5d < 0 (recent pullback)
    """
    df = df.copy()
    for col in ["entry_location_pct_20d", "above_5d_high", "above_10d_high", "ret_5d_neg"]:
        df[col] = np.nan

    for sym, g in df.groupby("symbol", sort=False):
        idx = g.index
        s   = pd.Series(g["close"].values, index=idx)

        # Rolling 20d range position
        min_20 = s.rolling(20, min_periods=5).min()
        max_20 = s.rolling(20, min_periods=5).max()
        rng_20 = (max_20 - min_20).replace(0, np.nan)
        loc_20 = (s - min_20) / rng_20
        df.loc[idx, "entry_location_pct_20d"] = loc_20.clip(0, 1).values

        # 5-day / 10-day breakout above prior structure (shift(1) = exclude today)
        prior_high_5  = s.shift(1).rolling(5,  min_periods=3).max()
        prior_high_10 = s.shift(1).rolling(10, min_periods=5).max()
        df.loc[idx, "above_5d_high"]  = (s > prior_high_5 ).astype(float).values
        df.loc[idx, "above_10d_high"] = (s > prior_high_10).astype(float).values

        # Recent pullback flag
        ret5 = g["ret_5d"].values if "ret_5d" in g.columns else np.zeros(len(g))
        df.loc[idx, "ret_5d_neg"] = (pd.Series(ret5, index=idx) < 0).astype(float).values

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIGNAL INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def _inject(df: pd.DataFrame, mask: pd.Series, strategy: str, size: float) -> pd.DataFrame:
    """Apply signal injection on flat rows matching mask."""
    # Only fire on days that are still FLAT (prevents double-signalling)
    active = mask & (df["final_direction"] == "FLAT")
    df = df.copy()
    df.loc[active, "final_direction"] = "LONG"
    df.loc[active, "strategy_used"]   = strategy
    df.loc[active, "mtf_entry_size"]  = size
    df.loc[active, "entry_quality"]   = 3   # pass quality gate
    return df, int(active.sum())


def build_early_trend_nq_mask(df: pd.DataFrame) -> pd.Series:
    """
    EARLY_TREND_NQ — NQ only, emerging trend before regime/R² confirmation.

    V2 redesign (post-diagnostic):
      Root cause of 0 candidates in v1: 'above_5d_high' is geometrically
      contradictory with 'entry_loc < 0.60' — if price is in the lower 60%
      of its 20d range, it cannot simultaneously be making a new 5-day high.
      Removed that condition entirely.

      Additional finding: all NQ flat rows with LONG momentum + positive slope
      have weekly_bias == LONG; they are flat because of TRANSITION or CHOP
      regime blocks (signal_reason: TRANSITION_FLAT, not bias-related).
      So this signal = early trend entry overriding regime block with
      tight R² + location screen as quality gate.

    Conditions (all backward-looking, no lookahead):
      - NQ symbol
      - baseline signal is FLAT (regime or quality blocked existing signal)
      - momentum_direction == LONG
      - slope_20d > 0                       (upward structural trend)
      - r2_20d < 0.65                       (early/not-yet-mature trend)
      - weekly_bias != SHORT                (LONG or NEUTRAL bias OK)
      - entry_location_pct_20d < 0.65      (price not extended in 20d range)
      - portfolio_regime != SHOCK           (allow TREND/TRANSITION/CHOP)
      - portfolio_sentiment_flag != HIGH_RISK
    """
    loc = df["entry_location_pct_20d"].fillna(1.0)
    r2  = df["r2_20d"].fillna(0.0)
    return (
        (df["symbol"] == "NQ")
        & (df["final_direction"] == "FLAT")
        & (df["momentum_direction"].fillna("NEUTRAL") == "LONG")
        & (df["slope_20d"].fillna(0.0) > 0)
        & (r2 < 0.65)
        & (df["weekly_bias"].fillna("NEUTRAL") != "SHORT")
        & (loc < 0.65)
        & (df["portfolio_regime"].fillna("CHOP") != "SHOCK")
        & (df["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )


def build_pullback_continuation_mask(df: pd.DataFrame) -> pd.Series:
    """
    PULLBACK_CONTINUATION — ES and NQ.

    V2 redesign (post-diagnostic):
      Original (TREND-only) found only 18 candidates across 7 years — too few.
      Diagnostic showed 24 additional candidates in TRANSITION regime with the
      same pullback characteristics. Expanding to TRANSITION increases universe
      to 42 rows while keeping the pullback quality filter (ret_5d < 0 inside
      intact medium-term uptrend) which is more selective than taking all
      TRANSITION rows (prior simulation showed blanket TRANSITION entry = -0.002).

    Conditions (all backward-looking, no lookahead):
      - baseline signal is FLAT
      - portfolio_regime in (TREND, TRANSITION)   [expanded from TREND-only]
      - momentum_direction == LONG                 (uptrend structure intact)
      - slope_20d > 0
      - ret_20d > 0                                (medium-term HH/HL intact)
      - ret_5d < 0                                 (short-term pullback from high)
      - entry_location_pct_20d in [0.20, 0.70]    (slightly wider than v1)
      - portfolio_sentiment_flag != HIGH_RISK
    """
    loc  = df["entry_location_pct_20d"].fillna(1.0)
    ret5 = df["ret_5d"].fillna(0.0)
    ret20 = df["ret_20d"].fillna(0.0)
    return (
        (df["final_direction"] == "FLAT")
        & (df["portfolio_regime"].fillna("CHOP").isin(["TREND", "TRANSITION"]))
        & (df["momentum_direction"].fillna("NEUTRAL") == "LONG")
        & (df["slope_20d"].fillna(0.0) > 0)
        & (ret20 > 0)
        & (ret5 < 0)
        & (loc >= 0.20)
        & (loc <= 0.70)
        & (df["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )


def build_relaxed_weekly_bias_nq_mask(df: pd.DataFrame) -> pd.Series:
    """
    RELAXED_WEEKLY_BIAS_NQ — NQ only.
    Recovers the high-quality REJECTED_NO_WEEKLY_BIAS bucket (3.73x MFE/MAE).
    Allows entries when weekly_bias == NEUTRAL, but ONLY with quality confirmation:
      - h4_exec_signal in (MOMENTUM, BREAKOUT), OR
      - signal_quality_bucket in (GOOD, STRONG, EXCEPTIONAL), OR
      - high_conviction == True
    Other conditions:
      - NQ symbol
      - baseline signal is FLAT
      - weekly_bias == NEUTRAL  (would normally be rejected)
      - momentum_direction == LONG
      - portfolio_regime in (TREND, TRANSITION)
      - portfolio_sentiment_flag != HIGH_RISK
    """
    h4_ok   = df["h4_exec_signal"].fillna("NONE").isin(["MOMENTUM", "BREAKOUT"])
    # trend_quality_bucket is the right column (signal_quality_bucket doesn't exist)
    qual_ok = df["trend_quality_bucket"].fillna("MODERATE").isin(
        ["GOOD", "STRONG", "EXCEPTIONAL"]
    )
    hc_ok   = df.get("high_conviction", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    return (
        (df["symbol"] == "NQ")
        & (df["final_direction"] == "FLAT")
        & (df["weekly_bias"].fillna("NEUTRAL") == "NEUTRAL")
        & (df["momentum_direction"].fillna("NEUTRAL") == "LONG")
        & (df["portfolio_regime"].fillna("CHOP").isin(["TREND", "TRANSITION"]))
        & (h4_ok | qual_ok | hc_ok)
        & (df["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def override_config(**kwargs):
    originals = {k: getattr(config, k) for k in kwargs if hasattr(config, k)}
    for k, v in kwargs.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in originals.items():
            setattr(config, k, v)


def run_experiment(
    name: str,
    df_injected: pd.DataFrame,
    df_base: pd.DataFrame,
    baseline_tl: pd.DataFrame,
    has_early: bool = False,
) -> dict:
    """
    Run the backtest with injected signals.
    has_early=True → apply 2-day hold for MTF_BREAKOUT (EARLY_TREND_NQ).
    """
    hold_cfg = _HOLD_WITH_EARLY if has_early else _HOLD_WITHOUT
    with override_config(MAX_HOLD_DAYS=hold_cfg):
        result = run_backtest(df_injected)

    stats = result["stats"]
    tl    = result["trade_log"]
    dr    = result["daily_records"]

    # Identify added trades (not in baseline)
    added_strats = {"MTF_RESET", "MTF_BREAKOUT"}
    if tl.empty:
        added_tl = pd.DataFrame()
    else:
        added_tl = tl[tl["strategy"].isin(added_strats)].copy()

    n_added    = len(added_tl)
    add_wr     = added_tl["win"].mean() if n_added > 0 else np.nan
    add_pnl    = added_tl["pnl"].sum()  if n_added > 0 else 0.0
    add_pf     = _profit_factor(added_tl)

    # MFE/MAE for added trades from daily_records
    add_mfe, add_mae = _compute_mfe_mae(added_tl, dr)

    # Stop rate for added trades
    stop_mask  = added_tl["exit_reason"].str.contains("STOP_LOSS", na=False)
    add_stop_r = stop_mask.mean() if n_added > 0 else np.nan

    # PnL by symbol and year
    pnl_es = tl[tl["symbol"] == "ES"]["pnl"].sum() if not tl.empty else 0.0
    pnl_nq = tl[tl["symbol"] == "NQ"]["pnl"].sum() if not tl.empty else 0.0

    year_pnl_raw = {}
    port_ret = result["portfolio_returns"]
    for yr in range(2019, 2027):
        yr_mask = port_ret.index.year == yr
        if yr_mask.any():
            yr_ret = float((1 + port_ret[yr_mask]).prod() - 1)
            year_pnl_raw[yr] = round(yr_ret * 100, 2)

    # Added trade entry locations
    add_loc_mean = np.nan
    if n_added > 0 and "entry_date" in added_tl.columns:
        locs = []
        feat = df_base[["date", "symbol", "entry_location_pct_20d"]].copy()
        feat = feat.rename(columns={"date": "entry_date"})
        merged = added_tl[["symbol", "entry_date"]].merge(feat, on=["symbol", "entry_date"], how="left")
        add_loc_mean = merged["entry_location_pct_20d"].mean()

    # Overlap with baseline
    if not baseline_tl.empty and not tl.empty:
        baseline_keys = set(zip(baseline_tl["symbol"], baseline_tl["entry_date"].astype(str)))
        exp_keys      = set(zip(tl["symbol"], tl["entry_date"].astype(str)))
        overlap_n     = len(baseline_keys & exp_keys)
    else:
        overlap_n = 0

    row = {
        "experiment":        name,
        "sharpe":            round(stats.get("sharpe_ratio", 0), 4),
        "annual_ret_pct":    round(stats.get("annualized_return_pct", 0), 2),
        "total_pnl_pct":     round(stats.get("total_return_pct", 0), 2),
        "max_dd_pct":        round(stats.get("max_drawdown_pct", 0), 2),
        "trade_count":       stats.get("total_trades", 0),
        "win_rate":          round(stats.get("overall_win_rate", 0), 4),
        "profit_factor":     round(stats.get("profit_factor", 0), 4),
        "added_trade_count": n_added,
        "added_win_rate":    round(add_wr, 4) if not np.isnan(add_wr) else np.nan,
        "added_pnl_pct":     round(add_pnl * 100, 3),
        "added_profit_factor": round(add_pf, 4),
        "added_mfe_mean":    round(add_mfe, 4) if not np.isnan(add_mfe) else np.nan,
        "added_mae_mean":    round(add_mae, 4) if not np.isnan(add_mae) else np.nan,
        "added_stop_rate":   round(add_stop_r, 4) if not np.isnan(add_stop_r) else np.nan,
        "added_entry_loc":   round(add_loc_mean, 4) if not np.isnan(add_loc_mean) else np.nan,
        "pnl_ES":            round(pnl_es, 4),
        "pnl_NQ":            round(pnl_nq, 4),
        "overlap_with_base": overlap_n,
        **{f"yr_{k}": v for k, v in year_pnl_raw.items()},
    }
    return row, added_tl


def _profit_factor(tl: pd.DataFrame) -> float:
    if tl.empty or "pnl" not in tl.columns:
        return np.nan
    wins  = tl[tl["pnl"] > 0]["pnl"].sum()
    loss  = abs(tl[tl["pnl"] < 0]["pnl"].sum())
    return wins / loss if loss > 1e-9 else np.nan


def _compute_mfe_mae(added_tl: pd.DataFrame, dr: pd.DataFrame) -> tuple:
    """Extract MFE (max cum_ret) and MAE (min cum_ret) for each added trade."""
    if added_tl.empty or dr.empty:
        return np.nan, np.nan
    mfes, maes = [], []
    added_strats = {"MTF_RESET", "MTF_BREAKOUT"}
    for _, trade in added_tl.iterrows():
        sym    = trade["symbol"]
        ed     = trade["entry_date"]
        xd     = trade["exit_date"]
        mask   = (
            (dr["symbol"]   == sym)
            & (dr["date"]   >= ed)
            & (dr["date"]   <= xd)
            & (dr["strategy"].isin(added_strats))
        )
        sub = dr.loc[mask, "cum_ret"]
        if len(sub) > 0:
            mfes.append(sub.max())
            maes.append(sub.min())
    return (np.mean(mfes) if mfes else np.nan,
            np.mean(maes) if maes else np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments(df_base: pd.DataFrame):
    results      = []
    all_trade_logs = []

    # ── Precompute masks on baseline df ───────────────────────────────────────
    m_early    = build_early_trend_nq_mask(df_base)
    m_pullback = build_pullback_continuation_mask(df_base)
    m_relaxed  = build_relaxed_weekly_bias_nq_mask(df_base)

    logger.info("Signal candidate counts: EARLY=%d  PULLBACK=%d  RELAXED=%d",
                m_early.sum(), m_pullback.sum(), m_relaxed.sum())

    # ── A: Baseline ───────────────────────────────────────────────────────────
    logger.info("=== A_BASELINE ===")
    r_base  = run_backtest(df_base)
    base_tl = r_base["trade_log"]
    base_dr = r_base["daily_records"]
    base_st = r_base["stats"]
    base_port = r_base["portfolio_returns"]
    # Build year-by-year for baseline
    yr_base = {}
    for yr in range(2019, 2027):
        yr_mask = base_port.index.year == yr
        if yr_mask.any():
            yr_base[yr] = round(float((1 + base_port[yr_mask]).prod() - 1) * 100, 2)
    pnl_es_b = base_tl[base_tl["symbol"]=="ES"]["pnl"].sum() if not base_tl.empty else 0.0
    pnl_nq_b = base_tl[base_tl["symbol"]=="NQ"]["pnl"].sum() if not base_tl.empty else 0.0
    row_a = {
        "experiment":        "A_BASELINE",
        "sharpe":            round(base_st.get("sharpe_ratio", 0), 4),
        "annual_ret_pct":    round(base_st.get("annualized_return_pct", 0), 2),
        "total_pnl_pct":     round(base_st.get("total_return_pct", 0), 2),
        "max_dd_pct":        round(base_st.get("max_drawdown_pct", 0), 2),
        "trade_count":       base_st.get("total_trades", 0),
        "win_rate":          round(base_st.get("overall_win_rate", 0), 4),
        "profit_factor":     round(base_st.get("profit_factor", 0), 4),
        "added_trade_count": 0,
        "added_win_rate":    np.nan,
        "added_pnl_pct":     0.0,
        "added_profit_factor": np.nan,
        "added_mfe_mean":    np.nan,
        "added_mae_mean":    np.nan,
        "added_stop_rate":   np.nan,
        "added_entry_loc":   np.nan,
        "pnl_ES":            round(pnl_es_b, 4),
        "pnl_NQ":            round(pnl_nq_b, 4),
        "overlap_with_base": 0,
        **{f"yr_{k}": v for k, v in yr_base.items()},
    }
    results.append(row_a)

    # ── B: EARLY_TREND_NQ only ────────────────────────────────────────────────
    logger.info("=== B_EARLY_TREND_NQ_ONLY ===")
    df_b = df_base.copy()
    df_b, n_b = _inject(df_b, m_early, "MTF_BREAKOUT", EARLY_TREND_SIZE)
    logger.info("  Injected %d EARLY_TREND_NQ rows", n_b)
    row_b, tl_b = run_experiment("B_EARLY_TREND_NQ_ONLY", df_b, df_base, base_tl, has_early=True)
    results.append(row_b); all_trade_logs.append(tl_b)

    # ── C: PULLBACK_CONTINUATION only ─────────────────────────────────────────
    logger.info("=== C_PULLBACK_CONTINUATION_ONLY ===")
    df_c = df_base.copy()
    df_c, n_c = _inject(df_c, m_pullback, "MTF_RESET", PULLBACK_SIZE)
    logger.info("  Injected %d PULLBACK_CONTINUATION rows", n_c)
    row_c, tl_c = run_experiment("C_PULLBACK_CONTINUATION_ONLY", df_c, df_base, base_tl)
    results.append(row_c); all_trade_logs.append(tl_c)

    # ── D: RELAXED_WEEKLY_BIAS_NQ only ────────────────────────────────────────
    logger.info("=== D_RELAXED_WEEKLY_BIAS_NQ ===")
    df_d = df_base.copy()
    df_d, n_d = _inject(df_d, m_relaxed, "MTF_RESET", RELAXED_BIAS_SIZE)
    logger.info("  Injected %d RELAXED_WEEKLY_BIAS_NQ rows", n_d)
    row_d, tl_d = run_experiment("D_RELAXED_WEEKLY_BIAS_NQ", df_d, df_base, base_tl)
    results.append(row_d); all_trade_logs.append(tl_d)

    # ── E: EARLY + PULLBACK ───────────────────────────────────────────────────
    logger.info("=== E_EARLY_PLUS_PULLBACK ===")
    df_e = df_base.copy()
    df_e, n_e1 = _inject(df_e, m_early,    "MTF_BREAKOUT", EARLY_TREND_SIZE)
    df_e, n_e2 = _inject(df_e, m_pullback,  "MTF_RESET",   PULLBACK_SIZE)
    logger.info("  Injected %d EARLY + %d PULLBACK rows", n_e1, n_e2)
    row_e, tl_e = run_experiment("E_EARLY_PLUS_PULLBACK", df_e, df_base, base_tl, has_early=True)
    results.append(row_e); all_trade_logs.append(tl_e)

    # ── F: EARLY + RELAXED ────────────────────────────────────────────────────
    logger.info("=== F_EARLY_PLUS_RELAXED ===")
    df_f = df_base.copy()
    df_f, n_f1 = _inject(df_f, m_early,   "MTF_BREAKOUT", EARLY_TREND_SIZE)
    df_f, n_f2 = _inject(df_f, m_relaxed,  "MTF_RESET",   RELAXED_BIAS_SIZE)
    logger.info("  Injected %d EARLY + %d RELAXED rows", n_f1, n_f2)
    row_f, tl_f = run_experiment("F_EARLY_PLUS_RELAXED", df_f, df_base, base_tl, has_early=True)
    results.append(row_f); all_trade_logs.append(tl_f)

    # ── G: PULLBACK + RELAXED ─────────────────────────────────────────────────
    logger.info("=== G_PULLBACK_PLUS_RELAXED ===")
    df_g = df_base.copy()
    df_g, n_g1 = _inject(df_g, m_pullback, "MTF_RESET", PULLBACK_SIZE)
    df_g, n_g2 = _inject(df_g, m_relaxed,  "MTF_RESET", RELAXED_BIAS_SIZE)
    logger.info("  Injected %d PULLBACK + %d RELAXED rows", n_g1, n_g2)
    row_g, tl_g = run_experiment("G_PULLBACK_PLUS_RELAXED", df_g, df_base, base_tl)
    results.append(row_g); all_trade_logs.append(tl_g)

    # ── H: ALL COMBINED ───────────────────────────────────────────────────────
    logger.info("=== H_ALL_COMBINED ===")
    df_h = df_base.copy()
    df_h, n_h1 = _inject(df_h, m_early,    "MTF_BREAKOUT", EARLY_TREND_SIZE)
    df_h, n_h2 = _inject(df_h, m_pullback,  "MTF_RESET",   PULLBACK_SIZE)
    df_h, n_h3 = _inject(df_h, m_relaxed,   "MTF_RESET",   RELAXED_BIAS_SIZE)
    logger.info("  Injected %d EARLY + %d PULLBACK + %d RELAXED rows", n_h1, n_h2, n_h3)
    row_h, tl_h = run_experiment("H_ALL_COMBINED", df_h, df_base, base_tl, has_early=True)
    results.append(row_h); all_trade_logs.append(tl_h)

    return pd.DataFrame(results), all_trade_logs, base_tl, base_dr, df_base



# ─────────────────────────────────────────────────────────────────────────────
# 6. DIAGNOSTICS FOR ADDED TRADES
# ─────────────────────────────────────────────────────────────────────────────

def build_trade_diagnostics(all_trade_logs, df_base, base_tl):
    """
    For each experiment's added trades:
      - entry_location_pct_20d (are we entering early?)
      - MFE, MAE (quality of the trade)
      - hold_duration
      - exit_reason distribution
      - whether a TREND trade follows within 3 days (leads into core trend)
    """
    diag_rows = []
    feat = df_base[["date", "symbol", "entry_location_pct_20d",
                    "ret_20d", "slope_20d", "r2_20d",
                    "portfolio_regime", "weekly_bias"]].copy()

    # Build a set of (symbol, date) → next TREND entry date (for sequencing check)
    if not base_tl.empty:
        trend_entries = base_tl[base_tl["strategy"] == "TREND"][["symbol", "entry_date"]].copy()
        trend_entries = trend_entries.sort_values(["symbol", "entry_date"])
    else:
        trend_entries = pd.DataFrame(columns=["symbol", "entry_date"])

    exp_names = [
        "B_EARLY_TREND_NQ_ONLY",
        "C_PULLBACK_CONTINUATION_ONLY",
        "D_RELAXED_WEEKLY_BIAS_NQ",
        "E_EARLY_PLUS_PULLBACK",
        "F_EARLY_PLUS_RELAXED",
        "G_PULLBACK_PLUS_RELAXED",
        "H_ALL_COMBINED",
    ]

    for exp_name, tl in zip(exp_names, all_trade_logs):
        if tl.empty:
            continue
        for _, trade in tl.iterrows():
            sym = trade["symbol"]
            ed  = trade["entry_date"]
            xd  = trade["exit_date"]

            # Entry location
            row_feat = feat[(feat["date"] == ed) & (feat["symbol"] == sym)]
            loc  = row_feat["entry_location_pct_20d"].values[0] if len(row_feat) > 0 else np.nan
            r2v  = row_feat["r2_20d"].values[0]    if len(row_feat) > 0 else np.nan
            regm = row_feat["portfolio_regime"].values[0] if len(row_feat) > 0 else ""
            biasv= row_feat["weekly_bias"].values[0]      if len(row_feat) > 0 else ""

            # Does a TREND trade follow within 3 days of this trade's exit?
            if not trend_entries.empty:
                fut_trend = trend_entries[
                    (trend_entries["symbol"] == sym) &
                    (trend_entries["entry_date"] > xd) &
                    (trend_entries["entry_date"] <= xd + pd.Timedelta(days=3))
                ]
                leads_to_trend = len(fut_trend) > 0
            else:
                leads_to_trend = False

            diag_rows.append({
                "experiment":        exp_name,
                "symbol":            sym,
                "strategy":          trade["strategy"],
                "entry_date":        ed,
                "exit_date":         xd,
                "holding_days":      trade["holding_days"],
                "pnl":               trade["pnl"],
                "win":               trade["win"],
                "exit_reason":       trade["exit_reason"],
                "entry_location_pct": round(loc, 4) if not np.isnan(loc) else np.nan,
                "r2_at_entry":       round(r2v, 4) if not np.isnan(r2v) else np.nan,
                "entry_regime":      regm,
                "entry_weekly_bias": biasv,
                "leads_to_trend":    leads_to_trend,
                "entry_weight":      trade.get("entry_weight", np.nan),
                "add_count":         trade.get("add_count", 0),
            })

    return pd.DataFrame(diag_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report(results: pd.DataFrame, trade_diag: pd.DataFrame, base_sharpe: float):
    baseline_dd = results.loc[results["experiment"] == "A_BASELINE", "max_dd_pct"].values[0]

    print("\n" + "=" * 70)
    print("EARLY TREND + PULLBACK CONTINUATION — FINAL REPORT")
    print("=" * 70)

    # Summary table
    cols = ["experiment", "sharpe", "annual_ret_pct", "max_dd_pct",
            "trade_count", "win_rate", "added_trade_count",
            "added_win_rate", "added_pnl_pct", "added_profit_factor",
            "added_mfe_mean", "added_mae_mean"]
    display = results[[c for c in cols if c in results.columns]].copy()
    display["delta_sharpe"] = (display["sharpe"] - base_sharpe).round(4)
    print(display.to_string(index=False))

    # Year-by-year consistency
    yr_cols = [c for c in results.columns if c.startswith("yr_")]
    if yr_cols:
        print("\n--- Annual Return % by Year ---")
        print(results[["experiment"] + yr_cols].to_string(index=False))

    # Q1: Which sleeve worked best?
    best_row = results.loc[results["sharpe"].idxmax()]
    print(f"\nQ1: Best sleeve: {best_row['experiment']} (Sharpe {best_row['sharpe']:.4f})")

    # Q2: Are early NQ trades real edge or noise?
    b_row = results[results["experiment"] == "B_EARLY_TREND_NQ_ONLY"]
    if len(b_row):
        b = b_row.iloc[0]
        is_edge = (b["sharpe"] > base_sharpe and b["added_win_rate"] >= 0.60
                   and not np.isnan(b.get("added_profit_factor", np.nan))
                   and b.get("added_profit_factor", 0) >= 1.2)
        verdict_b = "REAL EDGE" if is_edge else "NOISE or MARGINAL"
        print(f"\nQ2: EARLY_TREND_NQ → {verdict_b}")
        print(f"    Sharpe delta: {b['sharpe'] - base_sharpe:+.4f}  "
              f"WR: {b.get('added_win_rate', np.nan):.1%}  "
              f"PF: {b.get('added_profit_factor', np.nan):.2f}")

    # Q3: Pullbacks better than continuation?
    c_row = results[results["experiment"] == "C_PULLBACK_CONTINUATION_ONLY"]
    if len(b_row) and len(c_row):
        c = c_row.iloc[0]
        b = b_row.iloc[0]
        better = "PULLBACK" if c["sharpe"] > b["sharpe"] else "EARLY_TREND"
        print(f"\nQ3: Pullback vs Early Trend → {better} wins")
        print(f"    PULLBACK Sharpe: {c['sharpe']:.4f}  EARLY Sharpe: {b['sharpe']:.4f}")

    # Q4: Is weekly bias filter too strict?
    d_row = results[results["experiment"] == "D_RELAXED_WEEKLY_BIAS_NQ"]
    if len(d_row):
        d = d_row.iloc[0]
        too_strict = d["sharpe"] > base_sharpe + 0.01
        print(f"\nQ4: Weekly bias filter too strict? {'YES' if too_strict else 'NO — keep the filter'}")
        print(f"    RELAXED_WEEKLY_BIAS Sharpe: {d['sharpe']:.4f}  delta: {d['sharpe']-base_sharpe:+.4f}")

    # Q5: Does this improve trend capture materially?
    if not trade_diag.empty:
        early_loc = trade_diag[trade_diag["strategy"] == "MTF_BREAKOUT"]["entry_location_pct"]
        pull_loc  = trade_diag[trade_diag["strategy"] == "MTF_RESET"]["entry_location_pct"]
        print(f"\nQ5: Trend capture improvement:")
        print(f"    EARLY_TREND avg entry location: {early_loc.mean():.2%}  (baseline: ~86%)")
        print(f"    PULLBACK avg entry location:    {pull_loc.mean():.2%}")

    # Q6: Additional trend legs captured
    if not trade_diag.empty:
        leads = trade_diag["leads_to_trend"].sum()
        total = len(trade_diag)
        print(f"\nQ6: Added trades leading into core TREND position: {leads}/{total} ({leads/max(total,1):.1%})")

    # Q7: Stability across years
    h_row = results[results["experiment"] == "H_ALL_COMBINED"]
    if len(h_row) and yr_cols:
        h   = h_row.iloc[0]
        yrs = [h.get(c, np.nan) for c in yr_cols]
        pos = sum(1 for y in yrs if y > 0)
        print(f"\nQ7: H_ALL_COMBINED profitable in {pos}/{len(yr_cols)} years")
        for col, yr in zip(yr_cols, yrs):
            print(f"    {col.replace('yr_', '')}: {yr:+.2f}%")

    # Q8: Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    best_sharpe = results["sharpe"].max()
    best_dd     = results.loc[results["sharpe"].idxmax(), "max_dd_pct"]
    delta_s     = best_sharpe - base_sharpe
    delta_dd    = best_dd - baseline_dd

    # Check added-trade quality for best experiment
    best_exp    = results.loc[results["sharpe"].idxmax(), "experiment"]
    best_add_wr = results.loc[results["sharpe"].idxmax(), "added_win_rate"]
    best_add_pf = results.loc[results["sharpe"].idxmax(), "added_profit_factor"]
    add_ok      = (
        not np.isnan(best_add_wr) and best_add_wr >= 0.60
        and not np.isnan(best_add_pf) and best_add_pf >= 1.20
    )

    if delta_s > 0.05 and delta_dd <= 0.8 and add_ok:
        verdict = "ACCEPT_COMBINED" if "COMBINED" in best_exp else f"ACCEPT_{best_exp.split('_')[0]}"
    elif delta_s > 0.02 and delta_dd <= 0.8:
        verdict = "CONTINUE_TESTING"
    elif delta_s > 0.005 and add_ok:
        verdict = "CONTINUE_TESTING"
    elif delta_s < -0.02:
        verdict = "REJECT"
    else:
        verdict = "CONTINUE_TESTING"

    # Specific sub-verdicts
    print(f"Best experiment:  {best_exp}")
    print(f"Best Sharpe:      {best_sharpe:.4f}  (delta {delta_s:+.4f} vs baseline {base_sharpe:.4f})")
    print(f"MaxDD:            {best_dd:.1f}%  (delta {delta_dd:+.1f}%)")
    print(f"Added trade WR:   {best_add_wr:.1%}" if not np.isnan(best_add_wr) else "Added trade WR: N/A")
    print(f"Added trade PF:   {best_add_pf:.2f}" if not np.isnan(best_add_pf) else "Added trade PF: N/A")
    print()

    # Per-sleeve verdicts
    sleeve_names = {
        "B_EARLY_TREND_NQ_ONLY":        "EARLY_TREND_NQ",
        "C_PULLBACK_CONTINUATION_ONLY":  "PULLBACK_CONTINUATION",
        "D_RELAXED_WEEKLY_BIAS_NQ":      "RELAXED_WEEKLY_BIAS",
    }
    for exp_key, sleeve in sleeve_names.items():
        r = results[results["experiment"] == exp_key]
        if len(r):
            r = r.iloc[0]
            ds = r["sharpe"] - base_sharpe
            wr = r.get("added_win_rate", np.nan)
            pf = r.get("added_profit_factor", np.nan)
            ok = (not np.isnan(wr) and wr >= 0.60 and not np.isnan(pf) and pf >= 1.20)
            sv = "ACCEPT" if (ds > 0.02 and ok) else ("CONTINUE" if ds > 0 else "REJECT")
            print(f"  {sleeve:30s}: delta_sharpe={ds:+.4f}  WR={wr:.1%}  PF={pf:.2f}  → {sv}"
                  if not np.isnan(wr) else
                  f"  {sleeve:30s}: delta_sharpe={ds:+.4f}  → {sv}")

    print()
    print(f"  *** FINAL RECOMMENDATION: {verdict} ***")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== Early Trend + Pullback Continuation Engine ===")

    # Build pipeline + add entry features
    df_base = build_base_df()
    df_base = compute_entry_features(df_base)

    # Run all experiments
    results, all_trade_logs, base_tl, base_dr, df_base = run_all_experiments(df_base)

    # Add delta_sharpe column
    base_sharpe = results.loc[results["experiment"] == "A_BASELINE", "sharpe"].values[0]
    results["delta_sharpe"] = (results["sharpe"] - base_sharpe).round(4)

    # Build trade diagnostics
    trade_diag = build_trade_diagnostics(all_trade_logs, df_base, base_tl)

    # Add dollar values ($25k base, annualised over 7 years)
    results["final_dollar_25k"] = (
        25000 * (1 + results["annual_ret_pct"] / 100) ** 7
    ).round(0).astype(int)
    results["gain_dollar_25k"]  = results["final_dollar_25k"] - 25000
    results["maxdd_dollar_25k"] = (
        results["max_dd_pct"] / 100 * 25000
    ).round(0).astype(int)

    # Save outputs
    exp_path  = os.path.join(OUT_DIR, "early_pullback_experiments.csv")
    log_path  = os.path.join(OUT_DIR, "early_pullback_trade_log.csv")
    results.to_csv(exp_path, index=False)
    trade_diag.to_csv(log_path, index=False)

    logger.info("Saved experiments → %s", exp_path)
    logger.info("Saved trade log   → %s", log_path)

    # Print final report
    print_final_report(results, trade_diag, base_sharpe)

    # Summary table for console
    print("\n--- EXPERIMENT SUMMARY ($25k base, 7yr) ---")
    print(f"{'Experiment':<34} {'Sharpe':>7} {'Δ Sharpe':>9} {'Ann%':>6} "
          f"{'MaxDD%':>7} {'Trades':>7} {'Added':>6} {'Add WR':>7} "
          f"{'Add PF':>7} {'Final$':>9} {'Gain$':>8}")
    print("-" * 130)
    for _, r in results.iterrows():
        wr  = f"{r['added_win_rate']:.1%}" if not pd.isna(r.get("added_win_rate")) else "  —  "
        pf  = f"{r['added_profit_factor']:.2f}" if not pd.isna(r.get("added_profit_factor")) else "  —"
        print(
            f"{r['experiment']:<34} {r['sharpe']:7.4f} {r['delta_sharpe']:+9.4f} "
            f"{r['annual_ret_pct']:6.2f} {r['max_dd_pct']:7.2f} "
            f"{r['trade_count']:7d} {r['added_trade_count']:6d} {wr:>7} {pf:>7} "
            f"${r['final_dollar_25k']:>8,} ${r['gain_dollar_25k']:>7,}"
        )

    logger.info("All outputs saved to: %s", OUT_DIR)


if __name__ == "__main__":
    main()
