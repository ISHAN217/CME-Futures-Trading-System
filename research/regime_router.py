"""
regime_router.py — Regime-aware strategy activation flags for V2 architecture.

Adds six boolean flag columns that gate which strategy sleeves are allowed
on each row. All logic is strictly backward-looking.

Activation matrix
─────────────────
                      TREND  TRANSITION  CHOP  SHOCK
allow_trend_long        ✓        ✗         ✗     ✗
allow_bull_reset_long   ✓        ✓*        ✗     ✗    (* reduced size)
allow_stat_arb          ✓        ✓         ✓     ✗
allow_add_to_winner     ✓        ✗         ✗     ✗
allow_short             ✗        ✓         ✗     ✓
allow_shock_long        ✗        ✗         ✗     ✓

* BULL_RESET in TRANSITION: allowed but size is halved by the engine.

Output columns
──────────────
  allow_trend_long          bool-int  TREND_LONG entries permitted
  allow_bull_reset_long     bool-int  BULL_RESET_CONFIRMED entries permitted
  allow_stat_arb            bool-int  STAT_ARB entries permitted
  allow_add_to_winner       bool-int  Add-to-winner permitted
  allow_short               bool-int  Short sleeve permitted
  allow_shock_long          bool-int  SHOCK_BOUNCE long permitted
  router_size_scale         float     Regime-based size scalar (0.5 in TRANSITION)
  strategy_router_reason    str       Pipe-separated active strategies
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRANSITION_SIZE_SCALE = 0.50   # half-size for BULL_RESET in TRANSITION


def compute_regime_router(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime-activation flag columns to df.

    Requires:
      portfolio_regime, momentum_direction, weekly_bias, slope_20d,
      ret_5d, ret_20d, structure_break_count, portfolio_sentiment_flag,
      vol_regime, entry_location_pct_20d.
    """
    df = df.copy()

    regime    = df["portfolio_regime"].fillna("CHOP")
    mom_dir   = df["momentum_direction"].fillna("NEUTRAL")
    weekly    = df["weekly_bias"].fillna("NEUTRAL")
    sentiment = df["portfolio_sentiment_flag"].fillna("NORMAL")
    slope     = df["slope_20d"].fillna(0.0)
    ret20     = df["ret_20d"].fillna(0.0)
    ret5      = df["ret_5d"].fillna(0.0)
    bk_cnt    = df.get("structure_break_count", pd.Series(0, index=df.index)).fillna(0)
    loc20     = df.get("entry_location_pct_20d", pd.Series(0.5, index=df.index)).fillna(0.5)
    vol_reg   = df.get("vol_regime", pd.Series("NORMAL", index=df.index)).fillna("NORMAL")

    not_high_risk = sentiment != "HIGH_RISK"
    not_shock     = regime != "SHOCK"
    trend_ok      = regime == "TREND"
    trans_ok      = regime == "TRANSITION"
    shock_ok      = regime == "SHOCK"

    # ── allow_trend_long ──────────────────────────────────────────────────────
    # Full confirmation: TREND regime, bullish momentum, not at extreme late entry
    allow_trend_long = (
        trend_ok &
        (mom_dir == "LONG") &
        (weekly == "LONG") &
        not_high_risk
    )

    # ── allow_bull_reset_long ─────────────────────────────────────────────────
    # Pullback-reset entry: trend intact + recent dip + no structural break
    allow_bull_reset_long = (
        (trend_ok | trans_ok) &
        (slope > 0) &
        (ret20 > 0) &
        (ret5 < 0) &             # recent pullback
        (bk_cnt == 0) &          # structure still intact
        (loc20 < 0.80) &         # not at extreme high
        not_high_risk
    )

    # ── allow_stat_arb ────────────────────────────────────────────────────────
    # Any non-SHOCK regime is ok for spread trades
    allow_stat_arb = not_shock & not_high_risk

    # ── allow_add_to_winner ───────────────────────────────────────────────────
    # Only in clean TREND: structure intact, no SHOCK, not HIGH_RISK
    allow_add_to_winner = (
        trend_ok &
        (bk_cnt == 0) &
        not_high_risk
    )

    # ── allow_short ───────────────────────────────────────────────────────────
    # TRANSITION regime (potential failed rally) or SHOCK (continuation)
    # Also allow when TREND regime has turned clearly bearish (slope<0, ret20<0)
    bear_trend_in_trend = (
        trend_ok &
        (slope < 0) &
        (ret20 < 0) &
        (mom_dir == "NEUTRAL")
    )
    allow_short = (
        (trans_ok | shock_ok | bear_trend_in_trend) &
        not_high_risk
    )

    # ── allow_shock_long ──────────────────────────────────────────────────────
    allow_shock_long = shock_ok & not_high_risk

    # ── Size scalar ───────────────────────────────────────────────────────────
    # TRANSITION: reduce size for BULL_RESET to 50% of normal
    size_scale = pd.Series(1.0, index=df.index)
    size_scale[trans_ok] = _TRANSITION_SIZE_SCALE
    size_scale[shock_ok] = 0.75   # slight reduction in SHOCK

    # ── Reason string ─────────────────────────────────────────────────────────
    reasons = []
    for i in range(len(df)):
        parts = []
        if allow_trend_long.iloc[i]:       parts.append("TREND_LONG")
        if allow_bull_reset_long.iloc[i]:  parts.append("BULL_RESET")
        if allow_stat_arb.iloc[i]:         parts.append("SA")
        if allow_add_to_winner.iloc[i]:    parts.append("ADD")
        if allow_short.iloc[i]:            parts.append("SHORT")
        if allow_shock_long.iloc[i]:       parts.append("SHOCK_LONG")
        reasons.append("|".join(parts) or "BLOCKED")

    df["allow_trend_long"]      = allow_trend_long.astype(int)
    df["allow_bull_reset_long"] = allow_bull_reset_long.astype(int)
    df["allow_stat_arb"]        = allow_stat_arb.astype(int)
    df["allow_add_to_winner"]   = allow_add_to_winner.astype(int)
    df["allow_short"]           = allow_short.astype(int)
    df["allow_shock_long"]      = allow_shock_long.astype(int)
    df["router_size_scale"]     = size_scale.round(3)
    df["strategy_router_reason"]= reasons

    _log_summary(df)
    return df


def _log_summary(df: pd.DataFrame) -> None:
    n = max(len(df), 1)
    logger.info(
        "RegimeRouter — allow_trend=%.0f%% allow_reset=%.0f%% "
        "allow_sa=%.0f%% allow_short=%.0f%% allow_shock=%.0f%%  blocked=%.0f%%",
        100 * df["allow_trend_long"].mean(),
        100 * df["allow_bull_reset_long"].mean(),
        100 * df["allow_stat_arb"].mean(),
        100 * df["allow_short"].mean(),
        100 * df["allow_shock_long"].mean(),
        100 * (df["strategy_router_reason"] == "BLOCKED").mean(),
    )
