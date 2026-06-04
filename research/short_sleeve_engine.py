"""
short_sleeve_engine.py — Regime-confirmed short signal injection for V2.

Four short types (all regime-gated, NOT pullback shorts in bull trends):

  BEAR_TREND_SHORT
    Sustained downtrend on both timeframes.
    Regime: TRANSITION or (TREND + bear_trend_in_trend flag)
    Requires: slope < 0, ret_20d < 0, ret_60d < 0, mom==NEUTRAL or SHORT,
              loc20 > 0.40 (not deeply oversold), not HIGH_RISK

  FAILED_RALLY_SHORT
    Recent rally in bearish regime fails below prior swing high.
    Regime: TRANSITION or SHOCK
    Requires: ret_5d > 0 (recent bounce), but ret_20d < 0 (downtrend intact),
              loc20 > 0.55 (bounced into upper range), slope < 0, not HIGH_RISK

  SHOCK_CONTINUATION_SHORT
    After initial SHOCK drop, vol still elevated — continuation short.
    Regime: SHOCK only
    Requires: ret_5d < 0 (still falling), vol_regime in {ELEVATED, HIGH},
              slope < 0, not HIGH_RISK

  BEAR_RESET_CONFIRMED
    Symmetric to BULL_RESET: bounce in a downtrend forms a lower high.
    Regime: TRANSITION
    Requires: slope < 0, ret_20d < 0, ret_5d > 0 (bounce),
              loc20 > 0.50 (bounced into middle/upper range), bk_cnt >= 1,
              not HIGH_RISK

Output columns added to df:
  short_signal_type     str  BEAR_TREND / FAILED_RALLY / SHOCK_CONT /
                             BEAR_RESET / NONE
  short_signal_size     float  fraction of notional to short
  short_signal_reason   str  pipe-separated conditions that triggered
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Size parameters ──────────────────────────────────────────────────────────
BEAR_TREND_SIZE    = 0.08   # smallest — sustained trend, moderate conviction
FAILED_RALLY_SIZE  = 0.10   # failed rally has timing edge
SHOCK_CONT_SIZE    = 0.07   # shock continuation: smallest due to volatility risk
BEAR_RESET_SIZE    = 0.06   # lowest confidence, strictest conditions

# ─── Minimum z-score (momentum) to permit short ───────────────────────────────
_MIN_BEAR_SLOPE    = -0.0005   # slope must be meaningfully negative (not noise)


def compute_short_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject regime-confirmed short signals.

    Modifies rows where final_direction == 'FLAT' and short conditions are met.
    Does NOT touch rows that already have a direction.

    Requires columns (most already in pipeline):
      portfolio_regime, momentum_direction, weekly_bias, slope_20d,
      ret_5d, ret_20d, ret_60d, entry_location_pct_20d,
      structure_break_count, portfolio_sentiment_flag, vol_regime
    """
    df = df.copy()

    # ── Ensure short output columns exist ────────────────────────────────────
    if "short_signal_type"   not in df.columns: df["short_signal_type"]   = "NONE"
    if "short_signal_size"   not in df.columns: df["short_signal_size"]   = 0.0
    if "short_signal_reason" not in df.columns: df["short_signal_reason"] = ""

    regime    = df["portfolio_regime"].fillna("CHOP")
    mom_dir   = df["momentum_direction"].fillna("NEUTRAL")
    slope     = df["slope_20d"].fillna(0.0)
    ret20     = df["ret_20d"].fillna(0.0)
    ret5      = df["ret_5d"].fillna(0.0)
    ret60     = df["ret_60d"].fillna(0.0)
    sentiment = df["portfolio_sentiment_flag"].fillna("NORMAL")
    loc20     = df.get("entry_location_pct_20d",
                       pd.Series(0.5, index=df.index)).fillna(0.5)
    bk_cnt    = df.get("structure_break_count",
                       pd.Series(0, index=df.index)).fillna(0)
    vol_reg   = df.get("vol_regime",
                       pd.Series("NORMAL", index=df.index)).fillna("NORMAL")

    flat        = df["final_direction"] == "FLAT"
    not_hr      = sentiment != "HIGH_RISK"
    trend_ok    = regime == "TREND"
    trans_ok    = regime == "TRANSITION"
    shock_ok    = regime == "SHOCK"
    bear_slope  = slope < _MIN_BEAR_SLOPE
    bear_20d    = ret20 < 0
    bear_60d    = ret60 < 0

    # ── 1. BEAR_TREND_SHORT ──────────────────────────────────────────────────
    # Sustained downtrend confirmed on all timeframes. Enter continuation short
    # when price is not yet deeply oversold (loc > 0.40 — room to fall).
    bear_trend_in_trend = (
        trend_ok & bear_slope & bear_20d & (mom_dir == "NEUTRAL")
    )
    bear_trend_mask = (
        flat &
        (trans_ok | bear_trend_in_trend) &
        bear_slope &
        bear_20d &
        bear_60d &
        (mom_dir.isin(["NEUTRAL", "SHORT"])) &
        (loc20 > 0.40) &      # not already deeply oversold
        not_hr
    )

    # ── 2. FAILED_RALLY_SHORT ────────────────────────────────────────────────
    # Price bounced (ret_5d > 0) into upper range within a downtrend regime.
    # The rally has failed: slope still negative, 20d trend still down.
    # Only in TRANSITION or SHOCK — do NOT take this in TREND regime.
    failed_rally_mask = (
        flat &
        (trans_ok | shock_ok) &
        (ret5 > 0.005) &      # meaningful recent bounce
        bear_20d &            # 20d trend still bearish
        bear_slope &
        (loc20 > 0.55) &      # bounced into upper half of range
        not_hr
    )

    # ── 3. SHOCK_CONTINUATION_SHORT ─────────────────────────────────────────
    # Post-SHOCK regime: price still falling, vol still elevated → continuation.
    # Very tight: must still be falling AND in elevated vol.
    shock_cont_mask = (
        flat &
        shock_ok &
        (ret5 < -0.01) &       # still actively falling
        bear_slope &
        (vol_reg.isin(["ELEVATED", "HIGH"])) &
        not_hr
    )

    # ── 4. BEAR_RESET_CONFIRMED ──────────────────────────────────────────────
    # Symmetric to BULL_RESET: in a confirmed downtrend, price bounces (ret_5d>0)
    # into resistance — structure already broken (bk_cnt >= 1) confirming the
    # downtrend. Enter short on the bounce into a lower high.
    # Only TRANSITION regime — requires the downtrend is already the regime.
    bear_reset_mask = (
        flat &
        trans_ok &
        bear_slope &
        bear_20d &
        (ret5 > 0.003) &        # bounce present
        (loc20 > 0.50) &        # bounced into middle/upper range (lower high)
        (bk_cnt >= 1) &         # at least one structure break confirming trend
        not_hr
    )

    # ── Resolve conflicts: priority order (highest confidence first) ──────────
    # SHOCK_CONT > FAILED_RALLY > BEAR_RESET > BEAR_TREND
    # Later masks overwrite earlier ones (so write in reverse priority order)

    def _apply(mask, sig_type, size, reasons_fn):
        idxs = df.index[mask]
        df.loc[idxs, "short_signal_type"]   = sig_type
        df.loc[idxs, "short_signal_size"]   = size
        df.loc[idxs, "short_signal_reason"] = reasons_fn(idxs)

    _apply(bear_trend_mask, "BEAR_TREND",  BEAR_TREND_SIZE,
           lambda idx: _reasons(df, idx, slope, ret20, ret60, loc20, bk_cnt,
                                "bear_slope|bear_20d|bear_60d"))
    _apply(bear_reset_mask, "BEAR_RESET",  BEAR_RESET_SIZE,
           lambda idx: _reasons(df, idx, slope, ret20, ret60, loc20, bk_cnt,
                                "trans_ok|bear_slope|bear_20d|ret5_bounce|bk_cnt>=1"))
    _apply(failed_rally_mask, "FAILED_RALLY", FAILED_RALLY_SIZE,
           lambda idx: _reasons(df, idx, slope, ret20, ret60, loc20, bk_cnt,
                                "failed_rally|ret5_bounce|bear_20d|loc>0.55"))
    _apply(shock_cont_mask, "SHOCK_CONT",  SHOCK_CONT_SIZE,
           lambda idx: _reasons(df, idx, slope, ret20, ret60, loc20, bk_cnt,
                                "shock_cont|still_falling|elevated_vol"))

    # ── Inject into final_direction / strategy_used where signal fired ────────
    short_fired = df["short_signal_type"] != "NONE"
    df.loc[short_fired, "final_direction"] = "SHORT"
    df.loc[short_fired, "strategy_used"]   = df.loc[short_fired, "short_signal_type"]
    df.loc[short_fired, "mtf_entry_size"]  = df.loc[short_fired, "short_signal_size"]
    df.loc[short_fired, "entry_quality"]   = 3

    _log_summary(df, bear_trend_mask, failed_rally_mask,
                 shock_cont_mask, bear_reset_mask)
    return df


def _reasons(df, idx, slope, ret20, ret60, loc20, bk_cnt, base: str) -> pd.Series:
    out = []
    for i in idx:
        parts = [base,
                 f"sl={slope.loc[i]:.4f}",
                 f"r20={ret20.loc[i]:.3f}",
                 f"loc={loc20.loc[i]:.2f}"]
        out.append("|".join(parts))
    return pd.Series(out, index=idx)


def _log_summary(df, bt_mask, fr_mask, sc_mask, br_mask):
    logger.info(
        "ShortSleeve — BEAR_TREND=%d  FAILED_RALLY=%d  "
        "SHOCK_CONT=%d  BEAR_RESET=%d  total_shorts=%d",
        int(bt_mask.sum()), int(fr_mask.sum()),
        int(sc_mask.sum()), int(br_mask.sum()),
        int((df["short_signal_type"] != "NONE").sum()),
    )
