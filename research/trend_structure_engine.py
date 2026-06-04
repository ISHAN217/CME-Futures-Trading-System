"""
trend_structure_engine.py — Market structure analysis for the TREND strategy.

All features are strictly backward-looking (no lookahead).
Processed per-symbol independently.

Features computed
─────────────────
Swing structure (backward-looking rolling windows, shifted by 1 bar):
  swing_high_5d, swing_low_5d      prior 5-day high/low of close
  swing_high_10d, swing_low_10d    prior 10-day high/low
  swing_high_20d, swing_low_20d    prior 20-day high/low
  higher_high_flag                 10d swing high rising
  higher_low_flag                  10d swing low rising
  lower_high_flag                  10d swing high falling
  lower_low_flag                   10d swing low falling

Break of structure:
  break_of_structure_up            close > prior 10d high
  break_of_structure_down          close < prior 10d low
  failed_breakout_up               BOS-up in prior 3 days, then close reverses
  failed_breakdown_down            BOS-down in prior 3 days, then close reverses
  retest_success_up                post-BOS pullback holds near breakout level
  retest_success_down              post-BOS-down rally fails at prior low
  structure_break_count            consecutive days where BOS-down is active

Entry location:
  entry_location_pct_20d           (close − min20) / (max20 − min20)
  entry_location_pct_40d           (close − min40) / (max40 − min40)
  trend_maturity_bucket            EARLY / MIDDLE / LATE / VERY_LATE

Candle quality:
  close_location_value             (close − low) / (high − low)   [CLV]
  upper_wick_ratio                 (high − max(open,close)) / range
  lower_wick_ratio                 (min(open,close) − low) / range
  buyer_pressure_score             composite [0,1]
  seller_pressure_score            composite [0,1]

Trend efficiency:
  trend_efficiency_10              10d net move / 10d gross-distance
  trend_efficiency_20              20d net move / 20d gross-distance

Conflict score:
  structure_conflict_score         disagreement between HH/HL and slope/momentum [0,1]

Trend phase (one of 13 labels):
  trend_phase                      BULL_IMPULSE / BULL_PULLBACK / BULL_RESET_CONFIRMED /
                                   BULL_EXHAUSTION / BULL_BREAKDOWN_WARNING /
                                   BEAR_IMPULSE / BEAR_RALLY / BEAR_RESET_CONFIRMED /
                                   BEAR_EXHAUSTION / BEAR_BREAKDOWN_WARNING /
                                   RANGE / CHAOTIC / UNCLEAR

Structure exit signals:
  structure_exit_warning           first sign of deterioration (1 structural break)
  structure_exit_confirmed         confirmed break (2+ consecutive breaks)
  structure_reduce_flag            cut position to 50%
  structure_tighten_trail_flag     tighten trailing stop
  structure_break_count            consecutive days in structural breakdown
  structure_exit_reason            human-readable description
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Maturity thresholds (entry_location_pct_20d)
# ─────────────────────────────────────────────────────────────────────────────
_MATURITY_BINS  = [0.0,  0.40, 0.65, 0.80, 1.01]
_MATURITY_LABELS = ["EARLY", "MIDDLE", "LATE", "VERY_LATE"]


def _safe_div(a, b, fill=0.0):
    return np.where(np.abs(b) < 1e-9, fill, a / b)


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_structure(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values("date").reset_index(drop=True)
    cl = g["close"]
    hi = g["high"]
    lo = g["low"]
    op = g["open"]

    # ── Swing high / low (backward-looking: shift(1) so today not included) ──
    sh5  = cl.rolling(5,  min_periods=3).max().shift(1)
    sl5  = cl.rolling(5,  min_periods=3).min().shift(1)
    sh10 = cl.rolling(10, min_periods=5).max().shift(1)
    sl10 = cl.rolling(10, min_periods=5).min().shift(1)
    sh20 = cl.rolling(20, min_periods=10).max().shift(1)
    sl20 = cl.rolling(20, min_periods=10).min().shift(1)

    g["swing_high_5d"]  = sh5.round(4)
    g["swing_low_5d"]   = sl5.round(4)
    g["swing_high_10d"] = sh10.round(4)
    g["swing_low_10d"]  = sl10.round(4)
    g["swing_high_20d"] = sh20.round(4)
    g["swing_low_20d"]  = sl20.round(4)

    # ── HH / HL / LH / LL (comparing successive 10d swing windows) ───────────
    sh10_prev = sh10.shift(1)
    sl10_prev = sl10.shift(1)

    g["higher_high_flag"] = (sh10 > sh10_prev).astype(float)
    g["higher_low_flag"]  = (sl10 > sl10_prev).astype(float)
    g["lower_high_flag"]  = (sh10 < sh10_prev).astype(float)
    g["lower_low_flag"]   = (sl10 < sl10_prev).astype(float)

    # ── Break of structure ────────────────────────────────────────────────────
    bos_up   = (cl > sh10).astype(float)   # close broke prior 10d high
    bos_down = (cl < sl10).astype(float)   # close broke prior 10d low

    g["break_of_structure_up"]   = bos_up
    g["break_of_structure_down"] = bos_down

    # Failed breakout: BOS-up fired in prior 3 days, but now close < prior high
    bos_up_recent   = bos_up.rolling(3, min_periods=1).max().shift(1)
    bos_down_recent = bos_down.rolling(3, min_periods=1).max().shift(1)

    g["failed_breakout_up"]    = ((bos_up_recent > 0) & (cl < sh10)).astype(float)
    g["failed_breakdown_down"] = ((bos_down_recent > 0) & (cl > sl10)).astype(float)

    # Retest success: after BOS-up (within 5 bars), price pulls back to old high
    # then closes back above it (higher than yesterday)
    bos_up_5d   = bos_up.rolling(5, min_periods=1).max().shift(1)
    bos_down_5d = bos_down.rolling(5, min_periods=1).max().shift(1)

    near_prior_high  = (cl >= sh10 * 0.996) & (cl <= sh10 * 1.008)
    near_prior_low   = (cl >= sl10 * 0.992) & (cl <= sl10 * 1.004)

    g["retest_success_up"]   = ((bos_up_5d > 0) & near_prior_high &
                                (cl > cl.shift(1))).astype(float)
    g["retest_success_down"] = ((bos_down_5d > 0) & near_prior_low &
                                (cl < cl.shift(1))).astype(float)

    # ── Consecutive structure break count ─────────────────────────────────────
    # Counts consecutive days where bos_down is active (sustained breakdown)
    consec = pd.Series(0.0, index=g.index)
    cnt = 0.0
    for i, v in enumerate(bos_down):
        if v > 0:
            cnt += 1
        else:
            cnt = 0.0
        consec.iloc[i] = cnt
    g["structure_break_count"] = consec

    # ── Entry location ────────────────────────────────────────────────────────
    mn20 = cl.rolling(20, min_periods=5).min()
    mx20 = cl.rolling(20, min_periods=5).max()
    rng20 = (mx20 - mn20).replace(0.0, np.nan)
    loc20 = ((cl - mn20) / rng20).clip(0.0, 1.0)

    mn40 = cl.rolling(40, min_periods=10).min()
    mx40 = cl.rolling(40, min_periods=5).max()
    rng40 = (mx40 - mn40).replace(0.0, np.nan)
    loc40 = ((cl - mn40) / rng40).clip(0.0, 1.0)

    g["entry_location_pct_20d"] = loc20.round(4)
    g["entry_location_pct_40d"] = loc40.round(4)

    g["trend_maturity_bucket"] = pd.cut(
        loc20.fillna(0.5), bins=_MATURITY_BINS, labels=_MATURITY_LABELS,
        right=False, include_lowest=True,
    ).astype(str)

    # ── Candle quality features ───────────────────────────────────────────────
    full_range = (hi - lo).replace(0.0, np.nan)

    clv = _safe_div(cl - lo, hi - lo, fill=0.5)
    g["close_location_value"] = pd.Series(clv, index=g.index).clip(0.0, 1.0).round(4)

    upper_wick = _safe_div(hi - pd.concat([op, cl], axis=1).max(axis=1), hi - lo, fill=0.0)
    lower_wick = _safe_div(pd.concat([op, cl], axis=1).min(axis=1) - lo, hi - lo, fill=0.0)

    g["upper_wick_ratio"] = pd.Series(upper_wick, index=g.index).clip(0.0, 1.0).round(4)
    g["lower_wick_ratio"] = pd.Series(lower_wick, index=g.index).clip(0.0, 1.0).round(4)

    # Buyer / seller pressure score (5-day composite)
    clv_s    = pd.Series(clv, index=g.index)
    lwk_s    = pd.Series(lower_wick, index=g.index)
    pos_ret  = (cl > cl.shift(1)).astype(float)
    neg_ret  = (cl < cl.shift(1)).astype(float)

    buyer_s = (
        0.40 * clv_s.rolling(5, min_periods=2).mean()
        + 0.30 * lwk_s.rolling(5, min_periods=2).mean()   # lower wick = buyer support
        + 0.30 * pos_ret.rolling(5, min_periods=2).mean()
    ).clip(0.0, 1.0)

    seller_s = (
        0.40 * (1 - clv_s).rolling(5, min_periods=2).mean()
        + 0.30 * pd.Series(upper_wick, index=g.index).rolling(5, min_periods=2).mean()
        + 0.30 * neg_ret.rolling(5, min_periods=2).mean()
    ).clip(0.0, 1.0)

    g["buyer_pressure_score"]  = buyer_s.round(4)
    g["seller_pressure_score"] = seller_s.round(4)

    # ── Trend efficiency ──────────────────────────────────────────────────────
    def _efficiency(series: pd.Series, n: int) -> pd.Series:
        net   = series.diff(n).abs()
        gross = series.diff(1).abs().rolling(n, min_periods=max(1, n//2)).sum()
        return _safe_div(net.values, gross.values, fill=0.0)

    g["trend_efficiency_10"] = pd.Series(
        _efficiency(cl, 10), index=g.index).clip(0.0, 1.0).round(4)
    g["trend_efficiency_20"] = pd.Series(
        _efficiency(cl, 20), index=g.index).clip(0.0, 1.0).round(4)

    # ── Structure conflict score ──────────────────────────────────────────────
    # Disagreement between HH/HL pattern and slope/momentum direction
    bull_struct = ((g["higher_high_flag"] + g["higher_low_flag"]) / 2).fillna(0.5)
    bear_struct = ((g["lower_high_flag"] + g["lower_low_flag"]) / 2).fillna(0.5)

    slope_bull  = (g.get("slope_20d", pd.Series(0, index=g.index)).fillna(0) > 0).astype(float)
    mom_bull    = (g.get("momentum_direction",
                          pd.Series("NEUTRAL", index=g.index)).fillna("NEUTRAL") == "LONG").astype(float)

    # High conflict = structure says one thing, slope/momentum say another
    struct_bull = (bull_struct > 0.5).astype(float)
    conflict = (struct_bull - slope_bull).abs() * 0.5 + (struct_bull - mom_bull).abs() * 0.5
    g["structure_conflict_score"] = conflict.round(4)

    # ── Trend phase ───────────────────────────────────────────────────────────
    g["trend_phase"] = _compute_phase(g, loc20, bull_struct, bear_struct)

    # ── Structure exit signals ────────────────────────────────────────────────
    _add_exit_signals(g)

    return g


def _compute_phase(g: pd.DataFrame, loc20: pd.Series,
                   bull_struct: pd.Series, bear_struct: pd.Series) -> pd.Series:
    """Assign trend phase label row by row."""
    slope_pos = (g.get("slope_20d", pd.Series(0, index=g.index)).fillna(0) > 0)
    ret20_pos = (g.get("ret_20d",   pd.Series(0, index=g.index)).fillna(0) > 0)
    ret5_neg  = (g.get("ret_5d",    pd.Series(0, index=g.index)).fillna(0) < 0)
    ret5_pos  = ~ret5_neg
    mom_long  = (g.get("momentum_direction",
                        pd.Series("NEUTRAL", index=g.index)).fillna("NEUTRAL") == "LONG")
    bos_up    = g["break_of_structure_up"].astype(bool)
    bos_dn    = g["break_of_structure_down"].astype(bool)
    fb_up     = g["failed_breakout_up"].astype(bool)
    hh        = g["higher_high_flag"].astype(bool)
    hl        = g["higher_low_flag"].astype(bool)
    lh        = g["lower_high_flag"].astype(bool)
    ll        = g["lower_low_flag"].astype(bool)
    bk_cnt    = g["structure_break_count"]

    phases = []
    for i in range(len(g)):
        # --- bearish signals ---
        if bos_dn.iloc[i] and ll.iloc[i] and not ret20_pos.iloc[i]:
            phases.append("BEAR_IMPULSE")
        elif not ret20_pos.iloc[i] and ret5_pos.iloc[i] and lh.iloc[i]:
            phases.append("BEAR_RALLY")
        elif not ret20_pos.iloc[i] and lh.iloc[i] and ll.iloc[i] and not ret5_neg.iloc[i]:
            phases.append("BEAR_RESET_CONFIRMED")
        elif not ret20_pos.iloc[i] and hh.iloc[i]:
            phases.append("BEAR_EXHAUSTION")
        elif not slope_pos.iloc[i] and bos_up.iloc[i] and not hh.iloc[i]:
            phases.append("BEAR_BREAKDOWN_WARNING")

        # --- bullish signals ---
        elif bos_up.iloc[i] and hh.iloc[i] and hl.iloc[i] and slope_pos.iloc[i]:
            phases.append("BULL_IMPULSE")
        elif slope_pos.iloc[i] and ret5_neg.iloc[i] and hl.iloc[i]:
            phases.append("BULL_PULLBACK")
        elif slope_pos.iloc[i] and ret5_pos.iloc[i] and hl.iloc[i] and not hh.iloc[i]:
            phases.append("BULL_RESET_CONFIRMED")
        elif slope_pos.iloc[i] and loc20.iloc[i] > 0.80 and ll.iloc[i]:
            phases.append("BULL_EXHAUSTION")
        elif slope_pos.iloc[i] and bk_cnt.iloc[i] >= 2:
            phases.append("BULL_BREAKDOWN_WARNING")

        # --- neutral / unclear ---
        elif not hh.iloc[i] and not ll.iloc[i] and not hl.iloc[i] and not lh.iloc[i]:
            phases.append("RANGE")
        elif fb_up.iloc[i] and bos_dn.iloc[i]:
            phases.append("CHAOTIC")
        else:
            phases.append("UNCLEAR")

    return pd.Series(phases, index=g.index)


def _add_exit_signals(g: pd.DataFrame) -> None:
    """Add structure exit advisory signals in-place."""
    bk_cnt = g["structure_break_count"]
    fb_up  = g["failed_breakout_up"].astype(bool)
    ll     = g["lower_low_flag"].astype(bool)
    lh     = g["lower_high_flag"].astype(bool)
    bos_dn = g["break_of_structure_down"].astype(bool)

    # Warning: first sign of structure deterioration
    # = any of: 1 structure break, failed breakout, or lower-low while in uptrend
    slope_pos = (g.get("slope_20d", pd.Series(0, index=g.index)).fillna(0) > 0)
    warning   = (bk_cnt >= 1) | (fb_up & slope_pos) | (ll & lh)

    # Confirmed: 2+ consecutive structure breaks OR failed breakout + lower low
    confirmed = (bk_cnt >= 2) | (fb_up & ll)

    # Reduce flag: first break of structure but still in prior uptrend
    reduce    = (bk_cnt == 1) & slope_pos & ~confirmed

    # Tighten trail: price failed retest or lower high forming
    tighten   = (g["failed_breakout_up"].astype(bool)) | (lh & ~ll)

    g["structure_exit_warning"]       = warning.astype(float)
    g["structure_exit_confirmed"]     = confirmed.astype(float)
    g["structure_reduce_flag"]        = reduce.astype(float)
    g["structure_tighten_trail_flag"] = tighten.astype(float)

    # Reason text
    reasons = []
    for i in range(len(g)):
        parts = []
        if bk_cnt.iloc[i] >= 2:   parts.append(f"bos_dn_{int(bk_cnt.iloc[i])}x")
        if fb_up.iloc[i]:          parts.append("failed_bkout")
        if ll.iloc[i] and lh.iloc[i]: parts.append("ll_lh")
        reasons.append("|".join(parts) or "none")
    g["structure_exit_reason"] = reasons


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_trend_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all trend-structure features to df.
    Requires columns: date, symbol, close, high, low, open,
                      slope_20d, ret_5d, ret_20d, momentum_direction.
    Safe to call on a full multi-symbol df.
    """
    df = df.copy()
    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = _compute_structure(g)
        parts.append(g)

    df = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"])
    _log_summary(df)
    return df


def _log_summary(df: pd.DataFrame) -> None:
    for sym in sorted(df["symbol"].unique()):
        g = df[df["symbol"] == sym]
        n = max(len(g), 1)
        ph = g["trend_phase"].value_counts()
        mb = g["trend_maturity_bucket"].value_counts()
        warn = g["structure_exit_warning"].sum()
        conf = g["structure_exit_confirmed"].sum()
        logger.info(
            "TrendStructure %s — phase top3: %s | maturity: E=%d M=%d L=%d VL=%d"
            " | exit_warn=%d (%.0f%%) exit_conf=%d (%.0f%%)",
            sym,
            "/".join(f"{k}:{v}" for k, v in ph.head(3).items()),
            mb.get("EARLY", 0), mb.get("MIDDLE", 0),
            mb.get("LATE", 0), mb.get("VERY_LATE", 0),
            warn, 100 * warn / n, conf, 100 * conf / n,
        )
