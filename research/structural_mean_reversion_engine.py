"""
structural_mean_reversion_engine.py — CWT/EMD-confirmed structural MR. [NEW]

Identifies tradable range-bound environments using three independent filters:
  1. Range quality: stable oscillating range with boundary rejections
  2. CWT: organized oscillation (RANGE_OSCILLATORY or low-entropy UNCLEAR)
  3. EMD-proxy: flat trend + stable fast component (MA decomposition fallback)

Signals fire ONLY when:
  - portfolio_regime is CHOP or TRANSITION (never SHOCK, never TREND)
  - Range quality is high (oscillatory boundaries, breakouts fail)
  - CWT does not indicate chaos or breakout expansion
  - EMD-proxy indicates flat trend with controlled fast vol
  - Price is near range extreme (position_in_range ≤ 0.22 or ≥ 0.78)
  - No volatility shock (vol_regime != HIGH)

Design principle: be conservative about ES/NQ which are structurally bullish.
Long MR at lower boundary is the primary setup. Short MR at upper boundary
is allowed but filtered more strictly.

Output columns added to df:
  rq_rolling_high, rq_rolling_low, rq_range_midpoint, rq_range_width
  rq_range_width_z, rq_distance_to_upper, rq_distance_to_lower
  rq_position_in_range, rq_rolling_mean, rq_rolling_std, rq_price_zscore
  rq_mean_cross_freq, rq_upper_rejection_count, rq_lower_rejection_count
  rq_failed_breakout_count, rq_range_vol_stability, rq_quality_score, rq_range_label

  cwt_range_score, cwt_chaos_score, cwt_breakout_score, cwt_mr_allowed

  emd_fast_volatility, emd_fast_volatility_z, emd_stable_range_score
  emd_transition_score, emd_trend_deterioration_score, emd_label

  structural_mr_signal, structural_mr_direction, structural_mr_score
  structural_mr_bucket, structural_mr_reason, structural_mr_entry_weight
  structural_mr_target_type, structural_mr_stop_type, structural_mr_max_hold
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

_RQ_WINDOW = 20   # rolling range window (1 month)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RANGE QUALITY
# ════════════════════════════════════════════════════════════════════════════════

def _compute_range_quality(g: pd.DataFrame) -> pd.DataFrame:
    close = g["close"]
    w     = _RQ_WINDOW
    mp    = max(w // 2, 5)

    rolling_high = close.rolling(w, min_periods=mp).max()
    rolling_low  = close.rolling(w, min_periods=mp).min()
    range_pts    = (rolling_high - rolling_low).replace(0, np.nan)
    range_mid    = (rolling_high + rolling_low) / 2.0
    range_width  = (range_pts / close.replace(0, np.nan)).fillna(0.0)

    # Position in range: 0 = at rolling low, 1 = at rolling high
    pos_in_range = ((close - rolling_low) / range_pts).clip(0.0, 1.0).fillna(0.5)

    # Distances to boundaries (fraction of range width)
    dist_to_upper = ((rolling_high - close) / range_pts).fillna(0.5).clip(0, 1)
    dist_to_lower = ((close - rolling_low) / range_pts).fillna(0.5).clip(0, 1)

    # Rolling stats for z-score
    rolling_mean = close.rolling(w, min_periods=mp).mean()
    rolling_std  = close.rolling(w, min_periods=mp).std().replace(0, np.nan)
    price_zscore = ((close - rolling_mean) / rolling_std).fillna(0.0).clip(-4, 4)

    # Range width z-score (vs 1-year history)
    rw_mean = range_width.rolling(252, min_periods=63).mean()
    rw_std  = range_width.rolling(252, min_periods=63).std().replace(0, np.nan)
    range_width_z = ((range_width - rw_mean) / rw_std).fillna(0.0).clip(-3, 3)

    # Mean-cross frequency: price crosses rolling_mean → oscillation evidence
    above_mean   = (close > rolling_mean).astype(float)
    crosses      = above_mean.diff().abs()
    mean_cross_freq = crosses.rolling(w, min_periods=mp).sum() / w

    # Upper boundary rejection: near upper range, then reversed lower
    near_upper   = (pos_in_range.shift(1) >= 0.85).astype(float)
    fell_lower   = (close < close.shift(1)).astype(float)
    upper_rej    = near_upper * fell_lower
    upper_rej_count = upper_rej.rolling(w, min_periods=mp).sum() / w

    # Lower boundary rejection: near lower range, then reversed higher
    near_lower   = (pos_in_range.shift(1) <= 0.15).astype(float)
    rose_higher  = (close > close.shift(1)).astype(float)
    lower_rej    = near_lower * rose_higher
    lower_rej_count = lower_rej.rolling(w, min_periods=mp).sum() / w

    # Failed breakout: closed outside range yesterday, back inside today
    above_rng    = (close.shift(1) > rolling_high.shift(1)).astype(float).fillna(0)
    back_inside  = (close <= rolling_high).astype(float)
    failed_bk    = (above_rng * back_inside)
    failed_bk_count = failed_bk.rolling(w, min_periods=mp).sum() / w

    # Range volatility stability: low CV = stable range width
    rw_cv = (
        range_width.rolling(w, min_periods=mp).std()
        / range_width.rolling(w, min_periods=mp).mean().replace(0, np.nan)
    ).fillna(1.0)
    range_vol_stability = (1.0 - rw_cv).clip(0, 1)

    # Is price currently outside the range for ≥2 consecutive days?
    outside = ((close > rolling_high) | (close < rolling_low)).astype(float)
    outside_streak = outside.rolling(3, min_periods=1).sum()
    is_breaking = (outside_streak >= 2).astype(float)

    # Too narrow: range width < 0.5% (not worth trading)
    too_narrow = (range_width < 0.005).astype(float)

    # Anti-trend: high R² means trending, not ranging
    r2 = g["r2_20d"].fillna(0.0).clip(0, 1)

    quality_raw = (
        0.20 * mean_cross_freq.clip(0, 1)
        + 0.15 * upper_rej_count.clip(0, 1)
        + 0.15 * lower_rej_count.clip(0, 1)
        + 0.10 * failed_bk_count.clip(0, 1)
        + 0.15 * range_vol_stability
        + 0.10 * (1.0 - r2)            # reward non-trending
        + 0.15 * (1.0 - is_breaking)   # reward not-breaking
        - 0.30 * too_narrow            # penalise narrow range
    ).clip(0.0, 1.0)

    # Range label
    labels = []
    for i in range(len(g)):
        if too_narrow.iloc[i] > 0:
            labels.append("RANGE_TOO_NARROW")
        elif is_breaking.iloc[i] > 0:
            labels.append("RANGE_BREAKING")
        elif quality_raw.iloc[i] >= 0.50:
            labels.append("RANGE_STABLE")
        elif quality_raw.iloc[i] >= 0.35:
            labels.append("RANGE_WEAK")
        else:
            labels.append("RANGE_UNCLEAR")

    g["rq_rolling_high"]          = rolling_high.round(4)
    g["rq_rolling_low"]           = rolling_low.round(4)
    g["rq_range_midpoint"]        = range_mid.round(4)
    g["rq_range_width"]           = range_width.round(6)
    g["rq_range_width_z"]         = range_width_z.round(4)
    g["rq_distance_to_upper"]     = dist_to_upper.round(4)
    g["rq_distance_to_lower"]     = dist_to_lower.round(4)
    g["rq_position_in_range"]     = pos_in_range.round(4)
    g["rq_rolling_mean"]          = rolling_mean.round(4)
    g["rq_rolling_std"]           = rolling_std.round(6)
    g["rq_price_zscore"]          = price_zscore.round(4)
    g["rq_mean_cross_freq"]       = mean_cross_freq.fillna(0).round(4)
    g["rq_upper_rejection_count"] = upper_rej_count.fillna(0).round(4)
    g["rq_lower_rejection_count"] = lower_rej_count.fillna(0).round(4)
    g["rq_failed_breakout_count"] = failed_bk_count.fillna(0).round(4)
    g["rq_range_vol_stability"]   = range_vol_stability.fillna(0).round(4)
    g["rq_quality_score"]         = quality_raw.round(4)
    g["rq_range_label"]           = labels
    return g


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CWT FEATURE AUGMENTATION
# ════════════════════════════════════════════════════════════════════════════════

def _derive_cwt_mr_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    Derive MR-specific scores from existing CWT columns.
    Assumes cwt_chop_filter output columns are present.
    """
    label   = g["cwt_chop_label"].fillna("CWT_UNCLEAR")
    entropy = g["cwt_entropy"].fillna(0.5).clip(0, 1)
    slow_r  = g["cwt_slow_energy_ratio"].fillna(1 / 3).clip(0, 1)
    exp_flag= g.get("cwt_expansion_flag", pd.Series(0, index=g.index)).fillna(0)

    # cwt_range_score: high when organized oscillation
    cwt_range_score = np.where(
        label == "CWT_RANGE_OSCILLATORY", 0.90,
        np.where(
            (label == "CWT_UNCLEAR") & (entropy < 0.55), 0.65,
            np.where(
                label == "CWT_COMPRESSION", 0.40,
                np.where(label == "CWT_TREND_COHERENT", 0.20, 0.10)
            )
        )
    )

    # cwt_chaos_score: high when energy dispersed or expanding
    cwt_chaos_score = np.where(
        label == "CWT_CHAOTIC_NOISE", 0.90,
        np.where(
            label == "CWT_BREAKOUT_EXPANSION", 0.80,
            np.where(
                (label == "CWT_UNCLEAR") & (entropy > 0.70), 0.60,
                (entropy * 0.4).clip(0, 0.45)
            )
        )
    )

    # cwt_breakout_score: high when expansion after compression
    cwt_breakout_score = np.where(
        label == "CWT_BREAKOUT_EXPANSION", 1.0,
        np.where(
            exp_flag > 0, 0.80,
            np.where(slow_r > 0.60, 0.40, 0.10)
        )
    )

    # MR is blocked by CWT when chaotic or expanding
    mr_blocked = (
        label.isin(["CWT_CHAOTIC_NOISE", "CWT_BREAKOUT_EXPANSION"])
        | (entropy > 0.80)
    )

    g["cwt_range_score"]    = pd.Series(cwt_range_score,    index=g.index).round(4)
    g["cwt_chaos_score"]    = pd.Series(cwt_chaos_score,    index=g.index).round(4)
    g["cwt_breakout_score"] = pd.Series(cwt_breakout_score, index=g.index).round(4)
    g["cwt_mr_allowed"]     = (~mr_blocked).astype(int)
    return g


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EMD STRUCTURE FEATURES (MA DECOMPOSITION)
# ════════════════════════════════════════════════════════════════════════════════

def _compute_emd_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    EMD-proxy decomposition via EMA components.
    Reuses r2_20d and slope_20d (already in df) for trend assessment.

    fast_comp = close - EMA(5)          → very short-term residual
    emd_trend_r2, emd_trend_slope       → from existing r2_20d, slope_20d
    """
    close = g["close"]

    # Fast component: deviation from EMA(5)
    ema5      = close.ewm(span=5, min_periods=3, adjust=False).mean()
    fast_comp = (close - ema5) / close.replace(0, np.nan)   # normalized

    # Fast component volatility
    emd_fast_vol  = fast_comp.rolling(10, min_periods=5).std().fillna(0.0)
    fv_mean = emd_fast_vol.rolling(252, min_periods=63).mean()
    fv_std  = emd_fast_vol.rolling(252, min_periods=63).std().replace(0, np.nan)
    emd_fast_vol_z = ((emd_fast_vol - fv_mean) / fv_std).fillna(0.0).clip(-3, 3)

    # Use existing OLS features for trend assessment
    r2    = g["r2_20d"].fillna(0.0).clip(0, 1)
    slope = g["slope_20d"].fillna(0.0)

    # Normalized slope z-score (how strong is the trend direction)
    slope_abs  = slope.abs()
    slope_mean = slope_abs.rolling(252, min_periods=63).mean()
    slope_std  = slope_abs.rolling(252, min_periods=63).std().replace(0, np.nan)
    slope_z    = ((slope_abs - slope_mean) / slope_std).fillna(0.0).clip(0, 3)

    # emd_stable_range_score: trend flat + fast vol normal + low R²
    trend_flat   = (1.0 - r2.clip(0, 1)) * (1.0 - slope_z.clip(0, 3) / 3.0)
    vol_ctrl     = (1.0 - emd_fast_vol_z.clip(0, 3) / 3.0).clip(0, 1)
    # Fast comp centered near zero = stable oscillation (not drifting)
    fast_centered = (1.0 - fast_comp.abs().rolling(5, min_periods=2).mean().clip(0, 0.02) / 0.02).clip(0, 1).fillna(0.5)

    emd_stable_range_score = (
        0.40 * trend_flat.clip(0, 1)
        + 0.35 * vol_ctrl
        + 0.25 * fast_centered
    ).clip(0.0, 1.0)

    # emd_transition_score: recent R² slope change (directional instability)
    r2_delta  = r2.diff(5).abs().fillna(0)
    r2_delta_z = r2_delta.rolling(252, min_periods=63).rank(pct=True).fillna(0.5)
    emd_transition_score = r2_delta_z.clip(0, 1)

    # emd_trend_deterioration_score: R² dropping + slope weakening
    r2_rolling_mean = r2.rolling(10, min_periods=5).mean()
    r2_dropping = (r2 < r2_rolling_mean * 0.70).astype(float)
    slope_declining = (slope.diff(3).fillna(0) < 0).astype(float)
    emd_trend_detrn = (0.6 * r2_dropping + 0.4 * slope_declining).clip(0, 1)

    # EMD label
    labels = []
    for i in range(len(g)):
        r2_v  = float(r2.iloc[i])
        sl_z  = float(slope_z.iloc[i])
        ts_v  = float(emd_transition_score.iloc[i])
        fvz   = float(emd_fast_vol_z.iloc[i])
        sr    = float(emd_stable_range_score.iloc[i])
        td    = float(emd_trend_detrn.iloc[i])

        if r2_v > 0.50 and sl_z > 0.5:
            labels.append("EMD_TREND_INTACT")
        elif td > 0.60 and r2_v > 0.20:
            labels.append("EMD_TREND_DETERIORATING")
        elif ts_v > 0.80:
            labels.append("EMD_TRANSITION")
        elif fvz > 2.0:
            labels.append("EMD_CHAOTIC_FAST_NOISE")
        elif sr >= 0.42:
            labels.append("EMD_STABLE_RANGE")
        else:
            labels.append("EMD_UNCLEAR")

    g["emd_trend_slope"]          = slope.round(8)         # reuse slope_20d
    g["emd_trend_r2"]             = r2.round(4)            # reuse r2_20d
    g["emd_fast_volatility"]      = emd_fast_vol.round(6)
    g["emd_fast_volatility_z"]    = emd_fast_vol_z.round(4)
    g["emd_fast_energy_ratio"]    = (emd_fast_vol / (emd_fast_vol + r2 + 1e-9)).round(4)
    g["emd_slow_energy_ratio"]    = (r2 / (emd_fast_vol + r2 + 1e-9)).round(4)
    g["emd_stable_range_score"]   = emd_stable_range_score.round(4)
    g["emd_transition_score"]     = emd_transition_score.round(4)
    g["emd_trend_deterioration_score"] = emd_trend_detrn.round(4)
    g["emd_label"]                = labels
    return g


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STRUCTURAL MR SIGNAL
# ════════════════════════════════════════════════════════════════════════════════

def _mild_stabilization(g: pd.DataFrame, direction: str) -> pd.Series:
    """
    Mild price stabilization near range extreme.
    LONG: close ≥ open (not still falling hard) or tiny positive return today.
    SHORT: close ≤ open (not still surging hard) or flat/negative return today.
    """
    close  = g["close"]
    open_  = g.get("open", close.shift(1).fillna(close))
    ret    = g["ret_1d"].fillna(0)

    if direction == "LONG":
        # Not still falling hard: either close ≥ open, or ret > -0.3%
        return ((close >= open_.fillna(close)) | (ret > -0.003)).astype(float)
    else:
        return ((close <= open_.fillna(close)) | (ret < 0.003)).astype(float)


def _compute_mr_signals(
    g: pd.DataFrame,
    rq_threshold:  float = 0.42,
    pir_long:      float = 0.22,
    pir_short:     float = 0.78,
    zscore_long:   float = -1.30,
    zscore_short:  float = 1.30,
    require_cwt:   bool  = True,
    require_emd:   bool  = True,
    require_range: bool  = True,
) -> pd.DataFrame:
    regime     = g["portfolio_regime"].fillna("CHOP")
    vol_regime = g["vol_regime"].fillna("NORMAL")

    rq_score   = g["rq_quality_score"].fillna(0.0)
    rq_label   = g["rq_range_label"].fillna("RANGE_UNCLEAR")
    pos_in_r   = g["rq_position_in_range"].fillna(0.5)
    price_z    = g["rq_price_zscore"].fillna(0.0)

    cwt_allowed  = g.get("cwt_mr_allowed", pd.Series(1, index=g.index)).fillna(1).astype(int)
    cwt_range_s  = g.get("cwt_range_score", pd.Series(0.5, index=g.index)).fillna(0.5)

    emd_label   = g.get("emd_label", pd.Series("EMD_UNCLEAR", index=g.index)).fillna("EMD_UNCLEAR")
    emd_stable  = g.get("emd_stable_range_score", pd.Series(0.3, index=g.index)).fillna(0.3)

    # ── Base regime gate ──────────────────────────────────────────────────────
    in_valid_regime = regime.isin(["CHOP", "TRANSITION"]) & (vol_regime != "HIGH")

    # ── Range gate ────────────────────────────────────────────────────────────
    range_gate = (
        (rq_score >= rq_threshold)
        & (rq_label == "RANGE_STABLE")
    ) if require_range else pd.Series(True, index=g.index)

    # ── CWT gate ──────────────────────────────────────────────────────────────
    cwt_gate = (cwt_allowed > 0) if require_cwt else pd.Series(True, index=g.index)

    # ── EMD gate ──────────────────────────────────────────────────────────────
    emd_ok = (
        (emd_label == "EMD_STABLE_RANGE")
        | ((emd_label == "EMD_UNCLEAR") & (emd_stable >= 0.38))
    )
    emd_block = emd_label.isin([
        "EMD_TREND_INTACT", "EMD_TRANSITION",
        "EMD_TREND_DETERIORATING", "EMD_CHAOTIC_FAST_NOISE"
    ])
    emd_gate = (emd_ok & ~emd_block) if require_emd else pd.Series(True, index=g.index)

    base_gate = in_valid_regime & range_gate & cwt_gate & emd_gate

    # ── Directional gates ─────────────────────────────────────────────────────
    near_low  = (pos_in_r <= pir_long)  | (price_z <= zscore_long)
    near_high = (pos_in_r >= pir_short) | (price_z >= zscore_short)

    stab_long  = _mild_stabilization(g, "LONG")
    stab_short = _mild_stabilization(g, "SHORT")

    long_gate  = base_gate & near_low  & (stab_long  > 0)
    short_gate = base_gate & near_high & (stab_short > 0)

    # Prefer LONG over SHORT on the same day (ES/NQ structural bull bias)
    # If both fire, take LONG
    short_gate = short_gate & ~long_gate

    # ── Score ─────────────────────────────────────────────────────────────────
    any_cand = long_gate | short_gate

    base_s    = any_cand.astype(float) * 0.25
    rq_s      = rq_score * 0.15 * any_cand.astype(float)
    cwt_s     = cwt_range_s * 0.15 * any_cand.astype(float) if require_cwt else pd.Series(0.0, index=g.index)
    emd_s     = emd_stable  * 0.15 * any_cand.astype(float) if require_emd else pd.Series(0.0, index=g.index)

    # Positional extreme bonus
    long_pos_extreme = ((pir_long - pos_in_r) / max(pir_long, 0.01)).clip(0, 1)
    shrt_pos_extreme = ((pos_in_r - pir_short) / max(1 - pir_short, 0.01)).clip(0, 1)
    pos_s = (
        long_gate.astype(float) * long_pos_extreme
        + short_gate.astype(float) * shrt_pos_extreme
    ) * 0.15

    z_s = (price_z.abs() / max(abs(zscore_long), 0.1)).clip(0, 1) * 0.15 * any_cand.astype(float)

    scores = (base_s + rq_s + cwt_s + emd_s + pos_s + z_s).clip(0.0, 1.0)

    # ── Bucket ────────────────────────────────────────────────────────────────
    def _bucket(s):
        if   s >= 0.60: return "STRONG"
        elif s >= 0.45: return "MODERATE"
        elif s >= 0.30: return "WEAK"
        else:           return "NONE"

    buckets = scores.apply(_bucket)

    any_signal = any_cand & buckets.isin(["STRONG", "MODERATE"])

    direction_vals = np.where(
        long_gate  & any_signal, "LONG",
        np.where(short_gate & any_signal, "SHORT", "NONE")
    )
    direction_s = pd.Series(direction_vals, index=g.index)

    # ── Entry weight ──────────────────────────────────────────────────────────
    size     = getattr(config, "STRUCTURAL_MR_SIZE",     0.10)
    max_size = getattr(config, "STRUCTURAL_MR_MAX_SIZE",  0.15)
    max_hold = getattr(config, "STRUCTURAL_MR_MAX_HOLD",  2)
    fracs    = {"STRONG": 1.0, "MODERATE": 0.75}

    entry_weights = []
    for i in range(len(g)):
        if not any_signal.iloc[i]:
            entry_weights.append(0.0)
            continue
        b = buckets.iloc[i]
        w = min(size * fracs.get(b, 0.0), max_size)
        if direction_s.iloc[i] == "SHORT":
            w = -w
        entry_weights.append(w)

    # ── Reasons ───────────────────────────────────────────────────────────────
    reasons = []
    for i in range(len(g)):
        if not any_signal.iloc[i]:
            reasons.append("no_signal")
            continue
        parts = [
            f"rq={rq_score.iloc[i]:.2f}",
            f"pos={pos_in_r.iloc[i]:.2f}",
            rq_label.iloc[i],
        ]
        if require_cwt: parts.append(f"cwt={cwt_range_s.iloc[i]:.2f}")
        if require_emd: parts.append(emd_label.iloc[i])
        reasons.append("|".join(parts))

    g["structural_mr_signal"]        = any_signal.astype(bool)
    g["structural_mr_direction"]     = direction_s
    g["structural_mr_score"]         = scores.round(4)
    g["structural_mr_bucket"]        = buckets
    g["structural_mr_reason"]        = reasons
    g["structural_mr_entry_weight"]  = pd.Series(entry_weights, index=g.index).round(4)
    g["structural_mr_target_type"]   = "MIDPOINT"
    g["structural_mr_stop_type"]     = "ATR"
    g["structural_mr_max_hold"]      = max_hold
    return g


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def compute_structural_mean_reversion_signals(
    df: pd.DataFrame,
    rq_threshold:  float = 0.42,
    pir_long:      float = 0.22,
    pir_short:     float = 0.78,
    zscore_long:   float = -1.30,
    zscore_short:  float =  1.30,
    require_cwt:   bool  = True,
    require_emd:   bool  = True,
    require_range: bool  = True,
) -> pd.DataFrame:
    """
    Main entry point. Processes df per symbol.

    Requires df to already contain:
      CWT columns: cwt_chop_label, cwt_entropy, cwt_high/mid/slow_energy_ratio,
                   cwt_expansion_flag
      Regime:      portfolio_regime
      Vol:         vol_regime
      Features:    close, r2_20d, slope_20d, atr_pct
    """
    df = df.copy()
    parts = []

    for sym, g in df.groupby("symbol", sort=True):
        g = g.copy().sort_values("date").reset_index(drop=True)
        try:
            g = _compute_range_quality(g)
            g = _compute_emd_features(g)
            g = _derive_cwt_mr_features(g)
            g = _compute_mr_signals(
                g,
                rq_threshold  = rq_threshold,
                pir_long      = pir_long,
                pir_short     = pir_short,
                zscore_long   = zscore_long,
                zscore_short  = zscore_short,
                require_cwt   = require_cwt,
                require_emd   = require_emd,
                require_range = require_range,
            )
        except Exception as exc:
            logger.error("StructuralMR failed for %s: %s", sym, exc, exc_info=True)
            _add_neutral_columns(g)
        parts.append(g)

    result = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"])

    for sym in result["symbol"].unique():
        gs  = result[result["symbol"] == sym]
        sig = gs["structural_mr_signal"].sum()
        vc  = gs["structural_mr_bucket"].value_counts()
        rl  = gs["rq_range_label"].value_counts()
        el  = gs["emd_label"].value_counts()
        dl  = gs[gs["structural_mr_signal"]]["structural_mr_direction"].value_counts()
        logger.info(
            "StructuralMR %s — signals=%d  STRONG=%d MOD=%d  "
            "LONG=%d SHORT=%d  RangeStable=%d  EMD_Stable=%d  TrendIntact=%d",
            sym, sig,
            vc.get("STRONG", 0), vc.get("MODERATE", 0),
            dl.get("LONG", 0), dl.get("SHORT", 0),
            rl.get("RANGE_STABLE", 0),
            el.get("EMD_STABLE_RANGE", 0), el.get("EMD_TREND_INTACT", 0),
        )
    return result


def _add_neutral_columns(g: pd.DataFrame) -> None:
    for col, val in {
        "rq_quality_score": 0.0, "rq_range_label": "RANGE_UNCLEAR",
        "rq_position_in_range": 0.5, "rq_price_zscore": 0.0,
        "rq_range_midpoint": g["close"], "rq_rolling_high": g["close"],
        "rq_rolling_low": g["close"], "rq_range_width": 0.0,
        "rq_range_width_z": 0.0, "rq_rolling_mean": g["close"],
        "rq_rolling_std": 0.0, "rq_mean_cross_freq": 0.0,
        "rq_upper_rejection_count": 0.0, "rq_lower_rejection_count": 0.0,
        "rq_failed_breakout_count": 0.0, "rq_range_vol_stability": 0.0,
        "rq_distance_to_upper": 0.5, "rq_distance_to_lower": 0.5,
        "emd_label": "EMD_UNCLEAR", "emd_stable_range_score": 0.0,
        "emd_trend_slope": 0.0, "emd_trend_r2": 0.0,
        "emd_fast_volatility": 0.0, "emd_fast_volatility_z": 0.0,
        "emd_fast_energy_ratio": 0.5, "emd_slow_energy_ratio": 0.5,
        "emd_transition_score": 0.5, "emd_trend_deterioration_score": 0.0,
        "cwt_range_score": 0.5, "cwt_chaos_score": 0.5,
        "cwt_breakout_score": 0.0, "cwt_mr_allowed": 0,
        "structural_mr_signal": False, "structural_mr_direction": "NONE",
        "structural_mr_score": 0.0, "structural_mr_bucket": "NONE",
        "structural_mr_reason": "error", "structural_mr_entry_weight": 0.0,
        "structural_mr_target_type": "MIDPOINT",
        "structural_mr_stop_type": "ATR", "structural_mr_max_hold": 2,
    }.items():
        if col not in g.columns:
            g[col] = val
