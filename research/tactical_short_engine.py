"""
tactical_short_engine.py — Optional tactical short-signal sleeve. [NEW]

Operates as a SEPARATE module from TREND/STAT_ARB.
Does NOT modify existing long-side logic.

Why shorts are currently zero:
  ALLOW_EQUITY_SHORTS = False blocks all short signals in signal_engine.py.
  The signal pipeline never generates a SHORT final_direction.
  This module generates tactical short candidate columns that are evaluated
  independently in run_tactical_short.py (not in the main signal engine).

Three short setups:
  A. DOWNSIDE_MOMENTUM_SHORT
     — price breaks below support while momentum is already negative
  B. FAILED_RALLY_SHORT
     — counter-trend bounce fails with price below key MA
  C. COMPRESSION_BREAKDOWN_SHORT
     — tight range compresses then breaks down with vol expansion

Output columns per (date, symbol):
  ts_dm_signal        bool — downside momentum candidate
  ts_dm_score         float 0-1
  ts_fr_signal        bool — failed rally candidate
  ts_fr_score         float 0-1
  ts_cb_signal        bool — compression breakdown candidate
  ts_cb_score         float 0-1
  tactical_short_signal   bool — any short candidate active
  tactical_short_type     str  — DM | FR | CB | DM+FR | DM+CB | FR+CB | DM+FR+CB | NONE
  tactical_short_score    float 0-1 (best active score)
  tactical_short_bucket   str  — STRONG | MODERATE | WEAK | NONE
  tactical_short_reason   str  — human-readable rationale
  tactical_short_entry_weight  float — suggested weight (before portfolio cap)
  tactical_short_max_hold      int
  tactical_short_stop          float
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ── Rolling percentile rank ────────────────────────────────────────────────────

def _rank(s: pd.Series, w: int = 252, mp: int = 63, invert: bool = False) -> pd.Series:
    """Backward-looking rolling percentile rank [0,1]. No lookahead."""
    r = s.rolling(w, min_periods=mp).rank(pct=True).clip(0.0, 1.0)
    return (1.0 - r) if invert else r


def _roll_min(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(w // 2, 2)).min()


def _roll_max(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(w // 2, 2)).max()


# ════════════════════════════════════════════════════════════════════════════════
# COMPONENT SCORES (shared across setups)
# ════════════════════════════════════════════════════════════════════════════════

def _downside_momentum_score(g: pd.DataFrame) -> pd.Series:
    """
    Is short-term momentum genuinely negative?
    High = negative momentum on multiple timeframes.
    """
    ret5  = g["ret_5d"].fillna(0)
    ret20 = g["ret_20d"].fillna(0)
    mom   = g["momentum_score"].fillna(0)

    # ret_5d negative percentile rank (higher = more negative rel to history)
    neg5_rank = _rank(-ret5, 252, 63)
    # ret_20d negative
    neg20_rank = _rank(-ret20, 252, 63)
    # momentum_score below zero
    mom_neg = (mom < 0).astype(float)
    # slope negative
    slope_neg = (g["slope_20d"].fillna(0) < 0).astype(float)
    # momentum direction
    mom_dir_neg = (g["momentum_direction"].fillna("NEUTRAL") == "NEUTRAL").astype(float) * 0.5 + \
                  (g["momentum_direction"].fillna("NEUTRAL") == "LONG").astype(float) * 0.0

    return (
        0.30 * neg5_rank
        + 0.25 * neg20_rank
        + 0.20 * mom_neg
        + 0.15 * slope_neg
        + 0.10 * mom_dir_neg
    ).clip(0.0, 1.0)


def _breakdown_score(g: pd.DataFrame) -> pd.Series:
    """
    Has price broken below recent support?
    High = close below rolling 10d low with expanding vol.
    """
    close = g["close"]
    low10 = _roll_min(close, 10)
    low5  = _roll_min(close, 5)

    # Below 10d low (shifted 1 to avoid same-bar look)
    below10 = (close < low10.shift(1)).astype(float)
    below5  = (close < low5.shift(1)).astype(float)

    # Distance below 10d low in ATR units
    atr = g["atr_14"].replace(0, np.nan).fillna(close * 0.01)
    dist_below = ((low10.shift(1) - close) / atr).clip(0, 5) / 5.0  # 0→1

    # Volume confirmation (if available)
    vol_z = g.get("volume_z", pd.Series(0.0, index=g.index)).fillna(0)
    vol_conf = (vol_z > 0.5).astype(float)

    return (
        0.35 * below10
        + 0.25 * dist_below
        + 0.25 * below5
        + 0.15 * vol_conf
    ).clip(0.0, 1.0)


def _vol_expansion_score(g: pd.DataFrame) -> pd.Series:
    """
    Is volatility expanding (confirming directional move)?
    High = vol_ratio rising, atr_pct elevated, recent 1d returns large.
    """
    vol_ratio_rank = _rank(g["vol_ratio"].fillna(1.0), 252, 63)
    atr_pct_rank   = _rank(g["atr_pct"].fillna(0.005), 252, 63)
    # Recent 1d absolute return rank (large moves = vol expansion)
    abs_ret1 = g["ret_1d"].abs().fillna(0)
    abs_rank = _rank(abs_ret1, 252, 63)

    return (
        0.40 * vol_ratio_rank
        + 0.35 * atr_pct_rank
        + 0.25 * abs_rank
    ).clip(0.0, 1.0)


def _regime_support_score(g: pd.DataFrame) -> pd.Series:
    """
    Does the regime support a short trade?
    High = SHOCK or TRANSITION regime with negative momentum.
    Note: we explicitly allow shorts in adverse regimes since that's
    exactly when they have edge.
    """
    regime_map = {
        "SHOCK": 1.00,
        "TRANSITION": 0.70,
        "CHOP": 0.40,
        "TREND": 0.10,  # penalty — shorting into strong LONG trend is dangerous
    }
    reg_score = g["portfolio_regime"].fillna("CHOP").map(regime_map).fillna(0.40)

    # Sentiment confirms danger
    sent_map = {"HIGH_RISK": 1.0, "EVENT_ACTIVE": 0.6, "NORMAL": 0.3}
    sent_score = g["portfolio_sentiment_flag"].fillna("NORMAL").map(sent_map).fillna(0.3)

    # Vol regime confirms
    vol_map = {"HIGH": 1.0, "ELEVATED": 0.7, "NORMAL": 0.3}
    vol_score = g["vol_regime"].fillna("NORMAL").map(vol_map).fillna(0.3)

    return (
        0.50 * reg_score
        + 0.25 * sent_score
        + 0.25 * vol_score
    ).clip(0.0, 1.0)


def _extension_penalty(g: pd.DataFrame) -> pd.Series:
    """
    Is price already way oversold / extended to the downside?
    High penalty = already very extended — avoid chasing shorts.
    """
    ret5  = g["ret_5d"].fillna(0)
    ret20 = g["ret_20d"].fillna(0)

    # Percentile of negative returns (very negative → extended)
    neg5_rank  = _rank(-ret5,  252, 63)   # 1 = most negative
    neg20_rank = _rank(-ret20, 252, 63)

    # h4_zscore very negative = oversold on intraday basis
    h4_z = g["h4_zscore"].fillna(0)
    h4_oversold = ((-h4_z).clip(0, 4) / 4.0)  # 0→1 scale, high = very oversold

    # Combined: if ALL three very extended, penalize heavily
    return (
        0.40 * neg5_rank.clip(0.0, 1.0) * neg20_rank.clip(0.0, 1.0)  # both negative
        + 0.30 * h4_oversold
        + 0.30 * neg5_rank
    ).clip(0.0, 1.0)


def _weekly_bias_short_score(g: pd.DataFrame) -> pd.Series:
    """
    Weekly bias confirmation for shorts.
    High = NOT bullish (neutral or no bias).
    """
    not_long  = (g["weekly_bias"].fillna("NEUTRAL") != "LONG").astype(float)
    conf      = g["weekly_confidence"].fillna(0.5).clip(0, 1)
    # If bias is LONG with high confidence → strong penalty
    long_conf = (g["weekly_bias"] == "LONG").astype(float) * conf
    return (not_long * (1.0 - 0.5 * long_conf)).clip(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# SETUP A — DOWNSIDE MOMENTUM SHORT
# ════════════════════════════════════════════════════════════════════════════════

def _compute_dm_short(g: pd.DataFrame) -> tuple:
    """Returns (signal_series, score_series)."""
    dm_score  = _downside_momentum_score(g)
    bkd_score = _breakdown_score(g)
    vol_score = _vol_expansion_score(g)
    reg_score = _regime_support_score(g)
    wb_score  = _weekly_bias_short_score(g)
    ext_pen   = _extension_penalty(g)

    raw = (
        0.30 * dm_score
        + 0.25 * bkd_score
        + 0.20 * vol_score
        + 0.15 * reg_score
        + 0.10 * wb_score
        - 0.20 * ext_pen          # penalise already-extended shorts
    ).clip(0.0, 1.0)

    # Entry gate:
    # 1. ret_5d must be negative (short-term downside)
    # 2. momentum must be weakening (slope or momentum_score negative)
    # 3. regime must NOT be strongly bullish TREND
    gate = (
        (g["ret_5d"].fillna(0) < 0) &
        (g["momentum_score"].fillna(0) < 0.15) &    # not strongly long
        (g["portfolio_regime"].fillna("CHOP") != "TREND") &
        (raw > 0.30)                                 # minimum raw score
    )

    if getattr(config, "TACTICAL_SHORT_REQUIRE_CONFIRMATION", True):
        # Additional: price closed below 5d low
        close = g["close"]
        low5  = _roll_min(close, 5).shift(1)
        gate  = gate & (close < low5)

    return gate.astype(float), raw


# ════════════════════════════════════════════════════════════════════════════════
# SETUP B — FAILED RALLY SHORT
# ════════════════════════════════════════════════════════════════════════════════

def _compute_fr_short(g: pd.DataFrame) -> tuple:
    """Returns (signal_series, score_series)."""
    close  = g["close"]
    ma20   = close.rolling(20, min_periods=10).mean()
    ma10   = close.rolling(10, min_periods=5).mean()

    ret1  = g["ret_1d"].fillna(0)
    ret5  = g["ret_5d"].fillna(0)
    ret20 = g["ret_20d"].fillna(0)

    # 1. Larger trend down: medium-term negative or slope negative
    trend_down = (
        ((ret20 < 0) | (g["slope_20d"].fillna(0) < 0)).astype(float)
    )

    # 2. Recent bounce: 5d return was positive (bounce toward resistance)
    #    Then failed: today's 1d return negative
    bounce_then_fail = (
        (ret5.shift(1) > 0.005) &    # at least +0.5% bounce prior
        (ret1 < 0) &                  # failed today
        (close < ma10)                # still below 10d MA (bounce didn't recover)
    ).astype(float)

    # 3. Below 20d MA (price is below central tendency)
    below_ma20 = (close < ma20).astype(float)

    # 4. Momentum turning down: momentum_score dropped
    mom = g["momentum_score"].fillna(0)
    mom_falling = (mom < mom.shift(3)).astype(float)

    # 5. Vol controlled or expanding directionally
    vol_score = _vol_expansion_score(g) * 0.5 + 0.5  # partial vol score

    reg_score  = _regime_support_score(g)
    ext_pen    = _extension_penalty(g)
    wb_score   = _weekly_bias_short_score(g)

    raw = (
        0.25 * trend_down
        + 0.25 * bounce_then_fail
        + 0.20 * below_ma20
        + 0.15 * mom_falling
        + 0.10 * reg_score
        + 0.05 * wb_score
        - 0.15 * ext_pen
    ).clip(0.0, 1.0)

    # Entry gate:
    gate = (
        (ret20 < 0) &                                 # medium-term downtrend
        (ret5.shift(1) > 0.003) &                     # prior bounce ≥ +0.3%
        (ret1 < 0) &                                  # failed today
        (close < ma20) &                              # below MA
        (g["portfolio_regime"].fillna("CHOP") != "TREND") &
        (raw > 0.35)
    )

    return gate.astype(float), raw


# ════════════════════════════════════════════════════════════════════════════════
# SETUP C — COMPRESSION BREAKDOWN SHORT
# ════════════════════════════════════════════════════════════════════════════════

def _compute_cb_short(g: pd.DataFrame) -> tuple:
    """Returns (signal_series, score_series)."""
    close = g["close"]
    atr   = g["atr_14"].replace(0, np.nan).fillna(close * 0.01)

    # 1. Range compressed: rolling 10d range (high-low) vs ATR was low
    high10 = _roll_max(g["high"].fillna(close), 10)
    low10  = _roll_min(g["low"].fillna(close),  10)
    range10 = (high10 - low10) / atr   # range in ATR units
    # Low range = compressed (below 30th percentile historically)
    range_rank_inv = _rank(range10, 252, 63, invert=True)  # high = compressed
    compressed = (range10 < range10.rolling(60, min_periods=20).quantile(0.35))

    # 2. Price breaks below 10d range low
    below_range = (close < low10.shift(1)).astype(float)

    # 3. Vol expands AFTER compression (vol_ratio now rising from below)
    vol_ratio  = g["vol_ratio"].fillna(1.0)
    vol_expand = (vol_ratio > vol_ratio.shift(3)).astype(float)

    # 4. Breakdown confirmed: stayed below for ≥1 bar (today close < yesterday low10)
    #    (using shifted range low so it's the range BEFORE today)
    confirm    = (close < low10.shift(2)).astype(float)

    reg_score  = _regime_support_score(g)
    ext_pen    = _extension_penalty(g)

    raw = (
        0.30 * below_range
        + 0.25 * range_rank_inv.clip(0, 1)
        + 0.20 * vol_expand
        + 0.15 * reg_score
        + 0.10 * confirm
        - 0.15 * ext_pen
    ).clip(0.0, 1.0)

    # Entry gate:
    gate = (
        compressed.fillna(False) &
        (below_range > 0) &
        (vol_expand > 0) &
        (g["portfolio_regime"].fillna("CHOP") != "TREND") &
        (raw > 0.35)
    )

    return gate.astype(float), raw


# ════════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE + BUCKET
# ════════════════════════════════════════════════════════════════════════════════

def _tactical_short_score(dm_sig, dm_raw, fr_sig, fr_raw, cb_sig, cb_raw) -> pd.Series:
    """
    Best active short score across all setups.
    Bonus for multi-setup confirmation.
    """
    # Best individual score among active signals
    scores = pd.DataFrame({
        "dm": dm_sig * dm_raw,
        "fr": fr_sig * fr_raw,
        "cb": cb_sig * cb_raw,
    })
    best = scores.max(axis=1)

    # Confirmation bonus: +0.05 for each additional active signal
    n_active = (dm_sig > 0).astype(int) + (fr_sig > 0).astype(int) + (cb_sig > 0).astype(int)
    bonus = (n_active - 1).clip(0) * 0.05

    return (best + bonus).clip(0.0, 1.0)


def _score_to_bucket(score: pd.Series) -> pd.Series:
    def _b(s):
        if pd.isna(s) or s < 0.35:
            return "NONE"
        elif s < 0.50:
            return "WEAK"
        elif s < 0.65:
            return "MODERATE"
        else:
            return "STRONG"
    return score.apply(_b)


def _type_label(dm_sig, fr_sig, cb_sig, idx) -> pd.Series:
    types = []
    for i in idx:
        parts = []
        if dm_sig.loc[i] > 0: parts.append("DM")
        if fr_sig.loc[i] > 0: parts.append("FR")
        if cb_sig.loc[i] > 0: parts.append("CB")
        types.append("+".join(parts) if parts else "NONE")
    return pd.Series(types, index=idx)


def _reason_str(dm_s, fr_s, cb_s, reg_s, ext_pen, score) -> str:
    parts = []
    if dm_s > 0.35:  parts.append(f"DM={dm_s:.2f}")
    if fr_s > 0.35:  parts.append(f"FR={fr_s:.2f}")
    if cb_s > 0.35:  parts.append(f"CB={cb_s:.2f}")
    if reg_s > 0.5:  parts.append("regime_ok")
    if ext_pen > 0.6: parts.append("ext_warn")
    return "|".join(parts) or "no_signal"


# ════════════════════════════════════════════════════════════════════════════════
# SUGGESTED ENTRY WEIGHT
# ════════════════════════════════════════════════════════════════════════════════

def _entry_weight(bucket: str, size_mult: float = None) -> float:
    if size_mult is None:
        size_mult = getattr(config, "TACTICAL_SHORT_SIZE_MULT", 0.50)

    base = getattr(config, "VOL_SCALE_MAX", 0.30)
    bucket_frac = {"STRONG": 0.50, "MODERATE": 0.25, "WEAK": 0.00, "NONE": 0.00}
    w = -base * bucket_frac.get(bucket, 0.0) * size_mult
    max_w = getattr(config, "MAX_WEIGHT_PER_SYMBOL", 0.30)
    return max(w, -max_w)


# ════════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════

def _compute_short_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values("date").reset_index(drop=True)

    dm_sig, dm_raw = _compute_dm_short(g)
    fr_sig, fr_raw = _compute_fr_short(g)
    cb_sig, cb_raw = _compute_cb_short(g)

    reg_s  = _regime_support_score(g)
    ext_pen = _extension_penalty(g)

    score  = _tactical_short_score(dm_sig, dm_raw, fr_sig, fr_raw, cb_sig, cb_raw)
    bucket = _score_to_bucket(score)
    any_sig = ((dm_sig > 0) | (fr_sig > 0) | (cb_sig > 0)) & (bucket != "NONE")
    type_s  = _type_label(dm_sig, fr_sig, cb_sig, g.index)

    reasons = [
        _reason_str(dm_raw.iloc[i], fr_raw.iloc[i], cb_raw.iloc[i],
                    reg_s.iloc[i], ext_pen.iloc[i], score.iloc[i])
        for i in range(len(g))
    ]

    g["ts_dm_signal"] = dm_sig.astype(bool)
    g["ts_dm_score"]  = dm_raw.round(4)
    g["ts_fr_signal"] = fr_sig.astype(bool)
    g["ts_fr_score"]  = fr_raw.round(4)
    g["ts_cb_signal"] = cb_sig.astype(bool)
    g["ts_cb_score"]  = cb_raw.round(4)

    g["tactical_short_signal"]       = any_sig
    g["tactical_short_type"]         = type_s
    g["tactical_short_score"]        = score.round(4)
    g["tactical_short_bucket"]       = bucket
    g["tactical_short_reason"]       = reasons
    g["tactical_short_entry_weight"] = bucket.map(lambda b: _entry_weight(b))
    g["tactical_short_max_hold"]     = getattr(config, "TACTICAL_SHORT_MAX_HOLD", 3)
    g["tactical_short_stop"]         = getattr(config, "TACTICAL_SHORT_STOP", -0.015)

    return g


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def compute_tactical_shorts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tactical short signal columns to df.
    Processed per-symbol independently. Safe on multi-symbol df.
    """
    df = df.copy()
    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = _compute_short_features(g)
        parts.append(g)

    result = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"])

    # Log distribution
    for sym in result["symbol"].unique():
        g  = result[result["symbol"] == sym]
        n  = len(g)
        vs = g["tactical_short_bucket"].value_counts()
        sig_n = g["tactical_short_signal"].sum()
        logger.info(
            "TacticalShort %s — signals=%d (%.1f%%)  STRONG=%d  MOD=%d  WEAK=%d  NONE=%d",
            sym, sig_n, 100 * sig_n / max(n, 1),
            vs.get("STRONG", 0), vs.get("MODERATE", 0),
            vs.get("WEAK", 0), vs.get("NONE", 0),
        )
    return result
