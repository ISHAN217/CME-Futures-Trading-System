"""
trend_continuation_edge_engine.py — TREND continuation-edge scoring. [NEW]

Fixes the failure of the old trend_quality_score, which rewarded obvious/mature
trends and labelled late-stage EXCEPTIONAL entries that performed worst.

Key design change:
  OLD: rewarded strong, clean, aligned trend → measured MATURITY
  NEW: rewards FRESH continuation after pullback in intact trend;
       explicitly PENALISES maturity, extension, and exhaustion

The score is built from 8 components, some additive (good for continuation),
some subtractive (bad for continuation):

  ADDITIVE (edge present):
    A. trend_intact_score          — trend still structurally valid
    F. pullback_reset_score        — healthy pullback recently reset price
    G. momentum_reacceleration     — momentum picking up again
    H. weekly_bias_alignment       — structural weekly confirmation
    I. volatility_quality          — stable, non-explosive vol environment

  SUBTRACTIVE (edge consumed):
    B. trend_maturity_score        — trend has been running a long time
    C. price_extension_score       — price stretched above fair value
    D. momentum_exhaustion_score   — momentum rolling over
    E. volatility_exhaustion_score — vol expanding dangerously late in move

Composite formula (theoretical range −0.60 to +0.90, normalised to [0,1]):
  raw   =  0.30*intact + 0.25*pullback_reset + 0.15*reaccel
         + 0.10*wb_align + 0.10*vol_quality
         − 0.20*extension − 0.15*maturity
         − 0.15*exhaustion − 0.10*vol_exhaustion
  score = clip((raw + 0.60) / 1.50, 0, 1)

Buckets:
  NO_EDGE:     0.00–0.20
  WEAK:        0.20–0.40
  MODERATE:    0.40–0.60
  STRONG:      0.60–0.80
  EXCEPTIONAL: 0.80–1.00

Output columns per (date, symbol):
  trend_intact_score
  trend_maturity_score
  price_extension_score
  momentum_exhaustion_score
  volatility_exhaustion_score
  pullback_reset_score
  momentum_reacceleration_score
  trend_continuation_edge_score
  trend_continuation_edge_bucket
  trend_continuation_edge_reason
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Bucket thresholds ─────────────────────────────────────────────────────────
_EDGE_BUCKETS = [
    (0.80, "EXCEPTIONAL"),
    (0.60, "STRONG"),
    (0.40, "MODERATE"),
    (0.20, "WEAK"),
    (0.00, "NO_EDGE"),
]

# Composite formula weights
_W_POS = dict(intact=0.30, pullback=0.25, reaccel=0.15, wb=0.10, vol_q=0.10)
_W_NEG = dict(extension=0.20, maturity=0.15, exhaustion=0.15, vol_exh=0.10)
# Theoretical min = −(sum of negative weights) = −0.60
# Theoretical max = +(sum of positive weights) = +0.90
_NORM_SHIFT = 0.60     # shift so min = 0
_NORM_RANGE = 1.50     # range = 0.90 − (−0.60)


def _score_to_bucket(s: float) -> str:
    if pd.isna(s):
        return "MODERATE"
    for thr, label in _EDGE_BUCKETS:
        if s >= thr:
            return label
    return "NO_EDGE"


# ── Rolling percentile rank ───────────────────────────────────────────────────

def _rank(series: pd.Series, window: int = 252,
          min_periods: int = 63, invert: bool = False) -> pd.Series:
    """Backward-looking rolling percentile rank [0, 1]. No lookahead."""
    try:
        r = series.rolling(window, min_periods=min_periods).rank(pct=True)
    except Exception:
        r = series.rolling(window, min_periods=min_periods).apply(
            lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else 0.5, raw=True
        )
    r = r.clip(0.0, 1.0)
    return (1.0 - r) if invert else r


# ── Consecutive-true counter ──────────────────────────────────────────────────

def _consecutive_true(cond: pd.Series) -> pd.Series:
    """Vectorised: count consecutive True days (resets to 0 on False)."""
    cond_b = cond.fillna(False).astype(bool)
    groups = (cond_b != cond_b.shift(1, fill_value=False)).cumsum()
    count  = cond_b.groupby(groups).cumsum()
    return count.astype(float).where(cond_b, 0.0)


# ════════════════════════════════════════════════════════════════════════════════
# COMPONENT A — Trend intact
# ════════════════════════════════════════════════════════════════════════════════

def _trend_intact(g: pd.DataFrame) -> pd.Series:
    """
    Is the trend structurally still alive?
    High = multiple timeframes agree trend is ON.
    Uses: momentum_direction, ret_60d, slope_20d, r2_20d, weekly_bias, h4_exec_signal
    """
    mom_20d_pos = (g["momentum_direction"].fillna("NEUTRAL") == "LONG").astype(float)
    mom_60d_pos = (g["ret_60d"].fillna(0) > 0).astype(float)
    slope_pos   = (g["slope_20d"].fillna(0) > 0).astype(float)
    r2_rank     = _rank(g["r2_20d"].fillna(0), 252, 63)
    wb_align    = (g["weekly_bias"].fillna("NEUTRAL") == "LONG").astype(float)

    h4_map = {"MOMENTUM": 0.80, "BREAKOUT": 0.90, "PULLBACK": 0.85,
              "WAIT": 0.20, "NONE": 0.50}
    h4_conf = g["h4_exec_signal"].fillna("NONE").map(h4_map).fillna(0.50)

    return (
        0.20 * mom_20d_pos
        + 0.20 * mom_60d_pos
        + 0.15 * slope_pos
        + 0.20 * r2_rank
        + 0.15 * wb_align
        + 0.10 * h4_conf
    ).clip(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# COMPONENT B — Trend maturity (subtractive)
# ════════════════════════════════════════════════════════════════════════════════

def _trend_maturity(g: pd.DataFrame) -> pd.Series:
    """
    How long has the trend been running? High = mature/late.
    Uses: portfolio_regime, momentum_direction, close vs MA20.
    """
    # Consecutive TREND regime days
    in_trend    = (g["portfolio_regime"].fillna("CHOP") == "TREND")
    consec_trend = _consecutive_true(in_trend)
    trend_rank   = _rank(consec_trend, 252, 63)

    # Consecutive days with LONG momentum direction
    in_mom_long  = (g["momentum_direction"].fillna("NEUTRAL") == "LONG")
    consec_mom   = _consecutive_true(in_mom_long)
    mom_rank     = _rank(consec_mom, 252, 63)

    # Consecutive closes above 20d MA
    ma_20 = g["close"].rolling(20, min_periods=10).mean()
    above_ma = (g["close"] > ma_20)
    consec_ma  = _consecutive_true(above_ma)
    ma_rank    = _rank(consec_ma, 252, 63)

    return (
        0.40 * trend_rank
        + 0.40 * mom_rank
        + 0.20 * ma_rank
    ).clip(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# COMPONENT C — Price extension (subtractive)
# ════════════════════════════════════════════════════════════════════════════════

def _price_extension(g: pd.DataFrame) -> pd.Series:
    """
    Is price stretched above fair trend value? High = over-extended.
    Uses: close, atr_14, h4_zscore, ret_20d.
    """
    close = g["close"]
    ma_20 = close.rolling(20, min_periods=10).mean()
    std_20 = close.rolling(20, min_periods=10).std().replace(0, np.nan)

    # MA z-score (normalised distance above MA)
    ma_z      = ((close - ma_20) / std_20).fillna(0).clip(-4, 4)
    ma_z_rank = _rank(ma_z, 252, 63)

    # ATR-normalised distance
    atr_dist = ((close - ma_20) / g["atr_14"].replace(0, np.nan)).fillna(0).clip(-6, 6)
    atr_rank = _rank(atr_dist, 252, 63)

    # 20d return percentile (higher cumulative move → more extended)
    ret_20d_rank = _rank(g["ret_20d"].fillna(0), 252, 63)

    # 4H zscore absolute value (high abs z = overextended on intraday)
    h4_ext = g["h4_zscore"].abs().fillna(0).clip(0, 4) / 4.0  # normalise 0→1

    return (
        0.35 * ma_z_rank
        + 0.30 * atr_rank
        + 0.25 * ret_20d_rank
        + 0.10 * h4_ext
    ).clip(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# COMPONENT D — Momentum exhaustion (subtractive)
# ════════════════════════════════════════════════════════════════════════════════

def _momentum_exhaustion(g: pd.DataFrame) -> pd.Series:
    """
    Is momentum rolling over? High = decelerating/exhausted.
    Uses: momentum_score, ret_20d, ret_5d, ret_1d.
    """
    mom = g["momentum_score"].fillna(0)

    # 5-day change in momentum_score — negative = slowing
    mom_delta_5  = (mom - mom.shift(5)).clip(-2, 2)
    # Exhaustion: rank of *negative* delta (more negative = more exhausted)
    exh_mom_rank = _rank(-mom_delta_5, 252, 63)

    # 5-day change in 20d return — deceleration in medium-term trend
    ret20 = g["ret_20d"].fillna(0)
    ret20_delta  = (ret20 - ret20.shift(5)).clip(-0.20, 0.20)
    exh_ret_rank = _rank(-ret20_delta, 252, 63)

    # Short-term return acceleration: 5d average of ret_1d vs 10d average
    r1 = g["ret_1d"].fillna(0)
    accel = (r1.rolling(5,  min_periods=3).mean()
           - r1.rolling(10, min_periods=5).mean())
    exh_accel_rank = _rank(-accel, 252, 63)   # negative accel = exhaustion

    return (
        0.45 * exh_mom_rank
        + 0.35 * exh_ret_rank
        + 0.20 * exh_accel_rank
    ).clip(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# COMPONENT E — Volatility exhaustion (subtractive)
# ════════════════════════════════════════════════════════════════════════════════

def _vol_exhaustion(g: pd.DataFrame) -> pd.Series:
    """
    Is volatility expanding dangerously late in the move?
    High = vol expanding into strong trend (unstable, crowded).
    Uses: vol_ratio, atr_pct, vol_regime.
    """
    vol_ratio_rank = _rank(g["vol_ratio"].fillna(1.0),  252, 63)
    atr_pct_rank   = _rank(g["atr_pct"].fillna(0.005), 252, 63)

    vol_reg_score = g["vol_regime"].fillna("NORMAL").map(
        {"NORMAL": 0.0, "ELEVATED": 0.5, "HIGH": 1.0}
    ).fillna(0.3)

    return (
        0.35 * vol_ratio_rank
        + 0.35 * atr_pct_rank
        + 0.30 * vol_reg_score
    ).clip(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# COMPONENT F — Pullback reset (additive — the most novel component)
# ════════════════════════════════════════════════════════════════════════════════

def _pullback_reset(g: pd.DataFrame) -> pd.Series:
    """
    Has the trend recently reset via a healthy pullback?
    Best TREND entries occur AFTER a moderate pullback, not at the peak.
    High = recent shallow-to-moderate pullback with recovery underway.
    Uses: ret_5d, ret_20d, ret_1d, atr_pct, h4_exec_signal.
    """
    # 1. Pullback flag: short-term down within medium-term uptrend
    pb_flag = ((g["ret_5d"].fillna(0) < 0) &
               (g["ret_20d"].fillna(0) > 0)).astype(float)

    # 2. Pullback quality: moderate depth is optimal
    # ATR-normalised 5d return (in units of daily ATR as pct of price)
    atr_pct = g["atr_pct"].replace(0, np.nan).fillna(0.01)
    pb_depth = (g["ret_5d"].fillna(0) / atr_pct).clip(-8, 0)
    # Gaussian centred at −1.5 ATR (moderate healthy pullback)
    pb_quality = np.exp(-0.5 * ((pb_depth + 1.5) / 1.2) ** 2)
    pb_quality = pd.Series(pb_quality, index=g.index).clip(0, 1)

    # 3. Recovery: yesterday's 1d return positive (bounce underway)
    recovering = (g["ret_1d"].fillna(0) > 0).astype(float)

    # 4. 4H tactical pullback signal
    h4_pb = (g["h4_exec_signal"].fillna("NONE") == "PULLBACK").astype(float)

    # 5. MA proximity reset: h4_zscore came back toward zero from below
    h4_z = g["h4_zscore"].fillna(0)
    # Optimal zone: h4_zscore in [−1.5, 0] — pulled back but not crushed
    ma_proximity = ((-1.5 <= h4_z) & (h4_z <= 0.5)).astype(float)

    return (
        0.30 * pb_flag
        + 0.25 * pb_quality
        + 0.20 * recovering
        + 0.15 * h4_pb
        + 0.10 * ma_proximity
    ).clip(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# COMPONENT G — Momentum reacceleration (additive)
# ════════════════════════════════════════════════════════════════════════════════

def _momentum_reaccel(g: pd.DataFrame) -> pd.Series:
    """
    Is momentum picking up speed again (after a potential pause/pullback)?
    High = momentum recently inflected upward.
    Uses: momentum_score, ret_5d, ret_1d, h4_exec_signal.
    """
    mom = g["momentum_score"].fillna(0)

    # Momentum_score rising over past 5 bars
    mom_rising = (mom > mom.shift(5)).astype(float)

    # Short-term return improving
    ret5 = g["ret_5d"].fillna(0)
    ret5_rising = (ret5 > ret5.shift(5)).astype(float)

    # 4H signal: MOMENTUM or BREAKOUT (tactical confirmation of reaccel)
    h4_accel = g["h4_exec_signal"].fillna("NONE").isin(
        ["MOMENTUM", "BREAKOUT"]
    ).astype(float)

    # Recent days: fraction of last 3 daily returns that are positive
    r1 = g["ret_1d"].fillna(0)
    recent_pos = r1.rolling(3, min_periods=1).apply(
        lambda x: (x > 0).mean(), raw=True
    )

    return (
        0.35 * mom_rising
        + 0.25 * ret5_rising
        + 0.25 * h4_accel
        + 0.15 * recent_pos
    ).clip(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# COMPONENT H — Weekly bias alignment (additive, separate from trend_intact)
# ════════════════════════════════════════════════════════════════════════════════

def _weekly_bias_align(g: pd.DataFrame) -> pd.Series:
    """
    Structural weekly directional confirmation.
    High = weekly bias LONG with high confidence.
    """
    is_long  = (g["weekly_bias"].fillna("NEUTRAL") == "LONG")
    conf     = g["weekly_confidence"].fillna(0).clip(0, 1)
    return np.where(is_long, 0.60 + 0.40 * conf, 0.0).clip(0, 1)


# ════════════════════════════════════════════════════════════════════════════════
# COMPONENT I — Volatility quality (additive)
# ════════════════════════════════════════════════════════════════════════════════

def _vol_quality(g: pd.DataFrame) -> pd.Series:
    """
    Stable, non-explosive vol environment favours trend continuation.
    High = low vol regime, vol not expanding.
    """
    vol_reg_score = g["vol_regime"].fillna("NORMAL").map(
        {"NORMAL": 1.0, "ELEVATED": 0.5, "HIGH": 0.0}
    ).fillna(0.5)

    # Inverted vol_ratio rank: lower vol_ratio = more stable = better
    inv_vol_ratio = _rank(g["vol_ratio"].fillna(1.0), 252, 63, invert=True)

    return (0.60 * vol_reg_score + 0.40 * inv_vol_ratio).clip(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# REASON STRING
# ════════════════════════════════════════════════════════════════════════════════

def _build_reason(
    intact, maturity, extension, exhaustion, vol_exh,
    pullback, reaccel, score, bucket,
) -> str:
    parts = []
    # Positive drivers
    if intact >= 0.70:     parts.append("trend_on")
    elif intact <= 0.30:   parts.append("trend_fading")
    if pullback >= 0.60:   parts.append("pb_reset")
    elif pullback <= 0.20: parts.append("no_reset")
    if reaccel >= 0.60:    parts.append("reaccel")
    # Negative drags
    if maturity >= 0.70:   parts.append("mature")
    if extension >= 0.70:  parts.append("extended")
    if exhaustion >= 0.70: parts.append("exhausted")
    if vol_exh >= 0.70:    parts.append("vol_exh")
    return "|".join(parts) or "neutral"


# ════════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════

def _compute_edge_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values("date").reset_index(drop=True)

    intact    = _trend_intact(g)
    maturity  = _trend_maturity(g)
    extension = _price_extension(g)
    exhaustion= _momentum_exhaustion(g)
    vol_exh   = _vol_exhaustion(g)
    pullback  = _pullback_reset(g)
    reaccel   = _momentum_reaccel(g)
    wb        = _weekly_bias_align(g)
    vol_q     = _vol_quality(g)

    # Composite formula
    raw = (
        _W_POS["intact"]   * intact
        + _W_POS["pullback"] * pullback
        + _W_POS["reaccel"]  * reaccel
        + _W_POS["wb"]       * wb
        + _W_POS["vol_q"]    * vol_q
        - _W_NEG["extension"] * extension
        - _W_NEG["maturity"]  * maturity
        - _W_NEG["exhaustion"]* exhaustion
        - _W_NEG["vol_exh"]   * vol_exh
    )
    score = ((raw + _NORM_SHIFT) / _NORM_RANGE).clip(0.0, 1.0)

    g["trend_intact_score"]           = intact.round(4)
    g["trend_maturity_score"]         = maturity.round(4)
    g["price_extension_score"]        = extension.round(4)
    g["momentum_exhaustion_score"]    = exhaustion.round(4)
    g["volatility_exhaustion_score"]  = vol_exh.round(4)
    g["pullback_reset_score"]         = pullback.round(4)
    g["momentum_reacceleration_score"]= reaccel.round(4)
    g["trend_continuation_edge_score"]= score.round(4)
    g["trend_continuation_edge_bucket"]= score.apply(_score_to_bucket)

    reasons = []
    for i in range(len(g)):
        reasons.append(_build_reason(
            intact.iloc[i], maturity.iloc[i], extension.iloc[i],
            exhaustion.iloc[i], vol_exh.iloc[i], pullback.iloc[i],
            reaccel.iloc[i], score.iloc[i],
            g["trend_continuation_edge_bucket"].iloc[i],
        ))
    g["trend_continuation_edge_reason"] = reasons

    return g


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def compute_trend_continuation_edge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TREND continuation-edge scores to df.
    Processed per-symbol independently (no cross-symbol data needed).
    Safe to call on a full multi-symbol df.
    """
    df = df.copy()

    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = _compute_edge_features(g)
        parts.append(g)

    df = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"])

    _log_distribution(df)
    return df


def _log_distribution(df: pd.DataFrame) -> None:
    for sym in df["symbol"].unique():
        g  = df[df["symbol"] == sym]
        vc = g["trend_continuation_edge_bucket"].value_counts()
        n  = max(len(g), 1)
        logger.info(
            "TrendEdge %s — EXCEP: %d (%.0f%%)  STRONG: %d (%.0f%%)  "
            "MOD: %d (%.0f%%)  WEAK: %d (%.0f%%)  NO_EDGE: %d (%.0f%%)  "
            "avg_score=%.3f",
            sym,
            vc.get("EXCEPTIONAL", 0), 100 * vc.get("EXCEPTIONAL", 0) / n,
            vc.get("STRONG",      0), 100 * vc.get("STRONG",      0) / n,
            vc.get("MODERATE",    0), 100 * vc.get("MODERATE",    0) / n,
            vc.get("WEAK",        0), 100 * vc.get("WEAK",        0) / n,
            vc.get("NO_EDGE",     0), 100 * vc.get("NO_EDGE",     0) / n,
            g["trend_continuation_edge_score"].mean(),
        )
