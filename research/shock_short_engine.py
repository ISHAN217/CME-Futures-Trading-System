"""
shock_short_engine.py — SHOCK-regime-only short sleeve. [NEW]

Key design differences vs tactical_short_engine.py (rejected):
  OLD: blocked entries in SHOCK regime → missed best opportunities
       used -1.5% fixed stop → too tight for equity-index SHOCK rebounds
       fired on normal pullbacks in uptrends → structural bias mismatch

  NEW: ONLY enters in SHOCK/post-SHOCK contexts
       tests ATR-based stops and wider fixed stops
       three focused setups, all SHOCK-conditional

Pre-analysis findings (from shock_regime_audit):
  - ES: 54 SHOCK days 2019-2026, 59.3% negative
  - NQ: 54 SHOCK days 2019-2026, 57.4% negative
  - SHOCK next-day average is POSITIVE (+0.30/0.40%) — mean-reverting tendency
  - Negative SHOCK days close near lows (strength=0.128) — genuine breakdowns
  - 2020/2021 were genuine SHOCK continuation periods; 2024/2025 are noisier

Setups:
  A. SHOCK_CONTINUATION  — enter during SHOCK if negative + close near low
  B. POST_SHOCK_BOUNCE   — enter 1-5 days after SHOCK when bounce fails
  C. SHOCK_BREAKDOWN     — compression then SHOCK breakdown

Output columns:
  ss_cont_signal      bool
  ss_cont_score       float 0-1
  ss_bounce_signal    bool
  ss_bounce_score     float 0-1
  ss_breakdown_signal bool
  ss_breakdown_score  float 0-1
  shock_short_signal  bool   — any of the three
  shock_short_type    str    — CONT | BOUNCE | BREAKDOWN | combos
  shock_short_score   float  — best active score
  shock_short_bucket  str    — STRONG | MODERATE | WEAK | NONE
  shock_short_reason  str
  shock_short_entry_weight  float (negative)
  shock_short_stop_type     str
  shock_short_max_hold      int
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ── Rolling min/max helpers ────────────────────────────────────────────────────

def _rmin(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(w // 2, 2)).min()


def _rmax(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(w // 2, 2)).max()


def _rank(s: pd.Series, w: int = 252, mp: int = 63, invert: bool = False) -> pd.Series:
    r = s.rolling(w, min_periods=mp).rank(pct=True).clip(0.0, 1.0)
    return (1.0 - r) if invert else r


# ════════════════════════════════════════════════════════════════════════════════
# SHARED COMPONENTS
# ════════════════════════════════════════════════════════════════════════════════

def _shock_flag(g: pd.DataFrame) -> pd.Series:
    """Binary: is today a SHOCK regime day?"""
    return (g["portfolio_regime"].fillna("CHOP") == "SHOCK").astype(float)


def _neg_shock_flag(g: pd.DataFrame) -> pd.Series:
    """SHOCK day AND negative return."""
    return (
        (g["portfolio_regime"].fillna("CHOP") == "SHOCK") &
        (g["ret_1d"].fillna(0) < 0)
    ).astype(float)


def _close_weakness(g: pd.DataFrame) -> pd.Series:
    """
    Intraday close weakness: 0 = closed at high, 1 = closed at low.
    Uses OHLC if available, otherwise uses 1d return as proxy.
    """
    if "high" in g.columns and "low" in g.columns:
        hi   = g["high"].fillna(g["close"])
        lo   = g["low"].fillna(g["close"])
        rng  = (hi - lo).replace(0, np.nan)
        strength = (g["close"] - lo) / rng    # 0=at low, 1=at high
        return (1.0 - strength.fillna(0.5)).clip(0.0, 1.0)
    else:
        # Proxy: large negative 1d return = weakness
        r = g["ret_1d"].fillna(0).clip(-0.10, 0.0)
        return (-r / 0.10).clip(0.0, 1.0)


def _vol_expansion(g: pd.DataFrame) -> pd.Series:
    """Vol expanding directionally (confirming downside)."""
    vol_ratio_rank = _rank(g["vol_ratio"].fillna(1.0), 252, 63)
    atr_rank       = _rank(g["atr_pct"].fillna(0.005), 252, 63)
    return (0.6 * vol_ratio_rank + 0.4 * atr_rank).clip(0.0, 1.0)


def _days_since_shock(g: pd.DataFrame) -> pd.Series:
    """Days since most recent SHOCK regime day (0 = today is SHOCK)."""
    in_shock = (g["portfolio_regime"].fillna("CHOP") == "SHOCK")
    # Rolling: track last SHOCK position
    result = pd.Series(999, index=g.index, dtype=float)
    last_shock = -999
    for i, (idx, val) in enumerate(in_shock.items()):
        if val:
            last_shock = i
        result.iloc[i] = i - last_shock
    return result


def _not_in_strong_bull(g: pd.DataFrame) -> pd.Series:
    """
    True when we're NOT in a strong uptrend (safe to short structurally).
    Using: slope_20d, momentum_direction, weekly_bias.
    """
    slope_neg = (g["slope_20d"].fillna(0) <= 0).astype(float)
    not_mom_long = (g["momentum_direction"].fillna("NEUTRAL") != "LONG").astype(float) * 0.5
    not_wb_long  = (g["weekly_bias"].fillna("NEUTRAL") != "LONG").astype(float) * 0.5
    return (slope_neg * 0.5 + not_mom_long + not_wb_long).clip(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# SETUP A — SHOCK CONTINUATION SHORT
# ════════════════════════════════════════════════════════════════════════════════

def _compute_shock_continuation(g: pd.DataFrame) -> tuple:
    """
    Enter short on a negative SHOCK day where close is near low.
    Best signals: confirmed directional breakdown, not chaotic reversal.

    Signal fires when:
    - portfolio_regime == SHOCK
    - ret_1d < -threshold (genuine negative day)
    - close is near the day's low (weak close = continuation)
    - vol is expanding
    - price is below rolling 5d low (breakdown confirmed)
    - NOT a strong-momentum uptrend
    """
    shock_neg  = _neg_shock_flag(g)
    close_weak = _close_weakness(g)
    vol_exp    = _vol_expansion(g)
    no_bull    = _not_in_strong_bull(g)

    # Price below 5d rolling low (breakdown confirmed, not just intraday dip)
    close = g["close"]
    low5  = _rmin(close, 5).shift(1)
    below_support = (close < low5).astype(float).fillna(0)

    # Magnitude: larger negative return → stronger signal
    mag = (-g["ret_1d"].fillna(0)).clip(0, 0.08) / 0.08

    # Penalise chaotic shock: very wide range but small net move
    if "high" in g.columns and "low" in g.columns:
        daily_range = (g["high"].fillna(g["close"]) - g["low"].fillna(g["close"])) / g["close"]
        chaos_flag  = ((daily_range / g["atr_pct"].replace(0, np.nan).fillna(0.01)) > 2.5).astype(float)
    else:
        chaos_flag = pd.Series(0.0, index=g.index)

    raw = (
        0.30 * shock_neg
        + 0.25 * close_weak
        + 0.20 * vol_exp
        + 0.15 * below_support
        + 0.10 * mag
        - 0.20 * chaos_flag
    ).clip(0.0, 1.0)

    # Entry gate: must be genuine SHOCK continuation day
    gate = (
        (g["portfolio_regime"].fillna("CHOP") == "SHOCK") &
        (g["ret_1d"].fillna(0) < -0.005) &     # at least -0.5% return
        (close_weak > 0.40) &                   # close in lower 60% of range
        (raw > 0.35)
    )

    confirm = getattr(config, "SHOCK_SHORT_REQUIRE_DOWNSIDE_CONFIRMATION", True)
    if confirm:
        gate = gate & (below_support > 0)

    return gate.astype(float), raw


# ════════════════════════════════════════════════════════════════════════════════
# SETUP B — POST-SHOCK FAILED BOUNCE SHORT
# ════════════════════════════════════════════════════════════════════════════════

def _compute_post_shock_bounce(g: pd.DataFrame) -> tuple:
    """
    Enter short 1-5 days after a SHOCK when the bounce fails.

    Signal fires when:
    - SHOCK occurred 1 to 5 days ago
    - Price has bounced (ret_5d or ret_1d positive recently)
    - Bounce is failing: today's return is negative
    - Price remains below the pre-shock breakdown level
    - Volatility still elevated but not exploding
    - NOT a brand new SHOCK day (wait for the bounce first)
    """
    days_since = _days_since_shock(g)
    vol_exp    = _vol_expansion(g)

    close = g["close"]
    ret1  = g["ret_1d"].fillna(0)
    ret5  = g["ret_5d"].fillna(0)

    # Days since shock 1-5: post-shock window
    in_post_shock_window = ((days_since >= 1) & (days_since <= 5)).astype(float)

    # Bounce happened: 3d or 5d ret positive at some point
    bounce_occurred = (ret5.shift(1) > 0.003).astype(float).fillna(0)

    # Bounce failing: today negative, breaking back down
    bounce_failing = (ret1 < -0.003).astype(float)

    # Price below breakdown level (pre-shock low)
    low10_pre = _rmin(close, 10).shift(5)   # low from 10 days ending 5 days ago
    below_breakdown = (close < low10_pre.fillna(close)).astype(float).fillna(0)

    # Vol still elevated (post-shock vol environment)
    vol_elevated = (g["vol_regime"].fillna("NORMAL").isin(["ELEVATED", "HIGH"])).astype(float)

    # Not in a strong recovery (would invalidate the short)
    not_recovering = (g["slope_20d"].fillna(0) <= 0.002).astype(float)

    raw = (
        0.25 * in_post_shock_window
        + 0.20 * bounce_occurred
        + 0.25 * bounce_failing
        + 0.15 * below_breakdown
        + 0.10 * vol_elevated
        + 0.05 * not_recovering
    ).clip(0.0, 1.0)

    gate = (
        (days_since >= 1) & (days_since <= 5) &
        (bounce_occurred > 0) &
        (bounce_failing > 0) &
        (g["portfolio_regime"].fillna("CHOP").isin(["SHOCK", "TRANSITION", "CHOP"])) &
        (raw > 0.40)
    )

    return gate.astype(float), raw


# ════════════════════════════════════════════════════════════════════════════════
# SETUP C — SHOCK BREAKDOWN AFTER COMPRESSION
# ════════════════════════════════════════════════════════════════════════════════

def _compute_shock_breakdown(g: pd.DataFrame) -> tuple:
    """
    Compression → SHOCK entry as range breaks down.

    Signal fires when:
    - Recent rolling range was compressed (ATR percentile low)
    - Price breaks below compressed range low
    - Regime enters SHOCK or TRANSITION
    - Vol expands directionally after compression
    - Price stays below range (not immediate reversal)
    """
    close = g["close"]
    atr   = g["atr_14"].replace(0, np.nan).fillna(close * 0.01)

    # Rolling 10-bar range in ATR units
    hi10 = _rmax(g["high"].fillna(close), 10)
    lo10 = _rmin(g["low"].fillna(close),  10)
    range_atr = (hi10 - lo10) / atr

    # Compression: range below its 40th percentile (1-year window)
    compressed = (
        range_atr < range_atr.rolling(252, min_periods=63).quantile(0.40)
    ).astype(float).fillna(0)

    # Price breaks below compressed range low (1-bar lookback)
    breakdown = (close < lo10.shift(1)).astype(float).fillna(0)

    # Regime: SHOCK or TRANSITION at/after breakdown
    in_stress_regime = g["portfolio_regime"].fillna("CHOP").isin(
        ["SHOCK", "TRANSITION"]
    ).astype(float)

    # Vol expansion after compression
    vol_exp = _vol_expansion(g)
    # Vol expanding FROM a low level: vol_ratio recently crossed from below median
    vol_ratio = g["vol_ratio"].fillna(1.0)
    vol_was_low  = (vol_ratio.shift(5).fillna(1.0) < 1.0).astype(float)
    vol_now_high = (vol_ratio > 1.0).astype(float)
    vol_breakout = vol_was_low * vol_now_high

    # Not immediately reversed back into range
    not_reversed = (close < lo10.shift(2).fillna(close + 1)).astype(float).fillna(0)

    raw = (
        0.20 * compressed.shift(3).fillna(0)   # was compressed BEFORE breakdown
        + 0.25 * breakdown
        + 0.25 * in_stress_regime
        + 0.15 * vol_exp
        + 0.10 * vol_breakout
        + 0.05 * not_reversed
    ).clip(0.0, 1.0)

    gate = (
        (compressed.shift(3).fillna(0) > 0) &
        (breakdown > 0) &
        (in_stress_regime > 0) &
        (vol_breakout > 0) &
        (raw > 0.40)
    )

    return gate.astype(float), raw


# ════════════════════════════════════════════════════════════════════════════════
# SHOCK CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════════

def _classify_shock(g: pd.DataFrame) -> pd.Series:
    """
    Classify each SHOCK day:
    DOWNSIDE_CONTINUATION: negative, weak close, vol expanding, below support
    REVERSAL_SHOCK: large negative but strong close, tends to bounce
    CHAOTIC_SHOCK: high vol, mixed direction
    LOW_OPPORTUNITY_SHOCK: small magnitude, not clearly directional
    NON_SHOCK: not a SHOCK regime day
    """
    is_shock    = (g["portfolio_regime"].fillna("CHOP") == "SHOCK")
    ret         = g["ret_1d"].fillna(0)
    close_weak  = _close_weakness(g)
    vol_ratio   = g["vol_ratio"].fillna(1.0)

    if "high" in g.columns and "low" in g.columns:
        rng = (g["high"].fillna(g["close"]) - g["low"].fillna(g["close"])) / g["close"]
    else:
        rng = g["atr_pct"].fillna(0.01)

    labels = []
    for i in range(len(g)):
        if not is_shock.iloc[i]:
            labels.append("NON_SHOCK")
            continue
        r  = ret.iloc[i]
        cw = close_weak.iloc[i]
        vr = vol_ratio.iloc[i]
        rg = rng.iloc[i] if "high" in g.columns else g["atr_pct"].iloc[i]

        if r < -0.005 and cw > 0.55 and vr > 1.2:
            labels.append("DOWNSIDE_CONTINUATION")
        elif r < -0.010 and cw < 0.35:
            # Large negative move but closed strong → reversal candidate
            labels.append("REVERSAL_SHOCK")
        elif abs(r) > 0.005 and rg > 0.03:
            # Wide range, high vol, mixed
            labels.append("CHAOTIC_SHOCK")
        else:
            labels.append("LOW_OPPORTUNITY_SHOCK")

    return pd.Series(labels, index=g.index)


# ════════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE + BUCKET
# ════════════════════════════════════════════════════════════════════════════════

def _composite_score(
    cont_sig: pd.Series, cont_raw: pd.Series,
    bounce_sig: pd.Series, bounce_raw: pd.Series,
    breakdown_sig: pd.Series, breakdown_raw: pd.Series,
) -> pd.Series:
    """Best active score, with multi-setup confirmation bonus."""
    scores = pd.DataFrame({
        "cont":      cont_sig      * cont_raw,
        "bounce":    bounce_sig    * bounce_raw,
        "breakdown": breakdown_sig * breakdown_raw,
    })
    best   = scores.max(axis=1)
    n_act  = (cont_sig > 0).astype(int) + (bounce_sig > 0).astype(int) + (breakdown_sig > 0).astype(int)
    bonus  = (n_act - 1).clip(0) * 0.05
    return (best + bonus).clip(0.0, 1.0)


def _score_to_bucket(score: pd.Series) -> pd.Series:
    def _b(s):
        if pd.isna(s) or s < 0.35: return "NONE"
        elif s < 0.50: return "WEAK"
        elif s < 0.65: return "MODERATE"
        else: return "STRONG"
    return score.apply(_b)


def _type_label(cont_sig, bounce_sig, breakdown_sig, idx) -> pd.Series:
    labels = []
    for i in idx:
        parts = []
        if cont_sig.loc[i] > 0:      parts.append("CONT")
        if bounce_sig.loc[i] > 0:    parts.append("BOUNCE")
        if breakdown_sig.loc[i] > 0: parts.append("BREAKDOWN")
        labels.append("+".join(parts) if parts else "NONE")
    return pd.Series(labels, index=idx)


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY WEIGHT
# ════════════════════════════════════════════════════════════════════════════════

def _shock_entry_weight(bucket: str, size: float) -> float:
    """Negative weight for a short position. Size in fraction of capital."""
    frac = {"STRONG": 1.0, "MODERATE": 0.75, "WEAK": 0.0, "NONE": 0.0}
    max_size = getattr(config, "SHOCK_SHORT_MAX_SIZE", 0.15)
    w = -min(size * frac.get(bucket, 0.0), max_size)
    return w


# ════════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════

def _compute_symbol_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values("date").reset_index(drop=True)

    cont_sig,      cont_raw      = _compute_shock_continuation(g)
    bounce_sig,    bounce_raw    = _compute_post_shock_bounce(g)
    breakdown_sig, breakdown_raw = _compute_shock_breakdown(g)
    shock_class    = _classify_shock(g)

    score  = _composite_score(
        cont_sig, cont_raw, bounce_sig, bounce_raw, breakdown_sig, breakdown_raw
    )
    bucket = _score_to_bucket(score)
    any_sig = (
        ((cont_sig > 0) | (bounce_sig > 0) | (breakdown_sig > 0)) &
        (bucket.isin(["STRONG", "MODERATE"]))
    )
    type_s = _type_label(cont_sig, bounce_sig, breakdown_sig, g.index)

    # Reasons
    reasons = []
    for i in range(len(g)):
        parts = []
        if cont_raw.iloc[i] > 0.35:      parts.append(f"cont={cont_raw.iloc[i]:.2f}")
        if bounce_raw.iloc[i] > 0.35:    parts.append(f"bounce={bounce_raw.iloc[i]:.2f}")
        if breakdown_raw.iloc[i] > 0.35: parts.append(f"bkd={breakdown_raw.iloc[i]:.2f}")
        sc = shock_class.iloc[i]
        if sc != "NON_SHOCK": parts.append(sc)
        reasons.append("|".join(parts) or "no_signal")

    size = getattr(config, "SHOCK_SHORT_SIZE", 0.10)

    g["ss_cont_signal"]       = cont_sig.astype(bool)
    g["ss_cont_score"]        = cont_raw.round(4)
    g["ss_bounce_signal"]     = bounce_sig.astype(bool)
    g["ss_bounce_score"]      = bounce_raw.round(4)
    g["ss_breakdown_signal"]  = breakdown_sig.astype(bool)
    g["ss_breakdown_score"]   = breakdown_raw.round(4)
    g["shock_class"]          = shock_class
    g["shock_short_signal"]   = any_sig
    g["shock_short_type"]     = type_s
    g["shock_short_score"]    = score.round(4)
    g["shock_short_bucket"]   = bucket
    g["shock_short_reason"]   = reasons
    g["shock_short_entry_weight"] = bucket.map(lambda b: _shock_entry_weight(b, size))
    g["shock_short_stop_type"]    = getattr(config, "SHOCK_SHORT_USE_ATR_STOP", True) and "ATR" or "FIXED"
    g["shock_short_max_hold"]     = getattr(config, "SHOCK_SHORT_MAX_HOLD", 3)

    return g


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def compute_shock_short_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SHOCK short signal columns to df.
    Processed per-symbol independently (no cross-symbol data).
    """
    symbols_allowed = getattr(config, "SHOCK_SHORT_SYMBOLS", ["ES", "NQ"])
    df = df.copy()
    parts = []

    for sym, g in df.groupby("symbol", sort=True):
        if sym in symbols_allowed:
            g = _compute_symbol_features(g)
        else:
            # Add neutral columns
            for col in ["ss_cont_signal", "ss_bounce_signal", "ss_breakdown_signal",
                        "shock_short_signal", "shock_class"]:
                g[col] = False
            g["shock_short_type"]   = "NONE"
            g["shock_short_score"]  = 0.0
            g["shock_short_bucket"] = "NONE"
            g["shock_short_reason"] = "symbol_excluded"
        parts.append(g)

    result = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"])

    # Log distribution
    for sym in result["symbol"].unique():
        g   = result[result["symbol"] == sym]
        n   = len(g)
        sig = g["shock_short_signal"].sum()
        vc  = g["shock_short_bucket"].value_counts()
        sc  = g[g["portfolio_regime"] == "SHOCK"]["shock_class"].value_counts()
        logger.info(
            "ShockShort %s — signals=%d (%.1f%%) STRONG=%d MOD=%d | "
            "SHOCK classes: CONT=%d REV=%d CHAOTIC=%d LOW=%d",
            sym, sig, 100 * sig / max(n, 1),
            vc.get("STRONG", 0), vc.get("MODERATE", 0),
            sc.get("DOWNSIDE_CONTINUATION", 0), sc.get("REVERSAL_SHOCK", 0),
            sc.get("CHAOTIC_SHOCK", 0), sc.get("LOW_OPPORTUNITY_SHOCK", 0),
        )
    return result
