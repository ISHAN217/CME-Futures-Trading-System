"""
position_sizing.py — V3 entry weight computation. [NEW]

Replaces portfolio.py as the per-entry sizing engine.
Portfolio-level gross exposure cap applied separately in backtest.py.

Formula:
  weight = strategy_base × vol_adj × quality_mult × regime_mult
           × conviction_mult × sentiment_mult

Compared to V2 (portfolio.py):
  V2: base × vol_scalar × qual_scale × sentiment
  V3: same, but quality_mult (1-5 → 0.60-1.40) replaces a single
      qual_scale, and regime_mult and conviction_mult are factored in
      explicitly. The net result is ≈+30% higher average size with
      selective upscaling of quality-4/5 entries.

Strategy bases:
  TREND:       VOL_SCALE_MAX (0.30) × R²-quality scale
  PULLBACK:    TREND_PULLBACK_WEIGHT (0.15) × mr_strength scale
  MEAN_REVERT: MEAN_REVERT_WEIGHT (0.18) × mr_strength scale
  STAT_ARB:    STAT_ARB_NQ_WEIGHT (0.20) × sa_strength scale

Quality multiplier (entry_quality 1–5):
  1 → 0.60   (poor  — entered only when no other option)
  2 → 0.80   (below average)
  3 → 1.00   (average)
  4 → 1.20   (good)
  5 → 1.40   (excellent — LONG bias + strong 4H)

Regime multiplier:
  TREND:       1.00
  CHOP:        0.80
  TRANSITION:  0.50
  SHOCK:       0.00

Conviction multiplier:
  high_conviction == True  → HIGH_CONV_MULTIPLIER (1.20)

Sentiment multiplier: from SENTIMENT_MULTIPLIERS config dict.
"""

import numpy as np

import config

# Quality 1–5 → weight multiplier
_QUALITY_MULT = {1: 0.60, 2: 0.80, 3: 1.00, 4: 1.20, 5: 1.40}

# Regime → weight multiplier
_REGIME_MULT = {"TREND": 1.00, "CHOP": 0.80, "TRANSITION": 0.50, "SHOCK": 0.00}


def compute_entry_weight(row: dict, strategy: str) -> float:
    """
    Compute signed entry weight for a new position.

    Parameters
    ----------
    row      : dict — today's row (all features + signal_engine output)
    strategy : str  — TREND / PULLBACK / MEAN_REVERT / STAT_ARB

    Returns
    -------
    Signed float in [-MAX_WEIGHT_PER_SYMBOL, +MAX_WEIGHT_PER_SYMBOL].
    Returns 0.0 if direction is FLAT or strategy is unrecognised.
    """
    direction = row.get("final_direction", "FLAT")
    if direction == "FLAT":
        return 0.0

    sign = 1.0 if direction == "LONG" else -1.0

    # ── Strategy base ─────────────────────────────────────────────────────────
    if strategy == "TREND":
        # Scale base by R² quality of the trend (better trend → more size)
        r2         = float(row.get("r2_20d", 0.0) or 0.0)
        qual_scale = r2 / max(config.TREND_R2_MIN, 1e-6)
        base       = config.VOL_SCALE_MAX * min(qual_scale, 1.5)

    elif strategy == "PULLBACK":
        mr_str = float(row.get("mr_strength", 1.0) or 1.0)
        base   = config.TREND_PULLBACK_WEIGHT * min(mr_str, 1.5)

    elif strategy == "MEAN_REVERT":
        mr_str = float(row.get("mr_strength", 1.0) or 1.0)
        base   = config.MEAN_REVERT_WEIGHT * min(mr_str, 1.5)

    elif strategy == "STAT_ARB":
        sa_str = float(row.get("sa_strength", 1.0) or 1.0)
        base   = config.STAT_ARB_NQ_WEIGHT * min(sa_str, 1.5)

    elif strategy in ("MTF_RESET", "MTF_BREAKOUT"):
        # MTF-injected signals: use the per-row target size set by apply_mtf_modifications().
        # The vol/quality/regime multipliers below will still scale this base weight,
        # so the final weight is risk-adjusted just like any native strategy.
        base = float(row.get("mtf_entry_size",
                              getattr(config, "MTF_MAX_NEW_TRADE_SIZE", 0.10)))

    elif strategy == "BULL_RESET_CONFIRMED":
        # Pullback-reset long: uses the per-row target size (regime already gated upstream)
        base = float(row.get("mtf_entry_size", 0.08))

    elif strategy in ("BEAR_TREND", "FAILED_RALLY", "SHOCK_CONT", "BEAR_RESET"):
        # Regime-confirmed short sleeve: uses per-row target size.
        # Regime multiplier is bypassed below for shorts (SHOCK=0 was designed to
        # block longs; short entries in SHOCK should be allowed at reduced size).
        base = float(row.get("mtf_entry_size", 0.07))

    else:
        # Generic fallback: any experimental sleeve that sets mtf_entry_size explicitly.
        # This avoids having to enumerate every new strategy name here.
        mtf_size = float(row.get("mtf_entry_size", 0.0) or 0.0)
        if mtf_size < 1e-6:
            return 0.0
        base = mtf_size

    # ── Vol adjustment (pre-computed vol_scalar from volatility_engine) ───────
    vol_scalar = float(row.get("vol_scalar", 1.0) or 1.0)
    vol_adj    = max(vol_scalar, config.VOL_SCALE_MIN)  # floor at min

    # ── Quality multiplier ────────────────────────────────────────────────────
    quality = int(row.get("entry_quality", 3) or 3)
    quality = max(1, min(5, quality))
    q_mult  = _QUALITY_MULT.get(quality, 1.00)

    # ── Regime multiplier ─────────────────────────────────────────────────────
    regime = row.get("portfolio_regime", "CHOP") or "CHOP"
    # Short strategies use a separate regime table: SHOCK is 0.75 (reduced, not blocked)
    # because SHOCK=0.00 in _REGIME_MULT was designed to block longs only.
    if strategy in ("BEAR_TREND", "FAILED_RALLY", "SHOCK_CONT", "BEAR_RESET"):
        _SHORT_REGIME_MULT = {"TREND": 1.00, "TRANSITION": 0.75, "CHOP": 0.50, "SHOCK": 0.75}
        r_mult = _SHORT_REGIME_MULT.get(regime, 0.50)
    else:
        r_mult = _REGIME_MULT.get(regime, 0.80)

    # ── Conviction multiplier ─────────────────────────────────────────────────
    high_conv = bool(row.get("high_conviction", False))
    c_mult    = config.HIGH_CONV_MULTIPLIER if high_conv else 1.0

    # ── Sentiment multiplier ──────────────────────────────────────────────────
    flag   = row.get("portfolio_sentiment_flag", "NORMAL") or "NORMAL"
    s_mult = config.SENTIMENT_MULTIPLIERS.get(flag, 1.0)

    # ── Final weight (clipped to per-symbol cap) ──────────────────────────────
    weight = sign * base * vol_adj * q_mult * r_mult * c_mult * s_mult

    # ── CWT size multiplier [NEW] ────────────────────────────────────────────
    if config.USE_CWT_CHOP_FILTER:
        apply_cwt = (
            (strategy == "MEAN_REVERT" and config.CWT_APPLY_TO_MEAN_REVERT) or
            (strategy == "PULLBACK"    and config.CWT_APPLY_TO_PULLBACK)    or
            (strategy == "STAT_ARB"    and config.CWT_APPLY_TO_STAT_ARB)
        )
        if apply_cwt:
            cwt_mult = float(row.get("cwt_size_multiplier", 1.0) or 1.0)
            cwt_mult = max(0.0, min(1.0, cwt_mult))
            weight   = weight * cwt_mult

    # ── TREND continuation-edge sizing [NEW] ─────────────────────────────────
    if strategy == "TREND":
        if getattr(config, "USE_TREND_EDGE_MULTIPLIER", False):
            bucket = str(row.get("trend_continuation_edge_bucket", "MODERATE") or "MODERATE")
            mults  = getattr(config, "TREND_EDGE_MULTIPLIERS", {})
            mult   = float(mults.get(bucket, 1.0))
            weight = weight * mult

        elif getattr(config, "USE_TREND_EDGE_DIRECT_WEIGHT", False):
            bucket = str(row.get("trend_continuation_edge_bucket", "MODERATE") or "MODERATE")
            sched  = getattr(config, "TREND_EDGE_DIRECT_WEIGHT_SCHEDULE", {})
            target = float(sched.get(bucket, 0.15))
            # Direct weight overrides vol-scaled base; sign preserved
            weight = float(np.sign(weight)) * target if target > 0.0 else 0.0

    # ── Signal quality sizing multiplier [NEW] ────────────────────────────────
    if getattr(config, "USE_SIGNAL_QUALITY_SIZING", False):
        from signal_quality_engine import get_quality_multiplier
        mode = getattr(config, "QUALITY_SIZING_MODE", "multiplier")

        if mode == "multiplier":
            sq_mult = get_quality_multiplier(row, strategy)
            # Cap multiplier so position never exceeds the per-symbol hard cap
            weight  = weight * sq_mult

        elif mode == "direct_weight":
            # Direct weight mode: quality bucket assigns weight directly.
            # We still apply sign and cap at MAX_WEIGHT_PER_SYMBOL.
            # The vol-scaling from the base formula is replaced entirely.
            from signal_quality_engine import get_quality_multiplier as _gqm
            # Re-use multiplier dict but interpret values as target weights
            sq_target = get_quality_multiplier(row, strategy)
            if sq_target > 0:
                # Override base weight; sign already in `weight`
                weight = float(np.sign(weight)) * sq_target

    return float(np.clip(weight, -config.MAX_WEIGHT_PER_SYMBOL, config.MAX_WEIGHT_PER_SYMBOL))
