"""
portfolio.py — Vol-scaled dynamic position sizing. [CHANGED]

Core change from base system: vol_scalar from volatility_engine replaces
fixed-weight sizing. Every strategy now scales by actual market volatility.

WHY inverse-vol sizing?
  Fixed weights (0.25, 0.20, 0.15) ignore whether vol is 0.5% or 2%.
  In high-vol regimes, those weights take on 4× the dollar risk.
  vol_scalar = TARGET_VOL_DAILY / ewma_vol keeps daily P&L vol ≈ TARGET.
  This doesn't predict direction — it just ensures the bet size is
  appropriate for current conditions.

Sizing rules by strategy:

TREND:
  base   = VOL_SCALE_MAX * vol_scalar
           (vol_scalar already capped at 1.0 in volatility_engine)
  scale  = r2_20d / TREND_R2_MIN (reward high-quality trends with more size)
  raw    = sign * base * min(scale, 1.5) * sentiment

PULLBACK:
  base   = TREND_PULLBACK_WEIGHT * vol_scalar
  scale  = mr_strength (how far the dip went → reversion potential)
  raw    = sign * base * min(scale, 1.5) * sentiment

MEAN_REVERT (CHOP):
  base   = MEAN_REVERT_WEIGHT * vol_scalar
  scale  = mr_strength
  raw    = sign * base * min(scale, 1.5) * sentiment

STAT_ARB (NQ-only):
  base   = STAT_ARB_NQ_WEIGHT * vol_scalar
  raw    = sign * base * sa_strength * sentiment
  (SA_NQ_WEIGHT is slightly higher than MEAN_REVERT since only one leg)

High-conviction upscaling [NEW]:
  When high_conviction == True:
    weight *= HIGH_CONV_MULTIPLIER (1.20)
  High conviction = momentum_strength > 0.65 AND h4_quality >= 4.
  This is a +20% boost on the top 15-20% of entries. Max cap still applies.

TRANSITION regime:
  in_transition == True → weight *= TRANSITION_SIZE_MULT (0.50)
  Half size in transition zones, no new entries (enforced in signal_engine).

Portfolio constraints:
  Each symbol ≤ MAX_WEIGHT_PER_SYMBOL (0.30)
  ES + NQ gross combined ≤ MAX_COMBINED_EXPOSURE (0.50)

Output:
  raw_weight    pre-cap weight
  weight        final weight (signed: + long, - short)
  weight_reason diagnostic tag
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def compute_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    raw = df.apply(_compute_raw_weight, axis=1, result_type="expand")
    raw.columns = ["raw_weight", "weight_reason"]
    df = pd.concat([df, raw], axis=1)

    # Per-symbol cap
    df["capped_weight"] = df["raw_weight"].clip(
        -config.MAX_WEIGHT_PER_SYMBOL,
         config.MAX_WEIGHT_PER_SYMBOL,
    )

    # Portfolio-level gross exposure cap (date by date)
    df["weight"] = df["capped_weight"]
    for date, grp in df.groupby("date"):
        gross = grp["capped_weight"].abs().sum()
        if gross > config.MAX_COMBINED_EXPOSURE:
            scale = config.MAX_COMBINED_EXPOSURE / gross
            df.loc[grp.index, "weight"] = grp["capped_weight"] * scale
            df.loc[grp.index, "weight_reason"] = grp["weight_reason"].apply(
                lambda r: r if r in ("FLAT", "MISSING_VOL") else r + "_SCALED"
            )

    _log_weight_distribution(df)
    return df


def _compute_raw_weight(row: pd.Series) -> tuple:
    direction   = row.get("final_direction", "FLAT")
    strategy    = row.get("strategy_used",  "NONE")
    sentiment   = row.get("portfolio_sentiment_multiplier", 1.0) or 1.0
    high_conv   = bool(row.get("high_conviction",  False))
    in_trans    = bool(row.get("in_transition",    False))
    # entry_quality from signal_engine (distinct from h4_quality placeholder)
    _eq         = row.get("entry_quality", 0) or 0

    if direction == "FLAT":
        return 0.0, "FLAT"

    sign = 1.0 if direction == "LONG" else -1.0

    # Vol scalar (already range-capped to [VOL_SCALE_MIN/MAX, 1.0])
    vol_scalar = float(row.get("vol_scalar", 1.0) or 1.0)

    if strategy == "TREND":
        r2   = float(row.get("r2_20d", 0.0) or 0.0)
        base = config.VOL_SCALE_MAX * vol_scalar
        qual_scale = r2 / max(config.TREND_R2_MIN, 1e-6)
        raw = sign * base * min(qual_scale, 1.5) * sentiment

    elif strategy == "PULLBACK":
        mr_str = float(row.get("mr_strength", 1.0) or 1.0)
        base   = config.TREND_PULLBACK_WEIGHT * vol_scalar
        raw    = sign * base * min(mr_str, 1.5) * sentiment

    elif strategy == "MEAN_REVERT":
        mr_str = float(row.get("mr_strength", 1.0) or 1.0)
        base   = config.MEAN_REVERT_WEIGHT * vol_scalar
        raw    = sign * base * min(mr_str, 1.5) * sentiment

    elif strategy == "STAT_ARB":
        sa_str = float(row.get("sa_strength", 1.0) or 1.0)
        base   = config.STAT_ARB_NQ_WEIGHT * vol_scalar
        raw    = sign * base * min(sa_str, 1.5) * sentiment

    else:
        return 0.0, "UNKNOWN_STRATEGY"

    # High-conviction upscaling (+20%)
    if high_conv:
        raw  *= config.HIGH_CONV_MULTIPLIER
        label = f"{strategy}_ACTIVE_HIGHCONV"
    else:
        label = f"{strategy}_ACTIVE"

    # TRANSITION: half size
    if in_trans:
        raw  *= config.TRANSITION_SIZE_MULT
        label += "_TRANSITION"

    raw = float(np.clip(raw, -config.MAX_WEIGHT_PER_SYMBOL, config.MAX_WEIGHT_PER_SYMBOL))
    return raw, label


def _log_weight_distribution(df: pd.DataFrame) -> None:
    active = df[df["weight"].abs() > 1e-6]
    logger.info(
        "Portfolio weights — %d active symbol-days (%.1f%% of all)",
        len(active), 100 * len(active) / max(len(df), 1),
    )
    for sym, g in df.groupby("symbol"):
        act = g[g["weight"].abs() > 1e-6]
        if len(act) > 0:
            hc = act["high_conviction"].sum() if "high_conviction" in act.columns else 0
            logger.info(
                "  %s: %d active days  avg_|w|=%.3f  max_|w|=%.3f  "
                "high_conv_days=%d  avg_vol_scalar=%.3f",
                sym, len(act),
                act["weight"].abs().mean(),
                act["weight"].abs().max(),
                int(hc),
                act["vol_scalar"].mean() if "vol_scalar" in act.columns else np.nan,
            )
