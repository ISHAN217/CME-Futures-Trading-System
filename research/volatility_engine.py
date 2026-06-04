"""
volatility_engine.py — EWMA volatility and dynamic sizing support. [NEW MODULE]

Why EWMA instead of rolling std?
  Rolling std gives equal weight to all observations in the window.
  If there was a vol spike 15 days ago, it keeps inflating the estimate
  for 15 days even after vol has normalised. EWMA with λ=0.94 weights
  recent observations exponentially more, so vol estimates are faster
  to rise (risk control) and faster to fall (size recovery).

  Concretely: rolling 20d std has a "memory" of ~20 days regardless of
  what happens. EWMA(0.94) has an effective memory of 1/(1-0.94)≈17 days
  but puts 60% of the weight on the last 5 observations.

Features added:
  ewma_vol         already computed in features.py (EWMA std of returns)
  atr_14           already computed in features.py (ATR for stop reference)
  vol_regime       NORMAL / ELEVATED / HIGH based on vol_ratio
  vol_scalar       sizing multiplier: TARGET_VOL / ewma_vol (capped)
  atr_pct          ATR as % of close price (for stop distance normalisation)

The vol_scalar is the key input to portfolio.py for inverse-vol sizing.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add vol_regime, vol_scalar, atr_pct to df."""
    df = df.copy()

    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.copy().sort_values("date").reset_index(drop=True)
        g = _vol_features(g)
        parts.append(g)

    df = pd.concat(parts, ignore_index=True)
    _log_vol_stats(df)
    return df


def _vol_features(g: pd.DataFrame) -> pd.DataFrame:
    vol_ratio = g["vol_ratio"].fillna(1.0)
    ewma_vol  = g["ewma_vol"].fillna(g["vol_20d"])
    atr       = g["atr_14"].fillna(0.0)
    close     = g["close"].replace(0, np.nan)

    # Volatility regime classification
    g["vol_regime"] = np.where(
        vol_ratio > config.SHOCK_VOL_RATIO, "HIGH",
        np.where(vol_ratio > 1.3, "ELEVATED", "NORMAL")
    )

    # Vol scalar for inverse-vol position sizing
    # scalar = TARGET_VOL / ewma_vol → position × scalar keeps daily P&L vol ≈ TARGET
    # Capped: very low vol doesn't blow up position size beyond max
    safe_ewma = ewma_vol.replace(0, 1e-6)
    raw_scalar = config.TARGET_VOL_DAILY / safe_ewma
    g["vol_scalar"] = raw_scalar.clip(
        config.VOL_SCALE_MIN / config.VOL_SCALE_MAX,  # floor to min/max ratio
        1.0,                                            # cap at 1.0 (full base weight)
    )

    # ATR as % of close — normalised stop distance
    g["atr_pct"] = (atr / close).fillna(0.0)

    return g


def _log_vol_stats(df: pd.DataFrame) -> None:
    for sym in df["symbol"].unique():
        g = df[df["symbol"] == sym]
        vc = g["vol_regime"].value_counts()
        logger.info(
            "Vol regime %s — NORMAL: %d (%.0f%%)  ELEVATED: %d (%.0f%%)  "
            "HIGH: %d (%.0f%%)  avg_ewma_vol=%.4f  avg_scalar=%.3f",
            sym,
            vc.get("NORMAL",   0), 100 * vc.get("NORMAL",   0) / max(len(g), 1),
            vc.get("ELEVATED", 0), 100 * vc.get("ELEVATED", 0) / max(len(g), 1),
            vc.get("HIGH",     0), 100 * vc.get("HIGH",     0) / max(len(g), 1),
            g["ewma_vol"].mean(),
            g["vol_scalar"].mean(),
        )
