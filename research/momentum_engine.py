"""
momentum_engine.py — Multi-timeframe time-series momentum engine. [NEW MODULE]

Replaces raw weekly bias as the primary directional signal.

Why multi-timeframe momentum instead of raw weekly bias?
  The previous system used rolling 10d/20d/40d returns on daily data to proxy
  "weekly bias". This module formalises that into a properly normalised
  composite signal that:
    1. Is comparable across different volatility regimes (t-stat normalisation)
    2. Weights the 20-day horizon most heavily (most relevant for 5–7d windows)
    3. Uses 60d for structural context (prevents fighting long-term trend)
    4. Avoids overfitting: equal-round-number weights, no optimised parameters

Signal construction:
  mom_Nd = (ret_Nd) / (vol_20d × sqrt(N))  ← normalised like a t-statistic
  score  = 0.25 × clip(mom_5) + 0.50 × clip(mom_20) + 0.25 × clip(mom_60)
  output = clip(score, -1, +1)

  Clipping at ±3 before weighting prevents outlier-driven signals.
  Division by 3 maps the result to [-1, +1].

LONG/NEUTRAL only:
  score > MOM_LONG_THRESH  → LONG
  otherwise                → NEUTRAL
  No SHORT signal — backtest confirmed equity index shorts structurally fail.

Output per (date, symbol):
  momentum_score      float in [-1, +1]
  momentum_direction  LONG / NEUTRAL
  momentum_strength   float in [0, 1]  (abs of score)
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

_W  = config.MOM_WINDOWS   # [5, 20, 60]
_WT = config.MOM_WEIGHTS   # [0.25, 0.50, 0.25]


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.copy().sort_values("date").reset_index(drop=True)
        g = _symbol_momentum(g)
        parts.append(g)

    df = pd.concat(parts, ignore_index=True)
    _log_distribution(df)
    return df


def _symbol_momentum(g: pd.DataFrame) -> pd.DataFrame:
    vol = g["vol_20d"].replace(0, np.nan)

    components = []
    for w, weight in zip(_W, _WT):
        ret_col = f"ret_{w}d"

        if ret_col in g.columns:
            ret = g[ret_col]
        else:
            ret = g["close"].pct_change(w)

        # Normalise: return / (daily_vol × sqrt(holding_days)) ≈ t-statistic
        ann_factor = np.sqrt(w)
        raw        = ret / (vol * ann_factor)
        clipped    = raw.clip(-3, 3) / 3          # → roughly [-1, +1]
        components.append(clipped * weight)

    score = sum(components).clip(-1.0, 1.0)

    g["momentum_score"]    = score
    g["momentum_strength"] = score.abs()
    g["momentum_direction"] = np.where(
        score > config.MOM_LONG_THRESH, "LONG", "NEUTRAL"
    )
    return g


def _log_distribution(df: pd.DataFrame) -> None:
    for sym in df["symbol"].unique():
        g  = df[df["symbol"] == sym]
        vc = g["momentum_direction"].value_counts()
        logger.info(
            "Momentum %s — LONG: %d (%.0f%%)  NEUTRAL: %d (%.0f%%)  "
            "avg_strength=%.3f",
            sym,
            vc.get("LONG",    0), 100 * vc.get("LONG",    0) / max(len(g), 1),
            vc.get("NEUTRAL", 0), 100 * vc.get("NEUTRAL", 0) / max(len(g), 1),
            g["momentum_strength"].mean(),
        )
