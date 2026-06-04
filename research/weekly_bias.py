"""
weekly_bias.py — Simplified directional bias layer.

[CHANGED from base system] LONG/NEUTRAL only. No SHORT bias.

WHY: Weekly SHORT bias in ES/NQ was correct only 35-40% of the time
in the base system backtest. The structural bull market in equity index
futures means that "short weekly" often leads to trading against the
long-term trend for no edge. The weekly SHORT direction is permanently
disabled. All strategy entries are either LONG or FLAT.

The weekly bias is computed from:
  1. Regression slope sign over 40d (structural direction)
  2. Momentum_direction (from momentum_engine, already computed)
  3. R² quality (is the trend structured?)

Output:
  weekly_bias        LONG / NEUTRAL (no SHORT)
  weekly_confidence  float in [0, 1]
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def compute_weekly_bias(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.copy().sort_values("date").reset_index(drop=True)
        g = _symbol_bias(g)
        parts.append(g)

    df = pd.concat(parts, ignore_index=True)
    _log_distribution(df)
    return df


def _symbol_bias(g: pd.DataFrame) -> pd.DataFrame:
    mom_dir   = g.get("momentum_direction", pd.Series("NEUTRAL", index=g.index))
    mom_score = g.get("momentum_score",    pd.Series(0.0,        index=g.index))
    r2        = g["r2_20d"].fillna(0.0)
    slope     = g["slope_20d"].fillna(0.0)

    # LONG bias when:
    #   momentum direction is LONG
    #   AND slope is positive (price trending up in 40-day OLS)
    long_condition = (
        (mom_dir == "LONG")
        & (slope > 0)
    )

    g["weekly_bias"] = np.where(long_condition, "LONG", "NEUTRAL")

    # Confidence: how strong and structured is the LONG signal?
    # High R² + high momentum strength → high confidence
    g["weekly_confidence"] = np.where(
        g["weekly_bias"] == "LONG",
        (r2.clip(0, 1) * 0.5 + mom_score.clip(0, 1) * 0.5).clip(0, 1),
        0.0,
    )
    return g


def _log_distribution(df: pd.DataFrame) -> None:
    for sym in df["symbol"].unique():
        g  = df[df["symbol"] == sym]
        vc = g["weekly_bias"].value_counts()
        logger.info(
            "Weekly bias %s — LONG: %d (%.0f%%)  NEUTRAL: %d (%.0f%%)",
            sym,
            vc.get("LONG",    0), 100 * vc.get("LONG",    0) / max(len(g), 1),
            vc.get("NEUTRAL", 0), 100 * vc.get("NEUTRAL", 0) / max(len(g), 1),
        )
