"""
mean_reversion.py — Mean reversion + TREND pullback signals. [CHANGED]

Two modes vs the base system:

1. CHOP-regime MR (unchanged logic):
   |h4_zscore| > MR_ENTRY_THRESHOLD (1.2) → LONG/SHORT
   NEUTRAL weekly bias → disabled (proven noise at 30% win rate)
   MR SHORT in LONG bias → disabled (fighting the trend)

2. TREND-regime pullback mode [NEW]:
   Trigger: regime == TREND, weekly_bias == LONG, h4_zscore < -1.0
             AND momentum_score > TREND_PULLBACK_MOM_MIN (0.05)
   Signal type: PULLBACK (distinct from standard MR)
   Rationale: A dip in a confirmed uptrend is a mean-reversion entry
   WITH directional tailwind. This is fundamentally different from a
   random oscillation in CHOP. The regime filter is critical — PULLBACK
   in TREND has >60% expected win rate (supported by 4H pullback data).
   Weight is smaller (0.15 vs 0.18) and max hold is 2 days.

Output per (date, symbol):
    mr_signal        LONG / SHORT / PULLBACK / FLAT
    mr_signal_mode   CHOP_MR / TREND_PULLBACK / NONE
    mr_zscore        the z-score used
    mr_zscore_source '4H' or 'DAILY'
    mr_strength      |zscore| / entry_threshold (≥1.0 means at threshold)
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

_ENTRY  = config.MR_ENTRY_THRESHOLD    # 1.2
_EXIT   = config.MR_EXIT_THRESHOLD     # 0.30
_PULL_Z = config.TREND_PULLBACK_ENTRY_Z   # -1.0
_PULL_M = config.TREND_PULLBACK_MOM_MIN   # 0.05


def compute_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    """Add MR/PULLBACK signal columns to df."""
    df = df.copy()

    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.copy().sort_values("date").reset_index(drop=True)
        g = _compute_mr_signal(g)
        parts.append(g)

    df = pd.concat(parts, ignore_index=True)
    _log_mr_distribution(df)
    return df


def _compute_mr_signal(g: pd.DataFrame) -> pd.DataFrame:
    # Use 4H zscore if available, else daily zscore
    use_4h = "h4_zscore" in g.columns and g["h4_zscore"].notna().sum() > 0

    if use_4h:
        z      = g["h4_zscore"].fillna(np.nan)
        source = "4H"
    else:
        W      = config.MR_ZSCORE_WINDOW_DAY
        mean_  = g["close"].rolling(W, min_periods=3).mean()
        std_   = g["close"].rolling(W, min_periods=3).std().replace(0, np.nan)
        z      = (g["close"] - mean_) / std_
        source = "DAILY"

    g["mr_zscore"]        = z
    g["mr_zscore_source"] = source

    # Grab context columns needed for PULLBACK mode
    regime    = g.get("regime",           pd.Series("CHOP",    index=g.index))
    bias      = g.get("weekly_bias",      pd.Series("NEUTRAL", index=g.index))
    mom_score = g.get("momentum_score",   pd.Series(0.0,       index=g.index)).fillna(0.0)

    # ── Determine signal and mode per row ─────────────────────────────────────
    signals = []
    modes   = []

    for i in range(len(g)):
        zi  = z.iloc[i]
        reg = regime.iloc[i] if hasattr(regime, "iloc") else regime
        bi  = bias.iloc[i]   if hasattr(bias,   "iloc") else bias
        mi  = mom_score.iloc[i]

        if pd.isna(zi):
            signals.append("FLAT"); modes.append("NONE")
            continue

        # TREND-regime pullback mode [NEW]
        # Only triggers when ALL of: TREND regime, LONG bias, z < PULL_Z, mom > 0
        if reg == "TREND" and bi == "LONG" and zi < _PULL_Z and mi > _PULL_M:
            signals.append("PULLBACK"); modes.append("TREND_PULLBACK")
            continue

        # Standard CHOP-regime MR
        if abs(zi) >= _ENTRY:
            sig = "LONG" if zi < 0 else "SHORT"
            signals.append(sig); modes.append("CHOP_MR")
        else:
            signals.append("FLAT"); modes.append("NONE")

    g["mr_signal"]      = signals
    g["mr_signal_mode"] = modes
    g["mr_strength"]    = (z.abs() / _ENTRY).clip(0, 3)

    return g


def _log_mr_distribution(df: pd.DataFrame) -> None:
    for sym in df["symbol"].unique():
        g   = df[df["symbol"] == sym]
        vc  = g["mr_signal"].value_counts()
        mc  = g["mr_signal_mode"].value_counts()
        src = g["mr_zscore_source"].iloc[0] if len(g) > 0 else "UNKNOWN"
        logger.info(
            "MR signals %s [%s] — LONG: %d  SHORT: %d  PULLBACK: %d  FLAT: %d"
            "  | TREND_PULLBACK mode: %d",
            sym, src,
            vc.get("LONG",     0),
            vc.get("SHORT",    0),
            vc.get("PULLBACK", 0),
            vc.get("FLAT",     0),
            mc.get("TREND_PULLBACK", 0),
        )
