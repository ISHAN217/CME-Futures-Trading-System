"""
sentiment_layer.py — Shock/event risk proxy layer. (Same design as base system.)

This layer does NOT predict future direction. Its sole job is to
reduce or block new position entries when market conditions suggest
elevated risk of large, unpredictable moves.

Proxies used (all computable from price/volume data alone):
    1. Volatility spike:  vol_ratio > SENTIMENT_VOL_RATIO (2.0)
    2. Large daily move:  |ret_1d| > SENTIMENT_RET_SIGMA * vol_20d (2.5σ)
    3. Volume spike:      volume_z > SENTIMENT_VOL_Z (2.0)
    4. Overnight gap:     |gap_pct| > SENTIMENT_GAP_THRESH (1.5%)

Flag logic (per symbol):
    HIGH_RISK    if proxy 1 OR proxy 2 fires
    EVENT_ACTIVE if proxy 3 OR proxy 4 fires (but not HIGH_RISK)
    NORMAL       otherwise

Portfolio-level flag:
    Takes the WORSE of ES and NQ flags.
    If either symbol is HIGH_RISK, portfolio is HIGH_RISK.

Multipliers (from config.SENTIMENT_MULTIPLIERS):
    NORMAL       1.00
    EVENT_ACTIVE 0.50
    HIGH_RISK    0.00  (blocks all new entries)

Output per (date, symbol):
    sentiment_flag        NORMAL / EVENT_ACTIVE / HIGH_RISK
    sentiment_multiplier  float
    event_reason          human-readable

Portfolio output (merged to all symbols):
    portfolio_sentiment_flag
    portfolio_sentiment_multiplier

Extension hook:
    _get_api_sentiment(date) → plug in real news/sentiment API here.
    Returns one of the three flag strings, or None to use proxy flags.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

_RANK = {"NORMAL": 0, "EVENT_ACTIVE": 1, "HIGH_RISK": 2}
_RMAP = {0: "NORMAL", 1: "EVENT_ACTIVE", 2: "HIGH_RISK"}


def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.copy().sort_values("date").reset_index(drop=True)
        g = _symbol_sentiment(g)
        parts.append(g)

    df = pd.concat(parts, ignore_index=True)
    df = _portfolio_sentiment(df)
    _log_sentiment_distribution(df)
    return df


def _symbol_sentiment(g: pd.DataFrame) -> pd.DataFrame:
    ret       = g["ret_1d"].fillna(0)
    vol_20    = g["vol_20d"].fillna(g["vol_20d"].median()).replace(0, 1e-6)
    vol_ratio = g["vol_ratio"].fillna(1.0)
    vol_z     = g["volume_z"].fillna(0)
    gap       = g["gap_pct"].fillna(0).abs()

    high_risk = (
        (vol_ratio > config.SENTIMENT_VOL_RATIO)
        | (ret.abs() > config.SENTIMENT_RET_SIGMA * vol_20)
    )
    event_active = (
        (~high_risk)
        & (
            (vol_z > config.SENTIMENT_VOL_Z)
            | (gap > config.SENTIMENT_GAP_THRESH)
        )
    )

    flags = np.where(
        high_risk, "HIGH_RISK",
        np.where(event_active, "EVENT_ACTIVE", "NORMAL")
    )
    g["sentiment_flag"] = flags

    # API hook (returns None in default implementation → uses proxy flags)
    g["sentiment_flag"] = g.apply(
        lambda row: _get_api_sentiment(row["date"]) or row["sentiment_flag"],
        axis=1,
    )

    g["sentiment_multiplier"] = g["sentiment_flag"].map(config.SENTIMENT_MULTIPLIERS)

    # Human-readable reason
    reasons = []
    for _, row in g.iterrows():
        r   = row.get("ret_1d",   0) or 0
        vr  = row.get("vol_ratio", 1) or 1
        vz  = row.get("volume_z",  0) or 0
        gp  = abs(row.get("gap_pct", 0) or 0)
        v20 = row.get("vol_20d",   1e-6) or 1e-6

        flag = row["sentiment_flag"]
        if flag == "HIGH_RISK":
            if abs(r) > config.SENTIMENT_RET_SIGMA * v20:
                reasons.append(f"LARGE_MOVE({r:.2%})")
            else:
                reasons.append(f"VOL_SPIKE(ratio={vr:.2f})")
        elif flag == "EVENT_ACTIVE":
            if vz > config.SENTIMENT_VOL_Z:
                reasons.append(f"VOLUME_SPIKE(z={vz:.1f})")
            else:
                reasons.append(f"GAP({gp:.2%})")
        else:
            reasons.append("NORMAL")

    g["event_reason"] = reasons
    return g


def _portfolio_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Worst sentiment across ES and NQ governs the portfolio."""
    port = (
        df.groupby("date")["sentiment_flag"]
        .apply(lambda x: _RMAP[max(_RANK.get(f, 0) for f in x)])
        .reset_index()
        .rename(columns={"sentiment_flag": "portfolio_sentiment_flag"})
    )
    port["portfolio_sentiment_multiplier"] = port["portfolio_sentiment_flag"].map(
        config.SENTIMENT_MULTIPLIERS
    )
    return df.merge(port, on="date", how="left")


def _get_api_sentiment(date) -> Optional[str]:
    """
    Hook for external sentiment API.
    Return 'HIGH_RISK', 'EVENT_ACTIVE', 'NORMAL', or None.
    None = fall back to price-proxy flags.
    """
    return None


def _log_sentiment_distribution(df: pd.DataFrame) -> None:
    port = df.drop_duplicates("date")[["date", "portfolio_sentiment_flag"]]
    vc   = port["portfolio_sentiment_flag"].value_counts()
    n    = len(port)
    logger.info(
        "Portfolio sentiment — NORMAL: %d (%.0f%%)  EVENT: %d (%.0f%%)  HIGH_RISK: %d (%.0f%%)",
        vc.get("NORMAL",       0), 100 * vc.get("NORMAL",       0) / max(n, 1),
        vc.get("EVENT_ACTIVE", 0), 100 * vc.get("EVENT_ACTIVE", 0) / max(n, 1),
        vc.get("HIGH_RISK",    0), 100 * vc.get("HIGH_RISK",    0) / max(n, 1),
    )
