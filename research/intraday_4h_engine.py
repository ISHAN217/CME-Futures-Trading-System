"""
intraday_4h_engine.py — 4H execution timing with entry quality scoring.

[CHANGED] Added:
  h4_breakout_score   price breaking above recent N-bar high (momentum breakout)
  h4_quality          entry quality score 1–5 (gates entries and sizes)

Entry quality score (1–5):
  +2  weekly_bias == LONG (primary filter)
  +1  4H momentum positive (score > 0)
  +1  pullback present (z < -1.0) OR breakout (new high)
  -1  overextended (|z| > 2.0)
  Capped to [1, 5]

WHY quality scoring?
  Not all entries in LONG regime are equal. A pullback entry with positive
  4H momentum during a confirmed LONG bias week is higher quality than a
  flat-market entry with neutral 4H. Sizing up on quality ≥ 4 captures
  more of the edge on the best trades without adding risk on marginal ones.

Execution signals:
  MOMENTUM   clean upward 4H momentum, not overextended
  PULLBACK   zscore dipped below -1.0 (within an uptrend context)
  BREAKOUT   close above N-bar high (new near-term high)
  WAIT       momentum unclear or overextended — skip entry
  NONE       no 4H data available

All values aggregated to end-of-day (last 4H bar of each session).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

_W_Z    = config.H4_ZSCORE_WINDOW
_W_BRK  = config.H4_BREAKOUT_LOOKBACK
_PULL   = config.H4_PULLBACK_ZSCORE
_OVER   = config.H4_OVEREXTENSION_ZSCORE


def compute_4h_features(df_4h: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for sym, g in df_4h.groupby("symbol", sort=True):
        g = g.copy().sort_values("datetime").reset_index(drop=True)
        g = _bar_features(g)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def aggregate_4h_to_daily(df_4h_feat: pd.DataFrame) -> pd.DataFrame:
    last = (
        df_4h_feat.sort_values("datetime")
        .groupby(["date", "symbol"], sort=False)
        .last()
        .reset_index()
    )
    h4_cols = ["date", "symbol"] + [c for c in last.columns if c.startswith("h4_")]
    out = last[h4_cols].copy()
    out["date"] = pd.to_datetime(out["date"])
    logger.info("4H aggregated: %d symbol-days", len(out))
    return out


def merge_4h_into_daily(
    df_daily: pd.DataFrame,
    df_4h_daily: Optional[pd.DataFrame],
) -> pd.DataFrame:
    # h4_quality is a placeholder here (always 0); real quality computed in signal_engine
    # as entry_quality. We exclude h4_quality from the merge to avoid column conflicts.
    h4_cols = [
        "h4_zscore", "h4_momentum_1", "h4_momentum_3", "h4_momentum_6",
        "h4_momentum_score", "h4_vol", "h4_breakout_score",
        "h4_exec_signal",
    ]

    if df_4h_daily is None or df_4h_daily.empty:
        for c in h4_cols:
            df_daily[c] = np.nan if c not in ("h4_exec_signal",) else "NONE"
        logger.warning("No 4H data — all 4H signals set to NONE")
        return df_daily

    # Drop placeholder h4_quality from 4H daily before merging (avoid conflict)
    merge_cols = ["date", "symbol"] + h4_cols
    df_4h_merge = df_4h_daily[[c for c in merge_cols if c in df_4h_daily.columns]]

    df_daily = df_daily.merge(df_4h_merge, on=["date", "symbol"], how="left")
    for c in h4_cols:
        if c not in df_daily.columns:
            df_daily[c] = np.nan if c != "h4_exec_signal" else "NONE"
    df_daily["h4_exec_signal"] = df_daily["h4_exec_signal"].fillna("NONE")

    cov = (df_daily["h4_exec_signal"] != "NONE").sum()
    logger.info("4H coverage: %d / %d symbol-days (%.0f%%)",
                cov, len(df_daily), 100 * cov / max(len(df_daily), 1))
    return df_daily


# ── Internal ──────────────────────────────────────────────────────────────────

def _bar_features(g: pd.DataFrame) -> pd.DataFrame:
    ret = g["close"].pct_change(1)

    g["h4_momentum_1"] = ret
    g["h4_momentum_3"] = g["close"].pct_change(3)
    g["h4_momentum_6"] = g["close"].pct_change(6)
    g["h4_vol"]        = ret.rolling(_W_Z, min_periods=3).std()

    mean_ = g["close"].rolling(_W_Z, min_periods=3).mean()
    std_  = g["close"].rolling(_W_Z, min_periods=3).std().replace(0, np.nan)
    g["h4_zscore"] = (g["close"] - mean_) / std_

    safe_vol = g["h4_vol"].replace(0, np.nan)
    n1 = (g["h4_momentum_1"] / safe_vol).clip(-3, 3) / 3
    n3 = (g["h4_momentum_3"] / safe_vol).clip(-3, 3) / 3
    n6 = (g["h4_momentum_6"] / safe_vol).clip(-3, 3) / 3
    g["h4_momentum_score"] = (0.5 * n1 + 0.3 * n3 + 0.2 * n6).clip(-1, 1)

    # Breakout: close above highest close of past N bars (excludes current bar)
    rolling_high = g["close"].shift(1).rolling(_W_BRK, min_periods=2).max()
    g["h4_breakout_score"] = (g["close"] > rolling_high).astype(float)

    g["h4_exec_signal"] = g.apply(_classify_exec, axis=1)

    # Quality score will be computed in signal_engine (needs weekly_bias)
    # Placeholder here; aggregation carries it forward
    g["h4_quality"] = 0

    return g


def _classify_exec(row: pd.Series) -> str:
    z   = row.get("h4_zscore",          np.nan)
    mom = row.get("h4_momentum_score",  np.nan)
    brk = row.get("h4_breakout_score",  0)

    if np.isnan(z) or np.isnan(mom):
        return "NONE"
    if abs(z) >= _OVER:
        return "WAIT"
    if brk > 0:
        return "BREAKOUT"
    if z <= _PULL:
        return "PULLBACK"
    if mom > config.H4_MOMENTUM_THRESHOLD:
        return "MOMENTUM"
    return "WAIT"


def compute_entry_quality(row: pd.Series) -> int:
    """
    Compute 1–5 quality score for a potential entry.
    Called from signal_engine (needs weekly_bias context).
    """
    score = 1  # base

    bias   = row.get("weekly_bias",      "NEUTRAL")
    sig    = row.get("h4_exec_signal",   "NONE")
    z      = row.get("h4_zscore",        0.0) or 0.0
    mom    = row.get("h4_momentum_score", 0.0) or 0.0
    mom_s  = row.get("momentum_strength", 0.0) or 0.0

    if bias == "LONG":
        score += 2   # directional alignment is the most valuable factor

    if mom > 0.0:
        score += 1   # 4H momentum aligned

    if sig in ("PULLBACK", "BREAKOUT"):
        score += 1   # entry timing present

    if abs(z) >= _OVER:
        score -= 1   # overextended — penalty

    if mom_s >= 0.6:
        score += 1   # strong multi-timeframe momentum

    return int(np.clip(score, 1, 5))
