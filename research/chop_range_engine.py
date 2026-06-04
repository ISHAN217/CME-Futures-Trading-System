"""
chop_range_engine.py — Range-quality features for CHOP regime mean reversion.

Adds per-symbol rolling range features that score whether price is near a
high-quality range boundary and likely to revert.

Output columns:
  chop_range_high          rolling 20d high
  chop_range_low           rolling 20d low
  chop_range_mid           midpoint of 20d range
  chop_range_width_pct     range width as % of mid
  chop_range_quality       stability score [0,1] — 1 = tight/stable range
  chop_z_score             price position in range: -1=bottom, 0=mid, +1=top
  chop_boundary_dist_low   fractional distance from lower boundary [0,1]
  chop_boundary_dist_high  fractional distance from upper boundary [0,1]
  chop_vol_expanding       bool-int: ATR growing faster than 10d avg
  chop_failed_breakdown    bool-int: yesterday broke below low, today reclaimed
  chop_failed_breakout     bool-int: yesterday broke above high, today rejected
  chop_mr_score_long       composite long entry score [0,1]
  chop_mr_score_short      composite short entry score [0,1]
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RANGE_WIN  = 20   # rolling window for range boundaries
QUAL_WIN   = 10   # window for range stability
ATR_WIN    = 14   # ATR window
ATR_RATIO_THR = 1.25   # ATR expanding if ratio > this


def compute_chop_range_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    out_frames = []
    for sym, g in df.groupby("symbol", sort=False):
        out_frames.append(_per_symbol(g))
    return pd.concat(out_frames).sort_index()


def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values("date")
    close = g["close"].values.astype(float)
    high  = g["high"].values.astype(float) if "high" in g.columns else close
    low   = g["low"].values.astype(float)  if "low"  in g.columns else close
    n     = len(g)

    # ── Rolling range ──────────────────────────────────────────────────────────
    s_close = pd.Series(close)
    s_high  = pd.Series(high)
    s_low   = pd.Series(low)

    rng_high = s_close.rolling(RANGE_WIN, min_periods=5).max().values
    rng_low  = s_close.rolling(RANGE_WIN, min_periods=5).min().values
    rng_mid  = (rng_high + rng_low) / 2
    rng_w    = rng_high - rng_low
    rng_w_pct = np.where(rng_mid > 1e-9, rng_w / rng_mid, np.nan)

    # ── Range quality: CV of daily (high-low) range over QUAL_WIN ─────────────
    daily_rng = s_high - s_low
    daily_rng_mean = daily_rng.rolling(QUAL_WIN, min_periods=3).mean().values
    daily_rng_std  = daily_rng.rolling(QUAL_WIN, min_periods=3).std().values
    cv = np.where(daily_rng_mean > 1e-9, daily_rng_std / daily_rng_mean, 1.0)
    range_quality = np.clip(1.0 - cv / 0.6, 0.0, 1.0)   # 0=chaotic, 1=stable

    # ── Z-score in range: -1 = at low, 0 = at mid, +1 = at high ──────────────
    half_w = rng_w / 2
    z = np.where(half_w > 1e-9, (close - rng_mid) / half_w, 0.0)
    z = np.clip(z, -2.0, 2.0)

    bd_low  = np.clip((close - rng_low)  / np.where(rng_w > 1e-9, rng_w, 1.0), 0, 1)
    bd_high = np.clip((rng_high - close) / np.where(rng_w > 1e-9, rng_w, 1.0), 0, 1)

    # ── ATR and vol-expansion flag ─────────────────────────────────────────────
    atr = (s_high - s_low).rolling(ATR_WIN, min_periods=5).mean().values
    atr_avg = pd.Series(atr).rolling(ATR_WIN, min_periods=5).mean().values
    vol_expanding = (atr > ATR_RATIO_THR * atr_avg).astype(int)

    # ── Failed breakdown / failed breakout (1-day lag, backward-looking) ───────
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    prev_rng_low  = np.roll(rng_low, 1);  prev_rng_low[0]  = rng_low[0]
    prev_rng_high = np.roll(rng_high, 1); prev_rng_high[0] = rng_high[0]

    failed_bdown = ((prev_close < prev_rng_low) & (close >= rng_low)).astype(int)
    failed_bout  = ((prev_close > prev_rng_high) & (close <= rng_high)).astype(int)

    # ── Composite entry scores ─────────────────────────────────────────────────
    # Long: price at lower boundary, stable range, vol not expanding
    near_low    = np.clip((-z + 1) / 2, 0, 1)   # peaks at z=-1
    mr_long = (
        0.35 * near_low +
        0.25 * range_quality +
        0.20 * (1 - vol_expanding) +
        0.20 * failed_bdown
    ).clip(0, 1)

    near_high   = np.clip((z + 1) / 2, 0, 1)    # peaks at z=+1
    mr_short = (
        0.35 * near_high +
        0.25 * range_quality +
        0.20 * (1 - vol_expanding) +
        0.20 * failed_bout
    ).clip(0, 1)

    g["chop_range_high"]       = rng_high.round(4)
    g["chop_range_low"]        = rng_low.round(4)
    g["chop_range_mid"]        = rng_mid.round(4)
    g["chop_range_width_pct"]  = np.where(rng_w_pct is not None, rng_w_pct, np.nan).round(4)
    g["chop_range_quality"]    = range_quality.round(4)
    g["chop_z_score"]          = z.round(4)
    g["chop_boundary_dist_low"]  = bd_low.round(4)
    g["chop_boundary_dist_high"] = bd_high.round(4)
    g["chop_vol_expanding"]    = vol_expanding
    g["chop_failed_breakdown"] = failed_bdown
    g["chop_failed_breakout"]  = failed_bout
    g["chop_mr_score_long"]    = mr_long.round(4)
    g["chop_mr_score_short"]   = mr_short.round(4)

    return g


def _log_summary(df: pd.DataFrame) -> None:
    for sym in df["symbol"].unique():
        g = df[df["symbol"] == sym]
        chop = g[g["portfolio_regime"].fillna("CHOP") == "CHOP"]
        if len(chop) == 0:
            continue
        near_low_n  = (chop["chop_z_score"].fillna(0) < -0.5).sum()
        near_high_n = (chop["chop_z_score"].fillna(0) > 0.5).sum()
        avg_qual    = chop["chop_range_quality"].mean()
        logger.info(
            "ChopRange %s — CHOP rows=%d  near_low=%d  near_high=%d  avg_quality=%.2f",
            sym, len(chop), near_low_n, near_high_n, avg_qual,
        )
