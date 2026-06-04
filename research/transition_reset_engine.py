"""
transition_reset_engine.py — Selective TRANSITION regime entry signals.

Identifies the BEST TRANSITION rows: price reclaiming a broken level (long)
or failing to reclaim and rejecting (short). NOT a blanket TRANSITION entry.

Output columns:
  tr_sma10              10-day simple moving average
  tr_sma20              20-day simple moving average
  tr_reclaim_flag       1 if close crossed above sma10 today (was below yesterday)
  tr_failed_reclaim     1 if close crossed below sma10 today (was above yesterday)
  tr_above_sma10        1 if close > sma10
  tr_lower_wick_pct     lower wick as % of daily range (buying pressure)
  tr_upper_wick_pct     upper wick as % of daily range (selling pressure)
  tr_buyer_pressure_5d  rolling 5d mean lower wick pct (sustained buying)
  tr_seller_pressure_5d rolling 5d mean upper wick pct (sustained selling)
  tr_buyer_reclaim_score  composite long score [0,1]
  tr_seller_rejection_score composite short score [0,1]
  tr_reset_signal       LONG / SHORT / FLAT
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SMA_SHORT   = 10
SMA_LONG    = 20
PRESSURE_WIN = 5


def compute_transition_reset_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    out_frames = []
    for sym, g in df.groupby("symbol", sort=False):
        out_frames.append(_per_symbol(g))
    return pd.concat(out_frames).sort_index()


def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values("date")
    close = pd.Series(g["close"].values.astype(float), index=g.index)
    high  = pd.Series(g["high"].values.astype(float)  if "high"  in g.columns else g["close"].values, index=g.index)
    low_  = pd.Series(g["low"].values.astype(float)   if "low"   in g.columns else g["close"].values, index=g.index)
    open_ = pd.Series(g["open"].values.astype(float)  if "open"  in g.columns else g["close"].values, index=g.index)

    # ── SMAs ──────────────────────────────────────────────────────────────────
    sma10 = close.rolling(SMA_SHORT, min_periods=3).mean()
    sma20 = close.rolling(SMA_LONG,  min_periods=5).mean()

    above_sma10 = (close > sma10).astype(int)
    prev_above  = above_sma10.shift(1).fillna(0)

    # Reclaim: was below, now above
    reclaim_flag     = ((above_sma10 == 1) & (prev_above == 0)).astype(int)
    # Failed reclaim: was above, now below
    failed_reclaim   = ((above_sma10 == 0) & (prev_above == 1)).astype(int)

    # ── Candle wick pressure ───────────────────────────────────────────────────
    day_range = (high - low_).replace(0, np.nan)
    lower_wick = (np.minimum(close, open_) - low_).clip(lower=0)
    upper_wick = (high - np.maximum(close, open_)).clip(lower=0)

    lower_wick_pct = (lower_wick / day_range).fillna(0).clip(0, 1)
    upper_wick_pct = (upper_wick / day_range).fillna(0).clip(0, 1)

    buyer_pressure  = lower_wick_pct.rolling(PRESSURE_WIN, min_periods=2).mean()
    seller_pressure = upper_wick_pct.rolling(PRESSURE_WIN, min_periods=2).mean()

    # ── Contextual features ───────────────────────────────────────────────────
    slope  = pd.Series(g["slope_20d"].fillna(0).values,  index=g.index)
    ret5   = pd.Series(g["ret_5d"].fillna(0).values,     index=g.index)
    ret1   = close.pct_change(1).fillna(0)
    bk_cnt = pd.Series(g.get("structure_break_count", pd.Series(0, index=g.index)).fillna(0).values, index=g.index)

    # ── Buyer reclaim score (long) ─────────────────────────────────────────────
    # Reclaim of SMA10 after pullback, buying pressure building, slope not bearish
    buyer_score = (
        0.35 * reclaim_flag +
        0.20 * (slope > 0).astype(float) +
        0.20 * (ret5 < 0).astype(float) +           # came from a pullback
        0.15 * buyer_pressure.clip(0, 1) +
        0.10 * (ret1 > 0).astype(float)
    ).clip(0, 1)

    # ── Seller rejection score (short) ────────────────────────────────────────
    # Failed reclaim of SMA10, seller pressure returning, slope bearish
    seller_score = (
        0.35 * failed_reclaim +
        0.20 * (slope < 0).astype(float) +
        0.20 * (ret5 > 0).astype(float) +           # came from a bounce
        0.15 * seller_pressure.clip(0, 1) +
        0.10 * (ret1 < 0).astype(float)
    ).clip(0, 1)

    # ── Reset signal ──────────────────────────────────────────────────────────
    signal = pd.Series("FLAT", index=g.index)
    signal[buyer_score > 0.55]  = "LONG"
    signal[seller_score > 0.55] = "SHORT"
    # If both fire (edge case), buyer score wins when slope >= 0
    both = (buyer_score > 0.55) & (seller_score > 0.55)
    signal[both & (slope >= 0)]  = "LONG"
    signal[both & (slope < 0)]   = "SHORT"

    g["tr_sma10"]                  = sma10.round(4)
    g["tr_sma20"]                  = sma20.round(4)
    g["tr_reclaim_flag"]           = reclaim_flag
    g["tr_failed_reclaim"]         = failed_reclaim
    g["tr_above_sma10"]            = above_sma10
    g["tr_lower_wick_pct"]         = lower_wick_pct.round(4)
    g["tr_upper_wick_pct"]         = upper_wick_pct.round(4)
    g["tr_buyer_pressure_5d"]      = buyer_pressure.round(4)
    g["tr_seller_pressure_5d"]     = seller_pressure.round(4)
    g["tr_buyer_reclaim_score"]    = buyer_score.round(4)
    g["tr_seller_rejection_score"] = seller_score.round(4)
    g["tr_reset_signal"]           = signal

    return g


def _log_summary(df: pd.DataFrame) -> None:
    trans = df[df["portfolio_regime"].fillna("CHOP") == "TRANSITION"]
    if len(trans) == 0:
        return
    n_long  = (trans["tr_reset_signal"] == "LONG").sum()
    n_short = (trans["tr_reset_signal"] == "SHORT").sum()
    n_reclaim = trans["tr_reclaim_flag"].sum()
    n_failed  = trans["tr_failed_reclaim"].sum()
    logger.info(
        "TransitionReset — TRANSITION rows=%d  reclaim=%d  failed_reclaim=%d  "
        "reset_LONG=%d  reset_SHORT=%d",
        len(trans), n_reclaim, n_failed, n_long, n_short,
    )
