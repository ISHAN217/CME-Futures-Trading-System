"""
tier1_sentiment_engine.py — Cross-asset Tier 1 sentiment signals.

Validated signals (from diagnostic run 2026-05-24):
───────────────────────────────────────────────────
ES/NQ (equity instruments):
  VIX Term Structure  VIX3M/VIX ratio
    < 0.90  PANIC_BACKWD   avg=-0.317%  WR=41.9%  → BLOCK new longs
    0.90-0.97 BACKWD       avg=+0.610%  WR=63.4%  → BOOST size 1.20x
    0.97-1.10 NORMAL       avg=+0.104%  WR=54.5%  → full size
    > 1.10  DEEP_CONT      avg=+0.035%  WR=54.4%  → 0.90x (low-alpha calm)

  VIX Level (secondary filter)
    < 15    CALM           avg=+0.058%  → slight complacency warning
    > 40    PANIC          avg=+0.488%  → contrarian if ts_ratio >= 0.90

CL (crude oil):
  DXY 5-day return (dollar surge = CL headwind)
    > +1.5%               avg=-0.800%  WR=40.5%  → BLOCK new CL longs

  OVX level (crude oil implied vol)
    > 60    HIGH_FEAR      avg=+0.753%  WR=54.4%  → contrarian positive
    < 30    CALM           avg=-0.034%  WR=53.0%  → low-alpha environment

Output columns added to df (per row / per date):
  sent_vix            float  raw VIX level
  sent_vix3m          float  raw VIX3M level
  sent_ts_ratio       float  VIX3M / VIX
  sent_ts_regime      str    PANIC_BACK / BACKWD / NORMAL / DEEP_CONT
  sent_vix_regime     str    CALM / NORMAL / ELEVATED / FEAR / PANIC
  sent_eq_mult        float  size multiplier for ES/NQ entries (0.0 = block)
  sent_eq_signal      str    human-readable reason

  sent_dxy_r5         float  DXY 5-day pct return
  sent_dxy_block      bool   True → block new CL longs
  sent_ovx            float  raw OVX level
  sent_ovx_regime     str    CALM / NORMAL / ELEVATED / HIGH / PANIC
  sent_cl_mult        float  size multiplier for CL entries (0.0 = block)
  sent_cl_signal      str    human-readable reason

Design rules:
  - All computations strictly backward-looking (no future data)
  - Multipliers apply to mtf_entry_size before backtest sizing cap
  - 0.0 multiplier = hard block (no new entry that day)
  - Existing positions are NOT force-closed by sentiment (only entry-gated)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds — validated from diagnostic ─────────────────────────────────
_TS_PANIC_BACK   = 0.90    # VIX3M/VIX below this → panic backwardation (BLOCK)
_TS_BACKWD_HI    = 0.97    # 0.90-0.97 → mild backwardation (BOOST)
_TS_NORMAL_HI    = 1.10    # 0.97-1.10 → normal contango (NORMAL)
# > 1.10 → deep contango (CALM, slight caution)

_VIX_CALM        = 15.0    # below → complacency
_VIX_FEAR        = 30.0    # above → fear
_VIX_PANIC       = 40.0    # above → panic

_TS_BOOST_MULT   = 1.20    # backwardation recovery: boost by 20%
_TS_CALM_MULT    = 0.90    # deep contango: slight size reduction
_VIX_CALM_MULT   = 0.92    # complacency: slight caution

_DXY_SURGE_THRESH = 1.5    # DXY 5d pct return above this → block CL longs
_DXY_R5_WINDOW   = 5

_OVX_HIGH        = 60.0    # above → contrarian CL positive (keep entries)
_OVX_CALM        = 30.0    # below → low-alpha CL environment (slight caution)
_OVX_CALM_MULT   = 0.90    # reduce size when OVX very low


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_sentiment(sentiment_path: str) -> pd.DataFrame:
    """
    Load raw sentiment CSV (vix, vix3m, dxy, ovx columns, date index).
    Forward-fills up to 3 days for weekends/holidays.
    """
    path = Path(sentiment_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sentiment data not found: {path}\n"
            f"Run: python fetch_sentiment.py  (or pass correct path)"
        )
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    # forward-fill up to 3 trading days (handles weekends/holidays)
    df = df.ffill(limit=3)
    logger.info(
        "Sentiment loaded: %d rows  %s → %s  cols=%s",
        len(df), df.index[0].date(), df.index[-1].date(), df.columns.tolist()
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY (ES / NQ) SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ts_regime(ts_ratio: pd.Series) -> pd.Series:
    """Classify VIX term structure ratio into regime labels."""
    out = pd.Series("NORMAL", index=ts_ratio.index)
    out[ts_ratio < _TS_PANIC_BACK]  = "PANIC_BACK"
    out[ts_ratio.between(_TS_PANIC_BACK, _TS_BACKWD_HI)] = "BACKWD"
    out[ts_ratio > _TS_NORMAL_HI]   = "DEEP_CONT"
    return out


def _compute_vix_regime(vix: pd.Series) -> pd.Series:
    out = pd.Series("NORMAL", index=vix.index)
    out[vix < _VIX_CALM]  = "CALM"
    out[vix.between(_VIX_FEAR, _VIX_PANIC)] = "FEAR"
    out[vix >= _VIX_PANIC] = "PANIC"
    return out


def compute_equity_sentiment(sent_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute equity sentiment features from raw VIX/VIX3M data.
    Returns a DataFrame indexed by date with sent_* columns.
    """
    vix   = sent_raw["vix"].dropna()
    vix3m = sent_raw["vix3m"].dropna()

    idx = vix.index.intersection(vix3m.index)
    vix   = vix.reindex(idx)
    vix3m = vix3m.reindex(idx)

    ts_ratio   = (vix3m / vix).clip(0.5, 2.0)   # clip outliers
    ts_regime  = _compute_ts_regime(ts_ratio)
    vix_regime = _compute_vix_regime(vix)

    # ── Size multiplier ────────────────────────────────────────────────────
    mult = pd.Series(1.0, index=idx)

    # PANIC_BACK → block: fear still rising, not safe
    mult[ts_regime == "PANIC_BACK"] = 0.0

    # BACKWD → boost: fear peaked, recovery window
    mult[ts_regime == "BACKWD"] = _TS_BOOST_MULT

    # DEEP_CONT → do NOT reduce size.
    # Deep contango IS the normal bull-market structure (VIX3M >> VIX = calm).
    # This is when the system makes most of its money. Penalising it hurts Sharpe.
    # Diagnostic: DEEP_CONT avg ES return = +0.035%/day — still positive, 1136 days.
    # mult[ts_regime == "DEEP_CONT"] = _TS_CALM_MULT  ← removed after experiment

    # VIX calm override: reduce slightly even in contango
    calm_days = vix_regime == "CALM"
    mult[calm_days & (mult > 0)] = mult[calm_days & (mult > 0)].clip(upper=_VIX_CALM_MULT)

    # VIX PANIC + ts not panic_back = strong contrarian boost
    panic_ok = (vix_regime == "PANIC") & (ts_regime != "PANIC_BACK")
    mult[panic_ok] = np.maximum(mult[panic_ok], _TS_BOOST_MULT)

    # ── Signal label ───────────────────────────────────────────────────────
    signal = pd.Series("NORMAL", index=idx)
    signal[ts_regime == "PANIC_BACK"]                    = "BLOCKED_PANIC_BACKWD"
    signal[ts_regime == "BACKWD"]                        = "BOOSTED_FEAR_RECOVERY"
    signal[ts_regime == "DEEP_CONT"]                     = "CALM_LOW_ALPHA"
    signal[calm_days]                                    = "VIX_COMPLACENCY"
    signal[panic_ok]                                     = "PANIC_CONTRARIAN"

    out = pd.DataFrame({
        "sent_vix":        vix,
        "sent_vix3m":      vix3m,
        "sent_ts_ratio":   ts_ratio,
        "sent_ts_regime":  ts_regime,
        "sent_vix_regime": vix_regime,
        "sent_eq_mult":    mult,
        "sent_eq_signal":  signal,
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CL SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

def compute_cl_sentiment(sent_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CL sentiment features from DXY and OVX.
    Returns a DataFrame indexed by date with sent_cl_* columns.
    """
    dxy = sent_raw["dxy"].dropna()
    ovx = sent_raw["ovx"].dropna()

    idx = dxy.index.intersection(ovx.index)
    dxy = dxy.reindex(idx)
    ovx = ovx.reindex(idx)

    # DXY 5-day return (% change)
    dxy_r5 = dxy.pct_change(_DXY_R5_WINDOW) * 100

    # OVX regime
    ovx_regime = pd.Series("NORMAL", index=idx)
    ovx_regime[ovx < _OVX_CALM]  = "CALM"
    ovx_regime[ovx.between(45, 60)] = "ELEVATED"
    ovx_regime[ovx.between(60, 80)] = "HIGH"
    ovx_regime[ovx >= 80]         = "PANIC"

    # ── CL size multiplier ────────────────────────────────────────────────
    mult = pd.Series(1.0, index=idx)

    # DXY surge → block new CL longs
    dxy_block = dxy_r5 > _DXY_SURGE_THRESH
    mult[dxy_block] = 0.0

    # OVX calm → low-alpha environment, slight caution
    ovx_calm = ovx_regime == "CALM"
    mult[ovx_calm & (mult > 0)] = _OVX_CALM_MULT

    # OVX HIGH/PANIC → contrarian signal: keep entries (don't block)
    # Vol sizing already reduces position size; sentiment says edge is HIGHER, not lower
    # No multiplier change — we trust vol_scalar to size correctly
    # (Would be different if we wanted to ADD size, but that's aggressive)

    # ── Signal label ──────────────────────────────────────────────────────
    signal = pd.Series("NORMAL", index=idx)
    signal[dxy_block]                          = "BLOCKED_DXY_SURGE"
    signal[ovx_calm & ~dxy_block]              = "OVX_CALM_LOW_ALPHA"
    signal[(ovx_regime == "HIGH") & ~dxy_block]  = "OVX_HIGH_CONTRARIAN"
    signal[(ovx_regime == "PANIC") & ~dxy_block] = "OVX_PANIC_CONTRARIAN"

    out = pd.DataFrame({
        "sent_dxy":       dxy,
        "sent_dxy_r5":    dxy_r5,
        "sent_dxy_block": dxy_block.astype(int),
        "sent_ovx":       ovx,
        "sent_ovx_regime":ovx_regime,
        "sent_cl_mult":   mult,
        "sent_cl_signal": signal,
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY: merge sentiment into pipeline df
# ─────────────────────────────────────────────────────────────────────────────

def compute_tier1_sentiment(
    df: pd.DataFrame,
    sentiment_path: str = "/tmp/sentiment_raw.csv",
    eq_symbols: list = None,
    cl_symbols: list = None,
) -> pd.DataFrame:
    """
    Join Tier 1 sentiment features into pipeline df.

    eq_symbols: list of symbol names that are equity (default: ['ES','NQ'])
    cl_symbols: list of symbol names that are CL (default: ['CL'])
                — use None if not trading CL.

    Adds columns:
      sent_eq_mult, sent_eq_signal  (for equity rows)
      sent_cl_mult, sent_cl_signal  (for CL rows, if cl_symbols given)
    """
    if eq_symbols is None:
        eq_symbols = ["ES", "NQ"]

    sent_raw = load_sentiment(sentiment_path)

    eq_sent = compute_equity_sentiment(sent_raw)
    eq_sent.index = pd.to_datetime(eq_sent.index)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Default neutral values
    df["sent_eq_mult"]   = 1.0
    df["sent_eq_signal"] = "NORMAL"
    df["sent_ts_regime"] = "NORMAL"
    df["sent_vix_regime"]= "NORMAL"
    df["sent_ts_ratio"]  = np.nan

    # Merge equity sentiment (same for all equity symbols on a given date)
    eq_mask = df["symbol"].isin(eq_symbols)
    eq_dates = df.loc[eq_mask, "date"]
    merged = eq_dates.map(
        lambda d: eq_sent.loc[d] if d in eq_sent.index else None
    )
    for col in ["sent_eq_mult","sent_eq_signal","sent_ts_regime","sent_vix_regime","sent_ts_ratio"]:
        df.loc[eq_mask, col] = eq_dates.map(
            lambda d, c=col: eq_sent.at[d, c] if d in eq_sent.index else (1.0 if "mult" in c else "NORMAL")
        ).values

    # CL sentiment (optional)
    if cl_symbols:
        cl_sent = compute_cl_sentiment(sent_raw)
        cl_sent.index = pd.to_datetime(cl_sent.index)

        df["sent_cl_mult"]   = 1.0
        df["sent_cl_signal"] = "NORMAL"
        df["sent_ovx_regime"]= "NORMAL"

        cl_mask = df["symbol"].isin(cl_symbols)
        cl_dates = df.loc[cl_mask, "date"]
        for col in ["sent_cl_mult","sent_cl_signal","sent_ovx_regime"]:
            df.loc[cl_mask, col] = cl_dates.map(
                lambda d, c=col: cl_sent.at[d, c] if d in cl_sent.index else (1.0 if "mult" in c else "NORMAL")
            ).values

    # Logging
    eq_rows = df[eq_mask].drop_duplicates("date")
    blocked  = (eq_rows["sent_eq_mult"] == 0).sum()
    boosted  = (eq_rows["sent_eq_mult"] > 1.0).sum()
    reduced  = ((eq_rows["sent_eq_mult"] > 0) & (eq_rows["sent_eq_mult"] < 1.0)).sum()
    logger.info(
        "Tier1 equity sentiment: %d blocked  %d boosted  %d reduced  (of %d trading days)",
        blocked, boosted, reduced, len(eq_rows)
    )
    if cl_symbols:
        cl_rows = df[df["symbol"].isin(cl_symbols)].drop_duplicates("date")
        cl_blocked = (cl_rows["sent_cl_mult"] == 0).sum()
        logger.info("Tier1 CL sentiment: %d DXY-blocked days", cl_blocked)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# APPLY TO SIGNAL STACK  (call this after compute_tier1_sentiment)
# ─────────────────────────────────────────────────────────────────────────────

def apply_sentiment_to_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sentiment multipliers to mtf_entry_size for new entries only.

    - LONG entries on equity symbols: scale by sent_eq_mult
      mult=0.0 → flat the entry (don't enter)
    - LONG entries on CL symbols: scale by sent_cl_mult
    - Existing holds (days_held > 0) are NOT touched
    - SHORT entries are NOT blocked by equity sentiment
      (shorts are already in a fear regime — blocking them would be wrong)
    """
    df = df.copy()

    eq_syms = ["ES", "NQ"]
    cl_syms = ["CL"]   # only used if CL is in the df

    # ── Equity entries ────────────────────────────────────────────────────
    eq_new_long = (
        df["symbol"].isin(eq_syms) &
        (df["final_direction"] == "LONG") &
        (df.get("days_held", 0) == 0)   # new entry only
    )
    if "sent_eq_mult" in df.columns:
        mult = df.loc[eq_new_long, "sent_eq_mult"].fillna(1.0)
        # Block: set flat
        block = eq_new_long & (df["sent_eq_mult"] == 0.0)
        df.loc[block, "final_direction"] = "FLAT"
        df.loc[block, "strategy_used"]   = "NONE"
        df.loc[block, "mtf_entry_size"]  = 0.0
        # Scale non-blocked entries
        scale = eq_new_long & ~block & df["symbol"].isin(eq_syms)
        df.loc[scale, "mtf_entry_size"] = (
            df.loc[scale, "mtf_entry_size"].fillna(0.0) *
            df.loc[scale, "sent_eq_mult"].fillna(1.0)
        )

    # ── CL entries ────────────────────────────────────────────────────────
    if "sent_cl_mult" in df.columns:
        cl_new_long = (
            df["symbol"].isin(cl_syms) &
            (df["final_direction"] == "LONG") &
            (df.get("days_held", 0) == 0)
        )
        block_cl = cl_new_long & (df["sent_cl_mult"] == 0.0)
        df.loc[block_cl, "final_direction"] = "FLAT"
        df.loc[block_cl, "strategy_used"]   = "NONE"
        df.loc[block_cl, "mtf_entry_size"]  = 0.0
        scale_cl = cl_new_long & ~block_cl
        df.loc[scale_cl, "mtf_entry_size"] = (
            df.loc[scale_cl, "mtf_entry_size"].fillna(0.0) *
            df.loc[scale_cl, "sent_cl_mult"].fillna(1.0)
        )

    return df
