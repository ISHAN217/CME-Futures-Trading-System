"""
signal_quality_engine.py — Signal quality overlay for position sizing. [NEW]

Computes two composite quality scores using ONLY existing columns produced
by the upstream pipeline. No lookahead bias: all ranking windows are strictly
backward-looking (rolling percentile ranks).

TREND quality score (5 components):
  1. Directional strength    — multi-timeframe vol-adjusted momentum
  2. Trend cleanliness       — slope, R², efficiency ratio
  3. Multi-horizon alignment — timeframe agreement + weekly bias + 4H
  4. Volatility quality      — vol regime, vol ratio, ATR stability
  5. Entry timing quality    — h4 timing, price extension penalty

STAT_ARB quality score (5 components, two variants):
  1. Dislocation strength    — linear and hump-shaped z-score scoring
  2. Relationship stability  — beta stability, spread vol stability, correlation
  3. Mean-reversion evidence — z-score slope toward zero, recent reversions
  4. Regime compatibility    — not SHOCK, vol stable
  5. Tail-risk quality       — penalty for extreme z, vol explosion

Output columns per (date, symbol):
  trend_quality_score             float [0, 1]
  trend_quality_dir_strength      float [0, 1]  component
  trend_quality_cleanliness       float [0, 1]  component
  trend_quality_mh_align          float [0, 1]  component
  trend_quality_vol_quality       float [0, 1]  component
  trend_quality_entry_timing      float [0, 1]  component
  trend_quality_bucket            WEAK/MODERATE/GOOD/STRONG/EXCEPTIONAL
  trend_quality_reason            str

  statarb_quality_score_linear    float [0, 1]
  statarb_quality_score_hump      float [0, 1]
  statarb_quality_bucket_linear   str
  statarb_quality_bucket_hump     str
  statarb_quality_reason          str

Usage:
  from signal_quality_engine import compute_signal_quality
  df = compute_signal_quality(df)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ── Bucket thresholds ─────────────────────────────────────────────────────────
_BUCKETS = [
    (0.85, "EXCEPTIONAL"),
    (0.70, "STRONG"),
    (0.50, "GOOD"),
    (0.30, "MODERATE"),
    (0.00, "WEAK"),
]


def _score_to_bucket(score: float) -> str:
    if pd.isna(score):
        return "MODERATE"
    for threshold, label in _BUCKETS:
        if score >= threshold:
            return label
    return "WEAK"


# ── Rolling percentile rank ───────────────────────────────────────────────────

def _rolling_rank(
    series: pd.Series,
    window: int = 252,
    min_periods: int = 63,
    invert: bool = False,
) -> pd.Series:
    """
    Rolling percentile rank in [0, 1].
    invert=True → 0 = highest value (useful for 'lower is better' metrics).
    Uses pandas rolling().rank() which is O(n log n) per window.
    Strictly backward-looking — no lookahead bias.
    """
    try:
        ranked = series.rolling(window, min_periods=min_periods).rank(pct=True)
    except Exception:
        # Fallback for older pandas
        ranked = series.rolling(window, min_periods=min_periods).apply(
            lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else 0.5, raw=True
        )
    ranked = ranked.clip(0.0, 1.0)
    if invert:
        ranked = 1.0 - ranked
    return ranked


# ── Efficiency ratio ─────────────────────────────────────────────────────────

def _efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Efficiency ratio (Kaufman) = |directional move| / path length.
    1.0 = perfectly straight trend. 0.0 = random walk going nowhere.
    """
    directional = close.diff(window).abs()
    path = close.diff(1).abs().rolling(window, min_periods=window // 2).sum()
    er = (directional / path.replace(0, np.nan)).clip(0.0, 1.0)
    return er.fillna(0.5)


# ════════════════════════════════════════════════════════════════════════════════
# TREND QUALITY
# ════════════════════════════════════════════════════════════════════════════════

def _compute_trend_quality(g: pd.DataFrame) -> pd.DataFrame:
    """
    Per-symbol TREND quality computation.
    Adds trend_quality_* columns in-place.
    """
    RANK_WIN     = 252   # 1-year rolling window for percentile ranks
    RANK_MIN     = 63    # 3 months minimum before rank is meaningful
    close        = g["close"]
    vol_20d      = g["vol_20d"].replace(0, np.nan)

    # ── 1. Directional strength ───────────────────────────────────────────────
    # Vol-adjusted momentum for 5d, 20d, 60d timeframes.
    # Normalised like a t-statistic: ret / (vol × √n) ≈ signal-to-noise ratio.

    def _vol_adj_mom(ret_col: str, n: int) -> pd.Series:
        ret = g[ret_col] if ret_col in g.columns else close.pct_change(n)
        raw = ret / (vol_20d * np.sqrt(n))
        return _rolling_rank(raw.clip(-4, 4), RANK_WIN, RANK_MIN)

    mom_5_rank  = _vol_adj_mom("ret_5d",  5)
    mom_20_rank = _vol_adj_mom("ret_20d", 20)
    mom_60_rank = _vol_adj_mom("ret_60d", 60)

    dir_strength = (
        0.25 * mom_5_rank
        + 0.50 * mom_20_rank
        + 0.25 * mom_60_rank
    ).clip(0, 1)

    # ── 2. Trend cleanliness ─────────────────────────────────────────────────
    # Higher slope + higher R² + higher efficiency ratio = cleaner, more
    # tradeable trend structure.

    slope_rank = _rolling_rank(g["slope_20d"].fillna(0), RANK_WIN, RANK_MIN)
    r2_rank    = _rolling_rank(g["r2_20d"].fillna(0),    RANK_WIN, RANK_MIN)
    er         = _efficiency_ratio(close, window=20)
    er_rank    = _rolling_rank(er, RANK_WIN, RANK_MIN)

    trend_clean = (
        0.35 * slope_rank
        + 0.40 * r2_rank
        + 0.25 * er_rank
    ).clip(0, 1)

    # ── 3. Multi-horizon alignment ────────────────────────────────────────────
    # Are all timeframes pointing the same direction?
    # Structural (weekly_bias) alignment is the most important factor.

    # Timeframe agreement: 0/0.33/0.67/1.0 for 0/1/2/3 positive timeframes
    tf_positive = (
        (g["ret_5d"].fillna(0)  > 0).astype(float)
        + (g["ret_20d"].fillna(0) > 0).astype(float)
        + (g["ret_60d"].fillna(0) > 0).astype(float)
    ) / 3.0

    # Weekly bias alignment
    wb_align = (g["weekly_bias"].fillna("NEUTRAL") == "LONG").astype(float)

    # 4H execution signal confirmation
    h4_sig = g["h4_exec_signal"].fillna("NONE")
    h4_confirm = h4_sig.map({
        "MOMENTUM": 0.80,
        "BREAKOUT": 0.90,
        "PULLBACK": 0.85,   # pullback into trend — very good entry
        "WAIT":     0.20,
        "NONE":     0.50,
    }).fillna(0.50)

    mh_align = (
        0.30 * tf_positive
        + 0.40 * wb_align
        + 0.30 * h4_confirm
    ).clip(0, 1)

    # ── 4. Volatility quality ─────────────────────────────────────────────────
    # Stable, moderate volatility is ideal for trend-following.
    # Penalise high-vol regimes (signal noise increases) and very low-vol
    # (false breakouts).

    # vol_ratio rank inverted: lower vol_ratio = more stable vol = better
    inv_vol_ratio_rank = _rolling_rank(
        g["vol_ratio"].fillna(1.0), RANK_WIN, RANK_MIN, invert=True
    )

    # atr_pct rank inverted: lower ATR% relative to history = calmer market
    inv_atr_rank = _rolling_rank(
        g["atr_pct"].fillna(0), RANK_WIN, RANK_MIN, invert=True
    )

    # Vol regime flag: NORMAL=1.0, ELEVATED=0.5, HIGH=0.0
    vol_reg = g["vol_regime"].fillna("NORMAL")
    vol_norm = vol_reg.map({"NORMAL": 1.0, "ELEVATED": 0.5, "HIGH": 0.0}).fillna(0.5)

    vol_quality = (
        0.25 * inv_vol_ratio_rank
        + 0.25 * inv_atr_rank
        + 0.50 * vol_norm
    ).clip(0, 1)

    # ── 5. Entry timing quality ───────────────────────────────────────────────
    # Best entries: moderate pullback (not overextended), 4H signal actionable.

    # 4H z-score timing: optimal zone is mildly negative (pullback into trend).
    # Use a Gaussian-like scoring centred on z = -0.5.
    h4_z = g["h4_zscore"].fillna(0).clip(-4, 4)
    # Score = exp(-0.5 * ((z - (-0.5)) / 1.2)^2) → peak at z = -0.5
    # But we want 0-1 range: we then clip/normalize.
    h4_z_score_raw = np.exp(-0.5 * ((h4_z - (-0.5)) / 1.2) ** 2)
    # Heavy penalty if overextended (h4_z > 2.0)
    h4_z_score = np.where(h4_z.abs() >= 2.0, h4_z_score_raw * 0.3, h4_z_score_raw)
    h4_z_score = pd.Series(h4_z_score, index=g.index).clip(0, 1)

    # Price distance from 20d MA — prefer not overextended above MA
    ma_20 = close.rolling(20, min_periods=10).mean()
    ma_20_std = close.rolling(20, min_periods=10).std().replace(0, np.nan)
    price_ext_z = ((close - ma_20) / ma_20_std).fillna(0).clip(-3, 3)
    # Invert rank: lower extension = better entry (prefer pulling back to MA)
    inv_ext_rank = _rolling_rank(
        price_ext_z, RANK_WIN, RANK_MIN, invert=True
    )

    entry_timing = (
        0.50 * h4_z_score
        + 0.50 * inv_ext_rank
    ).clip(0, 1)

    # ── Composite TREND quality score ─────────────────────────────────────────
    trend_quality_score = (
        0.25 * dir_strength
        + 0.25 * trend_clean
        + 0.20 * mh_align
        + 0.15 * vol_quality
        + 0.15 * entry_timing
    ).clip(0, 1)

    # ── Assign ────────────────────────────────────────────────────────────────
    g["trend_quality_score"]        = trend_quality_score.round(4)
    g["trend_quality_dir_strength"] = dir_strength.round(4)
    g["trend_quality_cleanliness"]  = trend_clean.round(4)
    g["trend_quality_mh_align"]     = mh_align.round(4)
    g["trend_quality_vol_quality"]  = vol_quality.round(4)
    g["trend_quality_entry_timing"] = entry_timing.round(4)
    g["trend_quality_bucket"]       = trend_quality_score.apply(_score_to_bucket)
    g["trend_quality_reason"]       = _trend_reason_series(
        dir_strength, trend_clean, mh_align, vol_quality, entry_timing
    )

    return g


def _trend_reason_series(
    dir_s: pd.Series,
    clean: pd.Series,
    align: pd.Series,
    vol_q: pd.Series,
    timing: pd.Series,
) -> pd.Series:
    """Build human-readable reason string for each row."""
    reasons = []
    for i in range(len(dir_s)):
        parts = []
        d  = dir_s.iloc[i]  if not pd.isna(dir_s.iloc[i])  else 0.5
        c  = clean.iloc[i]  if not pd.isna(clean.iloc[i])  else 0.5
        a  = align.iloc[i]  if not pd.isna(align.iloc[i])  else 0.5
        v  = vol_q.iloc[i]  if not pd.isna(vol_q.iloc[i])  else 0.5
        t  = timing.iloc[i] if not pd.isna(timing.iloc[i]) else 0.5
        if d >= 0.70: parts.append("strong_mom")
        elif d <= 0.30: parts.append("weak_mom")
        if c >= 0.70: parts.append("clean_trend")
        elif c <= 0.30: parts.append("choppy")
        if a >= 0.70: parts.append("all_aligned")
        elif a <= 0.30: parts.append("misaligned")
        if v <= 0.30: parts.append("high_vol")
        if t >= 0.70: parts.append("good_timing")
        elif t <= 0.30: parts.append("extended")
        reasons.append("|".join(parts) or "neutral")
    return pd.Series(reasons, index=dir_s.index)


# ════════════════════════════════════════════════════════════════════════════════
# STAT ARB QUALITY
# ════════════════════════════════════════════════════════════════════════════════

def _compute_statarb_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    STAT_ARB quality — computed per date using pair-level columns.
    Result is broadcast back to all symbol rows on that date.
    """
    RANK_WIN = 252
    RANK_MIN = 63

    # Extract one row per date (use ES rows — spread features are identical)
    es_df = (
        df[df["symbol"] == "ES"]
        [["date", "spread_zscore", "beta", "ret_1d",
          "portfolio_regime", "vol_regime", "vol_ratio",
          "spread", "nq_rel_ret_5d"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    nq_df = (
        df[df["symbol"] == "NQ"]
        [["date", "ret_1d"]]
        .drop_duplicates("date")
        .sort_values("date")
        .rename(columns={"ret_1d": "nq_ret_1d"})
        .reset_index(drop=True)
        .copy()
    )

    pair = es_df.merge(nq_df, on="date", how="left")
    pair = pair.rename(columns={"ret_1d": "es_ret_1d"})

    z    = pair["spread_zscore"].fillna(0.0)
    absz = z.abs()

    # ── 1a. Dislocation — linear ──────────────────────────────────────────────
    # Higher |z| = more dislocated = stronger linear signal.
    disloc_linear_rank = _rolling_rank(absz, RANK_WIN, RANK_MIN)

    # ── 1b. Dislocation — hump-shaped ────────────────────────────────────────
    # Optimal dislocation: 1.5 ≤ |z| < 3.2. Extreme z penalised (likely break).
    def _hump_score(az: float) -> float:
        if az < 1.0:   return 0.20
        if az < 1.5:   return 0.40
        if az < 2.0:   return 0.65
        if az < 2.5:   return 0.85
        if az < 3.2:   return 1.00
        if az < 4.0:   return 0.60
        return 0.20   # |z| >= 4.0: likely structural break, dangerous

    disloc_hump = absz.apply(_hump_score)

    # ── 2. Relationship stability ─────────────────────────────────────────────
    # Rolling correlation between ES and NQ returns (20d)
    roll_corr = (
        pair["es_ret_1d"].rolling(20, min_periods=10)
        .corr(pair["nq_ret_1d"].fillna(0))
        .fillna(0.90)   # default: assume high correlation
        .clip(0, 1)
    )
    # Higher correlation = more stable relationship = better
    corr_rank = _rolling_rank(roll_corr, RANK_WIN, RANK_MIN)

    # Beta stability: rolling std of beta over 30d (lower std = more stable)
    beta_std = pair["beta"].fillna(pair["beta"].median()).rolling(30, min_periods=10).std()
    inv_beta_std_rank = _rolling_rank(beta_std.fillna(0), RANK_WIN, RANK_MIN, invert=True)

    # Spread vol stability: rolling std of spread (lower = more mean-reverting)
    spread_std = pair["spread"].fillna(0).rolling(30, min_periods=10).std()
    inv_spread_std_rank = _rolling_rank(spread_std.fillna(0), RANK_WIN, RANK_MIN, invert=True)

    rel_stability = (
        0.40 * corr_rank
        + 0.30 * inv_beta_std_rank
        + 0.30 * inv_spread_std_rank
    ).clip(0, 1)

    # ── 3. Mean-reversion evidence ────────────────────────────────────────────
    # Is the spread already starting to revert toward zero?
    # sign(z_t) * (z_t - z_{t-1}) < 0 → z moving toward zero (good sign)
    z_lag     = z.shift(1).fillna(z)
    z_delta   = z - z_lag      # positive = z drifting further from zero if z>0
    reverting = (np.sign(z) * z_delta < 0).astype(float)   # 1 = reverting

    # Recent reversion rate: fraction of last 5 days where z moved toward zero
    rev_rate_5d = reverting.rolling(5, min_periods=2).mean().fillna(0.5)

    # Zero-crossing frequency in last 20 days (more crossings = active reversion)
    def _zero_cross_rate(z_series: pd.Series, window: int = 20) -> pd.Series:
        signs = np.sign(z_series)
        crosses = (signs != signs.shift(1)).astype(float)
        return crosses.rolling(window, min_periods=5).mean().fillna(0.1)

    cross_rate = _zero_cross_rate(z, window=20)
    cross_rank = _rolling_rank(cross_rate, RANK_WIN, RANK_MIN)

    mr_evidence = (
        0.50 * rev_rate_5d.clip(0, 1)
        + 0.30 * cross_rank
        + 0.20 * reverting.rolling(3, min_periods=1).mean().clip(0, 1)
    ).clip(0, 1)

    # ── 4. Regime compatibility ───────────────────────────────────────────────
    # Not SHOCK; stable vol; not EVENT sentiment.
    not_shock  = (pair["portfolio_regime"].fillna("CHOP") != "SHOCK").astype(float)
    vol_stable = pair["vol_regime"].fillna("NORMAL").map(
        {"NORMAL": 1.0, "ELEVATED": 0.6, "HIGH": 0.0}
    ).fillna(0.6)
    # Not in extremely high vol_ratio (short-vol spike)
    vol_ratio_ok = (pair["vol_ratio"].fillna(1.0) < config.SHOCK_VOL_RATIO).astype(float)

    regime_compat = (
        0.50 * not_shock
        + 0.30 * vol_stable
        + 0.20 * vol_ratio_ok
    ).clip(0, 1)

    # ── 5. Tail-risk quality ──────────────────────────────────────────────────
    # Penalty for extreme z (>3.5) — may be regime break, not simple dislocation.
    # Penalty for correlation below 0.70 — relationship may be breaking down.
    # Penalty for spread vol explosion.
    no_extreme_z   = (absz < 3.5).astype(float) * 0.40 + (absz < 2.5).astype(float) * 0.60
    no_extreme_z   = no_extreme_z.clip(0, 1)

    corr_penalty   = (roll_corr >= 0.70).astype(float)   # 1 = correlation fine
    no_spread_blow = (inv_spread_std_rank >= 0.30).astype(float)  # 1 = spread vol not extreme

    tail_risk = (
        0.40 * no_extreme_z
        + 0.35 * corr_penalty
        + 0.25 * no_spread_blow
    ).clip(0, 1)

    # ── Composite scores ──────────────────────────────────────────────────────
    W = dict(disloc=0.25, stability=0.25, mr=0.20, regime=0.15, tail=0.15)

    sa_linear = (
        W["disloc"]   * disloc_linear_rank
        + W["stability"] * rel_stability
        + W["mr"]        * mr_evidence
        + W["regime"]    * regime_compat
        + W["tail"]      * tail_risk
    ).clip(0, 1)

    sa_hump = (
        W["disloc"]   * disloc_hump
        + W["stability"] * rel_stability
        + W["mr"]        * mr_evidence
        + W["regime"]    * regime_compat
        + W["tail"]      * tail_risk
    ).clip(0, 1)

    # ── Build reason string ───────────────────────────────────────────────────
    def _sa_reason(row_idx: int) -> str:
        parts = []
        az = absz.iloc[row_idx]
        if az < 1.5:  parts.append("mild_disloc")
        elif az > 3.2: parts.append("extreme_z")
        elif az > 2.5: parts.append("strong_disloc")
        else:          parts.append("good_disloc")

        cr = roll_corr.iloc[row_idx]
        if cr < 0.70:  parts.append("corr_break")
        elif cr > 0.90: parts.append("high_corr")

        rv = rev_rate_5d.iloc[row_idx]
        if rv > 0.60:  parts.append("reverting")
        elif rv < 0.30: parts.append("no_reversion")

        rg = pair["portfolio_regime"].iloc[row_idx]
        if rg == "SHOCK": parts.append("SHOCK")

        return "|".join(parts) or "neutral"

    pair["statarb_quality_score_linear"]  = sa_linear.round(4)
    pair["statarb_quality_score_hump"]    = sa_hump.round(4)
    pair["statarb_quality_bucket_linear"] = sa_linear.apply(_score_to_bucket)
    pair["statarb_quality_bucket_hump"]   = sa_hump.apply(_score_to_bucket)
    pair["statarb_quality_reason"]        = [
        _sa_reason(i) for i in range(len(pair))
    ]

    # ── Merge back to all symbol rows ─────────────────────────────────────────
    sa_cols = [
        "date",
        "statarb_quality_score_linear", "statarb_quality_score_hump",
        "statarb_quality_bucket_linear", "statarb_quality_bucket_hump",
        "statarb_quality_reason",
    ]
    pair_out = pair[sa_cols].copy()
    df = df.merge(pair_out, on="date", how="left")

    # Default fill for dates with no pair data
    for col in sa_cols[1:]:
        if col.endswith("_bucket_linear") or col.endswith("_bucket_hump"):
            df[col] = df[col].fillna("MODERATE")
        elif col.endswith("_reason"):
            df[col] = df[col].fillna("no_data")
        else:
            df[col] = df[col].fillna(0.5)

    return df


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def compute_signal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TREND and STAT_ARB quality scores to df.
    Processes per-symbol for TREND, per-date for STAT_ARB.
    Safe to call on the full multi-symbol df.
    """
    df = df.copy()

    # ── TREND quality (per-symbol) ────────────────────────────────────────────
    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.copy().sort_values("date").reset_index(drop=True)
        g = _compute_trend_quality(g)
        parts.append(g)
    df = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"])

    # ── STAT_ARB quality (cross-symbol / per date) ────────────────────────────
    # Only compute if we have both ES and NQ
    if df["symbol"].nunique() >= 2:
        df = _compute_statarb_quality(df)
    else:
        # Fallback: neutral values
        for col in [
            "statarb_quality_score_linear", "statarb_quality_score_hump"
        ]:
            df[col] = 0.5
        for col in [
            "statarb_quality_bucket_linear", "statarb_quality_bucket_hump"
        ]:
            df[col] = "MODERATE"
        df["statarb_quality_reason"] = "single_symbol"

    _log_quality_distribution(df)
    return df


def _log_quality_distribution(df: pd.DataFrame) -> None:
    for sym in df["symbol"].unique():
        g  = df[df["symbol"] == sym]
        vc = g["trend_quality_bucket"].value_counts()
        n  = len(g)
        logger.info(
            "TREND quality %s — EXCEP: %d (%.0f%%)  STRONG: %d (%.0f%%)  "
            "GOOD: %d (%.0f%%)  MOD: %d (%.0f%%)  WEAK: %d (%.0f%%)  "
            "avg_score=%.3f",
            sym,
            vc.get("EXCEPTIONAL", 0), 100 * vc.get("EXCEPTIONAL", 0) / max(n, 1),
            vc.get("STRONG",      0), 100 * vc.get("STRONG",      0) / max(n, 1),
            vc.get("GOOD",        0), 100 * vc.get("GOOD",        0) / max(n, 1),
            vc.get("MODERATE",    0), 100 * vc.get("MODERATE",    0) / max(n, 1),
            vc.get("WEAK",        0), 100 * vc.get("WEAK",        0) / max(n, 1),
            g["trend_quality_score"].mean(),
        )

    # SA quality (one log for overall)
    if "statarb_quality_bucket_hump" in df.columns:
        g   = df.drop_duplicates("date")
        vc  = g["statarb_quality_bucket_hump"].value_counts()
        n   = len(g)
        logger.info(
            "STAT_ARB quality (hump) — EXCEP: %d  STRONG: %d  GOOD: %d  "
            "MOD: %d  WEAK: %d  avg_hump=%.3f",
            vc.get("EXCEPTIONAL", 0), vc.get("STRONG", 0),
            vc.get("GOOD", 0),       vc.get("MODERATE", 0),
            vc.get("WEAK", 0),
            df["statarb_quality_score_hump"].mean(),
        )


# ── Quality sizing helpers ────────────────────────────────────────────────────

def get_quality_multiplier(row: dict, strategy: str) -> float:
    """
    Read quality multiplier from config based on strategy and quality bucket.
    Used by position_sizing.py when USE_SIGNAL_QUALITY_SIZING = True.
    Returns 1.0 (no change) if disabled or bucket not found.
    """
    if not getattr(config, "USE_SIGNAL_QUALITY_SIZING", False):
        return 1.0

    mode = getattr(config, "QUALITY_SIZING_MODE", "multiplier")

    if strategy == "TREND" and getattr(config, "APPLY_QUALITY_SIZING_TO_TREND", True):
        bucket = str(row.get("trend_quality_bucket", "GOOD") or "GOOD")
        mults  = getattr(config, "TREND_QUALITY_MULTIPLIERS",
                         {"WEAK": 0.50, "MODERATE": 0.75, "GOOD": 1.00,
                          "STRONG": 1.10, "EXCEPTIONAL": 1.25})
        return float(mults.get(bucket, 1.0))

    elif strategy == "STAT_ARB" and getattr(config, "APPLY_QUALITY_SIZING_TO_STAT_ARB", True):
        sa_mode = getattr(config, "STAT_ARB_QUALITY_MODE", "hump")
        bucket  = str(row.get(f"statarb_quality_bucket_{sa_mode}", "GOOD") or "GOOD")
        mults   = getattr(config, "STATARB_QUALITY_MULTIPLIERS",
                          {"WEAK": 0.50, "MODERATE": 0.75, "GOOD": 1.00,
                           "STRONG": 1.10, "EXCEPTIONAL": 1.25})
        return float(mults.get(bucket, 1.0))

    return 1.0
