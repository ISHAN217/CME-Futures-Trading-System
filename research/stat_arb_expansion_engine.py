"""
stat_arb_expansion_engine.py — Enhanced ES/NQ STAT_ARB quality scoring.

Builds on top of the existing stat_arb.py by adding:
  - Rolling beta (NQ vs ES) computed via rolling OLS
  - Rolling correlation and correlation stability
  - Beta stability score
  - Spread residual = NQ_ret − β × ES_ret
  - Spread z-score with hump-shaped quality
  - Relationship breakdown detection
  - Granular quality bucketing including DANGEROUS_EXTREME
  - Trade permission columns: statarb_allowed, statarb_direction, statarb_size_mult

Hump-shaped scoring philosophy:
  Small z-score  → weak signal (no strong dislocation)
  Moderate z     → good entry (clear dislocation, reversion expected)
  Extreme z      → relationship may be breaking down (not a safe entry)
  The optimal z peaks around 1.5–2.5; extremes (>3.5) are flagged DANGEROUS.

All computations are per-date cross-asset (wide pivot); output is merged back
to the original long-format df with NaN for ES rows (STAT_ARB is NQ-specific
in the current system).

Output columns (NQ rows — NaN on ES rows):
  sa_rolling_beta            rolling 20d beta of NQ vs ES
  sa_beta_stability          stability of rolling beta [0,1]
  sa_rolling_corr            rolling 20d correlation
  sa_corr_stability          stability of rolling correlation [0,1]
  sa_spread                  residual = NQ_ret − beta × ES_ret
  sa_spread_mean             rolling mean of spread
  sa_spread_vol              rolling std of spread
  sa_spread_zscore_exp       improved z-score (uses expanded spread engine)
  sa_spread_zscore_change    1-day change in spread z-score
  sa_spread_half_life        Ornstein-Uhlenbeck half-life estimate
  sa_hump_score              hump-shaped quality score [0,1]
  sa_reversion_score         strength of expected spread reversion [0,1]
  sa_breakout_score          breakout risk (spread diverging further) [0,1]
  sa_relationship_breakdown  score indicating relationship is failing [0,1]
  sa_quality_bucket_exp      WEAK/MODERATE/GOOD/STRONG/EXCEPTIONAL/DANGEROUS_EXTREME
  statarb_allowed_exp        1 if trade is permitted under expanded rules
  statarb_direction_exp      LONG / SHORT / FLAT
  statarb_size_mult_exp      position size multiplier [0, 1.5]
  statarb_reason_exp         human-readable decision reason
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
BETA_WINDOW        = 20    # rolling OLS window
CORR_WINDOW        = 20    # rolling correlation window
SPREAD_MEAN_WIN    = 30    # mean for z-score normalisation
SPREAD_STD_WIN     = 30    # std for z-score normalisation
STAB_WINDOW        = 10    # window for stability measurement
HL_WINDOW          = 30    # window for half-life estimation

HUMP_PEAK_Z        = 1.8   # z-score that gets maximum quality score
HUMP_WIDTH         = 0.70  # width parameter (wider = flatter hump)

DANGEROUS_Z_THR    = 3.5   # above this → DANGEROUS_EXTREME
STRONG_Z_THR       = 2.5
GOOD_Z_THR         = 1.5
MODERATE_Z_THR     = 1.0
WEAK_Z_THR         = 0.5

MIN_CORR_TO_TRADE  = 0.55  # minimum rolling correlation to allow trade
MAX_BREAKDOWN_TO_TRADE = 0.65  # max breakdown score to allow trade


# ─────────────────────────────────────────────────────────────────────────────
# Rolling OLS beta
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_beta(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta of y ~ x (no intercept term for speed)."""
    betas = pd.Series(np.nan, index=x.index)
    x_arr = x.values
    y_arr = y.values
    for i in range(window - 1, len(x_arr)):
        xi = x_arr[i - window + 1: i + 1]
        yi = y_arr[i - window + 1: i + 1]
        mask = np.isfinite(xi) & np.isfinite(yi)
        if mask.sum() < window // 2:
            continue
        xi_m, yi_m = xi[mask], yi[mask]
        denom = float(np.dot(xi_m, xi_m))
        if abs(denom) < 1e-12:
            continue
        betas.iloc[i] = float(np.dot(xi_m, yi_m)) / denom
    return betas


# ─────────────────────────────────────────────────────────────────────────────
# Half-life estimation (Ornstein-Uhlenbeck)
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_half_life(spread: pd.Series, window: int) -> pd.Series:
    """Estimate OU half-life from rolling regression of Δspread ~ spread(t-1)."""
    hl = pd.Series(np.nan, index=spread.index)
    s_arr = spread.values
    for i in range(window, len(s_arr)):
        s_win = s_arr[i - window: i]
        if not np.all(np.isfinite(s_win)):
            continue
        s_lag   = s_win[:-1]
        s_delta = np.diff(s_win)
        if np.std(s_lag) < 1e-12:
            continue
        try:
            slope, _, _, _, _ = scipy_stats.linregress(s_lag, s_delta)
            if slope < -1e-9:
                hl.iloc[i] = float(-np.log(2) / slope)
        except Exception:
            pass
    return hl.clip(1, 252)


# ─────────────────────────────────────────────────────────────────────────────
# Hump-shaped quality score
# ─────────────────────────────────────────────────────────────────────────────

def _hump_score(z: pd.Series) -> pd.Series:
    """Gaussian hump centred at HUMP_PEAK_Z; peaks at 1 when |z|==HUMP_PEAK_Z."""
    z_abs = z.abs().fillna(0)
    return np.exp(-0.5 * ((z_abs - HUMP_PEAK_Z) / HUMP_WIDTH) ** 2).clip(0, 1)


def _quality_bucket(z: pd.Series) -> pd.Series:
    z_abs = z.abs().fillna(0)
    buckets = pd.Series("NO_SIGNAL", index=z.index)
    buckets[z_abs > DANGEROUS_Z_THR] = "DANGEROUS_EXTREME"
    buckets[(z_abs > STRONG_Z_THR) & (z_abs <= DANGEROUS_Z_THR)] = "STRONG"
    buckets[(z_abs > GOOD_Z_THR)   & (z_abs <= STRONG_Z_THR)]    = "GOOD"
    buckets[(z_abs > MODERATE_Z_THR) & (z_abs <= GOOD_Z_THR)]    = "MODERATE"
    buckets[(z_abs > WEAK_Z_THR)   & (z_abs <= MODERATE_Z_THR)]  = "WEAK"
    return buckets


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_stat_arb_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add expanded STAT_ARB features to df.

    Requires: date, symbol, close columns.
    Returns df with new sa_* and statarb_*_exp columns added.
    NQ rows carry the computed features; ES rows get NaN.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Pivot to wide: one row per date, ES/NQ close as columns
    wide = (df.pivot_table(index="date", columns="symbol", values="close", aggfunc="first")
              .sort_index())

    if "ES" not in wide.columns or "NQ" not in wide.columns:
        logger.warning("StatArbExpansion: ES or NQ not found — skipping")
        return df

    es_cl = wide["ES"].ffill()
    nq_cl = wide["NQ"].ffill()

    # Daily log-returns (cleaner for beta/spread)
    es_ret = np.log(es_cl / es_cl.shift(1)).fillna(0)
    nq_ret = np.log(nq_cl / nq_cl.shift(1)).fillna(0)

    # ── Rolling beta ──────────────────────────────────────────────────────────
    beta = _rolling_beta(es_ret, nq_ret, BETA_WINDOW)
    beta = beta.clip(0.5, 3.0)   # reasonable bounds for NQ/ES beta

    # Beta stability: low std of rolling beta → stable
    beta_std  = beta.rolling(STAB_WINDOW, min_periods=3).std().fillna(0)
    beta_stab = (1 - (beta_std / 0.30).clip(0, 1)).fillna(0.5)

    # ── Rolling correlation ───────────────────────────────────────────────────
    corr  = es_ret.rolling(CORR_WINDOW, min_periods=10).corr(nq_ret).fillna(0.8)
    corr  = corr.clip(-1, 1)

    corr_std  = corr.rolling(STAB_WINDOW, min_periods=3).std().fillna(0)
    corr_stab = (1 - (corr_std / 0.20).clip(0, 1)).fillna(0.5)

    # ── Spread (residual of NQ ~ β × ES) ─────────────────────────────────────
    spread      = nq_ret - beta * es_ret
    spread_mean = spread.rolling(SPREAD_MEAN_WIN, min_periods=10).mean().fillna(0)
    spread_std  = spread.rolling(SPREAD_STD_WIN, min_periods=10).std().replace(0, np.nan)
    spread_std  = spread_std.fillna(spread_std.median() or 0.001)

    spread_z   = ((spread - spread_mean) / spread_std).clip(-5, 5)
    spread_dz  = spread_z.diff(1).fillna(0)

    # ── Half-life ─────────────────────────────────────────────────────────────
    half_life  = _rolling_half_life(spread, HL_WINDOW)

    # ── Hump-shaped quality score ─────────────────────────────────────────────
    hump_sc    = _hump_score(spread_z)
    bucket     = _quality_bucket(spread_z)

    # ── Reversion vs breakout scores ─────────────────────────────────────────
    # Reversion: spread at extreme, spread_dz converging back (z moving toward 0)
    reverting    = (spread_z * spread_dz < 0).astype(float)   # z and Δz have opposite signs
    reversion_sc = hump_sc * reverting

    # Breakout: spread moving further out (z and Δz same sign)
    diverging    = (spread_z * spread_dz > 0).astype(float)
    breakout_sc  = spread_z.abs() / 5.0 * diverging

    # ── Relationship breakdown score ──────────────────────────────────────────
    low_corr     = ((1 - corr.clip(0, 1)) * 0.5).clip(0, 1)   # high when corr low
    unstable_beta= (1 - beta_stab) * 0.5
    extreme_z    = (spread_z.abs() / DANGEROUS_Z_THR).clip(0, 1) * 0.3
    breakdown_sc = (low_corr + unstable_beta + extreme_z).clip(0, 1)

    # ── Trade permissions ─────────────────────────────────────────────────────
    # Allowed when: correlation OK, no breakdown, not DANGEROUS_EXTREME
    allowed = (
        (corr >= MIN_CORR_TO_TRADE) &
        (breakdown_sc < MAX_BREAKDOWN_TO_TRADE) &
        (bucket != "DANGEROUS_EXTREME") &
        (spread_z.abs() > WEAK_Z_THR)
    ).astype(int)

    direction = pd.Series("FLAT", index=wide.index)
    direction[allowed.astype(bool) & (spread_z < -WEAK_Z_THR)] = "LONG"   # NQ cheap vs ES
    direction[allowed.astype(bool) & (spread_z >  WEAK_Z_THR)] = "SHORT"  # NQ rich vs ES

    # Size multiplier: scaled by hump score; boost when reversion active
    size_mult = (hump_sc * (1 + 0.3 * reverting)).clip(0, 1.5)
    size_mult[~allowed.astype(bool)] = 0.0

    # Reason
    reasons = []
    for i in range(len(wide)):
        parts = []
        z_val = float(spread_z.iloc[i])
        b_val = bucket.iloc[i]
        parts.append(f"z={z_val:.2f}")
        parts.append(f"q={b_val}")
        if float(breakdown_sc.iloc[i]) >= MAX_BREAKDOWN_TO_TRADE:
            parts.append("breakdown_block")
        if float(corr.iloc[i]) < MIN_CORR_TO_TRADE:
            parts.append("low_corr_block")
        if reverting.iloc[i]:
            parts.append("reverting")
        reasons.append("|".join(parts))

    # ── Assemble wide result and merge back ───────────────────────────────────
    wide_out = pd.DataFrame({
        "sa_rolling_beta":        beta.round(4),
        "sa_beta_stability":      beta_stab.round(4),
        "sa_rolling_corr":        corr.round(4),
        "sa_corr_stability":      corr_stab.round(4),
        "sa_spread":              spread.round(6),
        "sa_spread_mean":         spread_mean.round(6),
        "sa_spread_vol":          spread_std.round(6),
        "sa_spread_zscore_exp":   spread_z.round(4),
        "sa_spread_zscore_change":spread_dz.round(4),
        "sa_spread_half_life":    half_life.round(1),
        "sa_hump_score":          hump_sc.round(4),
        "sa_reversion_score":     reversion_sc.round(4),
        "sa_breakout_score":      breakout_sc.clip(0, 1).round(4),
        "sa_relationship_breakdown": breakdown_sc.round(4),
        "sa_quality_bucket_exp":  bucket,
        "statarb_allowed_exp":    allowed,
        "statarb_direction_exp":  direction,
        "statarb_size_mult_exp":  size_mult.round(4),
        "statarb_reason_exp":     pd.Series(reasons, index=wide.index),
    }, index=wide.index)

    # Merge back — NQ rows only; ES stays NaN
    wide_out["_date"] = wide_out.index
    df = df.merge(wide_out.reset_index(), on="date", how="left")
    # Blank out ES rows for direction/allowed (STAT_ARB is NQ-only)
    es_mask = df["symbol"] == "ES"
    for col in ["statarb_allowed_exp", "statarb_direction_exp",
                "statarb_size_mult_exp", "statarb_reason_exp",
                "sa_quality_bucket_exp"]:
        if col in df.columns:
            df.loc[es_mask, col] = np.nan if col != "statarb_direction_exp" else "FLAT"

    # Drop helper column
    df = df.drop(columns=["_date"], errors="ignore")

    _log_summary(df)
    return df


def _log_summary(df: pd.DataFrame) -> None:
    nq = df[df["symbol"] == "NQ"]
    if "sa_quality_bucket_exp" not in nq.columns:
        return
    bkt = nq["sa_quality_bucket_exp"].value_counts()
    n   = max(len(nq), 1)
    logger.info(
        "StatArbExpansion NQ — EXCEP/STRONG/GOOD/MOD/WEAK/DANGER: "
        "%d/%d/%d/%d/%d/%d  avg_z=%.2f  avg_hump=%.3f  "
        "allowed=%d (%.0f%%)  breakdown_high=%d",
        bkt.get("EXCEPTIONAL", 0), bkt.get("STRONG", 0),
        bkt.get("GOOD", 0), bkt.get("MODERATE", 0),
        bkt.get("WEAK", 0), bkt.get("DANGEROUS_EXTREME", 0),
        nq["sa_spread_zscore_exp"].mean() if "sa_spread_zscore_exp" in nq.columns else 0,
        nq["sa_hump_score"].mean() if "sa_hump_score" in nq.columns else 0,
        nq["statarb_allowed_exp"].fillna(0).sum(),
        100 * nq["statarb_allowed_exp"].fillna(0).sum() / n,
        (nq["sa_relationship_breakdown"].fillna(0) >= MAX_BREAKDOWN_TO_TRADE).sum(),
    )
