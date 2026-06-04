"""
cwt_chop_filter.py — Continuous Wavelet Transform chop classification. [NEW]

Computes per-symbol, rolling-window CWT features for classifying market
regime into: TREND_COHERENT, RANGE_OSCILLATORY, CHAOTIC_NOISE, COMPRESSION,
BREAKOUT_EXPANSION, or UNCLEAR.

Works on past data only (causal / no lookahead).
Gracefully degrades if scipy/pywt are unavailable.
Never crashes the backtest.

Output columns added to df:
    cwt_total_energy       float, default 0.0
    cwt_high_energy        float, default 0.0
    cwt_mid_energy         float, default 0.0
    cwt_slow_energy        float, default 0.0
    cwt_high_energy_ratio  float, default 1/3
    cwt_mid_energy_ratio   float, default 1/3
    cwt_slow_energy_ratio  float, default 1/3
    cwt_entropy            float 0→1, default 0.5
    cwt_energy_z           float, default 0.0
    cwt_compression_flag   int 0/1, default 0
    cwt_expansion_flag     int 0/1, default 0
    cwt_chop_label         str, default "CWT_UNCLEAR"
    cwt_trade_allowed      int 0/1, default 1
    cwt_size_multiplier    float, default 1.0
    cwt_confidence         float 0→1, default 0.3
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ── Backend detection ─────────────────────────────────────────────────────────
try:
    from scipy.signal import cwt as scipy_cwt, ricker
    _CWT_BACKEND = "scipy"
except ImportError:
    try:
        import pywt
        _CWT_BACKEND = "pywt"
    except ImportError:
        _CWT_BACKEND = "none"
        logger.warning("CWT: neither scipy nor pywt available — returning neutral columns")

_SCALES = np.array(config.CWT_SCALES, dtype=float)
_N_SCALES = len(_SCALES)

# Pre-classify scales into bands
_HIGH_MASK = _SCALES <= config.CWT_HIGH_SCALE_MAX
_MID_MASK  = (_SCALES > config.CWT_HIGH_SCALE_MAX) & (_SCALES <= config.CWT_MID_SCALE_MAX)
_SLOW_MASK = _SCALES > config.CWT_MID_SCALE_MAX


def _neutral_row() -> dict:
    """Return default CWT values for rows with insufficient data."""
    return {
        "cwt_total_energy":      0.0,
        "cwt_high_energy":       0.0,
        "cwt_mid_energy":        0.0,
        "cwt_slow_energy":       0.0,
        "cwt_high_energy_ratio": 1.0 / 3,
        "cwt_mid_energy_ratio":  1.0 / 3,
        "cwt_slow_energy_ratio": 1.0 / 3,
        "cwt_entropy":           0.5,
        "cwt_energy_z":          0.0,
        "cwt_compression_flag":  0,
        "cwt_expansion_flag":    0,
        "cwt_chop_label":        "CWT_UNCLEAR",
        "cwt_trade_allowed":     1,
        "cwt_size_multiplier":   1.0,
        "cwt_confidence":        0.0,
    }


def _fallback_row() -> dict:
    """Fallback when backend=none or CWT computation fails."""
    row = _neutral_row()
    row["cwt_size_multiplier"] = 1.0
    row["cwt_trade_allowed"]   = 1
    row["cwt_confidence"]      = 0.0
    return row


def _compute_scale_energies_scipy(window: np.ndarray) -> np.ndarray:
    """
    Compute per-scale energy using scipy CWT (Ricker / Mexican hat wavelet).
    Returns array of shape (n_scales,) with mean power per scale.
    """
    w = window - window.mean()
    coeff = scipy_cwt(w, ricker, _SCALES)   # shape: (n_scales, n_times)
    power = coeff ** 2
    return power.mean(axis=1)               # (n_scales,)


def _compute_scale_energies_pywt(window: np.ndarray) -> np.ndarray:
    """
    Compute per-scale energy using pywt CWT (Morlet wavelet).
    Returns array of shape (n_scales,) with mean power per scale.
    """
    w = window - window.mean()
    energies = np.zeros(_N_SCALES)
    for i, s in enumerate(_SCALES):
        coeff, _ = pywt.cwt(w, [s], "morl")
        energies[i] = np.mean(np.abs(coeff[0]) ** 2)
    return energies


def _compute_scale_energies(window: np.ndarray) -> np.ndarray:
    if _CWT_BACKEND == "scipy":
        return _compute_scale_energies_scipy(window)
    elif _CWT_BACKEND == "pywt":
        return _compute_scale_energies_pywt(window)
    else:
        return np.zeros(_N_SCALES)


def _energy_to_features(scale_energies: np.ndarray) -> dict:
    """Convert per-scale energies to aggregate CWT features."""
    total = float(scale_energies.sum())

    if total < 1e-30:
        return {
            "cwt_total_energy":      0.0,
            "cwt_high_energy":       0.0,
            "cwt_mid_energy":        0.0,
            "cwt_slow_energy":       0.0,
            "cwt_high_energy_ratio": 1.0 / 3,
            "cwt_mid_energy_ratio":  1.0 / 3,
            "cwt_slow_energy_ratio": 1.0 / 3,
            "cwt_entropy":           0.5,
        }

    high = float(scale_energies[_HIGH_MASK].sum())
    mid  = float(scale_energies[_MID_MASK].sum())
    slow = float(scale_energies[_SLOW_MASK].sum())

    p = scale_energies / total   # normalized probabilities
    entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(_N_SCALES))
    entropy = float(np.clip(entropy, 0.0, 1.0))

    return {
        "cwt_total_energy":      total,
        "cwt_high_energy":       high,
        "cwt_mid_energy":        mid,
        "cwt_slow_energy":       slow,
        "cwt_high_energy_ratio": high / total,
        "cwt_mid_energy_ratio":  mid  / total,
        "cwt_slow_energy_ratio": slow / total,
        "cwt_entropy":           entropy,
    }


def _assign_label(
    feats: dict,
    energy_z: float,
    recent_compression: bool,
) -> dict:
    """
    Assign cwt_chop_label and related fields.
    Priority order: COMPRESSION > BREAKOUT_EXPANSION > TREND_COHERENT
                    > CHAOTIC_NOISE > RANGE_OSCILLATORY > UNCLEAR
    """
    entropy          = feats["cwt_entropy"]
    slow_ratio       = feats["cwt_slow_energy_ratio"]
    high_ratio       = feats["cwt_high_energy_ratio"]
    mid_ratio        = feats["cwt_mid_energy_ratio"]

    # 1. COMPRESSION
    if energy_z <= config.CWT_ENERGY_COMPRESSION_Z:
        conf = min(1.0, abs(energy_z - config.CWT_ENERGY_COMPRESSION_Z) * 2)
        return {
            "cwt_chop_label":      "CWT_COMPRESSION",
            "cwt_trade_allowed":   0,
            "cwt_size_multiplier": config.CWT_COMPRESSION_SIZE_MULT,
            "cwt_confidence":      conf,
            "cwt_compression_flag": 1,
            "cwt_expansion_flag":  0,
        }

    # 2. BREAKOUT_EXPANSION
    if recent_compression and energy_z >= config.CWT_ENERGY_EXPANSION_Z:
        conf = min(1.0, energy_z / config.CWT_ENERGY_EXPANSION_Z * 0.8)
        return {
            "cwt_chop_label":      "CWT_BREAKOUT_EXPANSION",
            "cwt_trade_allowed":   0,
            "cwt_size_multiplier": 0.25,
            "cwt_confidence":      conf,
            "cwt_compression_flag": 0,
            "cwt_expansion_flag":  1,
        }

    # 3. TREND_COHERENT
    if slow_ratio >= 0.50 and entropy <= config.CWT_ENTROPY_HIGH:
        return {
            "cwt_chop_label":      "CWT_TREND_COHERENT",
            "cwt_trade_allowed":   1,
            "cwt_size_multiplier": 1.0,
            "cwt_confidence":      slow_ratio,
            "cwt_compression_flag": 0,
            "cwt_expansion_flag":  0,
        }

    # 4. CHAOTIC_NOISE
    if entropy > config.CWT_ENTROPY_HIGH and high_ratio >= 0.30:
        conf = min(1.0, (entropy - config.CWT_ENTROPY_HIGH) * 5)
        return {
            "cwt_chop_label":      "CWT_CHAOTIC_NOISE",
            "cwt_trade_allowed":   0,
            "cwt_size_multiplier": config.CWT_CHAOTIC_SIZE_MULT,
            "cwt_confidence":      conf,
            "cwt_compression_flag": 0,
            "cwt_expansion_flag":  0,
        }

    # 5. RANGE_OSCILLATORY
    if entropy <= config.CWT_ENTROPY_HIGH and (high_ratio + mid_ratio) >= 0.50:
        return {
            "cwt_chop_label":      "CWT_RANGE_OSCILLATORY",
            "cwt_trade_allowed":   1,
            "cwt_size_multiplier": config.CWT_RANGE_SIZE_MULT,
            "cwt_confidence":      1.0 - entropy,
            "cwt_compression_flag": 0,
            "cwt_expansion_flag":  0,
        }

    # 6. UNCLEAR (default)
    return {
        "cwt_chop_label":      "CWT_UNCLEAR",
        "cwt_trade_allowed":   1,
        "cwt_size_multiplier": config.CWT_UNCLEAR_SIZE_MULT,
        "cwt_confidence":      0.30,
        "cwt_compression_flag": 0,
        "cwt_expansion_flag":  0,
    }


def _process_symbol(sym_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a single symbol's DataFrame, adding CWT feature columns.
    Input rows must be sorted by date ascending.
    """
    sym_df = sym_df.sort_values("date").copy()
    n = len(sym_df)

    returns = sym_df["ret_1d"].to_numpy(dtype=float)

    # Fill NaN returns with 0 (safe for CWT)
    returns = np.where(np.isfinite(returns), returns, 0.0)

    window_size  = config.CWT_WINDOW
    min_periods  = config.CWT_MIN_PERIODS

    # ── Pass 1: compute per-row energy features ───────────────────────────────
    feature_rows = []
    total_energies = np.zeros(n)

    for i in range(n):
        if _CWT_BACKEND == "none" or i < min_periods:
            feature_rows.append(None)
            continue

        start = max(0, i - window_size + 1)
        window = returns[start : i + 1]

        if len(window) < min_periods:
            feature_rows.append(None)
            continue

        try:
            scale_energies = _compute_scale_energies(window)
            feats = _energy_to_features(scale_energies)
            feature_rows.append(feats)
            total_energies[i] = feats["cwt_total_energy"]
        except Exception as exc:
            logger.debug("CWT computation error at index %d: %s", i, exc)
            feature_rows.append(None)

    # ── Pass 2: compute rolling z-score of total energy ──────────────────────
    energy_series = pd.Series(total_energies)
    roll_mean = energy_series.rolling(120, min_periods=30).mean()
    roll_std  = energy_series.rolling(120, min_periods=30).std()
    energy_z_arr = np.where(
        (roll_std > 1e-12) & roll_std.notna() & roll_mean.notna(),
        (energy_series - roll_mean) / roll_std,
        0.0,
    )

    # ── Pass 3: assign labels ─────────────────────────────────────────────────
    compression_flags = np.zeros(n, dtype=int)

    result_rows = []
    for i in range(n):
        energy_z = float(energy_z_arr[i])

        if feature_rows[i] is None:
            row_out = _fallback_row()
            row_out["cwt_energy_z"] = energy_z
            result_rows.append(row_out)
            continue

        feats = feature_rows[i]

        # Recent compression: any of last 5 bars had compression_flag == 1
        recent_comp = bool(compression_flags[max(0, i - 5) : i].any())

        label_feats = _assign_label(feats, energy_z, recent_comp)
        compression_flags[i] = label_feats["cwt_compression_flag"]

        row_out = {**feats, **label_feats, "cwt_energy_z": energy_z}
        result_rows.append(row_out)

    result_df = pd.DataFrame(result_rows, index=sym_df.index)
    sym_df = pd.concat([sym_df, result_df], axis=1)
    return sym_df


def compute_cwt_chop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Processes df per symbol, adds CWT feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ["date", "symbol", "ret_1d"]

    Returns
    -------
    pd.DataFrame with all original columns plus CWT columns.
    All CWT columns are filled — no NaNs.
    """
    if _CWT_BACKEND == "none":
        logger.warning("CWT backend unavailable — adding neutral CWT columns")
        neutral = _neutral_row()
        for col, val in neutral.items():
            if col not in df.columns:
                df[col] = val
        df["cwt_energy_z"] = 0.0
        return df

    original_index = df.index
    df = df.reset_index(drop=True)

    symbols = df["symbol"].unique()
    processed_parts = []

    for sym in symbols:
        sym_mask = df["symbol"] == sym
        sym_df   = df[sym_mask].copy()
        try:
            sym_df = _process_symbol(sym_df)
        except Exception as exc:
            logger.error("CWT failed for symbol %s: %s — using neutral values", sym, exc)
            neutral = _neutral_row()
            neutral["cwt_energy_z"] = 0.0
            for col, val in neutral.items():
                sym_df[col] = val
        processed_parts.append(sym_df)

    df_out = pd.concat(processed_parts).sort_values(["date", "symbol"])

    # Ensure all expected columns exist and have no NaNs
    cwt_cols_defaults = {
        "cwt_total_energy":      0.0,
        "cwt_high_energy":       0.0,
        "cwt_mid_energy":        0.0,
        "cwt_slow_energy":       0.0,
        "cwt_high_energy_ratio": 1.0 / 3,
        "cwt_mid_energy_ratio":  1.0 / 3,
        "cwt_slow_energy_ratio": 1.0 / 3,
        "cwt_entropy":           0.5,
        "cwt_energy_z":          0.0,
        "cwt_compression_flag":  0,
        "cwt_expansion_flag":    0,
        "cwt_chop_label":        "CWT_UNCLEAR",
        "cwt_trade_allowed":     1,
        "cwt_size_multiplier":   1.0,
        "cwt_confidence":        0.3,
    }
    for col, default in cwt_cols_defaults.items():
        if col not in df_out.columns:
            df_out[col] = default
        else:
            df_out[col] = df_out[col].fillna(default)

    # Restore original order
    df_out = df_out.reset_index(drop=True)

    n_labeled = (df_out["cwt_chop_label"] != "CWT_UNCLEAR").sum()
    label_counts = df_out["cwt_chop_label"].value_counts().to_dict()
    logger.info(
        "CWT chop filter complete — backend=%s  rows=%d  labeled(non-UNCLEAR)=%d  dist=%s",
        _CWT_BACKEND, len(df_out), n_labeled, label_counts,
    )

    return df_out
