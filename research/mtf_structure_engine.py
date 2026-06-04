"""
mtf_structure_engine.py — Phase 8: Multi-Timeframe Structure Features
======================================================================

Public entry point:
  compute_mtf_structure(daily_df, bars_by_tf) -> pd.DataFrame

Input:
  daily_df     — daily-bar DataFrame with date, symbol, close (from existing system)
  bars_by_tf   — dict {"1H": df_1h, "4H": df_4h, "12H": df_12h, "24H": df_24h}
                 Each is a parquet-loaded DataFrame with timestamp, open, high, low,
                 close, volume, symbol columns.

Output (one row per trading day per symbol):
  All structure labels, bias scores, MTF composite signals.

Design principles:
  - STRICTLY no lookahead: only completed bars inform each day's label
  - Higher-TF bar is "completed" when its close timestamp <= start of current bar
  - Swing detection: lookback-only (left pivots, N=5 bars)
  - Forward-fill completed structure labels to daily resolution
  - Weights: 24H=0.40  12H=0.30  4H=0.20  1H=0.10
"""

import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SWING_LOOKBACK = 5          # bars for swing high/low pivot detection
ROLLING_WINDOW = 20         # bars for rolling structure window
MIN_BARS_FOR_STRUCTURE = 10 # minimum bars required before classifying

TF_WEIGHTS = {"24H": 0.40, "12H": 0.30, "4H": 0.20, "1H": 0.10}
TF_ORDER   = ["24H", "12H", "4H", "1H"]

BIAS_BULLISH_THRESH  =  0.50
BIAS_BEARISH_THRESH  = -0.50

STRUCTURE_LABELS = ["UP_STRUCTURE","DOWN_STRUCTURE","RANGE_STRUCTURE",
                    "TRANSITION_STRUCTURE","CHAOTIC_STRUCTURE","INSUFFICIENT_DATA"]

BIAS_MAP = {
    "UP_STRUCTURE":         +1.0,
    "DOWN_STRUCTURE":       -1.0,
    "RANGE_STRUCTURE":       0.0,
    "TRANSITION_STRUCTURE":  0.0,
    "CHAOTIC_STRUCTURE":     0.0,
    "INSUFFICIENT_DATA":     0.0,
}

CONFIDENCE_PENALTY = {
    "UP_STRUCTURE":         0.0,
    "DOWN_STRUCTURE":       0.0,
    "RANGE_STRUCTURE":      0.15,
    "TRANSITION_STRUCTURE": 0.30,
    "CHAOTIC_STRUCTURE":    0.50,
    "INSUFFICIENT_DATA":    1.00,
}


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURE COMPUTATION PER TIMEFRAME
# ══════════════════════════════════════════════════════════════════════════════

def _compute_swing_points(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.DataFrame:
    """
    Lookahead-free swing high/low detection.
    Swing high at bar i: high[i] is max of last `lookback` bars (incl. i).
    Swing low  at bar i: low[i]  is min of last `lookback` bars (incl. i).
    This uses only past data — no future bars.
    """
    df = df.copy()
    df["rolling_high"] = df["high"].rolling(lookback, min_periods=lookback).max()
    df["rolling_low"]  = df["low"].rolling(lookback, min_periods=lookback).min()

    # A local high is confirmed when high[i] == rolling_max over [i-N+1 .. i]
    # i.e., high[i] is the peak of the last N bars
    df["swing_high_raw"] = (df["high"] == df["rolling_high"]).astype(float)
    df["swing_low_raw"]  = (df["low"]  == df["rolling_low"]).astype(float)

    # Extract swing high/low price series (price only when confirmed, NaN otherwise)
    df["swing_high_price"] = np.where(df["swing_high_raw"]==1, df["high"],  np.nan)
    df["swing_low_price"]  = np.where(df["swing_low_raw"]==1,  df["low"],   np.nan)

    # Forward-fill to get "most recent confirmed swing"
    df["last_swing_high"]  = df["swing_high_price"].ffill()
    df["last_swing_low"]   = df["swing_low_price"].ffill()

    # Prior swing (second-to-last confirmed)
    df["prior_swing_high"] = (
        df["swing_high_price"]
        .shift(1)
        .where(df["swing_high_raw"].shift(1)==1)
        .ffill()
    )
    df["prior_swing_low"]  = (
        df["swing_low_price"]
        .shift(1)
        .where(df["swing_low_raw"].shift(1)==1)
        .ffill()
    )

    # Structure flags
    df["higher_high_flag"]  = (df["last_swing_high"] > df["prior_swing_high"]).astype(float)
    df["higher_low_flag"]   = (df["last_swing_low"]  > df["prior_swing_low"]).astype(float)
    df["lower_high_flag"]   = (df["last_swing_high"] < df["prior_swing_high"]).astype(float)
    df["lower_low_flag"]    = (df["last_swing_low"]  < df["prior_swing_low"]).astype(float)

    return df


def _compute_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-bar structure features for one symbol/timeframe."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    if len(df) < MIN_BARS_FOR_STRUCTURE:
        df["structure_label"] = "INSUFFICIENT_DATA"
        df["structure_strength_score"]  = 0.0
        df["structure_direction_score"] = 0.0
        df["close_location_value"]      = 0.5
        df["range_position"]            = 0.5
        df["break_of_structure_up"]     = 0
        df["break_of_structure_down"]   = 0
        df["range_breakout_up"]         = 0
        df["range_breakout_down"]       = 0
        df["failed_breakout_up"]        = 0
        df["failed_breakdown_down"]     = 0
        return df

    # Swing points
    df = _compute_swing_points(df)

    # Rolling range
    df["range_high"] = df["high"].rolling(ROLLING_WINDOW, min_periods=5).max()
    df["range_low"]  = df["low"].rolling(ROLLING_WINDOW,  min_periods=5).min()
    df["range_size"] = (df["range_high"] - df["range_low"]).replace(0, np.nan)

    # Close location value  (0 = at low, 1 = at high)
    df["close_location_value"] = (
        (df["close"] - df["range_low"]) / df["range_size"]
    ).clip(0, 1).fillna(0.5)

    # Range position  (where close sits in rolling range)
    df["range_position"] = df["close_location_value"]

    # Break of structure
    # BoS up: close crosses above prior swing high
    # BoS down: close crosses below prior swing low
    df["break_of_structure_up"]   = (
        (df["close"] > df["prior_swing_high"].shift(1)) &
        (df["close"].shift(1) <= df["prior_swing_high"].shift(1))
    ).astype(int)
    df["break_of_structure_down"] = (
        (df["close"] < df["prior_swing_low"].shift(1)) &
        (df["close"].shift(1) >= df["prior_swing_low"].shift(1))
    ).astype(int)

    # Range breakout / breakdown  (close outside rolling range)
    rolling_h_prev = df["range_high"].shift(1)
    rolling_l_prev = df["range_low"].shift(1)
    df["range_breakout_up"]    = (df["close"] > rolling_h_prev).astype(int)
    df["range_breakout_down"]  = (df["close"] < rolling_l_prev).astype(int)

    # Failed breakout: breakout bar followed by close back inside range
    df["failed_breakout_up"]    = (
        df["range_breakout_up"].shift(1).fillna(0).astype(bool) &
        (df["close"] < rolling_h_prev)
    ).astype(int)
    df["failed_breakdown_down"] = (
        df["range_breakout_down"].shift(1).fillna(0).astype(bool) &
        (df["close"] > rolling_l_prev)
    ).astype(int)

    # ── Structure classification ──────────────────────────────────────────────
    df = _classify_structure(df)

    return df


def _classify_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each bar into a structure label."""
    df = df.copy()

    # Strength score: weighted combination of swing patterns
    hh = df["higher_high_flag"].fillna(0)
    hl = df["higher_low_flag"].fillna(0)
    lh = df["lower_high_flag"].fillna(0)
    ll = df["lower_low_flag"].fillna(0)
    bos_up   = df["break_of_structure_up"].fillna(0)
    bos_dn   = df["break_of_structure_down"].fillna(0)
    brk_up   = df["range_breakout_up"].fillna(0)
    brk_dn   = df["range_breakout_down"].fillna(0)
    fail_up  = df["failed_breakout_up"].fillna(0)
    fail_dn  = df["failed_breakdown_down"].fillna(0)
    clv      = df["close_location_value"].fillna(0.5)

    # Direction score  (+1 = pure up, -1 = pure down)
    up_score   = 0.35*hh + 0.25*hl + 0.20*bos_up  + 0.10*brk_up  + 0.10*clv
    down_score = 0.35*ll + 0.25*lh + 0.20*bos_dn  + 0.10*brk_dn  + 0.10*(1-clv)
    chaos_score= 0.5*fail_up + 0.5*fail_dn

    direction_score = up_score - down_score   # range -1 to +1
    df["structure_direction_score"] = direction_score.clip(-1, 1).fillna(0)

    # Strength = magnitude of directional commitment
    df["structure_strength_score"] = (
        0.40*np.abs(direction_score) +
        0.20*(hh*hl + ll*lh) +           # both conditions agree
        0.20*(bos_up + bos_dn) +
        0.20*(1 - chaos_score)
    ).clip(0, 1).fillna(0)

    # Need at least MIN_BARS_FOR_STRUCTURE non-null swing data
    valid_mask = df["prior_swing_high"].notna() & df["prior_swing_low"].notna()

    labels = pd.Series("INSUFFICIENT_DATA", index=df.index)

    # UP_STRUCTURE: HH + HL simultaneously, above rolling midpoint
    up_structure = (
        valid_mask &
        (direction_score > 0.20) &
        (hh + hl > 0) &
        (clv > 0.50)
    )
    # DOWN_STRUCTURE: LH + LL simultaneously, below rolling midpoint
    down_structure = (
        valid_mask &
        (direction_score < -0.20) &
        (lh + ll > 0) &
        (clv < 0.50)
    )
    # RANGE_STRUCTURE: neither clearly trending, low direction score
    range_structure = (
        valid_mask &
        (direction_score.abs() <= 0.15) &
        (chaos_score < 0.3)
    )
    # CHAOTIC_STRUCTURE: high failure rate, alternating breakouts
    chaotic = (
        valid_mask &
        (chaos_score >= 0.40)
    )
    # TRANSITION_STRUCTURE: direction changing (was up now showing LH, or vice versa)
    transition = (
        valid_mask &
        (direction_score.abs() <= 0.30) &
        (chaos_score < 0.40) &
        ~range_structure
    )

    # Apply in priority order
    labels[transition]    = "TRANSITION_STRUCTURE"
    labels[range_structure] = "RANGE_STRUCTURE"
    labels[down_structure]  = "DOWN_STRUCTURE"
    labels[up_structure]    = "UP_STRUCTURE"
    labels[chaotic]         = "CHAOTIC_STRUCTURE"

    df["structure_label"] = labels
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DAILY AGGREGATION — MAP TF STRUCTURE TO TRADING DAYS
# ══════════════════════════════════════════════════════════════════════════════

def _get_last_completed_tf_bar(tf_df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.Series:
    """
    Return the last bar whose timestamp < cutoff_ts (i.e., bar closed before cutoff).
    This avoids lookahead — we only see completed bars.
    """
    completed = tf_df[tf_df["timestamp"] < cutoff_ts]
    if len(completed) == 0:
        return None
    return completed.iloc[-1]


def _structure_features_from_bar(bar: pd.Series | None, tf: str) -> dict:
    """Extract structure features from a completed TF bar."""
    prefix = f"tf_{tf.lower()}_"
    if bar is None:
        return {
            f"structure_{tf.lower()}_label":           "INSUFFICIENT_DATA",
            f"structure_{tf.lower()}_direction_score": 0.0,
            f"structure_{tf.lower()}_strength_score":  0.0,
            f"structure_{tf.lower()}_clv":             0.5,
            f"structure_{tf.lower()}_range_position":  0.5,
            f"bias_{tf.lower()}":                      0,
        }
    label = bar.get("structure_label", "INSUFFICIENT_DATA")
    return {
        f"structure_{tf.lower()}_label":           label,
        f"structure_{tf.lower()}_direction_score": float(bar.get("structure_direction_score", 0)),
        f"structure_{tf.lower()}_strength_score":  float(bar.get("structure_strength_score",  0)),
        f"structure_{tf.lower()}_clv":             float(bar.get("close_location_value", 0.5)),
        f"structure_{tf.lower()}_range_position":  float(bar.get("range_position", 0.5)),
        f"bias_{tf.lower()}":                      BIAS_MAP.get(label, 0.0),
    }


def compute_daily_mtf_for_symbol(
    sym: str,
    daily_dates: list,
    tf_structures: dict,     # {"1H": df_with_structure, "4H": ..., "12H": ..., "24H": ...}
    daily_open_time: str = "00:00:00+00:00",   # assume day starts at UTC midnight
) -> pd.DataFrame:
    """
    For each trading day, find the last completed bar at each timeframe
    and compute MTF features.
    Returns one row per trading date with all MTF labels and scores.
    """
    rows = []
    for d in sorted(daily_dates):
        # Cutoff = start of trading day d (UTC midnight)
        # Any bar whose timestamp < cutoff is "completed" and usable
        cutoff = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta("1D")
        # Use end-of-day cutoff: bars up to and including close of day d
        # For daily production use: cutoff = next day start = d + 1 day
        # This means we see all bars that closed ON day d

        row = {"date": d, "symbol": sym}
        weighted_bias = 0.0
        total_confidence = 0.0
        chaos_count      = 0
        insuf_count      = 0

        for tf in TF_ORDER:
            tf_df = tf_structures.get(tf)
            if tf_df is None:
                row.update(_structure_features_from_bar(None, tf))
                insuf_count += 1
                continue

            last_bar = _get_last_completed_tf_bar(tf_df, cutoff)
            row.update(_structure_features_from_bar(last_bar, tf))

            label   = row.get(f"structure_{tf.lower()}_label", "INSUFFICIENT_DATA")
            weight  = TF_WEIGHTS[tf]
            bias    = BIAS_MAP.get(label, 0.0)
            penalty = CONFIDENCE_PENALTY.get(label, 0.5)

            weighted_bias      += weight * bias
            total_confidence   += weight * (1.0 - penalty)

            if label == "CHAOTIC_STRUCTURE":
                chaos_count += 1
            if label == "INSUFFICIENT_DATA":
                insuf_count += 1

        # MTF composite scores
        alignment_score = _compute_alignment_score(row)
        conflict_score  = _compute_conflict_score(row)

        bull_bias = max(0, weighted_bias)
        bear_bias = max(0, -weighted_bias)

        row["mtf_weighted_bias_score"]   = round(weighted_bias, 4)
        row["mtf_alignment_score"]       = round(alignment_score, 4)
        row["mtf_conflict_score"]        = round(conflict_score, 4)
        row["mtf_bullish_bias_score"]    = round(bull_bias, 4)
        row["mtf_bearish_bias_score"]    = round(bear_bias, 4)
        row["mtf_bias_confidence"]       = round(total_confidence, 4)
        row["mtf_chaos_count"]           = chaos_count
        row["mtf_insufficient_count"]    = insuf_count

        # Final bias
        if chaos_count >= 2 or conflict_score > 0.70:
            final_bias  = "NO_TRADE"
            reason      = "HIGH_CONFLICT_OR_CHAOS"
        elif insuf_count >= 3:
            final_bias  = "NO_TRADE"
            reason      = "INSUFFICIENT_DATA"
        elif weighted_bias >= BIAS_BULLISH_THRESH:
            final_bias  = "BULLISH"
            reason      = "WEIGHTED_BIAS_ABOVE_THRESHOLD"
        elif weighted_bias <= BIAS_BEARISH_THRESH:
            final_bias  = "BEARISH"
            reason      = "WEIGHTED_BIAS_BELOW_THRESHOLD"
        else:
            final_bias  = "NEUTRAL"
            reason      = "WEIGHTED_BIAS_IN_NEUTRAL_ZONE"

        row["mtf_final_bias"]           = final_bias
        row["mtf_reason"]               = reason

        # Entry / exit timing scores
        row["mtf_entry_timing_score"]   = _compute_entry_timing(row)
        row["mtf_exit_warning_score"]   = _compute_exit_warning(row)

        rows.append(row)

    return pd.DataFrame(rows)


def _compute_alignment_score(row: dict) -> float:
    """
    How well do the timeframes agree?
    1.0 = all agree  0.0 = completely split
    """
    biases = [
        row.get(f"bias_{tf.lower()}", 0.0)
        for tf in TF_ORDER
    ]
    # Alignment = (max - min absolute agreement)
    non_zero = [b for b in biases if b != 0.0]
    if len(non_zero) < 2:
        return 0.5
    signs = [np.sign(b) for b in non_zero]
    pct_agree = sum(1 for s in signs if s == signs[0]) / len(signs)
    return pct_agree


def _compute_conflict_score(row: dict) -> float:
    """
    How much do timeframes conflict?
    1.0 = all conflict, 0.0 = no conflict
    """
    biases = [row.get(f"bias_{tf.lower()}", 0.0) for tf in TF_ORDER]
    pos = sum(1 for b in biases if b > 0)
    neg = sum(1 for b in biases if b < 0)
    if pos + neg == 0:
        return 0.5  # all neutral
    conflict = min(pos, neg) / (pos + neg)
    return conflict * 2.0   # scale to 0-1


def _compute_entry_timing(row: dict) -> float:
    """
    Score for entry quality: higher when 1H aligns with higher TFs and
    1H is at a reset/pullback within the trend.
    """
    bias_1h   = row.get("bias_1h",  0.0)
    bias_4h   = row.get("bias_4h",  0.0)
    bias_12h  = row.get("bias_12h", 0.0)
    bias_24h  = row.get("bias_24h", 0.0)
    clv_1h    = row.get("structure_1h_clv", 0.5)
    align     = row.get("mtf_alignment_score", 0.5)
    conflict  = row.get("mtf_conflict_score",  0.5)

    score  = 0.0
    # Full alignment: 1H, 4H, 12H, 24H all same direction
    if bias_1h == bias_4h == bias_12h == bias_24h and bias_1h != 0:
        score += 0.40
    # 4H/12H/24H agree, 1H is at pullback extreme
    elif bias_4h == bias_12h == bias_24h and bias_4h != 0:
        if bias_4h > 0 and clv_1h < 0.35:   # bullish higher TF, 1H near lows = pullback
            score += 0.35
        elif bias_4h < 0 and clv_1h > 0.65:  # bearish, 1H near highs = failed bounce
            score += 0.35
        else:
            score += 0.15
    # Higher TFs agree but 1H diverges → entry timing poor
    if conflict > 0.50:
        score -= 0.20
    score += 0.20 * align
    return float(np.clip(score, 0, 1))


def _compute_exit_warning(row: dict) -> float:
    """
    Score for exit warning: higher when structure is deteriorating.
    """
    conflict = row.get("mtf_conflict_score",  0.0)
    chaos    = row.get("mtf_chaos_count", 0)
    strength_4h  = row.get("structure_4h_strength_score",  1.0)
    strength_12h = row.get("structure_12h_strength_score", 1.0)

    warning = (
        0.30 * conflict +
        0.25 * (chaos / 4.0) +
        0.25 * (1 - strength_4h) +
        0.20 * (1 - strength_12h)
    )
    return float(np.clip(warning, 0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def compute_mtf_structure(
    daily_df: pd.DataFrame,
    bars_by_tf: dict,         # {"1H": df, "4H": df, "12H": df, "24H": df}
    symbols: list = None,
) -> pd.DataFrame:
    """
    Compute MTF structure features for each symbol on each trading day.

    Parameters
    ----------
    daily_df   : DataFrame with columns [date, symbol, ...].  One row per day.
    bars_by_tf : Dict of DataFrames for each timeframe.  Each must have
                 [timestamp, open, high, low, close, volume, symbol] columns.
    symbols    : List of symbols to process (default: all in daily_df).

    Returns
    -------
    DataFrame with one row per (date, symbol) containing all MTF features.
    """
    if symbols is None:
        symbols = sorted(daily_df["symbol"].unique())

    all_parts = []
    for sym in symbols:
        logger.info("  Computing MTF structure for %s ...", sym)

        # Filter TF DataFrames to this symbol
        tf_structures = {}
        for tf, tf_df in bars_by_tf.items():
            if tf_df is None:
                continue
            sym_tf = tf_df[tf_df["symbol"] == sym].copy()
            if len(sym_tf) == 0:
                logger.warning("    No %s bars for symbol %s", tf, sym)
                continue
            sym_tf = sym_tf.sort_values("timestamp").reset_index(drop=True)
            sym_tf["timestamp"] = pd.to_datetime(sym_tf["timestamp"], utc=True)

            # Compute structure features for this TF
            logger.info("    Computing structure for %s %s (%d bars) ...", sym, tf, len(sym_tf))
            sym_tf = _compute_structure_features(sym_tf)
            tf_structures[tf] = sym_tf

        # Get trading dates for this symbol
        sym_daily = daily_df[daily_df["symbol"] == sym]
        trading_dates = sorted(sym_daily["date"].dt.date if hasattr(sym_daily["date"], "dt") else
                               pd.to_datetime(sym_daily["date"]).dt.date.unique())

        # Compute daily MTF features
        mtf_df = compute_daily_mtf_for_symbol(sym, trading_dates, tf_structures)
        all_parts.append(mtf_df)
        logger.info("    %s: %d trading days with MTF features", sym, len(mtf_df))

    if not all_parts:
        return pd.DataFrame()

    result = pd.concat(all_parts, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["symbol","date"]).reset_index(drop=True)
    return result


def load_tf_bars(out_dir: str, symbols: list = None) -> dict:
    """
    Convenience loader: reads the parquet files from Phase 7.
    Returns dict: {"1H": df, "4H": df, "12H": df, "24H": df}
    with all symbols combined.
    """
    import os
    import pyarrow.parquet as pq

    tf_dfs = {}
    for tf in ["1H", "4H", "12H", "24H"]:
        parts = []
        for sym in (symbols or ["ES","NQ"]):
            fpath = os.path.join(out_dir, f"{sym}_{tf}.parquet")
            if os.path.exists(fpath):
                df = pd.read_parquet(fpath)
                df["symbol"] = sym
                parts.append(df)
            else:
                logger.warning("  Missing %s_%s.parquet", sym, tf)
        if parts:
            tf_dfs[tf] = pd.concat(parts, ignore_index=True)
        else:
            tf_dfs[tf] = None
    return tf_dfs
