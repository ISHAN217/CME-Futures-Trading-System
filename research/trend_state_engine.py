"""
trend_state_engine.py — Trend-state framework for the CME futures trading system.
===================================================================================

Computes a richer trend-state classification beyond simple price-based momentum.
Trend = price structure + buyer/seller interest + participation + maturity + exhaustion.

Sections:
  A  Price structure   (swing H/L, HH/HL, BOS, retests)
  B  Buyer/seller pressure (CLV, wicks, persistence)
  C  Volume / participation  (up/down vol, breakout/pullback confirmation)
  D  Trend efficiency  (efficiency ratio, chop score)
  E  Pullback quality  (depth, volume, structure hold)
  F  Breakout / retest quality
  G  Exhaustion risk   (extension + weakening control)
  H  Cross-market ES/NQ confirmation
  I  Trend-state classification (13 states)

Main entry point:
    compute_trend_state_features(df: pd.DataFrame) -> pd.DataFrame

Input:  df from the main.py pipeline (after all existing features are computed)
Output: df with ~80 new columns appended; all existing columns preserved.

No lookahead: every feature at row i uses only data up to and including row i.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SWING_LOOKBACK   = 5    # rolling window for swing high/low detection
STRUCTURE_WINDOW = 20   # rolling window for range midpoint + progress
EFF_WINDOWS      = [5, 10, 20]
MA_WINDOWS       = [20, 50]
VOLUME_WINDOW    = 20
EXTENSION_WINDOW = 20

CLV_HIGH          = 0.70
CLV_LOW           = 0.30
WICK_SIGNIFICANT  = 0.30
VOLUME_CLIMAX_Z   = 2.0
EXTENSION_ATR     = 3.0
EXTENSION_Z       = 2.0

EXHAUS_MOD     = 0.35
EXHAUS_HIGH    = 0.60
EXHAUS_EXTREME = 0.80


# ─────────────────────────────────────────────────────────────────────────────
# A. PRICE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_price_structure(g: pd.DataFrame) -> pd.DataFrame:
    h  = g["high"].values
    lo = g["low"].values
    c  = g["close"].values
    n  = len(g)
    g  = g.copy()

    # Swing detection: lookahead-free rolling max/min
    roll_high = pd.Series(h).rolling(SWING_LOOKBACK, min_periods=SWING_LOOKBACK).max().values
    roll_low  = pd.Series(lo).rolling(SWING_LOOKBACK, min_periods=SWING_LOOKBACK).min().values

    swing_high = (h == roll_high).astype(float)
    swing_low  = (lo == roll_low).astype(float)
    g["swing_high"] = swing_high
    g["swing_low"]  = swing_low

    # Prior swing value: last confirmed swing level
    sh_val = np.where(swing_high > 0, h,  np.nan)
    sl_val = np.where(swing_low  > 0, lo, np.nan)
    prior_sh = pd.Series(sh_val).ffill().shift(1).values
    prior_sl = pd.Series(sl_val).ffill().shift(1).values
    g["prior_swing_high"] = prior_sh
    g["prior_swing_low"]  = prior_sl

    # Higher highs / higher lows / lower highs / lower lows
    g["higher_high_flag"] = ((swing_high > 0) & (h  > np.where(np.isnan(prior_sh), -np.inf, prior_sh))).astype(float)
    g["higher_low_flag"]  = ((swing_low  > 0) & (lo > np.where(np.isnan(prior_sl), -np.inf, prior_sl))).astype(float)
    g["lower_high_flag"]  = ((swing_high > 0) & (h  < np.where(np.isnan(prior_sh),  np.inf, prior_sh))).astype(float)
    g["lower_low_flag"]   = ((swing_low  > 0) & (lo < np.where(np.isnan(prior_sl),  np.inf, prior_sl))).astype(float)

    # Break of structure
    c_prev = np.concatenate([[c[0]], c[:-1]])
    psh    = np.where(np.isnan(prior_sh),  np.inf, prior_sh)
    psl    = np.where(np.isnan(prior_sl), -np.inf, prior_sl)
    g["break_of_structure_up"]   = ((c > psh) & (c_prev <= psh)).astype(float)
    g["break_of_structure_down"] = ((c < psl) & (c_prev >= psl)).astype(float)

    # Failed breakouts (poke through but close back inside)
    g["failed_breakout_up"]    = ((h  > psh) & (c < psh)).astype(float)
    g["failed_breakdown_down"] = ((lo < psl) & (c > psl)).astype(float)

    # Retest zones after BOS
    bos_up_level   = pd.Series(np.where(g["break_of_structure_up"].values   > 0, prior_sh, np.nan)).ffill().values
    bos_dn_level   = pd.Series(np.where(g["break_of_structure_down"].values  > 0, prior_sl, np.nan)).ffill().values
    zone_tol = 0.005
    retest_up = (~np.isnan(bos_up_level) &
                 (lo <= bos_up_level * (1 + zone_tol)) &
                 (lo >= bos_up_level * (1 - zone_tol)))
    retest_dn = (~np.isnan(bos_dn_level) &
                 (h  >= bos_dn_level * (1 - zone_tol)) &
                 (h  <= bos_dn_level * (1 + zone_tol)))
    g["retest_success_up"]   = (retest_up & (c > bos_up_level)).astype(float)
    g["retest_success_down"] = (retest_dn & (c < bos_dn_level)).astype(float)

    # Range midpoint
    roll_H = pd.Series(h).rolling(STRUCTURE_WINDOW, min_periods=5).max().values
    roll_L = pd.Series(lo).rolling(STRUCTURE_WINDOW, min_periods=5).min().values
    mid    = (roll_H + roll_L) / 2.0
    g["range_midpoint"]             = mid
    g["price_above_range_midpoint"] = (c > mid).astype(float)
    g["price_below_range_midpoint"] = (c < mid).astype(float)

    # Bars since last swing
    bssh = np.zeros(n); bssl = np.zeros(n)
    csh = 0; csl = 0
    for i in range(n):
        csh = 0 if swing_high[i] > 0 else csh + 1
        csl = 0 if swing_low[i]  > 0 else csl + 1
        bssh[i] = csh; bssl[i] = csl
    g["bars_since_last_swing_high"] = bssh
    g["bars_since_last_swing_low"]  = bssl

    # Structure progress: (bullish swings - bearish swings) / total
    hh = g["higher_high_flag"].rolling(STRUCTURE_WINDOW, min_periods=5).sum()
    hl = g["higher_low_flag"].rolling(STRUCTURE_WINDOW, min_periods=5).sum()
    lh = g["lower_high_flag"].rolling(STRUCTURE_WINDOW, min_periods=5).sum()
    ll = g["lower_low_flag"].rolling(STRUCTURE_WINDOW, min_periods=5).sum()
    total = (hh + hl + lh + ll).clip(lower=1)
    g["structure_progress_score"] = ((hh + hl) - (lh + ll)) / total   # [-1, +1]

    # Structure conflict: minority side / total
    bull_s = (hh + hl); bear_s = (lh + ll)
    minority = bull_s.clip(upper=bear_s) + bear_s.clip(upper=bull_s)
    g["structure_conflict_score"] = minority / total   # [0, 0.5]

    return g


# ─────────────────────────────────────────────────────────────────────────────
# B. BUYER / SELLER PRESSURE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pressure(g: pd.DataFrame) -> pd.DataFrame:
    h  = g["high"].values
    lo = g["low"].values
    o  = g["open"].values if "open" in g.columns else g["close"].values
    c  = g["close"].values
    g  = g.copy()

    rng = h - lo
    clv = np.where(rng > 1e-9, (c - lo) / rng, 0.5)
    g["close_location_value"] = clv
    g["close_near_high_flag"] = (clv >= CLV_HIGH).astype(float)
    g["close_near_low_flag"]  = (clv <= CLV_LOW).astype(float)

    body_top    = np.maximum(o, c)
    body_bottom = np.minimum(o, c)
    body_size   = body_top - body_bottom
    upper_wick  = h - body_top
    lower_wick  = body_bottom - lo

    g["upper_wick_ratio"] = np.where(rng > 1e-9, upper_wick / rng, 0.0)
    g["lower_wick_ratio"] = np.where(rng > 1e-9, lower_wick / rng, 0.0)
    g["body_ratio"]       = np.where(rng > 1e-9, body_size  / rng, 0.0)
    g["bullish_bar_flag"] = (c >= o).astype(float)
    g["bearish_bar_flag"] = (c <  o).astype(float)

    clv_s = pd.Series(clv, index=g.index)
    g["rolling_close_location_3"]  = clv_s.rolling(3,  min_periods=2).mean().values
    g["rolling_close_location_5"]  = clv_s.rolling(5,  min_periods=3).mean().values
    g["rolling_close_location_10"] = clv_s.rolling(10, min_periods=5).mean().values

    # Pressure scores: CLV weighted by wick confirmation
    buyer_raw  = clv * (1.0 + g["lower_wick_ratio"].values * 0.5)
    seller_raw = (1.0 - clv) * (1.0 + g["upper_wick_ratio"].values * 0.5)
    g["buyer_pressure_score"]  = pd.Series(buyer_raw,  index=g.index).rolling(5, min_periods=3).mean().clip(0, 1).values
    g["seller_pressure_score"] = pd.Series(seller_raw, index=g.index).rolling(5, min_periods=3).mean().clip(0, 1).values

    # Persistence: consecutive bars with CLV >= / <= 0.5
    bp = np.zeros(len(g)); sp = np.zeros(len(g))
    bc = 0; sc = 0
    for i in range(len(g)):
        if clv[i] >= 0.5:
            bc += 1; sc = 0
        else:
            sc += 1; bc = 0
        bp[i] = bc; sp[i] = sc
    g["buyer_pressure_persistence"]  = np.minimum(bp, 10) / 10.0
    g["seller_pressure_persistence"] = np.minimum(sp, 10) / 10.0

    return g


# ─────────────────────────────────────────────────────────────────────────────
# C. VOLUME / PARTICIPATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_volume(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    _zero = pd.Series(0.0, index=g.index)

    if "volume" not in g.columns or g["volume"].isna().all():
        for col in ["volume_zscore","volume_ratio_to_20","volume_on_up_bars",
                    "volume_on_down_bars","up_volume_ratio","down_volume_ratio",
                    "breakout_volume_confirmation","pullback_volume_quality",
                    "volume_climax_score","volume_divergence_score"]:
            g[col] = 0.0
        return g

    v   = g["volume"].fillna(0).values.astype(float)
    c   = g["close"].values
    o   = g["open"].values if "open" in g.columns else c

    vol_ma  = pd.Series(v).rolling(VOLUME_WINDOW, min_periods=5).mean().values
    vol_std = pd.Series(v).rolling(VOLUME_WINDOW, min_periods=5).std().values

    g["volume_zscore"]      = np.where(vol_std > 0, (v - vol_ma) / vol_std, 0.0)
    g["volume_ratio_to_20"] = np.where(vol_ma  > 0, v / vol_ma, 1.0)

    up_bar   = (c >= o).astype(float)
    down_bar = (c <  o).astype(float)
    vol_up   = pd.Series(v * up_bar,   index=g.index).rolling(10, min_periods=3).mean().values
    vol_down = pd.Series(v * down_bar, index=g.index).rolling(10, min_periods=3).mean().values
    vol_tot  = vol_up + vol_down + 1e-9

    g["volume_on_up_bars"]   = vol_up
    g["volume_on_down_bars"] = vol_down
    g["up_volume_ratio"]     = vol_up  / vol_tot
    g["down_volume_ratio"]   = vol_down / vol_tot

    bos_up   = g.get("break_of_structure_up",   _zero).values
    bos_down = g.get("break_of_structure_down",  _zero).values
    vol_z    = g["volume_zscore"].values
    g["breakout_volume_confirmation"] = np.where((bos_up + bos_down) > 0, vol_z.clip(-2, 4), 0.0)

    sp_col = g.get("structure_progress_score", _zero).values
    ret    = g.get("ret_1d", _zero).values
    in_up  = sp_col > 0.1
    pb_bar = (ret < 0) & in_up
    g["pullback_volume_quality"] = np.where(pb_bar, (vol_z < 0).astype(float), 0.0)

    # Volume climax + divergence (uses extension from Section G if present, else 0)
    ext_z    = g.get("upside_price_extension_z",   _zero).values
    bear_ext = g.get("downside_price_extension_z",  _zero).values
    clv      = g.get("close_location_value",         pd.Series(0.5, index=g.index)).values

    g["volume_climax_score"] = (
        np.where((vol_z > VOLUME_CLIMAX_Z) & (ext_z  > 1.5) & (clv < 0.5), 1.0, 0.0) +
        np.where((vol_z > VOLUME_CLIMAX_Z) & (bear_ext > 1.5) & (clv > 0.5), 1.0, 0.0)
    ).clip(0, 1)

    hh_flag = g.get("higher_high_flag", _zero).values
    ll_flag = g.get("lower_low_flag",   _zero).values
    vol_below = (g["volume_ratio_to_20"].values < 0.85).astype(float)
    raw_div = np.where(((hh_flag > 0) | (ll_flag > 0)) & (vol_below > 0), 1.0, 0.0)
    g["volume_divergence_score"] = pd.Series(raw_div, index=g.index).rolling(3, min_periods=1).mean().values

    return g


# ─────────────────────────────────────────────────────────────────────────────
# D. TREND EFFICIENCY
# ─────────────────────────────────────────────────────────────────────────────

def _compute_efficiency(g: pd.DataFrame) -> pd.DataFrame:
    g   = g.copy()
    c   = g["close"].values
    c_s = pd.Series(c, index=g.index)
    ret = np.abs(np.diff(c, prepend=c[0]))

    for n in EFF_WINDOWS:
        net   = (c_s - c_s.shift(n)).abs().values
        path  = pd.Series(ret).rolling(n, min_periods=max(1, n // 2)).sum().values
        g[f"trend_efficiency_{n}"] = np.clip(np.where(path > 1e-9, net / path, 0.5), 0.0, 1.0)

    cross_sum = pd.Series(0.0, index=g.index)
    for ma_w in MA_WINDOWS:
        ma    = c_s.rolling(ma_w, min_periods=ma_w // 2).mean()
        above = (c_s > ma).astype(int)
        cross_sum = cross_sum + above.diff().abs().rolling(20, min_periods=5).sum().fillna(0)
    g["moving_average_cross_count"] = cross_sum.values / len(MA_WINDOWS)

    mid = g.get("range_midpoint", c_s)
    above_mid = (c_s > mid).astype(int)
    g["midpoint_cross_count"] = above_mid.diff().abs().rolling(20, min_periods=5).sum().fillna(0).values

    eff_20     = g.get("trend_efficiency_20", pd.Series(0.5, index=g.index))
    cross_norm = (g["moving_average_cross_count"] / 10.0).clip(0, 1)
    g["chop_noise_score"]        = ((1.0 - eff_20) * 0.6 + cross_norm * 0.4).values
    g["trend_cleanliness_score"] = 1.0 - g["chop_noise_score"]

    return g


# ─────────────────────────────────────────────────────────────────────────────
# E. PULLBACK QUALITY
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pullback_quality(g: pd.DataFrame) -> pd.DataFrame:
    g   = g.copy()
    c   = g["close"].values
    lo  = g["low"].values
    ret = g.get("ret_1d", pd.Series(0.0, index=g.index)).values
    atr = g.get("atr",    pd.Series(np.abs(pd.Series(c).diff()).rolling(14).mean(), index=g.index)).values
    atr = np.where(atr > 0, atr, 1.0)

    sp  = g.get("structure_progress_score", pd.Series(0.0, index=g.index)).values
    in_up  = sp > 0.1
    pb_bar = in_up & (ret < 0)

    # Depth: cumulative negative return during pullback
    pb_depth_pct = np.zeros(len(g)); acc = 0.0
    for i in range(len(g)):
        if pb_bar[i]:
            acc += abs(ret[i])
        else:
            acc = 0.0
        pb_depth_pct[i] = acc

    g["pullback_depth_pct"] = pb_depth_pct
    g["pullback_depth_atr"] = pb_depth_pct / (atr / np.maximum(c, 1) + 1e-9)

    imp_5 = pd.Series(np.maximum(ret, 0), index=g.index).rolling(5, min_periods=3).sum().values
    g["pullback_depth_vs_prior_impulse"] = np.clip(
        np.where(imp_5 > 1e-9, pb_depth_pct / imp_5, 0.0), 0, 3)

    dur = np.zeros(len(g)); d = 0
    for i in range(len(g)):
        d = d + 1 if pb_bar[i] else 0
        dur[i] = d
    g["pullback_duration_bars"] = dur

    vol_ratio = g.get("volume_ratio_to_20", pd.Series(1.0, index=g.index)).values
    g["pullback_volume_ratio"]  = np.where(pb_bar, vol_ratio, 1.0)

    clv = g.get("close_location_value", pd.Series(0.5, index=g.index)).values
    g["pullback_close_quality"] = np.where(pb_bar, clv, 0.5)

    prior_sl = g.get("prior_swing_low", pd.Series(np.nan, index=g.index)).values
    g["pullback_holds_prior_swing_flag"]  = np.where(pb_bar, (lo > prior_sl).astype(float), 0.0)
    g["pullback_breaks_prior_swing_flag"] = np.where(pb_bar, (lo < prior_sl).astype(float), 0.0)

    holds    = g["pullback_holds_prior_swing_flag"].values
    low_vol  = (vol_ratio < 0.90).astype(float)
    good_clv = (clv > 0.40).astype(float)
    short_pb = (dur <= 3).astype(float)
    raw_h = np.where(pb_bar,
                     holds * 0.35 + low_vol * 0.25 + good_clv * 0.25 + short_pb * 0.15,
                     0.5)
    g["healthy_pullback_score"] = pd.Series(raw_h, index=g.index).rolling(3, min_periods=1).mean().values

    breaks   = g["pullback_breaks_prior_swing_flag"].values
    high_vol = (vol_ratio > 1.10).astype(float)
    bad_clv  = (clv < 0.40).astype(float)
    long_pb  = (dur > 5).astype(float)
    raw_b = np.where(pb_bar,
                     breaks * 0.40 + high_vol * 0.25 + bad_clv * 0.20 + long_pb * 0.15,
                     0.0)
    g["trend_breaking_pullback_score"] = pd.Series(raw_b, index=g.index).rolling(3, min_periods=1).mean().values

    bp = g.get("buyer_pressure_score", pd.Series(0.5, index=g.index)).values
    g["pullback_stabilization_score"]  = np.where(pb_bar, 0.0, np.where(in_up, bp, 0.5))
    eff5 = g.get("trend_efficiency_5", pd.Series(0.5, index=g.index)).values
    g["pullback_reacceleration_score"] = np.where(~pb_bar & in_up & (ret > 0), bp * eff5, 0.0)

    return g


# ─────────────────────────────────────────────────────────────────────────────
# F. BREAKOUT / RETEST QUALITY
# ─────────────────────────────────────────────────────────────────────────────

def _compute_breakout_quality(g: pd.DataFrame) -> pd.DataFrame:
    g    = g.copy()
    _z   = pd.Series(0.0, index=g.index)
    h    = g["high"].values
    lo   = g["low"].values
    vol_z = g.get("volume_zscore", _z).values
    clv  = g.get("close_location_value", pd.Series(0.5, index=g.index)).values
    bos_u = g.get("break_of_structure_up",   _z).values
    bos_d = g.get("break_of_structure_down",  _z).values
    fail_u = g.get("failed_breakout_up",     _z).values
    fail_d = g.get("failed_breakdown_down",  _z).values

    g["breakout_up_flag"]   = bos_u
    g["breakout_down_flag"] = bos_d
    g["breakout_close_strength"] = np.where((bos_u + bos_d) > 0, clv, 0.5)
    g["breakout_volume_z"]       = np.where((bos_u + bos_d) > 0, vol_z, 0.0)

    atr       = g.get("atr", pd.Series(1.0, index=g.index)).values
    bar_range = h - lo
    g["breakout_range_expansion"] = np.where(
        (bos_u + bos_d) > 0,
        np.where(atr > 0, bar_range / atr, 1.0),
        1.0
    )

    ret = g.get("ret_1d", _z).values
    prior_bos_u = pd.Series(bos_u).shift(1).fillna(0).values
    prior_bos_d = pd.Series(bos_d).shift(1).fillna(0).values
    g["breakout_followthrough_1bar"] = np.where(
        prior_bos_u > 0, (ret > 0).astype(float),
        np.where(prior_bos_d > 0, (ret < 0).astype(float), 0.5)
    )
    g["breakout_followthrough_3bar"] = pd.Series(
        g["breakout_followthrough_1bar"].values, index=g.index
    ).rolling(3, min_periods=1).mean().values

    g["breakout_retest_success"] = g.get("retest_success_up", _z).values
    g["breakout_retest_failure"] = (
        (g.get("retest_success_up", _z).values == 0).astype(float) *
        pd.Series(bos_u).shift(1).rolling(5, min_periods=1).max().fillna(0).values
    )

    g["failed_breakout_score"] = (
        pd.Series(fail_u, index=g.index).rolling(5, min_periods=1).sum().clip(0, 1).values * 0.6 +
        pd.Series(fail_d, index=g.index).rolling(5, min_periods=1).sum().clip(0, 1).values * 0.4
    )

    bo_str  = g["breakout_close_strength"].values
    bo_vol  = (vol_z > 0.5).astype(float) * np.where((bos_u + bos_d) > 0, 1.0, 0.0)
    bo_rng  = (g["breakout_range_expansion"].values > 1.0).astype(float)
    bo_foll = g["breakout_followthrough_1bar"].values
    raw_bq = np.where(bos_u > 0,
                      bo_str * 0.35 + bo_vol * 0.30 + bo_rng * 0.20 + bo_foll * 0.15,
                      0.0)
    g["breakout_quality_score"] = pd.Series(raw_bq, index=g.index).rolling(3, min_periods=1).mean().values

    return g


# ─────────────────────────────────────────────────────────────────────────────
# G. EXHAUSTION RISK
# ─────────────────────────────────────────────────────────────────────────────

def _compute_exhaustion(g: pd.DataFrame) -> pd.DataFrame:
    g   = g.copy()
    _z  = pd.Series(0.0, index=g.index)
    c   = g["close"].values
    h   = g["high"].values
    lo  = g["low"].values
    ret = g.get("ret_1d", _z).values
    atr = g.get("atr",    pd.Series(1.0, index=g.index)).values
    atr = np.where(atr > 0, atr, 1.0)
    ret_s = pd.Series(ret, index=g.index)

    # Price extension in ATR units from rolling range
    roll_min = pd.Series(lo).rolling(EXTENSION_WINDOW, min_periods=5).min().values
    roll_max = pd.Series(h).rolling(EXTENSION_WINDOW, min_periods=5).max().values
    g["upside_price_extension_atr"]   = np.clip((c - roll_min) / atr, 0, 10)
    g["downside_price_extension_atr"] = np.clip((roll_max - c) / atr, 0, 10)

    # Z-score of cumulative 5-bar return relative to 60-bar baseline
    cum_5    = ret_s.rolling(5,  min_periods=3).sum()
    base_mn  = ret_s.rolling(60, min_periods=30).mean()
    base_std = ret_s.rolling(60, min_periods=30).std().replace(0, 0.01)
    ret_z    = (cum_5 - base_mn) / base_std
    g["upside_price_extension_z"]   = ret_z.clip(-4,  4).values
    g["downside_price_extension_z"] = (-ret_z).clip(-4, 4).values

    # Return acceleration (short-term vs long-term momentum)
    mom_5  = ret_s.rolling(5,  min_periods=3).mean()
    mom_20 = ret_s.rolling(20, min_periods=10).mean()
    accel  = mom_5 - mom_20
    ac_std = accel.rolling(60, min_periods=20).std().replace(0, 0.001)
    g["upside_return_acceleration_z"]   = (accel  / ac_std).clip(-3, 3).values
    g["downside_return_acceleration_z"] = (-accel / ac_std).clip(-3, 3).values

    # Consecutive bars & bars since last pullback/bounce
    cu = cd = spb = sbk = 0
    cup = np.zeros(len(g)); cdn = np.zeros(len(g))
    sipb = np.zeros(len(g)); sibk = np.zeros(len(g))
    for i in range(len(g)):
        if   ret[i] > 0: cu += 1; cd = 0;  spb += 1; sbk = 0
        elif ret[i] < 0: cd += 1; cu = 0;  sbk += 1; spb = 0
        else:             spb += 1; sbk += 1
        cup[i] = cu; cdn[i] = cd
        sipb[i] = spb; sibk[i] = sbk
    g["consecutive_up_bars"]     = cup
    g["consecutive_down_bars"]   = cdn
    g["bars_since_last_pullback"] = sipb
    g["bars_since_last_bounce"]   = sibk

    # Wick / close signals after extension
    clv  = g.get("close_location_value", pd.Series(0.5, index=g.index)).values
    uwk  = g.get("upper_wick_ratio",     _z).values
    lwk  = g.get("lower_wick_ratio",     _z).values
    rz   = ret_z.values

    ext_h = (g["upside_price_extension_atr"].values   > EXTENSION_ATR) | (rz >  EXTENSION_Z)
    ext_l = (g["downside_price_extension_atr"].values > EXTENSION_ATR) | (rz < -EXTENSION_Z)

    g["weak_close_after_extension"]  = (ext_h & (clv < 0.40)).astype(float)
    g["upper_wick_after_extension"]  = (ext_h & (uwk > WICK_SIGNIFICANT)).astype(float)
    g["strong_close_after_selloff"]  = (ext_l & (clv > 0.60)).astype(float)
    g["lower_wick_after_selloff"]    = (ext_l & (lwk > WICK_SIGNIFICANT)).astype(float)

    vol_z = g.get("volume_zscore", _z).values
    g["volume_climax_after_rally"] = (ext_h & (vol_z > VOLUME_CLIMAX_Z)).astype(float)
    g["selling_volume_climax"]     = (ext_l & (vol_z > VOLUME_CLIMAX_Z)).astype(float)

    # Momentum deceleration
    mom_dec = mom_5 - mom_5.shift(3)
    md_std  = mom_dec.rolling(60, min_periods=20).std().replace(0, 0.001)
    dec_z   = (mom_dec / md_std).clip(-3, 3)
    g["momentum_deceleration_up"]          = (-dec_z).clip(0, 3).values
    g["downside_momentum_deceleration"]    = dec_z.clip(0, 3).values

    fu  = g.get("failed_breakout_up",    _z).values
    fd  = g.get("failed_breakdown_down", _z).values
    g["failed_breakout_after_extension"]  = (ext_h & (fu > 0)).astype(float)
    g["failed_breakdown_after_extension"] = (ext_l & (fd > 0)).astype(float)

    # Cross-market divergence placeholder (filled after _compute_cross_market)
    g["cross_market_bullish_divergence"] = 0.0
    g["cross_market_bearish_divergence"] = 0.0

    # ── Composite exhaustion scores ───────────────────────────────────────
    up_ext_atr = g["upside_price_extension_atr"].values
    dn_ext_atr = g["downside_price_extension_atr"].values
    bull_exh = (
        np.clip(up_ext_atr / 5.0, 0, 1)                         * 0.20 +
        np.clip(rz / 3.0, 0, 1)                                  * 0.15 +
        g["weak_close_after_extension"].values                    * 0.20 +
        g["upper_wick_after_extension"].values                    * 0.15 +
        g["volume_climax_after_rally"].values                     * 0.15 +
        np.clip(g["momentum_deceleration_up"].values / 2.0, 0, 1) * 0.10 +
        g["failed_breakout_after_extension"].values               * 0.05
    ).clip(0, 1)

    bear_exh = (
        np.clip(dn_ext_atr / 5.0, 0, 1)                               * 0.20 +
        np.clip(-rz / 3.0, 0, 1)                                       * 0.15 +
        g["strong_close_after_selloff"].values                          * 0.20 +
        g["lower_wick_after_selloff"].values                            * 0.15 +
        g["selling_volume_climax"].values                               * 0.15 +
        np.clip(g["downside_momentum_deceleration"].values / 2.0, 0, 1) * 0.10 +
        g["failed_breakdown_after_extension"].values                    * 0.05
    ).clip(0, 1)

    g["bull_exhaustion_score"] = pd.Series(bull_exh, index=g.index).rolling(3, min_periods=1).mean().values
    g["bear_exhaustion_score"] = pd.Series(bear_exh, index=g.index).rolling(3, min_periods=1).mean().values

    max_exh = np.maximum(g["bull_exhaustion_score"].values,
                         g["bear_exhaustion_score"].values)
    g["exhaustion_label"] = np.where(
        max_exh >= EXHAUS_EXTREME, "EXTREME_EXHAUSTION",
        np.where(max_exh >= EXHAUS_HIGH,    "HIGH_EXHAUSTION",
        np.where(max_exh >= EXHAUS_MOD,     "MODERATE_EXHAUSTION",
                                            "NO_EXHAUSTION")))

    return g


# ─────────────────────────────────────────────────────────────────────────────
# H. CROSS-MARKET ES/NQ CONFIRMATION
# ─────────────────────────────────────────────────────────────────────────────

_CROSS_COLS = [
    "ES_NQ_structure_alignment",
    "ES_new_high_flag", "NQ_new_high_flag",
    "ES_new_low_flag",  "NQ_new_low_flag",
    "cross_market_bull_confirm", "cross_market_bear_confirm",
    "cross_market_divergence",
    "NQ_relative_strength_vs_ES", "NQ_relative_strength_deceleration",
    "ES_NQ_spread_zscore", "trend_participation_breadth",
    "cross_market_bullish_divergence", "cross_market_bearish_divergence",
]

def _compute_cross_market(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    syms = df["symbol"].unique() if "symbol" in df.columns else []

    if "ES" not in syms or "NQ" not in syms:
        for col in _CROSS_COLS:
            df[col] = 0.0
        return df

    keep = [c for c in ["ret_1d","close","structure_progress_score",
                         "higher_high_flag","lower_low_flag",
                         "bull_exhaustion_score","bear_exhaustion_score",
                         "buyer_pressure_score","seller_pressure_score"]
            if c in df.columns]

    es = df[df["symbol"] == "ES"][["date"] + keep].set_index("date")
    nq = df[df["symbol"] == "NQ"][["date"] + keep].set_index("date")
    al = es.join(nq, lsuffix="_ES", rsuffix="_NQ", how="inner")

    if al.empty:
        for col in _CROSS_COLS:
            df[col] = 0.0
        return df

    # New highs / lows (20-bar rolling)
    for sym, sfx in [("ES", "_ES"), ("NQ", "_NQ")]:
        cc = f"close{sfx}"
        if cc in al.columns:
            mx = al[cc].rolling(20, min_periods=5).max()
            mn = al[cc].rolling(20, min_periods=5).min()
            al[f"{sym}_new_high_flag"] = (al[cc] == mx).astype(float)
            al[f"{sym}_new_low_flag"]  = (al[cc] == mn).astype(float)
        else:
            al[f"{sym}_new_high_flag"] = 0.0
            al[f"{sym}_new_low_flag"]  = 0.0

    sp_es = al.get("structure_progress_score_ES", pd.Series(0.0, index=al.index))
    sp_nq = al.get("structure_progress_score_NQ", pd.Series(0.0, index=al.index))
    bp_es = al.get("buyer_pressure_score_ES",  pd.Series(0.5, index=al.index))
    bp_nq = al.get("buyer_pressure_score_NQ",  pd.Series(0.5, index=al.index))
    sl_es = al.get("seller_pressure_score_ES", pd.Series(0.5, index=al.index))
    sl_nq = al.get("seller_pressure_score_NQ", pd.Series(0.5, index=al.index))

    al["cross_market_bull_confirm"] = ((sp_es > 0.1) & (sp_nq > 0.1) & (bp_es > 0.5) & (bp_nq > 0.5)).astype(float)
    al["cross_market_bear_confirm"] = ((sp_es < -0.1) & (sp_nq < -0.1) & (sl_es > 0.5) & (sl_nq > 0.5)).astype(float)

    al["cross_market_divergence"] = np.where(
        sp_es.fillna(0).apply(np.sign) != sp_nq.fillna(0).apply(np.sign), 1.0,
        np.where(al["ES_new_high_flag"] != al["NQ_new_high_flag"], 0.5, 0.0)
    )

    if "ret_1d_ES" in al.columns and "ret_1d_NQ" in al.columns:
        rs = al["ret_1d_NQ"] - al["ret_1d_ES"]
        al["NQ_relative_strength_vs_ES"]          = rs.rolling(5, min_periods=3).mean().values
        al["NQ_relative_strength_deceleration"]   = (rs.rolling(5, min_periods=3).mean() -
                                                      rs.rolling(20, min_periods=10).mean()).values
    else:
        al["NQ_relative_strength_vs_ES"]         = 0.0
        al["NQ_relative_strength_deceleration"]  = 0.0

    if "close_ES" in al.columns and "close_NQ" in al.columns:
        ratio    = np.log((al["close_NQ"] / al["close_ES"]).replace(0, np.nan))
        ratio_mn = ratio.rolling(60, min_periods=20).mean()
        ratio_sd = ratio.rolling(60, min_periods=20).std().replace(0, 0.001)
        al["ES_NQ_spread_zscore"] = ((ratio - ratio_mn) / ratio_sd).clip(-3, 3).values
    else:
        al["ES_NQ_spread_zscore"] = 0.0

    al["trend_participation_breadth"] = al["cross_market_bull_confirm"] - al["cross_market_bear_confirm"]
    al["ES_NQ_structure_alignment"]   = al["trend_participation_breadth"]

    al["cross_market_bullish_divergence"] = ((al["ES_new_high_flag"] > 0) & (al["NQ_new_high_flag"] == 0)).astype(float)
    al["cross_market_bearish_divergence"] = ((al["ES_new_low_flag"]  > 0) & (al["NQ_new_low_flag"]  == 0)).astype(float)

    # Merge cross-market features back into df for each symbol
    cross_use = [c for c in _CROSS_COLS if c in al.columns]
    al_reset  = al[cross_use].reset_index()  # 'date' becomes column

    for sym in ["ES", "NQ"]:
        mask  = df["symbol"] == sym
        dates = df.loc[mask, "date"].reset_index()
        merged = dates.merge(al_reset, on="date", how="left").set_index("index")
        for col in cross_use:
            if col in merged.columns:
                df.loc[mask, col] = merged[col].fillna(0.0).values
            else:
                df.loc[mask, col] = 0.0

    # Update exhaustion scores with cross-market divergence
    if "bull_exhaustion_score" in df.columns:
        div_bull = df.get("cross_market_bullish_divergence", pd.Series(0.0, index=df.index))
        df["bull_exhaustion_score"] = (df["bull_exhaustion_score"] * 0.88 + div_bull * 0.12).clip(0, 1)
    if "bear_exhaustion_score" in df.columns:
        div_bear = df.get("cross_market_bearish_divergence", pd.Series(0.0, index=df.index))
        df["bear_exhaustion_score"] = (df["bear_exhaustion_score"] * 0.88 + div_bear * 0.12).clip(0, 1)

    for col in _CROSS_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# I. TREND STATE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _classify_trend_state(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    def _col(name, default=0.0):
        return g.get(name, pd.Series(default, index=g.index)).values

    sp     = _col("structure_progress_score",    0.0)
    bp     = _col("buyer_pressure_score",         0.5)
    sp_sel = _col("seller_pressure_score",        0.5)
    eff    = _col("trend_efficiency_20",          0.5)
    chop   = _col("chop_noise_score",             0.5)
    bull_e = _col("bull_exhaustion_score",        0.0)
    bear_e = _col("bear_exhaustion_score",        0.0)
    hpb    = _col("healthy_pullback_score",       0.5)
    bpb    = _col("trend_breaking_pullback_score",0.0)
    cf     = _col("structure_conflict_score",     0.0)
    bos_u  = _col("break_of_structure_up",        0.0)
    bos_d  = _col("break_of_structure_down",      0.0)
    phold  = _col("pullback_holds_prior_swing_flag", 0.0)
    cbull  = _col("cross_market_bull_confirm",    0.0)
    cbear  = _col("cross_market_bear_confirm",    0.0)
    dur    = _col("pullback_duration_bars",       0.0)
    reg    = g.get("regime", pd.Series("CHOP", index=g.index)).values
    mtf    = g.get("mtf_final_bias", pd.Series("NEUTRAL", index=g.index)).values

    states = []; conf_v = []; opp_v = []; risk_v = []
    eperm = []; aperm = []; rflag = []; ewarn = []; rsns = []

    for i in range(len(g)):
        spi   = float(sp[i]);   bpi = float(bp[i]); sei = float(sp_sel[i])
        effi  = float(eff[i]);  chi = float(chop[i]); cfi = float(cf[i])
        bulli = float(bull_e[i]); beari = float(bear_e[i])
        hpbi  = float(hpb[i]); bpbi = float(bpb[i])
        bsui  = float(bos_u[i]); bsdi = float(bos_d[i])
        phi   = float(phold[i]); cbui = float(cbull[i]); cbei = float(cbear[i])
        duri  = float(dur[i])
        regi  = str(reg[i]) if isinstance(reg[i], str) else "CHOP"
        mtfi  = str(mtf[i]) if isinstance(mtf[i], str) else "NEUTRAL"

        # Priority 1 — CHAOTIC
        if chi > 0.72 and cfi > 0.35:
            s = "CHAOTIC";         cv = 0.4; ov = 0.1; rv = 0.8
            ep = ap = False; rf = ew = True
            rsn = f"chop={chi:.2f} conflict={cfi:.2f}"

        # Priority 2 — RANGE (weak structure + low efficiency)
        elif abs(spi) < 0.15 and effi < 0.45:
            s = "RANGE";           cv = 0.45; ov = 0.2; rv = 0.4
            ep = ap = rf = ew = False
            rsn = f"sp={spi:.2f} eff={effi:.2f}"

        # Priority 3 — BULL states
        elif spi > 0.15 or (spi > 0.0 and regi == "TREND" and mtfi == "BULLISH"):

            if bpbi > 0.50 or (bsdi > 0 and spi < 0.30):
                s = "BULL_BREAKDOWN_WARNING"; cv = 0.5 + bpbi * 0.3; ov = 0.1; rv = 0.75
                ep = ap = False; rf = ew = True
                rsn = f"brk_pb={bpbi:.2f} bos_dn={bsdi:.0f}"

            elif bulli >= EXHAUS_HIGH:
                s = "BULL_EXHAUSTION"; cv = 0.5 + bulli * 0.3; ov = 0.1; rv = 0.70
                ep = ap = False; rf = bulli > EXHAUS_EXTREME; ew = bulli > EXHAUS_EXTREME
                rsn = f"bull_exh={bulli:.2f}"

            elif duri > 0 and phi > 0 and bpbi < 0.30:
                s = "BULL_PULLBACK";  cv = 0.4 + hpbi * 0.3; ov = 0.3; rv = 0.30
                ep = ap = rf = ew = False
                rsn = f"pb_dur={duri:.0f} holds={phi:.0f} hpb={hpbi:.2f}"

            elif duri == 0 and hpbi > 0.50 and bpbi < 0.20 and bpi > 0.50:
                s = "BULL_RESET_CONFIRMED"; cv = min(0.5 + hpbi * 0.25 + (bpi-0.5)*0.25, 0.9)
                ov = min(0.7 + cbui * 0.2, 0.9); rv = 0.25
                ep = ap = True; rf = ew = False
                rsn = f"hpb={hpbi:.2f} bp={bpi:.2f} bos_up={bsui:.0f}"

            elif bpi > 0.55 and effi > 0.50 and bulli < EXHAUS_MOD:
                s = "BULL_IMPULSE";   cv = min(0.5 + spi*0.3 + effi*0.2, 0.9); ov = 0.60; rv = 0.30
                ep = ap = True; rf = ew = False
                rsn = f"bp={bpi:.2f} eff={effi:.2f} exh={bulli:.2f}"

            else:
                s = "UNCLEAR";        cv = 0.30; ov = 0.30; rv = 0.40
                ep = ap = rf = ew = False
                rsn = f"sp={spi:.2f} bull-unclear"

        # Priority 4 — BEAR states
        elif spi < -0.15 or (spi < 0.0 and regi == "TREND" and mtfi == "BEARISH"):

            if bpbi > 0.50 or (bsui > 0 and spi > -0.30):
                s = "BEAR_BREAKDOWN_WARNING"; cv = 0.5 + bpbi * 0.3; ov = 0.1; rv = 0.75
                ep = ap = False; rf = ew = True
                rsn = f"brk={bpbi:.2f} bos_up={bsui:.0f}"

            elif beari >= EXHAUS_HIGH:
                s = "BEAR_EXHAUSTION"; cv = 0.5 + beari * 0.3; ov = 0.1; rv = 0.70
                ep = ap = False; rf = beari > EXHAUS_EXTREME; ew = beari > EXHAUS_EXTREME
                rsn = f"bear_exh={beari:.2f}"

            elif duri > 0:
                s = "BEAR_RALLY";     cv = 0.40; ov = 0.20; rv = 0.40
                ep = ap = rf = ew = False
                rsn = f"bear_rally dur={duri:.0f}"

            elif duri == 0 and sei > 0.50 and bpbi < 0.20:
                s = "BEAR_RESET_CONFIRMED"; cv = min(0.5 + (sei-0.5)*0.4, 0.9)
                ov = min(0.5 + cbei * 0.2, 0.7); rv = 0.30
                ep = ap = True; rf = ew = False
                rsn = f"sel={sei:.2f} bear_reset"

            elif sei > 0.55 and effi > 0.50 and beari < EXHAUS_MOD:
                s = "BEAR_IMPULSE";   cv = min(0.5 + abs(spi)*0.3, 0.9); ov = 0.50; rv = 0.35
                ep = ap = True; rf = ew = False
                rsn = f"sel={sei:.2f} eff={effi:.2f}"

            else:
                s = "UNCLEAR";        cv = 0.30; ov = 0.30; rv = 0.40
                ep = ap = rf = ew = False
                rsn = f"sp={spi:.2f} bear-unclear"

        else:
            s = "UNCLEAR";            cv = 0.30; ov = 0.30; rv = 0.40
            ep = ap = rf = ew = False
            rsn = f"sp={spi:.2f} reg={regi}"

        states.append(s)
        conf_v.append(float(np.clip(cv, 0.0, 1.0)))
        opp_v.append(float(ov))
        risk_v.append(float(rv))
        eperm.append(bool(ep)); aperm.append(bool(ap))
        rflag.append(bool(rf)); ewarn.append(bool(ew))
        rsns.append(rsn)

    g["trend_state"]             = states
    g["trend_state_confidence"]  = conf_v
    g["trend_opportunity_score"] = opp_v
    g["trend_risk_score"]        = risk_v
    g["trend_entry_permission"]  = eperm
    g["trend_add_permission"]    = aperm
    g["trend_reduce_flag"]       = rflag
    g["trend_exit_warning_flag"] = ewarn
    g["trend_state_reason"]      = rsns

    return g


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def compute_trend_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all trend-state features and append to df.

    Parameters
    ----------
    df : pd.DataFrame
        Output of main.py pipeline (after all existing features computed).
        Must contain at minimum: date, symbol, open, high, low, close, ret_1d

    Returns
    -------
    pd.DataFrame with ~85 new columns appended.
    All existing columns preserved. No lookahead.
    """
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    logger.info("Trend state engine: %d rows, symbols=%s",
                len(df), list(df["symbol"].unique()))

    # Per-symbol feature computation
    groups = []
    for sym, g in df.groupby("symbol", sort=True):
        g = _compute_price_structure(g)
        g = _compute_pressure(g)
        g = _compute_exhaustion(g)        # needs structure for prior_swing refs
        g = _compute_efficiency(g)
        g = _compute_volume(g)            # needs extension_z from exhaustion
        g = _compute_pullback_quality(g)
        g = _compute_breakout_quality(g)
        groups.append(g)
        logger.info("  %s: per-symbol features computed", sym)

    df = pd.concat(groups, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)

    # Cross-market (both symbols must be present)
    df = _compute_cross_market(df)
    logger.info("  Cross-market ES/NQ features computed")

    # Per-symbol state classification (needs cross-market in df)
    groups2 = []
    for sym, g in df.groupby("symbol", sort=True):
        g = _classify_trend_state(g)
        groups2.append(g)

    df = pd.concat(groups2, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)

    dist = df["trend_state"].value_counts()
    logger.info("  Trend state distribution:\n%s", dist.to_string())

    return df
