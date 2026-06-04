"""
features_4h.py — 4H bar feature computation for the 4H resolution system.

Design philosophy:
  - Compute everything from raw OHLCV at 4H frequency
  - Join daily context (regime, ATR, trend) from the daily frame
  - Keep it simple: breakout + momentum + volume + session filter
  - No lookahead: all features use shift(1) before rolling

Symbols:
  - Works for CL, ES, NQ — session_filter adapts per instrument
  - CL liquid hours: 08:00, 12:00, 16:00 UTC (Nymex + CME hours)
  - ES/NQ liquid hours: 14:00, 18:00 UTC (RTH + early afternoon)
"""

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

BREAKOUT_LOOKBACK = 8       # bars for rolling high/low (8 × 4H = 2 calendar days)
EMA_FAST          = 2       # bars  (~8H)
EMA_MED           = 5       # bars  (~20H)
EMA_SLOW          = 12      # bars  (~48H)
ATR_PERIOD        = 14      # bars for 4H ATR
VOL_LOOKBACK      = 20      # bars for volume z-score

# Liquid session UTC hours by asset class (matched to actual 4H bar start times)
# CL Databento 4H bars start at: 00, 04, 08, 12, 16, 20 UTC
# ES yfinance 4H bars start at:  04, 08, 12, 16, 20 UTC (approximately)
_LIQUID_HOURS = {
    "CL": {8, 12, 16},        # European + Nymex open + US afternoon (highest vol)
    "ES": {8, 12, 16, 20},    # Pre-market + RTH open (12 UTC) + US afternoon
    "NQ": {8, 12, 16, 20},
}
_LIQUID_HOURS_DEFAULT = {8, 12, 16, 20}


# ── Helper ────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD, min_periods=5).mean()


def _daily_regime(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime and context features from daily OHLCV.
    Returns df_daily with added columns:
      daily_regime, daily_above_sma200, daily_adx, daily_atr_pct,
      daily_trend_score  (-1 to +1)
    """
    results = []
    for sym, grp in df_daily.groupby("symbol"):
        grp = grp.copy().sort_values("date").reset_index(drop=True)

        # SMA 200
        grp["_sma200"] = grp["close"].rolling(200, min_periods=50).mean()
        grp["daily_above_sma200"] = (grp["close"] > grp["_sma200"]).astype(int)

        # ADX (14-period)
        prev_close = grp["close"].shift(1)
        tr = pd.concat([
            grp["high"] - grp["low"],
            (grp["high"] - prev_close).abs(),
            (grp["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean()
        up   = (grp["high"] - grp["high"].shift(1)).clip(lower=0)
        down = (grp["low"].shift(1) - grp["low"]).clip(lower=0)
        dm_pos = up.where(up > down, 0.0)
        dm_neg = down.where(down > up, 0.0)
        di_pos = 100 * dm_pos.ewm(span=14, adjust=False).mean() / atr14.clip(lower=1e-8)
        di_neg = 100 * dm_neg.ewm(span=14, adjust=False).mean() / atr14.clip(lower=1e-8)
        dx = (100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-8))
        grp["daily_adx"] = dx.ewm(span=14, adjust=False).mean()

        # Daily ATR %
        grp["daily_atr_pct"] = atr14 / grp["close"]

        # 50-day momentum score (price vs SMA50)
        sma50 = grp["close"].rolling(50, min_periods=20).mean()
        grp["_mom50"] = (grp["close"] - sma50) / sma50.clip(lower=1e-8)

        # Regime label
        adx     = grp["daily_adx"]
        above   = grp["daily_above_sma200"]
        mom50   = grp["_mom50"]

        cond_bull = above == 1
        cond_bear = above == 0
        cond_trend = adx > 20

        grp["daily_regime"] = "NEUTRAL"
        grp.loc[cond_bull & cond_trend,  "daily_regime"] = "BULL_TREND"
        grp.loc[cond_bear & cond_trend,  "daily_regime"] = "BEAR_TREND"
        grp.loc[cond_bull & ~cond_trend, "daily_regime"] = "BULL_RANGE"
        grp.loc[cond_bear & ~cond_trend, "daily_regime"] = "BEAR_RANGE"

        # Trend score: +1 (full bull trend), 0 (neutral), -1 (full bear)
        grp["daily_trend_score"] = (grp["daily_above_sma200"] * 2 - 1) * (
            (grp["daily_adx"] / 40).clip(0, 1)
        )

        results.append(grp)

    out = pd.concat(results).sort_values(["symbol", "date"]).reset_index(drop=True)
    drop_cols = [c for c in ["_sma200", "_mom50"] if c in out.columns]
    out.drop(columns=drop_cols, inplace=True)
    return out


# ── Main feature computation ──────────────────────────────────────────────────

def compute_4h_features(df_4h: pd.DataFrame, df_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute all 4H features and join daily context.

    Parameters
    ----------
    df_4h : DataFrame
        4H OHLCV bars. Required columns: datetime, date, open, high, low, close, volume, symbol
        datetime must be timezone-aware UTC.
    df_daily : DataFrame (optional)
        Daily OHLCV for context. Required columns: date, symbol, open, high, low, close, volume.
        If None, daily context columns will be NaN.

    Returns
    -------
    DataFrame with original 4H columns plus:
        hb_roll_high, hb_roll_low     — rolling N-bar high/low (shifted)
        hb_breakout_long              — close > rolling high (0/1)
        hb_breakdown_short            — close < rolling low (0/1)
        hb_ema_fast, hb_ema_med, hb_ema_slow
        hb_ema_bull, hb_ema_bear      — EMA stack aligned (0/1)
        hb_ema_score                  — −1 to +1 alignment score
        hb_vol_zscore                 — volume vs 20-bar mean/std
        hb_atr, hb_atr_pct           — 4H ATR (price, %)
        hb_is_liquid                  — session liquid flag (0/1)
        hb_signal_long, hb_signal_short — filtered directional signal
        daily_regime, daily_above_sma200, daily_adx, daily_atr_pct, daily_trend_score
    """
    df = df_4h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["date"]     = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    # ── Per-symbol 4H features ─────────────────────────────────────────────
    parts = []
    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("datetime").reset_index(drop=True)

        # Rolling N-bar high/low (shifted: use previous N bars, not current)
        grp["hb_roll_high"] = grp["high"].shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=4).max()
        grp["hb_roll_low"]  = grp["low"].shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=4).min()

        grp["hb_breakout_long"]   = (grp["close"] > grp["hb_roll_high"]).astype(int)
        grp["hb_breakdown_short"] = (grp["close"] < grp["hb_roll_low"]).astype(int)

        # EMA stack
        grp["hb_ema_fast"] = _ema(grp["close"], EMA_FAST)
        grp["hb_ema_med"]  = _ema(grp["close"], EMA_MED)
        grp["hb_ema_slow"] = _ema(grp["close"], EMA_SLOW)

        grp["hb_ema_bull"] = (
            (grp["hb_ema_fast"] > grp["hb_ema_med"]) &
            (grp["hb_ema_med"]  > grp["hb_ema_slow"])
        ).astype(int)
        grp["hb_ema_bear"] = (
            (grp["hb_ema_fast"] < grp["hb_ema_med"]) &
            (grp["hb_ema_med"]  < grp["hb_ema_slow"])
        ).astype(int)

        # EMA score: fraction of pairs aligned in bull direction
        pairs = [
            grp["hb_ema_fast"] > grp["hb_ema_med"],
            grp["hb_ema_med"]  > grp["hb_ema_slow"],
            grp["hb_ema_fast"] > grp["hb_ema_slow"],
        ]
        grp["hb_ema_score"] = sum(p.astype(float) for p in pairs) / len(pairs) * 2 - 1  # −1..+1

        # Volume z-score
        vol_mean = grp["volume"].rolling(VOL_LOOKBACK, min_periods=5).mean()
        vol_std  = grp["volume"].rolling(VOL_LOOKBACK, min_periods=5).std().clip(lower=1)
        grp["hb_vol_zscore"] = (grp["volume"] - vol_mean) / vol_std

        # 4H ATR
        grp["hb_atr"]     = _atr(grp)
        grp["hb_atr_pct"] = grp["hb_atr"] / grp["close"].clip(lower=1e-8)

        # Session liquid flag
        hour_utc = grp["datetime"].dt.hour
        liquid_hrs = _LIQUID_HOURS.get(sym, _LIQUID_HOURS_DEFAULT)
        grp["hb_is_liquid"] = hour_utc.isin(liquid_hrs).astype(int)

        # Net signals (entry conditions)
        vol_ok = grp["hb_vol_zscore"] > 0.0          # any above-average volume
        liquid = grp["hb_is_liquid"] == 1
        atr_ok = grp["hb_atr_pct"] > 0.001           # need valid ATR

        # Breakout signal: close > N-bar rolling high + EMA bull
        grp["hb_signal_long"]  = (
            (grp["hb_breakout_long"]   == 1) &
            (grp["hb_ema_bull"]        == 1) &
            vol_ok & liquid & atr_ok
        ).astype(int)
        grp["hb_signal_short"] = (
            (grp["hb_breakdown_short"] == 1) &
            (grp["hb_ema_bear"]        == 1) &
            vol_ok & liquid & atr_ok
        ).astype(int)

        # Momentum signal: EMA alignment only (no breakout required)
        # More frequent, captures trend continuation not just initial breakouts
        prev_close = grp["close"].shift(1)
        up_bar   = grp["close"] > prev_close
        down_bar = grp["close"] < prev_close

        grp["hb_signal_mom_long"]  = (
            (grp["hb_ema_bull"]   == 1) &
            up_bar & vol_ok & liquid & atr_ok
        ).astype(int)
        grp["hb_signal_mom_short"] = (
            (grp["hb_ema_bear"]   == 1) &
            down_bar & vol_ok & liquid & atr_ok
        ).astype(int)

        parts.append(grp)

    df = pd.concat(parts).sort_values(["symbol", "datetime"]).reset_index(drop=True)

    # ── Join daily context ─────────────────────────────────────────────────
    if df_daily is not None:
        df_daily_ctx = _daily_regime(df_daily.copy())
        df_daily_ctx["date"] = pd.to_datetime(df_daily_ctx["date"])

        ctx_cols = ["date", "symbol", "daily_regime", "daily_above_sma200",
                    "daily_adx", "daily_atr_pct", "daily_trend_score"]
        df_daily_ctx = df_daily_ctx[ctx_cols].drop_duplicates(["date", "symbol"])

        df = df.merge(df_daily_ctx, on=["date", "symbol"], how="left")

        # Forward-fill over weekends / missing days
        df = df.sort_values(["symbol", "datetime"])
        for col in ["daily_regime", "daily_above_sma200", "daily_adx",
                    "daily_atr_pct", "daily_trend_score"]:
            df[col] = df.groupby("symbol")[col].ffill()
    else:
        for col in ["daily_regime", "daily_above_sma200", "daily_adx",
                    "daily_atr_pct", "daily_trend_score"]:
            df[col] = np.nan

    return df.reset_index(drop=True)
