"""
features.py — Daily feature engineering.

Adds to base system:
  ewma_vol    EWMA volatility (λ=0.94) — responds faster to vol changes than rolling std
  atr_14      14-period Average True Range — natural stop reference
  rel_ret_5d  NQ excess return vs ES over 5 days — NQ dominance indicator
  ret_60d     60-day return for long-term momentum window

All features are computed from PAST data only (no lookahead).
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import config

logger = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()

    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.copy().sort_values("date").reset_index(drop=True)
        g = _symbol_features(g)
        parts.append(g)

    df = pd.concat(parts, ignore_index=True)
    df = _pair_features(df)

    logger.info("Features computed: %d rows, %d symbols", len(df), df["symbol"].nunique())
    return df


def _symbol_features(g: pd.DataFrame) -> pd.DataFrame:
    g["log_close"] = np.log(g["close"].clip(lower=1e-9))

    # Returns
    g["ret_1d"]  = g["close"].pct_change(1)
    g["ret_5d"]  = g["close"].pct_change(5)
    g["ret_10d"] = g["close"].pct_change(10)
    g["ret_20d"] = g["close"].pct_change(20)
    g["ret_60d"] = g["close"].pct_change(60)   # for long-term momentum window

    # Rolling volatility
    g["vol_5d"]  = g["ret_1d"].rolling(5,  min_periods=3).std()
    g["vol_20d"] = g["ret_1d"].rolling(20, min_periods=10).std()
    g["vol_ratio"] = (g["vol_5d"] / g["vol_20d"].replace(0, np.nan)).fillna(1.0)

    # EWMA volatility (RiskMetrics, λ=0.94)
    # ewm with alpha=1-λ gives: var_t = (1-λ)r²_t + λ*var_{t-1}
    g["ewma_vol"] = (
        g["ret_1d"].ewm(alpha=1 - config.EWMA_LAMBDA, adjust=False)
        .var()
        .apply(np.sqrt)
    )

    # ATR (14-period)
    g["atr_14"] = _compute_atr(g["high"], g["low"], g["close"], config.ATR_PERIOD)

    # Gap and range
    prev = g["close"].shift(1)
    g["gap_pct"]   = (g["open"] - prev) / prev.replace(0, np.nan)
    g["range_pct"] = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)

    # Volume z-score
    vol_m = g["volume"].rolling(20, min_periods=5).mean()
    vol_s = g["volume"].rolling(20, min_periods=5).std().replace(0, np.nan)
    g["volume_z"] = (g["volume"] - vol_m) / vol_s

    # Trend quality via rolling OLS
    g = _rolling_ols(g, window=20)
    return g


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range — maximum of (H-L), |H-prev_C|, |L-prev_C|."""
    prev_c = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period // 2, adjust=False).mean()


def _rolling_ols(g: pd.DataFrame, window: int) -> pd.DataFrame:
    n      = len(g)
    slopes = np.full(n, np.nan)
    r2s    = np.full(n, np.nan)
    lc     = g["log_close"].values

    for i in range(window - 1, n):
        y = lc[i - window + 1 : i + 1]
        if np.isnan(y).any():
            continue
        x = np.arange(window, dtype=float)
        try:
            sl, _, r, _, _ = scipy_stats.linregress(x, y)
            slopes[i] = sl
            r2s[i]    = r * r
        except Exception:
            pass

    g["slope_20d"] = slopes
    g["r2_20d"]    = r2s
    return g


def _pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """ES/NQ spread features + relative return (NQ dominance indicator)."""
    es = df[df["symbol"] == "ES"][["date", "log_close", "ret_5d"]].rename(
        columns={"log_close": "log_es", "ret_5d": "ret5_es"}
    ).sort_values("date")
    nq = df[df["symbol"] == "NQ"][["date", "log_close", "ret_5d"]].rename(
        columns={"log_close": "log_nq", "ret_5d": "ret5_nq"}
    ).sort_values("date")

    pair = es.merge(nq, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Rolling OLS beta (60-day, strictly excludes today → no lookahead)
    n     = len(pair)
    betas = np.full(n, np.nan)
    W     = config.SA_BETA_WINDOW
    for i in range(W, n):
        x = pair["log_es"].iloc[i - W : i].values
        y = pair["log_nq"].iloc[i - W : i].values
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        try:
            sl, _, _, _, _ = scipy_stats.linregress(x, y)
            if sl > 0:
                betas[i] = sl
        except Exception:
            pass

    pair["beta"]   = betas
    pair["spread"] = pair["log_nq"] - pair["beta"] * pair["log_es"]

    W_z = config.SA_ZSCORE_WINDOW
    s_m = pair["spread"].rolling(W_z, min_periods=W_z // 2).mean()
    s_s = pair["spread"].rolling(W_z, min_periods=W_z // 2).std().replace(0, np.nan)
    pair["spread_zscore"] = (pair["spread"] - s_m) / s_s

    # NQ excess return vs ES over 5 days (positive = NQ outperforming)
    pair["nq_rel_ret_5d"] = pair["ret5_nq"] - pair["ret5_es"]

    spread_cols = pair[["date", "beta", "spread", "spread_zscore", "nq_rel_ret_5d"]]
    df = df.merge(spread_cols, on="date", how="left")
    return df
