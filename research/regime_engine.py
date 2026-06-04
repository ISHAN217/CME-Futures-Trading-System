"""
regime_engine.py — Daily regime classification with TRANSITION state.

[CHANGED] Added TRANSITION regime between TREND and CHOP.

Four regimes:
  TREND:       R² > 0.25, positive slope, vol not spiking
               → full trend and pullback entries
  TRANSITION:  R² between 0.10 and 0.25 (gray zone), or recent regime change
               → half-size, no new trend entries, allow MR/SA exits only
  CHOP:        R² < 0.10, no directional structure
               → MR + SA (NQ dominance) full activation
  SHOCK:       |ret_1d| > 2.5σ OR vol_ratio > 2.0
               → exit all, no new entries

WHY TRANSITION matters:
  The base system used a "2-day buffer" to prevent position carry-over losses
  at regime boundaries. Making TRANSITION an explicit regime with its own
  rules is cleaner: the diagnostics clearly show when the system was in
  TRANSITION, how long those periods lasted, and whether they were correctly
  cautious. A buffer hidden in the signal engine is opaque; TRANSITION is not.

Portfolio regime: worst of ES and NQ
  If either symbol is SHOCK → portfolio is SHOCK
  Then TRANSITION → TRANSITION (one TRANSITION doesn't doom both to CHOP)

Output per (date, symbol):
  regime              TREND / TRANSITION / CHOP / SHOCK
  regime_confidence   float in [0, 1]
  portfolio_regime    same hierarchy applied across both symbols
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

_RANK = {"TREND": 0, "TRANSITION": 1, "CHOP": 2, "SHOCK": 3}
_RMAP = {0: "TREND", 1: "TRANSITION", 2: "CHOP", 3: "SHOCK"}


def compute_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.copy().sort_values("date").reset_index(drop=True)
        g = _classify(g)
        parts.append(g)

    df = pd.concat(parts, ignore_index=True)
    df = _portfolio_regime(df)
    _log_distribution(df)
    return df


def _classify(g: pd.DataFrame) -> pd.DataFrame:
    ret       = g["ret_1d"].fillna(0)
    vol_20    = g["vol_20d"].fillna(g["vol_20d"].median()).replace(0, 1e-6)
    vol_ratio = g["vol_ratio"].fillna(1.0)
    r2        = g["r2_20d"].fillna(0.0)
    slope     = g["slope_20d"].fillna(0.0)

    is_shock = (
        (ret.abs() > config.SHOCK_RETURN_SIGMA * vol_20)
        | (vol_ratio > config.SHOCK_VOL_RATIO)
    )

    is_trend = (
        (~is_shock)
        & (r2 >= config.TREND_R2_MIN)
        & (slope >= 0)
    )

    # TRANSITION: between TREND and CHOP thresholds, or R² heading lower
    is_transition = (
        (~is_shock)
        & (~is_trend)
        & (r2 >= config.TRANSITION_R2_MIN)
    )

    # CHOP: not SHOCK, not TREND, not TRANSITION
    # (implicitly: r2 < TRANSITION_R2_MIN)

    regime = np.where(
        is_shock,      "SHOCK",
        np.where(
            is_trend,     "TREND",
            np.where(
                is_transition, "TRANSITION",
                "CHOP"
            )
        )
    )
    g["regime"] = regime

    # Confidence
    trend_conf = (r2.clip(0, 1) * (1 - vol_ratio.clip(0, 2) / 2).clip(0, 1))
    chop_conf  = (1 - r2).clip(0, 1)
    shock_conf = (ret.abs() / vol_20 / config.SHOCK_RETURN_SIGMA).clip(0, 1)
    trans_conf = 1 - (r2 - config.TRANSITION_R2_MIN) / (
        config.TREND_R2_MIN - config.TRANSITION_R2_MIN + 1e-6
    )

    g["regime_confidence"] = np.where(
        is_shock,      shock_conf,
        np.where(is_trend, trend_conf,
        np.where(is_transition, trans_conf.clip(0, 1), chop_conf))
    )
    return g


def _portfolio_regime(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("date")["regime"]
        .apply(lambda x: _RMAP[max(_RANK.get(r, 0) for r in x)])
        .reset_index()
        .rename(columns={"regime": "portfolio_regime"})
    )
    return df.merge(daily, on="date", how="left")


def _log_distribution(df: pd.DataFrame) -> None:
    port = df.drop_duplicates("date")
    vc   = port["portfolio_regime"].value_counts()
    n    = len(port)
    logger.info(
        "Regime — TREND: %d (%.0f%%)  TRANSITION: %d (%.0f%%)  "
        "CHOP: %d (%.0f%%)  SHOCK: %d (%.0f%%)",
        vc.get("TREND",      0), 100 * vc.get("TREND",      0) / n,
        vc.get("TRANSITION", 0), 100 * vc.get("TRANSITION", 0) / n,
        vc.get("CHOP",       0), 100 * vc.get("CHOP",       0) / n,
        vc.get("SHOCK",      0), 100 * vc.get("SHOCK",      0) / n,
    )
