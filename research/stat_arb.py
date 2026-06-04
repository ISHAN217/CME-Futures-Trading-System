"""
stat_arb.py — NQ dominance stat arb. [CHANGED from base system]

Spread definition (unchanged):
    spread = log(NQ) - beta * log(ES)
    beta   = 60-day rolling OLS of log(NQ) ~ log(ES)

Entry: spread_zscore < -SA_ENTRY_THRESHOLD (-1.5)
    NQ is cheap relative to ES → expect mean reversion upward.
    LONG NQ only. No ES SHORT hedge (SA_NO_ES_HEDGE = True).

WHY NQ-only?
    The base system already established that the ES SHORT leg in the pair
    trade lost at 35% win rate, dragging down the SA strategy by ~5%.
    NQ has structural outperformance vs ES (secular tech growth premium).
    When NQ is cheap relative to ES, both tend to go up, but NQ goes up MORE.
    Running NQ-only captures the mean-reversion AND the structural alpha.

SHORT NQ direction disabled (SA_NQ_ONLY = True):
    When the spread is high (NQ expensive), shorting NQ fights the structural
    drift. Entry rate for the SHORT direction at z > +1.5 is lower because
    NQ keeps going. Disabled entirely.

Exit [CHANGED]:
    SA_EXIT_THRESHOLD = 0.40 (was 0.50 in base system).
    WHY: Competition windows are 5-7 days. A 0.50 exit is too loose —
    it keeps positions open a day longer on average, which consumes the
    competition window. Exiting at 0.40 books profits earlier and frees
    capital for the next signal.

NQ dominance enhancement [NEW]:
    nq_rel_ret_5d > 0 (NQ outperforming ES recently) is a confirmation
    signal. Trades confirmed by NQ dominance get sa_strength bonus of +0.2.

Output per (date, symbol):
    sa_signal        ENTRY / FLAT
    sa_es_direction  FLAT (always — hedge disabled)
    sa_nq_direction  LONG / FLAT
    sa_strength      |spread_zscore| / SA_ENTRY_THRESHOLD + nq_dom bonus
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

_ENTRY = config.SA_ENTRY_THRESHOLD   # 1.5
_EXIT  = config.SA_EXIT_THRESHOLD    # 0.40


def compute_stat_arb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SA signals. spread_zscore already computed in features.py.
    """
    df   = df.copy()
    dates = df["date"].unique()
    sa_daily = _compute_daily_sa(df, dates)
    df = df.merge(sa_daily, on="date", how="left")
    _log_sa_distribution(df)
    return df


def _compute_daily_sa(df: pd.DataFrame, dates) -> pd.DataFrame:
    # Use ES rows for spread signal (one per date)
    es_df = (
        df[df["symbol"] == "ES"][["date", "spread_zscore"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # NQ dominance: nq_rel_ret_5d from the NQ row
    nq_df = (
        df[df["symbol"] == "NQ"][["date", "nq_rel_ret_5d"]]
        .drop_duplicates("date")
        if "nq_rel_ret_5d" in df.columns
        else pd.DataFrame(columns=["date", "nq_rel_ret_5d"])
    )

    if not nq_df.empty:
        es_df = es_df.merge(nq_df, on="date", how="left")
    else:
        es_df["nq_rel_ret_5d"] = np.nan

    records = []
    for _, row in es_df.iterrows():
        date = row["date"]
        z    = row["spread_zscore"]
        nq_r = row.get("nq_rel_ret_5d", np.nan)

        if pd.isna(z):
            records.append(_flat_record(date)); continue

        if z < -_ENTRY:
            # NQ cheap relative to ES → expect NQ to revert up
            base_strength = abs(z) / _ENTRY
            # NQ dominance bonus: if NQ has outperformed recently, +0.2
            dom_bonus = 0.2 if (not pd.isna(nq_r) and nq_r > 0) else 0.0
            records.append({
                "date":            date,
                "sa_signal":       "ENTRY",
                "sa_es_direction": "FLAT",          # hedge disabled
                "sa_nq_direction": "LONG",
                "sa_strength":     min(base_strength + dom_bonus, 3.0),
                "sa_nq_dominant":  bool(dom_bonus > 0),
            })
        else:
            # No SHORT NQ direction (SA_NQ_ONLY = True always)
            records.append(_flat_record(date))

    return pd.DataFrame(records)


def _flat_record(date) -> dict:
    return {
        "date":            date,
        "sa_signal":       "FLAT",
        "sa_es_direction": "FLAT",
        "sa_nq_direction": "FLAT",
        "sa_strength":     0.0,
        "sa_nq_dominant":  False,
    }


def _log_sa_distribution(df: pd.DataFrame) -> None:
    es_only = df[df["symbol"] == "ES"].drop_duplicates("date")
    vc  = es_only["sa_signal"].value_counts()
    dom = es_only.get("sa_nq_dominant", pd.Series(False))
    n   = len(es_only)
    logger.info(
        "SA signals (NQ-only): ENTRY=%d (%.0f%%)  FLAT=%d (%.0f%%)  "
        "NQ_dominant_entries=%d",
        vc.get("ENTRY", 0), 100 * vc.get("ENTRY", 0) / max(n, 1),
        vc.get("FLAT",  0), 100 * vc.get("FLAT",  0) / max(n, 1),
        int(dom.sum()) if len(dom) > 0 else 0,
    )
