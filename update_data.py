#!/usr/bin/env python3
"""
update_data.py — Pull missing 1-min bars from IBKR and append to parquet.

Fetches ES + NQ 1-min bars from the day after the last parquet date
up to the current session, aligns them, and appends to the existing file.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import ib_insync as ib

PARQUET = ("/Users/ishanbhardwaj/cme_competition_system/output/mtf/"
           "ES_NQ_1min_aligned.parquet")


def to_aligned_df(es_bars, nq_bars, contract_es, contract_nq):
    """Convert ib_insync bar lists to aligned parquet-format DataFrame."""
    es_df = ib.util.df(es_bars).copy()
    nq_df = ib.util.df(nq_bars).copy()

    # Timestamps are already tz-aware from IBKR
    es_df["timestamp"] = pd.to_datetime(es_df["date"]).dt.tz_convert("UTC")
    nq_df["timestamp"] = pd.to_datetime(nq_df["date"]).dt.tz_convert("UTC")

    es_df = es_df.rename(columns={
        "open":   "ES_open",   "high":   "ES_high",
        "low":    "ES_low",    "close":  "ES_close",
        "volume": "ES_volume",
    })[["timestamp","ES_open","ES_high","ES_low","ES_close","ES_volume"]]
    es_df["ES_active_contract"] = contract_es

    nq_df = nq_df.rename(columns={
        "open":   "NQ_open",   "high":   "NQ_high",
        "low":    "NQ_low",    "close":  "NQ_close",
        "volume": "NQ_volume",
    })[["timestamp","NQ_open","NQ_high","NQ_low","NQ_close","NQ_volume"]]
    nq_df["NQ_active_contract"] = contract_nq

    merged = pd.merge(es_df, nq_df, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


async def main():
    # Load existing parquet
    print("Loading existing parquet …")
    existing = pd.read_parquet(PARQUET)
    last_ts  = existing["timestamp"].max()
    print(f"  Last timestamp : {last_ts}")
    print(f"  Rows           : {len(existing):,}")

    # Connect
    print("\nConnecting to IBKR (port 7496) …")
    ibc = ib.IB()
    await ibc.connectAsync("127.0.0.1", 7496, clientId=77)
    print("  Connected ✓")

    # Front-month contracts
    es = ib.Future("ES", "20260618", "CME")
    nq = ib.Future("NQ", "20260618", "CME")
    await ibc.qualifyContractsAsync(es, nq)
    print(f"  ES: {es.localSymbol}   NQ: {nq.localSymbol}")

    # How many calendar days are we missing?
    now_utc   = datetime.now(timezone.utc)
    days_back = (now_utc - last_ts).days + 2   # +2 for safety
    days_back = min(days_back, 30)             # IBKR limit for 1-min bars
    print(f"\nPulling last {days_back} calendar days of 1-min bars …")

    es_bars = await ibc.reqHistoricalDataAsync(
        es, endDateTime="", durationStr=f"{days_back} D",
        barSizeSetting="1 min", whatToShow="TRADES",
        useRTH=False, formatDate=2)

    nq_bars = await ibc.reqHistoricalDataAsync(
        nq, endDateTime="", durationStr=f"{days_back} D",
        barSizeSetting="1 min", whatToShow="TRADES",
        useRTH=False, formatDate=2)

    ibc.disconnect()

    print(f"  ES bars fetched: {len(es_bars)}")
    print(f"  NQ bars fetched: {len(nq_bars)}")

    if len(es_bars) == 0 or len(nq_bars) == 0:
        print("No bars returned — nothing to append.")
        return

    # Convert to aligned DataFrame
    new_df = to_aligned_df(es_bars, nq_bars, es.localSymbol, nq.localSymbol)
    print(f"  Aligned rows   : {len(new_df):,}")

    # Keep only rows AFTER the last existing timestamp
    new_df = new_df[new_df["timestamp"] > last_ts].copy()
    print(f"  New rows       : {len(new_df):,}")

    if len(new_df) == 0:
        print("\nParquet already up to date — nothing to append.")
        return

    # Append
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Validate no duplicates
    dupes = combined.duplicated(subset=["timestamp"]).sum()
    if dupes:
        print(f"  ⚠️  Removing {dupes} duplicate timestamps")
        combined = combined.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Save
    combined.to_parquet(PARQUET, index=False)
    new_last = combined["timestamp"].max()
    print(f"\n✅  Parquet updated")
    print(f"   Rows     : {len(existing):,}  →  {len(combined):,}  (+{len(new_df):,})")
    print(f"   Coverage : {combined['timestamp'].min()}  →  {new_last}")


if __name__ == "__main__":
    asyncio.run(main())
