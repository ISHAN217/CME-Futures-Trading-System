#!/usr/bin/env python3
"""update_cl_data.py — Pull missing CL 1-min bars from IBKR and append."""
import asyncio, pandas as pd
from datetime import datetime, timezone
import ib_insync as ib

PARQUET = "output/mtf/CL_1min_continuous.parquet"

async def main():
    print("Loading existing CL parquet …")
    existing = pd.read_parquet(PARQUET)
    existing["ts_et"] = pd.to_datetime(existing["ts_et"])
    last_ts = existing["timestamp"].max()
    print(f"  Last bar : {last_ts}  |  Rows: {len(existing):,}")

    print("\nConnecting to IBKR …")
    ibc = ib.IB()
    await ibc.connectAsync("127.0.0.1", 7496, clientId=78)
    print("  Connected ✓")

    # CLN6 — July 2026 front month (conId verified)
    cl = ib.Future(conId=304037461, exchange="NYMEX")
    await ibc.qualifyContractsAsync(cl)
    print(f"  CL contract: {cl.localSymbol}")

    now_utc   = datetime.now(timezone.utc)
    days_back = min((now_utc - last_ts).days + 3, 30)
    print(f"\nPulling last {days_back} calendar days …")

    bars = await ibc.reqHistoricalDataAsync(
        cl, endDateTime="", durationStr=f"{days_back} D",
        barSizeSetting="1 min", whatToShow="TRADES",
        useRTH=False, formatDate=2, timeout=120)
    ibc.disconnect()

    print(f"  Bars fetched: {len(bars)}")
    if not len(bars):
        print("No bars returned."); return

    df = ib.util.df(bars).copy()
    df["timestamp"] = pd.to_datetime(df["date"]).dt.tz_convert("UTC")
    df["ts_et"]     = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.rename(columns={
        "open":"CL_open","high":"CL_high","low":"CL_low",
        "close":"CL_close","volume":"CL_volume"
    })
    df["hour_et"]    = df["ts_et"].dt.hour
    df["min_et"]     = df["ts_et"].dt.minute
    df["trade_date"] = df["ts_et"].dt.date.astype(str)
    df = df[["timestamp","ts_et","CL_open","CL_high","CL_low",
             "CL_close","CL_volume","hour_et","min_et","trade_date"]]

    new = df[df["timestamp"] > last_ts].copy()
    print(f"  New rows   : {len(new):,}")
    if not len(new):
        print("Already up to date."); return

    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    # Ensure consistent dtypes
    combined["trade_date"] = combined["trade_date"].astype(str)
    combined["ts_et"]      = pd.to_datetime(combined["ts_et"]).dt.tz_convert("America/New_York")
    combined.to_parquet(PARQUET, index=False)
    print(f"\n✅ CL parquet updated")
    print(f"   {len(existing):,} → {len(combined):,}  (+{len(new):,})")
    print(f"   Coverage: {combined['timestamp'].min()} → {combined['timestamp'].max()}")

if __name__ == "__main__":
    asyncio.run(main())
