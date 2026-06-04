#!/usr/bin/env python3
"""
backtest_target.py — Compare profit target multiples (1.5R / 2.0R / 2.5R / 3.0R)

Tests how changing the profit target changes WR, EV, and Sharpe for both
ORB and PDH/L strategies. Optimising for max EV (competition objective),
not max Sharpe (long-run objective).
"""

import asyncio, sys, time as _time
import numpy as np
import pandas as pd
from datetime import time
from orb_live import run_full_backtest, ORBConfig

DATA = ("/Users/ishanbhardwaj/cme_competition_system/output/mtf/"
        "ES_NQ_1min_aligned.parquet")

MULTIPLES = [1.5, 2.0, 2.5, 3.0]

# PDH/L simulation reused from backtest_pdhl
OR_START     = time(9, 30);   OR_END       = time(10, 0)
ORB_CUTOFF   = time(10, 30);  WATCH_START  = time(10, 0)
ENTRY_CUTOFF = time(11, 0);   EOD_CLOSE    = time(15, 30)
RTH_START    = time(9, 30);   RTH_END      = time(16, 0)
MIN_ES = 8.0; MAX_ES = 60.0; MIN_NQ = 30.0; MAX_NQ = 150.0
SLIP = 0.25


def simulate_pdhl_at_mult(df, mult):
    """Run PDH/L simulation with a specific profit multiple. Returns trade list."""
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    trades = []

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue

        or_b = day[(day["time_et"] >= OR_START) & (day["time_et"] < OR_END)]
        if len(or_b) == 0: continue

        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min()
        nq_H = or_b["NQ_high"].max(); nq_L = or_b["NQ_low"].min()
        es_R = es_H - es_L;           nq_R = nq_H - nq_L
        if not (MIN_ES <= es_R <= MAX_ES and MIN_NQ <= nq_R <= MAX_NQ): continue

        # ORB check
        post = day[(day["time_et"] >= OR_END) & (day["time_et"] < ORB_CUTOFF)]
        es_b = nq_b = None
        for _, r in post.iterrows():
            if es_b is None:
                if r["ES_high"] > es_H: es_b = "L"
                elif r["ES_low"]  < es_L: es_b = "S"
            if nq_b is None:
                if r["NQ_high"] > nq_H: nq_b = "L"
                elif r["NQ_low"]  < nq_L: nq_b = "S"
            if es_b and nq_b: break
        if es_b and nq_b and es_b == nq_b: continue   # ORB fired — skip

        prev  = by_d.get(dates[i-1])
        if prev is None: continue
        prth  = prev[(prev["time_et"] >= RTH_START) & (prev["time_et"] <= RTH_END)]
        if len(prth) == 0: continue

        es_PDH = prth["ES_high"].max(); es_PDL = prth["ES_low"].min()
        nq_PDH = prth["NQ_high"].max(); nq_PDL = prth["NQ_low"].min()

        watch = day[(day["time_et"] >= WATCH_START) & (day["time_et"] < ENTRY_CUTOFF)]
        es_up = es_dn = nq_up = nq_dn = False
        direction = None; entry_row = None

        for _, r in watch.iterrows():
            if not es_up and r["ES_high"] > es_PDH: es_up = True
            if not es_dn and r["ES_low"]  < es_PDL: es_dn = True
            if not nq_up and r["NQ_high"] > nq_PDH: nq_up = True
            if not nq_dn and r["NQ_low"]  < nq_PDL: nq_dn = True
            if es_up and nq_up and direction is None: direction = "LONG";  entry_row = r; break
            if es_dn and nq_dn and direction is None: direction = "SHORT"; entry_row = r; break
            if (es_up and nq_dn) or (es_dn and nq_up): direction = "CONFLICT"; break

        if direction in (None, "CONFLICT"): continue
        entry_t = entry_row["time_et"]

        for sym, pdh, pdl, rng, col_h, col_l, col_c in [
            ("ES", es_PDH, es_PDL, es_R, "ES_high", "ES_low", "ES_close"),
            ("NQ", nq_PDH, nq_PDL, nq_R, "NQ_high", "NQ_low", "NQ_close"),
        ]:
            entry_px = pdh if direction == "LONG" else pdl
            stop_px  = entry_px - rng if direction == "LONG" else entry_px + rng
            tgt_px   = entry_px + rng * mult if direction == "LONG" else entry_px - rng * mult

            after = day[day["time_et"] > entry_t]
            exit_px = exit_r = None
            for _, bar in after.iterrows():
                t = bar["time_et"]
                if t >= EOD_CLOSE:
                    exit_px = bar[col_c]; exit_r = "EOD"; break
                if direction == "LONG":
                    if bar[col_l] <= stop_px: exit_px = stop_px; exit_r = "STOP"; break
                    if bar[col_h] >= tgt_px:  exit_px = tgt_px;  exit_r = "TGT";  break
                else:
                    if bar[col_h] >= stop_px: exit_px = stop_px; exit_r = "STOP"; break
                    if bar[col_l] <= tgt_px:  exit_px = tgt_px;  exit_r = "TGT";  break
            if exit_px is None: continue

            sign    = 1 if direction == "LONG" else -1
            net_pts = sign * (exit_px - entry_px) - SLIP - (SLIP if exit_r in ("STOP","EOD") else 0.0)
            r_val   = net_pts / rng if rng > 0 else 0.0
            trades.append(dict(r=r_val, win=r_val > 0, exit_r=exit_r))

    return trades


async def run_orb_at_mult(df, mult):
    cfg = ORBConfig(
        or_start=time(9,30), or_end=time(10,0),
        entry_cutoff=time(10,30), eod_close=time(15,30),
        min_range_mes=8.0, max_range_mes=60.0,
        min_range_mnq=30.0, max_range_mnq=150.0,
        profit_mult=mult, slip_ticks=1, commission_rt=0.74,
    )
    result = await run_full_backtest(df, cfg, starting_capital=25_000.0)
    s = result["stats"]
    tl = pd.DataFrame(result["trade_log"]) if result["trade_log"] else pd.DataFrame()
    wr = float((tl["pnl_%"] > 0).mean()) if len(tl) else 0.0
    return s, wr


def main():
    print("Loading data …")
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    df = df.sort_values("ts_et").reset_index(drop=True)
    print(f"  {len(df):,} bars   {sorted(df['date_et'].unique())[0]} → {sorted(df['date_et'].unique())[-1]}\n")

    # ── ORB at each multiple ──────────────────────────────────────────────────
    print("Testing ORB at each profit multiple (this takes ~60s) …")
    orb_results = {}
    for mult in MULTIPLES:
        t0 = _time.time()
        s, wr = asyncio.run(run_orb_at_mult(df, mult))
        elapsed = _time.time() - t0
        ev = wr * mult - (1 - wr) * 1.0
        orb_results[mult] = dict(sh=s["sharpe_ratio"], ann=s["annualized_return_pct"],
                                 wr=wr, ev=ev, t=s["total_trades"], elapsed=elapsed)
        print(f"  ORB {mult}R  done in {elapsed:.0f}s")

    # ── PDH/L at each multiple ────────────────────────────────────────────────
    print("\nTesting PDH/L at each profit multiple …")
    pdhl_results = {}
    for mult in MULTIPLES:
        trades = simulate_pdhl_at_mult(df, mult)
        if not trades:
            pdhl_results[mult] = dict(wr=0, ev=0, n=0)
            continue
        tdf = pd.DataFrame(trades)
        wr  = tdf["win"].mean()
        ev  = wr * mult - (1 - wr)
        pdhl_results[mult] = dict(wr=wr, ev=ev, n=len(tdf))

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("ORB — PROFIT TARGET COMPARISON")
    print("=" * 68)
    print(f"  {'Target':>8}  {'WR':>7}  {'EV':>8}  {'Sharpe':>8}  {'Ann Ret':>9}  {'Trades':>7}")
    print(f"  {'─'*60}")
    for mult in MULTIPLES:
        r = orb_results[mult]
        marker = " ◀ competition" if r["ev"] == max(v["ev"] for v in orb_results.values()) else ""
        print(f"  {mult}R      {r['wr']*100:5.1f}%   {r['ev']:+6.3f}R   "
              f"{r['sh']:7.3f}   {r['ann']:+8.1f}%   {r['t']:6d}{marker}")

    print()
    print("=" * 68)
    print("PDH/L — PROFIT TARGET COMPARISON")
    print("=" * 68)
    print(f"  {'Target':>8}  {'WR':>7}  {'EV':>8}  {'Legs':>7}")
    print(f"  {'─'*40}")
    for mult in MULTIPLES:
        r = pdhl_results[mult]
        marker = " ◀ competition" if r["ev"] == max(v["ev"] for v in pdhl_results.values()) else ""
        print(f"  {mult}R      {r['wr']*100:5.1f}%   {r['ev']:+6.3f}R   {r['n']:6d}{marker}")

    # ── Recommendation ────────────────────────────────────────────────────────
    best_orb  = max(orb_results,  key=lambda m: orb_results[m]["ev"])
    best_pdhl = max(pdhl_results, key=lambda m: pdhl_results[m]["ev"])

    print()
    print("=" * 68)
    print("COMPETITION RECOMMENDATION  (max EV, not max Sharpe)")
    print("=" * 68)
    print(f"  ORB  optimal target  : {best_orb}R   "
          f"EV={orb_results[best_orb]['ev']:+.3f}R  WR={orb_results[best_orb]['wr']*100:.1f}%")
    print(f"  PDH/L optimal target : {best_pdhl}R   "
          f"EV={pdhl_results[best_pdhl]['ev']:+.3f}R  WR={pdhl_results[best_pdhl]['wr']*100:.1f}%")

    if best_orb == best_pdhl:
        print(f"\n  ✅  Both systems agree: use {best_orb}R profit target")
    else:
        print(f"\n  ⚠️   Systems disagree — ORB wants {best_orb}R, PDH/L wants {best_pdhl}R")
        print(f"      Recommend: use {best_orb}R for both (ORB is the primary system)")
    print()


if __name__ == "__main__":
    main()
