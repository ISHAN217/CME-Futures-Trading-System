#!/usr/bin/env python3
"""
backtest_nq_filter.py — Test 3 alternatives to the 150pt NQ OR range cap

Baseline : ES 8–60pts  NQ 30–150pts  (both legs)
Option A  : ES 8–60pts  NQ 30–175pts  (both legs)
Option B  : ES 8–60pts  NQ 30–200pts  (both legs)
Option C  : ES 8–60pts  NQ no max     (both legs)
Option D  : ES 8–60pts  NQ any range  — but trade ES leg ONLY when NQ > 150pts
"""

import numpy as np
import pandas as pd
from datetime import time

DATA = ("/Users/ishanbhardwaj/cme_competition_system/output/mtf/"
        "ES_NQ_1min_aligned.parquet")

OR_START   = time(9, 30);  OR_END     = time(10, 0)
ORB_CUTOFF = time(10, 30); EOD_CLOSE  = time(15, 30)
MIN_ES = 8.0; MAX_ES = 60.0; MIN_NQ = 30.0
SLIP   = 0.25


def simulate_day(day, direction, entry_t, entries, pmult=3.0):
    """
    entries: list of (sym, entry_px, stop_px, tgt_px, col_h, col_l, col_c)
    Returns list of r_val per leg.
    """
    results = []
    after = day[day["time_et"] > entry_t]
    for (sym, ep, sp, tp, col_h, col_l, col_c) in entries:
        rng = abs(ep - sp)
        exit_px = exit_r = None
        for _, bar in after.iterrows():
            t = bar["time_et"]
            if t >= EOD_CLOSE:
                sign = 1 if direction == "LONG" else -1
                net  = sign * (bar[col_c] - ep) - SLIP * 2
                exit_r = "EOD"; exit_px = bar[col_c]; break
            if direction == "LONG":
                if bar[col_l] <= sp: exit_px = sp;  exit_r = "STOP"; break
                if bar[col_h] >= tp: exit_px = tp;  exit_r = "TGT";  break
            else:
                if bar[col_h] >= sp: exit_px = sp;  exit_r = "STOP"; break
                if bar[col_l] <= tp: exit_px = tp;  exit_r = "TGT";  break
        if exit_px is None:
            continue
        if exit_r == "EOD":
            sign = 1 if direction == "LONG" else -1
            net  = sign * (exit_px - ep) - SLIP * 2
            r_val = net / rng if rng > 0 else 0
        elif exit_r == "STOP":
            r_val = -1.0 - SLIP * 2 / rng
        else:
            r_val = pmult - SLIP / rng
        results.append(dict(sym=sym, r=r_val, exit=exit_r, win=r_val > 0))
    return results


def run_scenario(df, label, max_nq, es_only_when_nq_wide=False, pmult=3.0):
    """
    max_nq             : max allowed NQ OR range (float or inf)
    es_only_when_nq_wide: if True, trade ES leg only when NQ > 150pts
    """
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    trades = []
    signal_days = 0

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue

        or_b = day[(day["time_et"] >= OR_START) & (day["time_et"] < OR_END)]
        if len(or_b) == 0: continue

        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min(); es_R = es_H - es_L
        nq_H = or_b["NQ_high"].max(); nq_L = or_b["NQ_low"].min(); nq_R = nq_H - nq_L

        es_ok = MIN_ES <= es_R <= MAX_ES
        nq_ok = MIN_NQ <= nq_R <= max_nq

        if not es_ok:
            continue

        nq_wide = (nq_R > 150)  # NQ above original cap

        # Determine which legs to trade
        if es_only_when_nq_wide:
            # Always need ES valid; NQ leg only if also valid (≤150)
            if not es_ok:
                continue
            use_nq = nq_ok and not nq_wide   # NQ leg only if within original cap
            use_es = True
        else:
            # Both legs required — standard dual-confirmation
            if not nq_ok:
                continue
            use_es = use_nq = True

        # ORB breakout detection
        post = day[(day["time_et"] >= OR_END) & (day["time_et"] < ORB_CUTOFF)]
        es_b = nq_b = None; entry_row = None

        for _, r in post.iterrows():
            if use_es and es_b is None:
                if r["ES_high"] > es_H: es_b = "L"
                elif r["ES_low"] < es_L: es_b = "S"
            if use_nq and nq_b is None:
                if r["NQ_high"] > nq_H: nq_b = "L"
                elif r["NQ_low"] < nq_L: nq_b = "S"
            # Confirm: ES must break; NQ must agree if trading it
            es_broke = (es_b is not None)
            nq_broke = (nq_b is not None) if use_nq else True
            if es_broke and nq_broke:
                # Direction conflict check
                if use_nq and es_b != nq_b:
                    es_b = nq_b = None  # reset on conflict
                    continue
                entry_row = r; break

        if entry_row is None: continue
        direction = "LONG" if es_b == "L" else "SHORT"

        # Build entries list
        entries = []
        if use_es:
            ep = es_H if direction == "LONG" else es_L
            sp = es_L if direction == "LONG" else es_H
            tp = round(ep + pmult * es_R, 2) if direction == "LONG" else round(ep - pmult * es_R, 2)
            entries.append(("ES", ep, sp, tp, "ES_high", "ES_low", "ES_close"))
        if use_nq:
            ep = nq_H if direction == "LONG" else nq_L
            sp = nq_L if direction == "LONG" else nq_H
            tp = round(ep + pmult * nq_R, 2) if direction == "LONG" else round(ep - pmult * nq_R, 2)
            entries.append(("NQ", ep, sp, tp, "NQ_high", "NQ_low", "NQ_close"))

        if not entries: continue

        day_trades = simulate_day(day, direction, entry_row["time_et"], entries, pmult)
        if day_trades:
            signal_days += 1
            for t in day_trades:
                t["date"] = d
                trades.append(t)

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["r","win","exit","sym"])
    if len(tdf) == 0:
        return dict(label=label, n_days=0, n_legs=0, wr=0, avg_r=0, ev=0, sharpe=0)

    wr    = tdf["win"].mean()
    avg_r = tdf["r"].mean()
    ev    = wr * pmult - (1 - wr) * 1.0

    # Daily returns for Sharpe
    day_r = tdf.groupby("date")["r"].sum()
    sharpe = day_r.mean() / day_r.std(ddof=1) * np.sqrt(252) if day_r.std() > 0 else 0

    # Break out wide-NQ days separately for option D
    extra = ""
    if es_only_when_nq_wide and "sym" in tdf.columns:
        es_only = tdf[tdf["sym"] == "ES"]
        both    = tdf.groupby("date").filter(lambda x: len(x) == 2)
        if len(es_only):
            extra = (f"\n    ES-only legs (NQ wide): n={len(es_only)}  "
                     f"WR={es_only['win'].mean()*100:.1f}%  "
                     f"AvgR={es_only['r'].mean():+.3f}R")

    return dict(label=label, n_days=signal_days, n_legs=len(tdf),
                wr=wr, avg_r=avg_r, ev=ev, sharpe=round(sharpe,3), extra=extra)


def main():
    print("Loading data …")
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    df = df.sort_values("ts_et").reset_index(drop=True)
    dates = sorted(df["date_et"].unique())
    print(f"  {len(df):,} bars   {dates[0]} → {dates[-1]}  ({len(dates)} days)\n")

    scenarios = [
        ("Baseline  (NQ max 150pts)",  150,   False),
        ("Option A  (NQ max 175pts)",  175,   False),
        ("Option B  (NQ max 200pts)",  200,   False),
        ("Option C  (NQ no max)",      9999,  False),
        ("Option D  (ES-only when NQ > 150)", 9999, True),
    ]

    results = []
    for label, max_nq, es_only in scenarios:
        print(f"Running {label} …")
        r = run_scenario(df, label, max_nq, es_only_when_nq_wide=es_only)
        results.append(r)

    print()
    print("=" * 74)
    print("NQ FILTER BACKTEST  —  ORB system  (3R target, 2018–2026)")
    print("=" * 74)
    print(f"  {'Scenario':<38} {'Days':>5} {'Legs':>5} {'WR':>6} {'AvgR':>7} {'EV':>7} {'Sharpe':>7}")
    print(f"  {'─'*68}")
    for r in results:
        print(f"  {r['label']:<38} {r['n_days']:>5} {r['n_legs']:>5} "
              f"{r['wr']*100:>5.1f}% {r['avg_r']:>+6.3f}R {r['ev']:>+6.3f}R {r['sharpe']:>7.3f}")
        if r.get("extra"):
            print(r["extra"])

    print()
    print("KEY METRICS EXPLAINED")
    print("  WR     = win rate per leg")
    print("  AvgR   = average R per leg (higher = better edge per trade)")
    print("  EV     = expected value at 3R target (WR×3 - (1-WR)×1)")
    print("  Sharpe = annualised Sharpe on daily R returns")
    print()

    # Days added by each option vs baseline
    base_days = results[0]["n_days"]
    print("ADDITIONAL SIGNAL DAYS vs BASELINE (150pt cap)")
    print(f"  {'─'*40}")
    for r in results[1:]:
        extra_days = r["n_days"] - base_days
        wr_delta   = (r["wr"] - results[0]["wr"]) * 100
        print(f"  {r['label']:<38}  +{extra_days:3d} days  WR {wr_delta:+.1f}pp")

    print()


if __name__ == "__main__":
    main()
