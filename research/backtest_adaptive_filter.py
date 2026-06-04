#!/usr/bin/env python3
"""
backtest_adaptive_filter.py — Test adaptive NQ OR range filter vs static cap

Instead of a fixed 150pt NQ max, use a rolling window to adapt to
the current volatility regime.

Approaches tested:
  Baseline    : static NQ max 150pts
  Static-175  : static NQ max 175pts (our quick-fix option)
  Pct-80      : skip if NQ OR > 80th pct of last 20 days' NQ OR ranges
  Pct-75      : skip if NQ OR > 75th pct of last 20 days' NQ OR ranges
  Mult-1.5x   : skip if NQ OR > 1.5x rolling 20-day median NQ OR
  Mult-2.0x   : skip if NQ OR > 2.0x rolling 20-day median NQ OR
  No-filter   : no NQ cap at all (ceiling)
"""

import numpy as np
import pandas as pd
from datetime import time
from collections import deque

DATA = ("/Users/ishanbhardwaj/cme_competition_system/output/mtf/"
        "ES_NQ_1min_aligned.parquet")

OR_START   = time(9, 30);  OR_END    = time(10, 0)
ORB_CUTOFF = time(10, 30); EOD_CLOSE = time(15, 30)
MIN_ES=8.0; MAX_ES=60.0; MIN_NQ=30.0
SLIP=0.25; PMULT=3.0
ROLL_WINDOW = 20   # trading days for rolling stats


def run_scenario(df, label, filter_fn):
    """
    filter_fn(es_R, nq_R, nq_history) -> bool  (True = trade, False = skip)
    nq_history is a deque of last ROLL_WINDOW NQ OR ranges (valid OR days only)
    """
    dates  = sorted(df["date_et"].unique())
    by_d   = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    trades = []
    nq_hist = deque(maxlen=ROLL_WINDOW)   # rolling NQ OR ranges

    for d in dates:
        day = by_d.get(d)
        if day is None: continue

        or_b = day[(day["time_et"] >= OR_START) & (day["time_et"] < OR_END)]
        if len(or_b) == 0: continue

        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min(); es_R = es_H - es_L
        nq_H = or_b["NQ_high"].max(); nq_L = or_b["NQ_low"].min(); nq_R = nq_H - nq_L

        # Always add to history (even if we skip — reflects actual market)
        if MIN_NQ <= nq_R:
            nq_hist.append(nq_R)

        # ES min/max filter
        if not (MIN_ES <= es_R <= MAX_ES): continue
        # NQ min filter
        if nq_R < MIN_NQ: continue

        # Adaptive filter
        if not filter_fn(es_R, nq_R, nq_hist): continue

        # ORB breakout
        post = day[(day["time_et"] >= OR_END) & (day["time_et"] < ORB_CUTOFF)]
        es_b = nq_b = None; entry_row = None
        for _, r in post.iterrows():
            if es_b is None:
                if r["ES_high"] > es_H: es_b = "L"
                elif r["ES_low"] < es_L: es_b = "S"
            if nq_b is None:
                if r["NQ_high"] > nq_H: nq_b = "L"
                elif r["NQ_low"] < nq_L: nq_b = "S"
            if es_b and nq_b: entry_row = r; break
        if not (es_b and nq_b and es_b == nq_b): continue

        direction = "LONG" if es_b == "L" else "SHORT"
        et = entry_row["time_et"]
        after = day[day["time_et"] > et]

        for sym, H, L, R, ch, cl, cc in [
            ("ES", es_H, es_L, es_R, "ES_high", "ES_low", "ES_close"),
            ("NQ", nq_H, nq_L, nq_R, "NQ_high", "NQ_low", "NQ_close"),
        ]:
            ep = H if direction == "LONG" else L
            sp = L if direction == "LONG" else H
            tp = round(ep + PMULT * R, 2) if direction == "LONG" else round(ep - PMULT * R, 2)
            for _, bar in after.iterrows():
                t = bar["time_et"]
                if t >= EOD_CLOSE:
                    sign = 1 if direction == "LONG" else -1
                    net  = sign * (bar[cc] - ep) - SLIP * 2
                    r_val = net / R if R > 0 else 0; why = "EOD"; break
                if direction == "LONG":
                    if bar[cl] <= sp: r_val = -1.0 - SLIP*2/R; why = "STOP"; break
                    if bar[ch] >= tp: r_val = PMULT - SLIP/R;  why = "TGT";  break
                else:
                    if bar[ch] >= sp: r_val = -1.0 - SLIP*2/R; why = "STOP"; break
                    if bar[cl] <= tp: r_val = PMULT - SLIP/R;  why = "TGT";  break
            else:
                continue
            trades.append(dict(date=d, sym=sym, r=r_val, win=r_val>0, exit=why,
                               year=d.year))

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["r","win","date"])
    if len(tdf) == 0:
        return dict(label=label, n_days=0, n_legs=0, wr=0, avg_r=0, ev=0, sharpe=0,
                    tdf=tdf)

    wr    = tdf["win"].mean()
    avg_r = tdf["r"].mean()
    ev    = wr * PMULT - (1 - wr) * 1.0
    dr    = tdf.groupby("date")["r"].sum()
    sh    = dr.mean() / dr.std(ddof=1) * np.sqrt(252) if dr.std() > 0 else 0

    return dict(label=label, n_days=len(tdf)//2, n_legs=len(tdf),
                wr=wr, avg_r=avg_r, ev=ev, sharpe=round(sh, 3), tdf=tdf)


def main():
    print("Loading data …")
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    df = df.sort_values("ts_et").reset_index(drop=True)
    dates = sorted(df["date_et"].unique())
    print(f"  {len(df):,} bars   {dates[0]} → {dates[-1]}\n")

    scenarios = [
        ("Baseline   (static 150pts)",
         lambda es_R, nq_R, h: nq_R <= 150),

        ("Static-175 (static 175pts)",
         lambda es_R, nq_R, h: nq_R <= 175),

        ("Pct-80     (< 80th pct last 20d)",
         lambda es_R, nq_R, h: (nq_R <= np.percentile(list(h), 80))
                                 if len(h) >= 5 else nq_R <= 150),

        ("Pct-75     (< 75th pct last 20d)",
         lambda es_R, nq_R, h: (nq_R <= np.percentile(list(h), 75))
                                 if len(h) >= 5 else nq_R <= 150),

        ("Mult-1.5x  (< 1.5x rolling median)",
         lambda es_R, nq_R, h: (nq_R <= np.median(list(h)) * 1.5)
                                 if len(h) >= 5 else nq_R <= 150),

        ("Mult-2.0x  (< 2.0x rolling median)",
         lambda es_R, nq_R, h: (nq_R <= np.median(list(h)) * 2.0)
                                 if len(h) >= 5 else nq_R <= 150),

        ("No-filter  (no NQ cap)",
         lambda es_R, nq_R, h: True),
    ]

    results = []
    for label, fn in scenarios:
        print(f"  Running {label} …")
        r = run_scenario(df, label, fn)
        results.append(r)

    print()
    print("=" * 80)
    print(f"ADAPTIVE NQ FILTER BACKTEST  (3R target, {dates[0]} – {dates[-1]})")
    print("=" * 80)
    print(f"  {'Scenario':<36} {'Days':>5} {'Legs':>5} {'WR':>6} "
          f"{'AvgR':>7} {'EV':>7} {'Sharpe':>7}")
    print(f"  {'─'*72}")
    for r in results:
        print(f"  {r['label']:<36} {r['n_days']:>5} {r['n_legs']:>5} "
              f"{r['wr']*100:>5.1f}% {r['avg_r']:>+6.3f}R "
              f"{r['ev']:>+6.3f}R {r['sharpe']:>7.3f}")

    # Year-by-year for best adaptive vs baseline
    best_adaptive = max(results[2:6], key=lambda r: r["sharpe"])
    base  = results[0]

    print()
    print("=" * 80)
    print(f"YEAR-BY-YEAR COMPARISON  —  Baseline vs {best_adaptive['label'].strip()}")
    print("=" * 80)
    print(f"  {'Year':<6} {'Base Days':>9} {'Base WR':>8} {'Base EV':>8} │ "
          f"{'Adap Days':>9} {'Adap WR':>8} {'Adap EV':>8}")
    print(f"  {'─'*64}")

    base_tdf = base["tdf"]
    adap_tdf = best_adaptive["tdf"]

    all_years = sorted(set(base_tdf["year"].unique()) | set(adap_tdf["year"].unique()))
    for yr in all_years:
        bs = base_tdf[base_tdf["year"] == yr]
        ad = adap_tdf[adap_tdf["year"] == yr]
        b_n  = len(bs)//2;  b_wr = bs["win"].mean()*100 if len(bs) else 0
        b_ev = bs["win"].mean()*PMULT-(1-bs["win"].mean()) if len(bs) else 0
        a_n  = len(ad)//2;  a_wr = ad["win"].mean()*100 if len(ad) else 0
        a_ev = ad["win"].mean()*PMULT-(1-ad["win"].mean()) if len(ad) else 0
        print(f"  {yr:<6} {b_n:>9} {b_wr:>7.1f}% {b_ev:>+7.3f}R │ "
              f"{a_n:>9} {a_wr:>7.1f}% {a_ev:>+7.3f}R")

    # Would it have traded this week?
    print()
    print("=" * 80)
    print("THIS WEEK (May 26-29)  —  Would each scenario have traded?")
    print("=" * 80)
    from datetime import date
    week_start = date(2026, 5, 26)
    print(f"  {'Scenario':<36} {'Trades':>7}")
    print(f"  {'─'*46}")
    for r in results:
        if len(r["tdf"]) == 0:
            wk = 0
        else:
            wk = len(r["tdf"][pd.to_datetime(r["tdf"]["date"]) >= pd.Timestamp(week_start)]) // 2
        print(f"  {r['label']:<36} {wk:>7} signal days")

    print()
    print("RECOMMENDATION")
    print(f"  Best adaptive: {best_adaptive['label'].strip()}")
    print(f"  Sharpe: {best_adaptive['sharpe']:.3f} vs baseline {base['sharpe']:.3f}")
    wr_delta = (best_adaptive['wr'] - base['wr']) * 100
    day_delta = best_adaptive['n_days'] - base['n_days']
    print(f"  WR delta: {wr_delta:+.1f}pp   Signal days delta: {day_delta:+d}")
    print()


if __name__ == "__main__":
    main()
