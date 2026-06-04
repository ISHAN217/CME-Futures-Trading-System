#!/usr/bin/env python3
"""
backtest_weekly_hl.py — Prior Week High/Low as structural layer

Tests whether adding Prior Week H/L (PWH/PWL) as a confluence filter
improves WR and EV over the base ORB + PDH/L system.

Questions answered:
  1. 3-way confluence: ORB + PDH/L + PWH/PWL same direction → WR?
  2. Is breaking PWH/PWL at entry better than trading below it?
  3. Does PWH/PWL as resistance (ahead of trade) hurt WR?
  4. PWH/PWL standalone signal (when ORB misses and PDH/L misses)?
  5. Optimal use: filter, context, or standalone?
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = ("/Users/ishanbhardwaj/cme_competition_system/output/mtf/"
        "ES_NQ_1min_aligned.parquet")

OR_START   = time(9, 30);  OR_END    = time(10, 0)
ORB_CUTOFF = time(10, 30); EOD_CLOSE = time(15, 30)
RTH_START  = time(9, 30);  RTH_END   = time(16, 0)
WATCH_END  = time(11, 0)
MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175
SLIP=0.25; PMULT=3.0


def compute_weekly_hl(by_d, dates):
    """Pre-compute ES + NQ RTH H/L for every ISO (year, week)."""
    weekly: dict = {}
    for d in dates:
        d_ts = pd.Timestamp(d)
        iso  = d_ts.isocalendar()
        key  = (iso.year, int(iso.week))
        day  = by_d.get(d)
        if day is None: continue
        rth  = day[(day["time_et"] >= RTH_START) & (day["time_et"] <= RTH_END)]
        if len(rth) == 0: continue
        es_h = rth["ES_high"].max(); es_l = rth["ES_low"].min()
        nq_h = rth["NQ_high"].max(); nq_l = rth["NQ_low"].min()
        if key not in weekly:
            weekly[key] = [es_h, es_l, nq_h, nq_l]
        else:
            weekly[key][0] = max(weekly[key][0], es_h)
            weekly[key][1] = min(weekly[key][1], es_l)
            weekly[key][2] = max(weekly[key][2], nq_h)
            weekly[key][3] = min(weekly[key][3], nq_l)
    return weekly


def prior_week_hl(d, weekly):
    """Return (PWH_ES, PWL_ES, PWH_NQ, PWL_NQ) or (None,...) if unavailable."""
    d_ts = pd.Timestamp(d)
    iso  = d_ts.isocalendar()
    pw   = int(iso.week) - 1
    yr   = int(iso.year)
    if pw == 0:
        yr -= 1
        # last ISO week of the previous year
        pw = int(pd.Timestamp(f"{yr}-12-28").isocalendar().week)
    key = (yr, pw)
    if key in weekly:
        h = weekly[key]
        return h[0], h[1], h[2], h[3]
    return None, None, None, None


def sim_leg(after, direction, ep, sp, rng, ch, cl, cc):
    for _, bar in after.iterrows():
        t = bar["time_et"]
        if t >= EOD_CLOSE:
            sign = 1 if direction == "LONG" else -1
            return round((sign*(bar[cc]-ep) - SLIP*2) / rng, 3), "EOD"
        if direction == "LONG":
            if bar[cl] <= sp: return round(-1.0 - SLIP*2/rng, 3), "STOP"
            if bar[ch] >= ep + rng*PMULT: return round(PMULT - SLIP/rng, 3), "TGT"
        else:
            if bar[ch] >= sp: return round(-1.0 - SLIP*2/rng, 3), "STOP"
            if bar[cl] <= ep - rng*PMULT: return round(PMULT - SLIP/rng, 3), "TGT"
    return 0.0, "NONE"


def main():
    print("Loading data …")
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    df = df.sort_values("ts_et").reset_index(drop=True)
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars   {dates[0]} → {dates[-1]}")

    print("Pre-computing weekly H/L …")
    weekly = compute_weekly_hl(by_d, dates)
    print(f"  {len(weekly)} weeks computed\n")

    trades = []  # one record per leg

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue

        # Opening range
        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b)==0: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not (MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        # Prior week levels
        PWH_ES, PWL_ES, PWH_NQ, PWL_NQ = prior_week_hl(d, weekly)
        if PWH_ES is None: continue  # no prior week data

        # Prior day levels
        prev = by_d.get(dates[i-1])
        if prev is None: continue
        prth = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if len(prth)==0: continue
        PDH_ES=prth["ES_high"].max(); PDL_ES=prth["ES_low"].min()
        PDH_NQ=prth["NQ_high"].max(); PDL_NQ=prth["NQ_low"].min()

        # ── ORB detection ─────────────────────────────────────────────
        post = day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUTOFF)]
        es_b=nq_b=None; entry_row=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: entry_row=r; break
        orb_fired = bool(es_b and nq_b and es_b==nq_b)

        # ── PDH/L detection (10:00–11:00) ────────────────────────────
        watch = day[(day["time_et"]>=OR_END)&(day["time_et"]<WATCH_END)]
        eu=ed=nu=nd=False; pdhl_dir=None; pdhl_row=None
        for _,r in watch.iterrows():
            if not eu and r["ES_high"]>PDH_ES: eu=True
            if not ed and r["ES_low"]<PDL_ES:  ed=True
            if not nu and r["NQ_high"]>PDH_NQ: nu=True
            if not nd and r["NQ_low"]<PDL_NQ:  nd=True
            if eu and nu and pdhl_dir is None: pdhl_dir="LONG";  pdhl_row=r; break
            if ed and nd and pdhl_dir is None: pdhl_dir="SHORT"; pdhl_row=r; break
            if (eu and nd) or (ed and nu): break

        # ── PWH/PWL detection (10:00–11:00, same window as PDH/L) ────
        pw_eu=pw_ed=pw_nu=pw_nd=False; pwhl_dir=None; pwhl_row=None
        for _,r in watch.iterrows():
            if not pw_eu and r["ES_high"]>PWH_ES: pw_eu=True
            if not pw_ed and r["ES_low"]<PWL_ES:  pw_ed=True
            if not pw_nu and r["NQ_high"]>PWH_NQ: pw_nu=True
            if not pw_nd and r["NQ_low"]<PWL_NQ:  pw_nd=True
            if pw_eu and pw_nu and pwhl_dir is None: pwhl_dir="LONG";  pwhl_row=r; break
            if pw_ed and pw_nd and pwhl_dir is None: pwhl_dir="SHORT"; pwhl_row=r; break
            if (pw_eu and pw_nd) or (pw_ed and pw_nu): break

        # ── Determine signal category ─────────────────────────────────
        # Priority: use the best available signal with most confluence
        direction = None; entry_row_use = None; sig_type = None

        orb_dir = ("LONG" if es_b=="L" else "SHORT") if orb_fired else None
        pdhl_ok = (pdhl_dir is not None)
        pwhl_ok = (pwhl_dir is not None)

        if orb_fired:
            # Check PDH/L and PWH/PWL agreement
            pdhl_agrees = pdhl_ok and pdhl_dir == orb_dir
            pwhl_agrees = pwhl_ok and pwhl_dir == orb_dir
            pdhl_conflicts = pdhl_ok and pdhl_dir != orb_dir
            pwhl_conflicts = pwhl_ok and pwhl_dir != orb_dir

            if pdhl_conflicts or pwhl_conflicts:
                # Any conflict → skip (back-test shows WR=43.8% on conflict days)
                sig_type = "CONFLICT_SKIP"
                continue  # don't trade

            if pdhl_agrees and pwhl_agrees:
                sig_type = "3WAY"       # ORB + PDH/L + PWH/PWL all agree
            elif pdhl_agrees:
                sig_type = "ORB_PDHL"   # ORB + PDH/L (current best)
            elif pwhl_agrees:
                sig_type = "ORB_PWHL"   # ORB + PWH/PWL (new)
            else:
                sig_type = "ORB_ONLY"   # ORB alone

            direction = orb_dir
            entry_row_use = entry_row

        elif pdhl_ok and pwhl_ok and pdhl_dir == pwhl_dir:
            sig_type  = "PDHL_PWHL"     # both PDH/L and PWH/PWL, no ORB
            direction = pdhl_dir
            entry_row_use = pdhl_row
        elif pdhl_ok:
            sig_type  = "PDHL_ONLY"
            direction = pdhl_dir
            entry_row_use = pdhl_row
        elif pwhl_ok:
            sig_type  = "PWHL_ONLY"     # PWH/PWL standalone (new)
            direction = pwhl_dir
            entry_row_use = pwhl_row
        else:
            continue

        if direction is None or entry_row_use is None: continue
        et = entry_row_use["time_et"]
        after = day[day["time_et"] > et]

        # ── Proximity: is PWH ahead as resistance? ────────────────────
        # For LONG: distance from ES entry to PWH_ES expressed in es_R units
        # Negative = we're already above PWH (broke through)
        ep_es = es_H if direction=="LONG" else es_L
        if direction == "LONG":
            pwh_dist_r = (PWH_ES - ep_es) / es_R   # >0 = PWH ahead, <0 = already broke
        else:
            pwh_dist_r = (ep_es - PWL_ES) / es_R

        # ── Simulate both legs ────────────────────────────────────────
        use_pdhl_entry = sig_type in ("PDHL_ONLY","PDHL_PWHL","3WAY","ORB_PDHL")
        use_pwhl_entry = (sig_type == "PWHL_ONLY" or
                          (sig_type == "ORB_PWHL" and not use_pdhl_entry) or
                          sig_type == "PDHL_PWHL")

        for sym, PDH, PDL, H, L, R, ch, cl, cc in [
            ("ES", PDH_ES, PDL_ES, es_H, es_L, es_R, "ES_high","ES_low","ES_close"),
            ("NQ", PDH_NQ, PDL_NQ, nq_H, nq_L, nq_R, "NQ_high","NQ_low","NQ_close"),
        ]:
            PWH = PWH_ES if sym=="ES" else PWH_NQ
            PWL = PWL_ES if sym=="ES" else PWL_NQ

            if use_pdhl_entry:
                ep = PDH if direction=="LONG" else PDL
                sp = ep - R if direction=="LONG" else ep + R
            elif use_pwhl_entry:
                ep = PWH if direction=="LONG" else PWL
                sp = ep - R if direction=="LONG" else ep + R
            else:
                ep = H if direction=="LONG" else L
                sp = L if direction=="LONG" else H

            rv, why = sim_leg(after, direction, ep, sp, R, ch, cl, cc)
            trades.append(dict(
                date=d, sym=sym, sig_type=sig_type, direction=direction,
                r=rv, win=rv>0, exit=why, year=d.year,
                pwh_dist_r=pwh_dist_r,
            ))

    tdf = pd.DataFrame(trades)
    W = 72

    def sec(title):
        print("\n" + "="*W)
        print(f"  {title}")
        print("="*W)

    def stats_row(label, sub, pad=30):
        if len(sub)==0:
            print(f"  {label:<{pad}}  n=   0  —")
            return
        wr=sub["win"].mean(); ar=sub["r"].mean()
        ev=wr*PMULT-(1-wr)
        n=len(sub)//2
        dr=sub.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        print(f"  {label:<{pad}}  n={n:4d}  WR={wr*100:5.1f}%  "
              f"AvgR={ar:+.3f}R  EV={ev:+.3f}R  Sh={sh:.2f}")

    # ── 1. Signal type comparison ─────────────────────────────────────────────
    sec("1. SIGNAL TYPE COMPARISON  —  All categories vs baseline")
    print(f"  {'Category':<30}  {'Days':>4}  {'WR':>6}  {'AvgR':>7}  {'EV':>7}  {'Sharpe':>7}")
    print(f"  {'─'*64}")
    order = ["3WAY","ORB_PDHL","ORB_PWHL","ORB_ONLY","PDHL_PWHL","PDHL_ONLY","PWHL_ONLY"]
    labels = {
        "3WAY":      "3-Way (ORB+PDH/L+PWH/PWL) ✨",
        "ORB_PDHL":  "ORB + PDH/L  (current best)",
        "ORB_PWHL":  "ORB + PWH/PWL  (new)",
        "ORB_ONLY":  "ORB only  (baseline)",
        "PDHL_PWHL": "PDH/L + PWH/PWL  (no ORB)",
        "PDHL_ONLY": "PDH/L only  (current fallback)",
        "PWHL_ONLY": "PWH/PWL only  (new standalone)",
    }
    for st in order:
        sub = tdf[tdf["sig_type"]==st]
        stats_row(labels[st], sub, pad=34)

    # ── 2. PWH/PWL proximity effect ───────────────────────────────────────────
    sec("2. PWH/PWL PROXIMITY EFFECT  —  Does resistance ahead hurt WR?")
    print("  (For LONG trades: PWH_dist_R = distance to prior week high in R-units)")
    print(f"  {'Bucket':<25}  {'Days':>4}  {'WR':>6}  {'AvgR':>7}  {'EV':>7}")
    print(f"  {'─'*56}")
    orb_only = tdf[tdf["sig_type"].isin(["ORB_ONLY","ORB_PDHL","ORB_PWHL","3WAY"])]
    bins = [(-99,-1),(-1,0),(0,1),(1,2),(2,5),(5,99)]
    labels2 = [
        "Already 1R+ above PWH",
        "Just above PWH (<1R)",
        "PWH within 1R ahead",
        "PWH 1–2R ahead",
        "PWH 2–5R ahead",
        "PWH 5R+ ahead (far)",
    ]
    for (lo,hi), lbl in zip(bins, labels2):
        sub = orb_only[(orb_only["pwh_dist_r"]>=lo)&(orb_only["pwh_dist_r"]<hi)]
        if len(sub)==0: continue
        wr=sub["win"].mean(); ar=sub["r"].mean(); ev=wr*PMULT-(1-wr)
        n=len(sub)//2
        print(f"  {lbl:<25}  n={n:4d}  WR={wr*100:5.1f}%  AvgR={ar:+.3f}R  EV={ev:+.3f}R")

    # ── 3. 3-Way year-by-year ─────────────────────────────────────────────────
    sec("3. 3-WAY CONFLUENCE  —  Year-by-year consistency")
    three = tdf[tdf["sig_type"]=="3WAY"]
    print(f"  {'Year':<6}  {'Days':>4}  {'WR':>6}  {'AvgR':>7}  {'EV':>7}")
    print(f"  {'─'*44}")
    for yr, sub in three.groupby("year"):
        wr=sub["win"].mean(); ar=sub["r"].mean(); ev=wr*PMULT-(1-wr)
        flag = "  ⚠️" if ev < 1.0 else ""
        print(f"  {yr:<6}  {len(sub)//2:>4}  {wr*100:>5.1f}%  {ar:>+6.3f}R  {ev:>+6.3f}R{flag}")

    # ── 4. PWH/PWL standalone year-by-year ───────────────────────────────────
    sec("4. PWH/PWL STANDALONE  —  Year-by-year (new signal type)")
    pwhl = tdf[tdf["sig_type"]=="PWHL_ONLY"]
    print(f"  {'Year':<6}  {'Days':>4}  {'WR':>6}  {'AvgR':>7}  {'EV':>7}")
    print(f"  {'─'*44}")
    for yr, sub in pwhl.groupby("year"):
        wr=sub["win"].mean(); ar=sub["r"].mean(); ev=wr*PMULT-(1-wr)
        flag = "  ⚠️" if ev < 0.5 else ""
        print(f"  {yr:<6}  {len(sub)//2:>4}  {wr*100:>5.1f}%  {ar:>+6.3f}R  {ev:>+6.3f}R{flag}")

    # ── 5. Signal frequency with PWH/PWL added ───────────────────────────────
    sec("5. SIGNAL FREQUENCY  —  How many more days does PWH/PWL add?")
    for st in order:
        n = len(tdf[tdf["sig_type"]==st])//2
        pct = n / (len(dates)/8.1/52)   # per competition week expected
        print(f"  {labels[st]:<34}  {n:4d} days total  "
              f"~{n/8.1:.0f}/yr  ~{n/8.1/52:.2f}/wk")

    # ── 6. Combined system scorecard ─────────────────────────────────────────
    sec("6. COMBINED SYSTEM SCORECARD  —  Optimal priority hierarchy")
    print("  Proposed hierarchy:")
    print("    1. 3-WAY (ORB+PDH/L+PWH/PWL agree)  → MAX size")
    print("    2. ORB + PDH/L                       → full size")
    print("    3. ORB + PWH/PWL                     → full size")
    print("    4. PDH/L + PWH/PWL (no ORB)          → full size")
    print("    5. ORB only                           → standard size")
    print("    6. PDH/L only                         → standard size")
    print("    7. PWH/PWL only                       → smaller size")
    print("    8. Any conflict                       → SKIP")
    print()

    all_tradeable = tdf[tdf["sig_type"].isin(
        ["3WAY","ORB_PDHL","ORB_PWHL","ORB_ONLY","PDHL_PWHL","PDHL_ONLY","PWHL_ONLY"]
    )]
    stats_row("Full combined system", all_tradeable, pad=24)

    # ── 7. IS / OOS split ────────────────────────────────────────────────────
    sec("7. WALK-FORWARD: 3-WAY CONFLUENCE  (IS 2018-2021 vs OOS 2022-2026)")
    three_is  = three[three["year"] <= 2021]
    three_oos = three[three["year"] >= 2022]
    stats_row("3-Way IS  (2018-2021)", three_is,  pad=24)
    stats_row("3-Way OOS (2022-2026)", three_oos, pad=24)
    pwhl_is   = pwhl[pwhl["year"] <= 2021]
    pwhl_oos  = pwhl[pwhl["year"] >= 2022]
    stats_row("PWH/PWL standalone IS ", pwhl_is,  pad=24)
    stats_row("PWH/PWL standalone OOS", pwhl_oos, pad=24)

    print()
    print("="*W)
    print("  RECOMMENDATION")
    print("="*W)
    three_wr  = three_oos["win"].mean() if len(three_oos) else 0
    pwhl_ev   = (pwhl_oos["win"].mean()*PMULT-(1-pwhl_oos["win"].mean())) if len(pwhl_oos) else 0
    base_wr   = tdf[tdf["sig_type"]=="ORB_PDHL"]["win"].mean() if len(tdf[tdf["sig_type"]=="ORB_PDHL"]) else 0
    print(f"  3-Way OOS WR          : {three_wr*100:.1f}%  (vs ORB+PDH/L: {base_wr*100:.1f}%)")
    print(f"  PWH/PWL standalone EV : {pwhl_ev:+.3f}R")
    print()
    if three_wr > base_wr + 0.03:
        print("  ✅ 3-Way confluence meaningfully outperforms → ADD to system")
    elif three_wr > base_wr:
        print("  🟡 3-Way confluence marginally better → ADD as size modifier")
    else:
        print("  ⚠️  3-Way confluence does NOT outperform → use as context only")
    if pwhl_ev > 0.8:
        print("  ✅ PWH/PWL standalone has positive EV → ADD as new signal type")
    else:
        print("  ⚠️  PWH/PWL standalone edge weak → skip or use as filter only")
    print()


if __name__ == "__main__":
    main()
