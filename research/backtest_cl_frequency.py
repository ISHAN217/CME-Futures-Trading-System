#!/usr/bin/env python3
"""
backtest_cl_frequency.py — Find the best trade-frequency vs WR balance for CL

Tests several approaches to add more trades while keeping WR as high as possible:
  1. Signal tier breakdown  — granular WR by confluence level
  2. Day-of-week effect     — does WR vary by day?
  3. OR range bucketing     — does CL OR range size predict WR?
  4. Proximity to PWH/PWL   — approaching the level vs breaking it
  5. Tiered sizing model    — combine all tiers, weight by confidence
  6. Session time buckets   — does entry time within 10:00-11:00 matter?
"""

import numpy as np
import pandas as pd
from datetime import time
from collections import deque

CL_DATA   = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/CL_1min_continuous.parquet"
ESNG_DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"

RTH_START = time(9, 30); RTH_END = time(14, 30)
OR_START  = time(9, 30); OR_END  = time(10, 0)
ENTRY_END = time(11, 0); FLAT    = time(14, 0)
SLIP = 0.02; PMULT = 3.0


def load():
    cl = pd.read_parquet(CL_DATA)
    cl["ts_et"]   = pd.to_datetime(cl["ts_et"])
    if cl["ts_et"].dt.tz is None:
        cl["ts_et"] = cl["ts_et"].dt.tz_localize("America/New_York")
    else:
        cl["ts_et"] = cl["ts_et"].dt.tz_convert("America/New_York")
    cl["date_et"] = cl["ts_et"].dt.date
    cl["time_et"] = cl["ts_et"].dt.time
    cl = cl.sort_values("ts_et").reset_index(drop=True)
    return cl


def weekly_hl(by_d, dates):
    w = {}
    for d in dates:
        ts  = pd.Timestamp(d); iso = ts.isocalendar()
        key = (int(iso.year), int(iso.week))
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)]
        if not len(rth): continue
        h = rth["CL_high"].max(); l = rth["CL_low"].min()
        if key not in w: w[key] = [h, l]
        else: w[key][0]=max(w[key][0],h); w[key][1]=min(w[key][1],l)
    return w


def get_pw(d, w):
    ts = pd.Timestamp(d); iso = ts.isocalendar()
    pw = int(iso.week)-1; yr = int(iso.year)
    if pw == 0:
        yr -= 1; pw = int(pd.Timestamp(f"{yr}-12-28").isocalendar().week)
    key = (yr, pw)
    return (w[key][0], w[key][1]) if key in w else (None, None)


def sim(day, entry_t, direction, ep, sp, tp):
    rng = abs(ep - sp)
    for _, bar in day[day["time_et"] > entry_t].iterrows():
        t = bar["time_et"]
        if t >= FLAT:
            s = 1 if direction=="LONG" else -1
            return round((s*(bar["CL_close"]-ep) - SLIP*2)/rng, 3), "FLAT"
        if direction=="LONG":
            if bar["CL_low"]  <= sp: return round(-1.0-SLIP*2/rng, 3), "STOP"
            if bar["CL_high"] >= tp: return round(PMULT-SLIP/rng, 3),  "TGT"
        else:
            if bar["CL_high"] >= sp: return round(-1.0-SLIP*2/rng, 3), "STOP"
            if bar["CL_low"]  <= tp: return round(PMULT-SLIP/rng, 3),  "TGT"
    return 0.0, "NONE"


def run_granular(cl, dates, by_d, wkly):
    """Run with maximum signal detail captured."""
    trades = []

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d);
        if day is None: continue

        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b) < 5: continue
        or_H = or_b["CL_high"].max(); or_L = or_b["CL_low"].min()
        or_R = or_H - or_L
        if or_R < 0.10: continue

        # Prior day H/L
        prev = by_d.get(dates[i-1])
        if prev is None: continue
        prev_rth = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(prev_rth): continue
        PDH = prev_rth["CL_high"].max(); PDL = prev_rth["CL_low"].min()

        # Prior week H/L
        PWH, PWL = get_pw(d, wkly)
        if PWH is None: continue

        # OR range bucket
        if or_R < 0.20:   or_bucket = "tight (<20t)"
        elif or_R < 0.40: or_bucket = "normal (20-40t)"
        elif or_R < 0.60: or_bucket = "wide (40-60t)"
        else:             or_bucket = "very_wide (60t+)"

        # Day of week
        dow = pd.Timestamp(d).day_name()

        # Pre-market proximity to PWH/PWL
        # (pre-market high/low before 9:30 ET)
        pm = day[(day["time_et"] < OR_START)]
        pm_near_pwh = False; pm_near_pwl = False
        if len(pm):
            pm_h = pm["CL_high"].max(); pm_l = pm["CL_low"].min()
            pm_near_pwh = abs(pm_h - PWH) <= or_R * 0.5
            pm_near_pwl = abs(pm_l - PWL) <= or_R * 0.5

        watch = day[(day["time_et"]>=OR_END)&(day["time_et"]<ENTRY_END)]
        long_fired = short_fired = False
        pdh_broken = pdl_broken = pwh_broken = pwl_broken = False

        for _, bar in watch.iterrows():
            t = bar["time_et"]

            # Track what levels have broken so far
            if bar["CL_high"] > PDH: pdh_broken = True
            if bar["CL_low"]  < PDL: pdl_broken = True
            if bar["CL_high"] > PWH: pwh_broken = True
            if bar["CL_low"]  < PWL: pwl_broken = True

            # Proximity: how close is price to PWH when OR breaks?
            dist_to_pwh_r = (PWH - bar["CL_high"]) / or_R  # +ve = above
            dist_to_pwl_r = (bar["CL_low"] - PWL) / or_R

            orb_long  = bar["CL_high"] > or_H
            orb_short = bar["CL_low"]  < or_L

            # LONG signal
            if not long_fired and orb_long:
                long_fired = True
                ep = PDH if pdh_broken else or_H
                sp = ep - or_R
                tp = ep + or_R * PMULT

                # Determine signal tier
                if pdh_broken and pwh_broken:
                    tier = "T1_PDH+PWH"
                elif pwh_broken:
                    tier = "T2_PWH_only"
                elif pdh_broken:
                    tier = "T3_PDH_only"
                else:
                    tier = "T4_OR_only"

                rv, why = sim(day, t, "LONG", ep, sp, tp)
                trades.append(dict(
                    date=d, year=d.year, dow=dow,
                    dir="LONG", tier=tier,
                    or_bucket=or_bucket,
                    r=rv, win=rv>0, exit=why,
                    or_R=or_R, entry_min=(t.hour*60+t.minute)-600,
                    pm_near=pm_near_pwh,
                    dist_pwh_r=dist_to_pwh_r,
                ))

            # SHORT signal
            if not short_fired and orb_short:
                short_fired = True
                ep = PDL if pdl_broken else or_L
                sp = ep + or_R
                tp = ep - or_R * PMULT

                if pdl_broken and pwl_broken:
                    tier = "T1_PDH+PWH"
                elif pwl_broken:
                    tier = "T2_PWH_only"
                elif pdl_broken:
                    tier = "T3_PDH_only"
                else:
                    tier = "T4_OR_only"

                rv, why = sim(day, t, "SHORT", ep, sp, tp)
                trades.append(dict(
                    date=d, year=d.year, dow=dow,
                    dir="SHORT", tier=tier,
                    or_bucket=or_bucket,
                    r=rv, win=rv>0, exit=why,
                    or_R=or_R, entry_min=(t.hour*60+t.minute)-600,
                    pm_near=pm_near_pwl,
                    dist_pwh_r=dist_to_pwh_r,
                ))

    return pd.DataFrame(trades)


def pstats(sub, label, pad=30):
    if not len(sub):
        print(f"  {label:<{pad}}  n=   0  —"); return
    wr=sub["win"].mean(); ar=sub["r"].mean()
    ev=wr*PMULT-(1-wr)
    n=len(sub); yr=4.83
    dr=sub.groupby("date")["r"].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    print(f"  {label:<{pad}}  n={n:4d} ({n/yr:.0f}/yr)  "
          f"WR={wr*100:5.1f}%  AvgR={ar:+.3f}  EV={ev:+.3f}  Sh={sh:.2f}")


def main():
    print("Loading …")
    cl = load()
    dates  = sorted(cl["date_et"].unique())
    by_d   = {d: g.reset_index(drop=True) for d, g in cl.groupby("date_et")}
    wkly   = weekly_hl(by_d, dates)
    print("Running granular backtest …\n")
    tdf = run_granular(cl, dates, by_d, wkly)

    W = 76
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. Tier breakdown ─────────────────────────────────────────────────────
    sec("1. SIGNAL TIER BREAKDOWN  —  WR by confluence level")
    tier_order = ["T1_PDH+PWH","T2_PWH_only","T3_PDH_only","T4_OR_only"]
    tier_labels = {
        "T1_PDH+PWH": "Tier 1: OR + PDH/L + PWH/PWL",
        "T2_PWH_only":"Tier 2: OR + PWH/PWL (no PDH/L)",
        "T3_PDH_only":"Tier 3: OR + PDH/L (no PWH/PWL)",
        "T4_OR_only": "Tier 4: OR only (no confirmation)",
    }
    cumulative_n = 0
    cumulative_wins = 0
    print(f"  {'Tier':<34}  {'n':>4}  {'WR':>6}  {'AvgR':>7}  {'EV':>7}  "
          f"{'Sh':>6}  {'Cumul WR':>9}")
    print(f"  {'─'*72}")
    for t in tier_order:
        sub = tdf[tdf["tier"]==t]
        if not len(sub):
            print(f"  {tier_labels[t]:<34}  n=   0"); continue
        wr=sub["win"].mean(); ar=sub["r"].mean()
        ev=wr*PMULT-(1-wr)
        dr=sub.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        cumulative_n += len(sub); cumulative_wins += sub["win"].sum()
        cum_wr = cumulative_wins/cumulative_n if cumulative_n > 0 else 0
        flag = "✅" if wr > 0.55 else ("🟡" if wr > 0.45 else "❌")
        print(f"  {flag} {tier_labels[t]:<33}  n={len(sub):4d}  "
              f"WR={wr*100:5.1f}%  AvgR={ar:+.3f}  EV={ev:+.3f}  "
              f"Sh={sh:.2f}  CumWR={cum_wr*100:.1f}%")

    # ── 2. Day-of-week ────────────────────────────────────────────────────────
    sec("2. WIN RATE BY DAY OF WEEK  (Tiers 1+2 only)")
    top2 = tdf[tdf["tier"].isin(["T1_PDH+PWH","T2_PWH_only"])]
    for dow in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
        pstats(top2[top2["dow"]==dow], dow)

    # ── 3. OR range bucket ────────────────────────────────────────────────────
    sec("3. WIN RATE BY CL OR RANGE  (does tight range = better WR?)")
    for bkt in ["tight (<20t)","normal (20-40t)","wide (40-60t)","very_wide (60t+)"]:
        pstats(tdf[tdf["or_bucket"]==bkt], bkt)
    print()
    print("  Tiers 1+2 only:")
    for bkt in ["tight (<20t)","normal (20-40t)","wide (40-60t)","very_wide (60t+)"]:
        pstats(top2[top2["or_bucket"]==bkt], f"  {bkt}")

    # ── 4. Pre-market proximity to PWH/PWL ────────────────────────────────────
    sec("4. PRE-MARKET PROXIMITY TO PWH/PWL  (opens near level = better setup?)")
    pstats(tdf[tdf["pm_near"]==True],  "Pre-mkt within 0.5R of PWH/PWL")
    pstats(tdf[tdf["pm_near"]==False], "Pre-mkt NOT near PWH/PWL")
    print()
    pstats(top2[top2["pm_near"]==True],  "Tiers 1+2 + PM near level")
    pstats(top2[top2["pm_near"]==False], "Tiers 1+2 + PM not near")

    # ── 5. Entry time analysis ────────────────────────────────────────────────
    sec("5. ENTRY TIME WITHIN WINDOW  (minutes after 10:00 ET)")
    tdf["entry_bucket"] = pd.cut(tdf["entry_min"],
                                  bins=[-1,5,10,20,35,60],
                                  labels=["0-5min","5-10min","10-20min","20-35min","35-60min"])
    for bkt in ["0-5min","5-10min","10-20min","20-35min","35-60min"]:
        pstats(tdf[tdf["entry_bucket"]==bkt], bkt)

    # ── 6. TIERED COMBINED SYSTEM ─────────────────────────────────────────────
    sec("6. TIERED COMBINED SYSTEM  —  best trades-per-week vs WR balance")
    configs = [
        ("T1 only",           ["T1_PDH+PWH"]),
        ("T1 + T2",           ["T1_PDH+PWH","T2_PWH_only"]),
        ("T1 + T2 + T3",      ["T1_PDH+PWH","T2_PWH_only","T3_PDH_only"]),
        ("All tiers",         ["T1_PDH+PWH","T2_PWH_only","T3_PDH_only","T4_OR_only"]),
        ("T1+T2, normal OR",  None),   # special: T1+T2 + tight/normal OR only
        ("T1+T2, Mon+Tue+Fri",None),   # special: T1+T2 + best days
    ]
    print(f"  {'Config':<28}  {'n':>4}  {'n/yr':>5}  {'WR':>6}  "
          f"{'AvgR':>7}  {'EV':>7}  {'Sharpe':>7}")
    print(f"  {'─'*70}")

    for label, tiers in configs:
        if tiers is not None:
            sub = tdf[tdf["tier"].isin(tiers)]
        elif label == "T1+T2, normal OR":
            sub = tdf[tdf["tier"].isin(["T1_PDH+PWH","T2_PWH_only"]) &
                      tdf["or_bucket"].isin(["tight (<20t)","normal (20-40t)"])]
        else:  # best days
            sub = tdf[tdf["tier"].isin(["T1_PDH+PWH","T2_PWH_only"]) &
                      tdf["dow"].isin(["Monday","Tuesday","Friday"])]
        if not len(sub):
            print(f"  {label:<28}  n=   0"); continue
        wr=sub["win"].mean(); ar=sub["r"].mean(); ev=wr*PMULT-(1-wr)
        dr=sub.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        yr=4.83
        flag = "✅" if wr>0.58 else ("🟡" if wr>0.50 else "❌")
        print(f"  {flag} {label:<27}  n={len(sub):4d} ({len(sub)/yr:4.0f}/yr)  "
              f"WR={wr*100:5.1f}%  AvgR={ar:+.3f}  EV={ev:+.3f}  Sh={sh:.2f}")

    # ── 7. Year-by-year for best combined ─────────────────────────────────────
    sec("7. YEAR-BY-YEAR  —  T1+T2 combined (best trade-off)")
    t12 = tdf[tdf["tier"].isin(["T1_PDH+PWH","T2_PWH_only"])]
    print(f"  {'Year':<6}  {'n':>4}  {'WR':>6}  {'AvgR':>8}  {'EV':>8}")
    print(f"  {'─'*42}")
    for yr, sub in t12.groupby("year"):
        wr=sub["win"].mean(); ar=sub["r"].mean(); ev=wr*PMULT-(1-wr)
        flag = " ⚠️" if ev < 0.8 else ""
        print(f"  {yr:<6}  {len(sub):>4}  {wr*100:>5.1f}%  {ar:>+7.3f}R  {ev:>+7.3f}R{flag}")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    sec("8. SWEET SPOT RECOMMENDATION")
    t12 = tdf[tdf["tier"].isin(["T1_PDH+PWH","T2_PWH_only"])]
    t12_wr = t12["win"].mean()*100 if len(t12) else 0
    t1_wr  = tdf[tdf["tier"]=="T1_PDH+PWH"]["win"].mean()*100
    n_t1   = len(tdf[tdf["tier"]=="T1_PDH+PWH"])
    n_t12  = len(t12)
    yr = 4.83

    print(f"  Tier 1 only (PDH/L + PWH/PWL):  "
          f"WR={t1_wr:.1f}%  n={n_t1} total ({n_t1/yr:.0f}/yr = {n_t1/yr/52:.2f}/wk)")
    print(f"  Tier 1+2    (+ PWH/PWL-only):   "
          f"WR={t12_wr:.1f}%  n={n_t12} total ({n_t12/yr:.0f}/yr = {n_t12/yr/52:.2f}/wk)")

    extra_n   = n_t12 - n_t1
    t2_wr     = tdf[tdf["tier"]=="T2_PWH_only"]["win"].mean()*100
    print(f"\n  Adding Tier 2 adds {extra_n} trades ({extra_n/yr:.0f}/yr) at WR={t2_wr:.1f}%")

    if t12_wr > t1_wr - 3:
        print(f"  ✅ Tier 1+2 keeps WR within 3pp of Tier 1 — WORTH ADDING")
    else:
        print(f"  ⚠️  Tier 1+2 drops WR by {t1_wr-t12_wr:.1f}pp — marginal benefit")

    print(f"\n  For comparison — ES/NQ system signal days/week: 2.2")
    print(f"  CL T1+T2 signal days/week: {n_t12/yr/52:.2f}")
    print()


if __name__ == "__main__":
    main()
