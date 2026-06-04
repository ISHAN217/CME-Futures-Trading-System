#!/usr/bin/env python3
"""
backtest_weekly_trades.py — Weekly trade frequency analysis across all signal types

Shows exactly how many trades fire per week, by signal type, and the
weekly P&L distribution — the most competition-relevant view.
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
    weekly = {}
    for d in dates:
        d_ts = pd.Timestamp(d)
        iso  = d_ts.isocalendar()
        key  = (int(iso.year), int(iso.week))
        day  = by_d.get(d)
        if day is None: continue
        rth  = day[(day["time_et"] >= RTH_START) & (day["time_et"] <= RTH_END)]
        if len(rth) == 0: continue
        es_h=rth["ES_high"].max(); es_l=rth["ES_low"].min()
        nq_h=rth["NQ_high"].max(); nq_l=rth["NQ_low"].min()
        if key not in weekly:
            weekly[key] = [es_h, es_l, nq_h, nq_l]
        else:
            weekly[key][0] = max(weekly[key][0], es_h)
            weekly[key][1] = min(weekly[key][1], es_l)
            weekly[key][2] = max(weekly[key][2], nq_h)
            weekly[key][3] = min(weekly[key][3], nq_l)
    return weekly


def prior_week_hl(d, weekly):
    d_ts = pd.Timestamp(d)
    iso  = d_ts.isocalendar()
    pw   = int(iso.week) - 1
    yr   = int(iso.year)
    if pw == 0:
        yr -= 1
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


def run(df, dates, by_d, weekly):
    trades = []

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue

        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b)==0: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not (MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        PWH_ES,PWL_ES,PWH_NQ,PWL_NQ = prior_week_hl(d, weekly)
        if PWH_ES is None: continue

        prev = by_d.get(dates[i-1])
        if prev is None: continue
        prth = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if len(prth)==0: continue
        PDH_ES=prth["ES_high"].max(); PDL_ES=prth["ES_low"].min()
        PDH_NQ=prth["NQ_high"].max(); PDL_NQ=prth["NQ_low"].min()

        # ORB
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
        orb_dir   = ("LONG" if es_b=="L" else "SHORT") if orb_fired else None

        # PDH/L
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

        # PWH/PWL
        pw_eu=pw_ed=pw_nu=pw_nd=False; pwhl_dir=None; pwhl_row=None
        for _,r in watch.iterrows():
            if not pw_eu and r["ES_high"]>PWH_ES: pw_eu=True
            if not pw_ed and r["ES_low"]<PWL_ES:  pw_ed=True
            if not pw_nu and r["NQ_high"]>PWH_NQ: pw_nu=True
            if not pw_nd and r["NQ_low"]<PWL_NQ:  pw_nd=True
            if pw_eu and pw_nu and pwhl_dir is None: pwhl_dir="LONG";  pwhl_row=r; break
            if pw_ed and pw_nd and pwhl_dir is None: pwhl_dir="SHORT"; pwhl_row=r; break
            if (pw_eu and pw_nd) or (pw_ed and pw_nu): break

        # Signal classification
        direction=None; entry_row_use=None; sig_type=None
        pdhl_ok = pdhl_dir is not None
        pwhl_ok = pwhl_dir is not None

        if orb_fired:
            pdhl_agrees   = pdhl_ok and pdhl_dir == orb_dir
            pwhl_agrees   = pwhl_ok and pwhl_dir == orb_dir
            any_conflicts = (pdhl_ok and pdhl_dir != orb_dir) or \
                            (pwhl_ok and pwhl_dir != orb_dir)
            if any_conflicts: continue
            if pdhl_agrees and pwhl_agrees: sig_type="3WAY"
            elif pdhl_agrees:               sig_type="ORB_PDHL"
            elif pwhl_agrees:               sig_type="ORB_PWHL"
            else:                           sig_type="ORB_ONLY"
            direction=orb_dir; entry_row_use=entry_row
        elif pdhl_ok and pwhl_ok and pdhl_dir==pwhl_dir:
            sig_type="PDHL_PWHL"; direction=pdhl_dir; entry_row_use=pdhl_row
        elif pdhl_ok:
            sig_type="PDHL_ONLY"; direction=pdhl_dir; entry_row_use=pdhl_row
        elif pwhl_ok:
            sig_type="PWHL_ONLY"; direction=pwhl_dir; entry_row_use=pwhl_row
        else:
            continue

        if direction is None or entry_row_use is None: continue
        et  = entry_row_use["time_et"]
        after = day[day["time_et"]>et]

        use_pdhl = sig_type in ("PDHL_ONLY","PDHL_PWHL","3WAY","ORB_PDHL")
        use_pwhl = sig_type in ("PWHL_ONLY",) or \
                   (sig_type=="ORB_PWHL") or \
                   (sig_type=="PDHL_PWHL" and not use_pdhl)

        d_ts  = pd.Timestamp(d)
        iso_w = (int(d_ts.isocalendar().year), int(d_ts.isocalendar().week))

        for sym,PDH,PDL,H,L,R,ch,cl,cc in [
            ("ES",PDH_ES,PDL_ES,es_H,es_L,es_R,"ES_high","ES_low","ES_close"),
            ("NQ",PDH_NQ,PDL_NQ,nq_H,nq_L,nq_R,"NQ_high","NQ_low","NQ_close"),
        ]:
            PWH = PWH_ES if sym=="ES" else PWH_NQ
            PWL = PWL_ES if sym=="ES" else PWL_NQ
            if use_pdhl:
                ep = PDH if direction=="LONG" else PDL
                sp = ep-R if direction=="LONG" else ep+R
            elif use_pwhl:
                ep = PWH if direction=="LONG" else PWL
                sp = ep-R if direction=="LONG" else ep+R
            else:
                ep = H if direction=="LONG" else L
                sp = L if direction=="LONG" else H

            rv, why = sim_leg(after, direction, ep, sp, R, ch, cl, cc)
            trades.append(dict(
                date=d, iso_week=iso_w, year=d.year,
                sym=sym, sig_type=sig_type,
                direction=direction, r=rv, win=rv>0, exit=why,
            ))

    return pd.DataFrame(trades)


def main():
    print("Loading …")
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    df = df.sort_values("ts_et").reset_index(drop=True)
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}

    weekly_hl = compute_weekly_hl(by_d, dates)
    tdf = run(df, dates, by_d, weekly_hl)

    # Add week label
    tdf["week_key"] = tdf["iso_week"].astype(str)

    # ── Daily signal days (one row per day, not per leg) ─────────────────────
    day_sig = tdf.drop_duplicates(subset=["date","sig_type"])[["date","iso_week","year","sig_type"]]

    # ── Weekly aggregation ────────────────────────────────────────────────────
    # Per week: count signal days, sum R, win/loss
    week_df = tdf.groupby("iso_week").agg(
        signal_days   = ("date",    lambda x: x.nunique()),
        total_R       = ("r",       "sum"),
        legs          = ("r",       "count"),
        wins          = ("win",     "sum"),
    ).reset_index()
    week_df["WR"]         = week_df["wins"] / week_df["legs"]
    week_df["win_week"]   = week_df["total_R"] > 0
    week_df["year"]       = week_df["iso_week"].apply(lambda x: x[0])

    total_weeks = len(week_df)

    W = 70
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. Overall weekly trade count distribution ────────────────────────────
    sec("1. WEEKLY SIGNAL DAY DISTRIBUTION  (how many days per week do we trade?)")
    dist = week_df["signal_days"].value_counts().sort_index()
    print(f"  Total trading weeks in dataset : {total_weeks}")
    print(f"  Weeks with ≥1 signal day       : {(week_df['signal_days']>0).sum()}  "
          f"({(week_df['signal_days']>0).mean()*100:.1f}%)")
    print()
    print(f"  {'Signal days/week':>18}  {'# Weeks':>8}  {'% of Weeks':>11}  {'Avg weekly R':>13}")
    print(f"  {'─'*56}")
    for n_days in sorted(dist.index):
        sub   = week_df[week_df["signal_days"]==n_days]
        pct   = len(sub)/total_weeks*100
        avg_r = sub["total_R"].mean()
        print(f"  {n_days:>18}  {len(sub):>8}  {pct:>10.1f}%  {avg_r:>+12.2f}R")

    # ── 2. Weekly signal type breakdown ──────────────────────────────────────
    sec("2. WHICH SIGNAL TYPES FIRE IN A TYPICAL WEEK")
    sig_types = ["3WAY","ORB_PDHL","ORB_PWHL","ORB_ONLY","PDHL_PWHL","PDHL_ONLY","PWHL_ONLY"]
    labels = {
        "3WAY":      "3-Way (ORB+PDH/L+PWH/PWL)",
        "ORB_PDHL":  "ORB + PDH/L",
        "ORB_PWHL":  "ORB + PWH/PWL",
        "ORB_ONLY":  "ORB only",
        "PDHL_PWHL": "PDH/L + PWH/PWL",
        "PDHL_ONLY": "PDH/L only",
        "PWHL_ONLY": "PWH/PWL only",
    }
    print(f"  {'Signal Type':<30}  {'Days/yr':>7}  {'Days/wk':>8}  "
          f"{'Wks it fires':>12}  {'% of wks':>9}")
    print(f"  {'─'*68}")
    for st in sig_types:
        sub    = tdf[tdf["sig_type"]==st]
        n_days = sub["date"].nunique()
        per_yr = n_days / 8.1
        per_wk = per_yr / 52
        # Weeks where this signal fired at least once
        wks_fired = sub.drop_duplicates("iso_week")["iso_week"].nunique()
        pct_wks   = wks_fired / total_weeks * 100
        print(f"  {labels[st]:<30}  {per_yr:>7.1f}  {per_wk:>8.2f}  "
              f"{wks_fired:>12}  {pct_wks:>8.1f}%")

    # ── 3. Weekly P&L distribution ────────────────────────────────────────────
    sec("3. WEEKLY P&L DISTRIBUTION  (in R, both legs combined)")
    print(f"  Total weeks with trades : {total_weeks}")
    print(f"  Winning weeks           : {week_df['win_week'].sum()}  "
          f"({week_df['win_week'].mean()*100:.1f}%)")
    print(f"  Losing weeks            : {(~week_df['win_week']).sum()}  "
          f"({(~week_df['win_week']).mean()*100:.1f}%)")
    print()
    pcts = [5,10,25,50,75,90,95]
    vals = np.percentile(week_df["total_R"], pcts)
    print(f"  Weekly R percentiles:")
    print(f"  {'─'*40}")
    for p,v in zip(pcts,vals):
        print(f"    {p:>3}th pctile : {v:>+7.2f}R")
    print()
    print(f"  Mean weekly R  : {week_df['total_R'].mean():+.2f}R")
    print(f"  Median weekly R: {week_df['total_R'].median():+.2f}R")
    print(f"  Worst week     : {week_df['total_R'].min():+.2f}R")
    print(f"  Best week      : {week_df['total_R'].max():+.2f}R")
    print(f"  Std dev        : {week_df['total_R'].std():.2f}R")

    # ── 4. Zero-trade weeks ───────────────────────────────────────────────────
    # Need to count weeks where market was open but no signal fired
    # Approximate: total ISO weeks in dataset - weeks with trades
    all_iso_weeks = set()
    for d in dates:
        d_ts = pd.Timestamp(d)
        iso  = d_ts.isocalendar()
        all_iso_weeks.add((int(iso.year), int(iso.week)))
    zero_trade_weeks = len(all_iso_weeks) - total_weeks

    sec("4. ZERO-TRADE WEEKS  (weeks where system produces nothing)")
    print(f"  Total ISO weeks in dataset : {len(all_iso_weeks)}")
    print(f"  Weeks with ≥1 trade        : {total_weeks}  ({total_weeks/len(all_iso_weeks)*100:.1f}%)")
    print(f"  Zero-trade weeks           : {zero_trade_weeks}  "
          f"({zero_trade_weeks/len(all_iso_weeks)*100:.1f}%)")
    print(f"  Expected zero-trade weeks per competition (1 wk): "
          f"{zero_trade_weeks/len(all_iso_weeks)*100:.1f}% chance")

    # ── 5. Year-by-year weekly stats ─────────────────────────────────────────
    sec("5. YEAR-BY-YEAR WEEKLY PERFORMANCE")
    print(f"  {'Year':<6}  {'Wks traded':>10}  {'Win wks':>8}  {'WR':>6}  "
          f"{'Avg wkly R':>11}  {'Best wk':>8}  {'Worst wk':>9}")
    print(f"  {'─'*66}")
    for yr, sub in week_df.groupby("year"):
        wr  = sub["win_week"].mean()*100
        avg = sub["total_R"].mean()
        bst = sub["total_R"].max()
        wst = sub["total_R"].min()
        print(f"  {yr:<6}  {len(sub):>10}  {sub['win_week'].sum():>8}  "
              f"{wr:>5.1f}%  {avg:>+10.2f}R  {bst:>+7.2f}R  {wst:>+8.2f}R")

    # ── 6. Competition week simulation ────────────────────────────────────────
    sec("6. COMPETITION WEEK SIMULATION  (5-day window Jun 8-12 equivalent)")
    # Simulate many random 5-day blocks and report outcome distribution
    all_signal_days = tdf.drop_duplicates("date").sort_values("date")
    day_returns = tdf.groupby("date")["r"].sum().reset_index()
    day_returns.columns = ["date","total_R"]

    # Rolling 5-signal-day windows
    dr_sorted = day_returns.sort_values("date")["total_R"].values
    windows = [dr_sorted[i:i+5].sum() for i in range(len(dr_sorted)-4)]
    windows = np.array(windows)

    print(f"  Based on {len(windows)} rolling 5-signal-day windows:")
    print()
    print(f"  Probability of positive week  : {(windows>0).mean()*100:.1f}%")
    print(f"  Probability of negative week  : {(windows<0).mean()*100:.1f}%")
    print()
    for p,v in zip([5,10,25,50,75,90,95], np.percentile(windows,[5,10,25,50,75,90,95])):
        bar = "█"*max(0,int((v+10)/2))
        print(f"    {p:>3}th pctile : {v:>+7.2f}R  {bar}")
    print()
    print(f"  At 5% per leg on $25k:")
    for p,v in zip([10,25,50,75,90], np.percentile(windows,[10,25,50,75,90])):
        dollar = v * 0.05 * 25000 / 2   # /2 because R is per leg, 2 legs
        print(f"    {p:>3}th pctile : {v:>+6.2f}R  →  ${dollar:>+8,.0f}")

    # ── 7. High-confidence week identification ────────────────────────────────
    sec("7. WHAT MAKES A GREAT COMPETITION WEEK?")
    week_types = tdf.groupby("iso_week")["sig_type"].apply(list).reset_index()
    week_types["has_3way"]  = week_types["sig_type"].apply(lambda x: "3WAY" in x)
    week_types["has_pwhl"]  = week_types["sig_type"].apply(
        lambda x: any(s in x for s in ["PWHL_ONLY","ORB_PWHL","3WAY","PDHL_PWHL"]))
    week_types["has_pdhl"]  = week_types["sig_type"].apply(
        lambda x: any(s in x for s in ["PDHL_ONLY","ORB_PDHL","3WAY","PDHL_PWHL"]))
    week_types["n_signals"] = week_types["sig_type"].apply(
        lambda x: len(set([s for s in x])))

    wk_r = week_df.set_index("iso_week")["total_R"]
    week_types = week_types.merge(
        pd.Series(wk_r, name="total_R"), on="iso_week", how="left")

    print(f"  Weeks containing a 3-Way signal:")
    sub3 = week_types[week_types["has_3way"]]
    if len(sub3):
        sub3_r = sub3["total_R"].dropna()
        print(f"    n={len(sub3)}  Avg R={sub3_r.mean():+.2f}  "
              f"Win%={( sub3_r>0).mean()*100:.1f}%  "
              f"Worst={sub3_r.min():+.2f}R")

    print(f"  Weeks containing a PWH/PWL signal (any type):")
    subp = week_types[week_types["has_pwhl"]]
    if len(subp):
        subp_r = subp["total_R"].dropna()
        print(f"    n={len(subp)}  Avg R={subp_r.mean():+.2f}  "
              f"Win%={(subp_r>0).mean()*100:.1f}%  "
              f"Worst={subp_r.min():+.2f}R")

    print(f"  Weeks with ONLY ORB-only signals (weakest):")
    sub_orb = week_types[week_types["sig_type"].apply(
        lambda x: all(s=="ORB_ONLY" for s in x))]
    if len(sub_orb):
        sub_orb_r = sub_orb["total_R"].dropna()
        print(f"    n={len(sub_orb)}  Avg R={sub_orb_r.mean():+.2f}  "
              f"Win%={(sub_orb_r>0).mean()*100:.1f}%  "
              f"Worst={sub_orb_r.min():+.2f}R")

    print()
    print("="*W)
    print("  SUMMARY FOR COMPETITION WEEK")
    print("="*W)
    print(f"  Expected signal days per week         : "
          f"{tdf['date'].nunique()/8.1/52:.1f}")
    print(f"  Chance of at least 1 signal day       : "
          f"{total_weeks/len(all_iso_weeks)*100:.1f}%")
    print(f"  Chance of a 3-Way or PWH/PWL signal   : "
          f"{len(subp)/len(all_iso_weeks)*100:.1f}%")
    print(f"  Chance of winning week (when trading)  : "
          f"{week_df['win_week'].mean()*100:.1f}%")
    print(f"  Expected weekly R (signal weeks only)  : "
          f"{week_df['total_R'].mean():+.2f}R")
    print(f"  At 5% risk: expected $         "
          f": ${week_df['total_R'].mean()*0.05*25000/2:+,.0f}")
    print()


if __name__ == "__main__":
    main()
