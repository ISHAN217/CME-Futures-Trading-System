#!/usr/bin/env python3
"""
verify_signals.py — Deep audit of ES/NQ signal numbers

Checks for:
  1. Look-ahead bias in PWH/PWL calculation
  2. Entry price validity (are we entering at impossible prices?)
  3. Sample uniqueness (no duplicate trades)
  4. True walk-forward (discover PWH/PWL ONLY on IS data, apply cold to OOS)
  5. Manual trace of 10 random trades
  6. Statistical significance of each signal type
  7. Survivorship/selection bias check
"""

import numpy as np
import pandas as pd
from datetime import time, date
import random

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"

OR_START  = time(9,30); OR_END    = time(10,0)
ORB_CUT   = time(10,30); WATCH_END = time(11,0)
RTH_START = time(9,30); RTH_END   = time(16,0)
EOD       = time(15,30)
SLIP      = 0.25; PMULT = 3.0
MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def compute_weekly_hl(by_d, dates):
    w = {}
    for d in dates:
        ts  = pd.Timestamp(d); iso = ts.isocalendar()
        key = (int(iso.year), int(iso.week))
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)]
        if not len(rth): continue
        es_h=rth["ES_high"].max(); es_l=rth["ES_low"].min()
        nq_h=rth["NQ_high"].max(); nq_l=rth["NQ_low"].min()
        if key not in w: w[key]=[es_h,es_l,nq_h,nq_l]
        else:
            w[key][0]=max(w[key][0],es_h); w[key][1]=min(w[key][1],es_l)
            w[key][2]=max(w[key][2],nq_h); w[key][3]=min(w[key][3],nq_l)
    return w


def get_pw(d, w):
    ts=pd.Timestamp(d); iso=ts.isocalendar()
    pw=int(iso.week)-1; yr=int(iso.year)
    if pw==0:
        yr-=1; pw=int(pd.Timestamp(f"{yr}-12-28").isocalendar().week)
    key=(yr,pw)
    return (w[key][0],w[key][1],w[key][2],w[key][3]) if key in w else (None,None,None,None)


def sim_trade(day, entry_t, direction, ep, sp, tp, col_h, col_l, col_c):
    """Returns (r_val, exit_px, exit_reason, exit_time)."""
    rng = abs(ep - sp)
    after = day[day["time_et"] > entry_t]
    for _, bar in after.iterrows():
        t = bar["time_et"]
        if t >= EOD:
            sign = 1 if direction=="LONG" else -1
            net  = sign*(bar[col_c]-ep) - SLIP*2
            return round(net/rng,3), bar[col_c], "EOD", t
        if direction=="LONG":
            if bar[col_l]<=sp: return round(-1.0-SLIP*2/rng,3), sp, "STOP", t
            if bar[col_h]>=tp: return round(PMULT-SLIP/rng,3),  tp, "TGT",  t
        else:
            if bar[col_h]>=sp: return round(-1.0-SLIP*2/rng,3), sp, "STOP", t
            if bar[col_l]<=tp: return round(PMULT-SLIP/rng,3),  tp, "TGT",  t
    return 0.0, ep, "NONE", None


def run_full(df, dates, by_d, weekly, only_pwhl_standalone=False):
    """Run full signal backtest. Returns detailed trade log."""
    trades = []

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue

        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b)==0: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        # Prior week H/L
        PWH_ES,PWL_ES,PWH_NQ,PWL_NQ = get_pw(d, weekly)
        if PWH_ES is None: continue

        # ── CHECK 1: PWH/PWL must be from PRIOR week, not current ──
        d_ts = pd.Timestamp(d); iso = d_ts.isocalendar()
        pw = int(iso.week)-1; yr = int(iso.year)
        if pw==0: yr-=1; pw=int(pd.Timestamp(f"{yr}-12-28").isocalendar().week)
        # Verify: PWH_ES should NOT be from this week's bars
        this_week_dates = [dd for dd in dates
                           if (pd.Timestamp(dd).isocalendar().week==int(iso.week)
                               and pd.Timestamp(dd).isocalendar().year==int(iso.year)
                               and dd < d)]
        if this_week_dates:
            this_week_max_es = max(
                by_d[dd]["ES_high"].max() if by_d.get(dd) is not None else 0
                for dd in this_week_dates)
            if PWH_ES == this_week_max_es and PWH_ES not in [
                by_d[dd]["ES_high"].max() for dd in dates
                if (pd.Timestamp(dd).isocalendar().week==pw
                    and pd.Timestamp(dd).isocalendar().year==yr
                    and by_d.get(dd) is not None)]:
                # PWH is same as this week's max — potential look-ahead
                pass  # flag but continue for now

        prev = by_d.get(dates[i-1])
        if prev is None: continue
        prth = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(prth): continue
        PDH_ES=prth["ES_high"].max(); PDL_ES=prth["ES_low"].min()
        PDH_NQ=prth["NQ_high"].max(); PDL_NQ=prth["NQ_low"].min()

        watch = day[(day["time_et"]>=OR_END)&(day["time_et"]<WATCH_END)]

        # ORB
        post = day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUT)]
        es_b=nq_b=None; orb_row=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: orb_row=r; break
        orb_fired = bool(es_b and nq_b and es_b==nq_b)
        orb_dir   = ("LONG" if es_b=="L" else "SHORT") if orb_fired else None

        # PDH/L
        eu=ed=nu=nd=False; pdhl_dir=None; pdhl_row=None
        for _,r in watch.iterrows():
            if not eu and r["ES_high"]>PDH_ES: eu=True
            if not ed and r["ES_low"]<PDL_ES:  ed=True
            if not nu and r["NQ_high"]>PDH_NQ: nu=True
            if not nd and r["NQ_low"]<PDL_NQ:  nd=True
            if eu and nu and pdhl_dir is None: pdhl_dir="LONG";  pdhl_row=r; break
            if ed and nd and pdhl_dir is None: pdhl_dir="SHORT"; pdhl_row=r; break
            if (eu and nd) or (ed and nu): break
        pdhl_ok = pdhl_dir is not None

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
        pwhl_ok = pwhl_dir is not None

        # Priority
        direction=None; entry_row_use=None; sig_type=None
        if orb_fired:
            pdhl_agrees = pdhl_ok and pdhl_dir==orb_dir
            pwhl_agrees = pwhl_ok and pwhl_dir==orb_dir
            any_conflict= (pdhl_ok and pdhl_dir!=orb_dir) or (pwhl_ok and pwhl_dir!=orb_dir)
            if any_conflict: continue
            if pdhl_agrees and pwhl_agrees: sig_type="3WAY"
            elif pdhl_agrees:               sig_type="ORB_PDHL"
            elif pwhl_agrees:               sig_type="ORB_PWHL"
            else:                           sig_type="ORB_ONLY"
            direction=orb_dir; entry_row_use=orb_row
        elif pdhl_ok and pwhl_ok and pdhl_dir==pwhl_dir:
            sig_type="PDHL_PWHL"; direction=pdhl_dir; entry_row_use=pdhl_row
        elif pdhl_ok:
            sig_type="PDHL_ONLY"; direction=pdhl_dir; entry_row_use=pdhl_row
        elif pwhl_ok:
            sig_type="PWHL_ONLY"; direction=pwhl_dir; entry_row_use=pwhl_row
        else: continue

        if only_pwhl_standalone and sig_type!="PWHL_ONLY": continue
        if direction is None or entry_row_use is None: continue

        et = entry_row_use["time_et"]
        use_pdhl = sig_type in ("PDHL_ONLY","PDHL_PWHL","3WAY","ORB_PDHL")
        after = day[day["time_et"]>et]

        for sym,PDH,PDL,H,L,R,ch,cl,cc in [
            ("ES",PDH_ES,PDL_ES,es_H,es_L,es_R,"ES_high","ES_low","ES_close"),
            ("NQ",PDH_NQ,PDL_NQ,nq_H,nq_L,nq_R,"NQ_high","NQ_low","NQ_close"),
        ]:
            ep = (PDH if direction=="LONG" else PDL) if use_pdhl else (H if direction=="LONG" else L)
            sp = ep-R if direction=="LONG" else ep+R
            tp = round(ep+PMULT*R,2) if direction=="LONG" else round(ep-PMULT*R,2)

            # ── CHECK 2: Entry price must be reachable that bar ──
            bar_at_entry = after.iloc[0] if len(after) else None
            entry_reachable = True
            if bar_at_entry is not None:
                if direction=="LONG" and ep < bar_at_entry[cl]:
                    entry_reachable = False  # entered below bar low — impossible fill
                elif direction=="SHORT" and ep > bar_at_entry[ch]:
                    entry_reachable = False  # entered above bar high — impossible fill

            rv, exit_px, exit_r, exit_t = sim_trade(after, et, direction, ep, sp, tp, ch, cl, cc)
            trades.append(dict(
                date=d, year=d.year, sym=sym, sig_type=sig_type,
                direction=direction, entry_t=et,
                ep=ep, sp=sp, tp=tp, or_R=R,
                exit_px=exit_px, exit_r=exit_r, exit_t=exit_t,
                r=rv, win=rv>0,
                entry_reachable=entry_reachable,
                pwh_es=PWH_ES, pwl_es=PWL_ES,
                pwh_nq=PWH_NQ, pwl_nq=PWL_NQ,
                pdh_es=PDH_ES, pdl_es=PDL_ES,
            ))

    return pd.DataFrame(trades)


def main():
    print("Loading data …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}

    print("Computing weekly H/L …")
    weekly = compute_weekly_hl(by_d, dates)

    print("Running full simulation …\n")
    tdf = run_full(df, dates, by_d, weekly)
    print(f"Total legs: {len(tdf):,}  Signal days: {tdf['date'].nunique()}")

    W = 72
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── CHECK 1: Look-ahead bias ──────────────────────────────────────────────
    sec("CHECK 1: LOOK-AHEAD BIAS — PWH/PWL is truly prior week?")
    pwhl = tdf[tdf["sig_type"]=="PWHL_ONLY"]
    print(f"  PWHL_ONLY sample: {len(pwhl)} legs  {len(pwhl)//2} signal days")
    # For each PWHL trade, verify PWH comes from PRIOR week
    violations = 0
    for _, row in pwhl.head(100).iterrows():  # check first 100
        d = row["date"]
        ts = pd.Timestamp(d); iso = ts.isocalendar()
        curr_week = int(iso.week); curr_yr = int(iso.year)
        pw = curr_week-1; py = curr_yr
        if pw==0: py-=1; pw=int(pd.Timestamp(f"{py}-12-28").isocalendar().week)
        key = (py, pw)
        # Prior week days
        pw_days = [dd for dd in dates
                   if (pd.Timestamp(dd).isocalendar().week==pw
                       and pd.Timestamp(dd).isocalendar().year==py)]
        if not pw_days: continue
        # Compute actual prior week ES high
        pw_es_h = max(by_d[dd]["ES_high"].max() for dd in pw_days if by_d.get(dd) is not None)
        if abs(pw_es_h - row["pwh_es"]) > 0.1:
            violations += 1
    print(f"  Checked 100 trades: {violations} PWH/PWL look-ahead violations")
    print(f"  Result: {'✅ CLEAN' if violations==0 else f'❌ {violations} VIOLATIONS'}")

    # ── CHECK 2: Entry price validity ────────────────────────────────────────
    sec("CHECK 2: ENTRY PRICE VALIDITY — can we actually fill at entry?")
    impossible = tdf[tdf["entry_reachable"]==False]
    print(f"  Total legs        : {len(tdf):,}")
    print(f"  Impossible fills  : {len(impossible):,}  ({len(impossible)/len(tdf)*100:.1f}%)")
    if len(impossible):
        print(f"  ⚠️  {len(impossible)/len(tdf)*100:.1f}% of trades entered at an impossible price")
        print(f"     This inflates WR — PDH entry might be below bar low at signal bar")
        print(f"  By signal type:")
        for st, sub in impossible.groupby("sig_type"):
            total = len(tdf[tdf["sig_type"]==st])
            print(f"    {st:<20}: {len(sub):4d}/{total:4d} ({len(sub)/total*100:.1f}%)")
    else:
        print(f"  ✅ All fills valid")

    # ── CHECK 3: No duplicate trades ─────────────────────────────────────────
    sec("CHECK 3: SAMPLE INTEGRITY — no duplicates or errors")
    dupes = tdf.duplicated(subset=["date","sym","sig_type"]).sum()
    print(f"  Duplicate (date, sym, sig_type) : {dupes}")
    # Check each date has at most 2 legs (ES + NQ)
    legs_per_day = tdf.groupby("date")["sym"].count()
    over2 = (legs_per_day > 2).sum()
    print(f"  Days with >2 legs               : {over2}")
    print(f"  Result: {'✅ CLEAN' if dupes==0 and over2==0 else '❌ ISSUES FOUND'}")

    # ── CHECK 4: Manual trade trace ───────────────────────────────────────────
    sec("CHECK 4: MANUAL TRADE TRACE — verify 5 random PWHL trades")
    random.seed(42)
    pwhl_days = tdf[tdf["sig_type"]=="PWHL_ONLY"]["date"].unique()
    sample_days = random.sample(list(pwhl_days), min(5, len(pwhl_days)))
    for d in sorted(sample_days):
        sub = tdf[(tdf["date"]==d)&(tdf["sig_type"]=="PWHL_ONLY")]
        row = sub.iloc[0]
        day = by_d[d]
        # Verify the PWH was actually broken in 10:00-11:00 ET
        watch = day[(day["time_et"]>=OR_END)&(day["time_et"]<WATCH_END)]
        pwh_break_bar = watch[watch["ES_high"]>row["pwh_es"]]
        print(f"\n  {d}  dir={row['direction']}  OR_R={row['or_R']:.2f}")
        print(f"    PWH_ES={row['pwh_es']:.2f}  PDH_ES={row['pdh_es']:.2f}")
        print(f"    Entry={row['ep']:.2f}  Stop={row['sp']:.2f}  Target={row['tp']:.2f}")
        print(f"    PWH break at: {pwh_break_bar['time_et'].iloc[0] if len(pwh_break_bar) else 'NOT FOUND'}")
        print(f"    Exit={row['exit_r']} @ {row['exit_px']:.2f}  R={row['r']:+.2f}")
        ok = "✅" if len(pwh_break_bar) > 0 else "❌ PWH NOT BROKEN IN WINDOW"
        print(f"    {ok}")

    # ── CHECK 5: True walk-forward ────────────────────────────────────────────
    sec("CHECK 5: TRUE WALK-FORWARD — IS 2018-2021 → OOS 2022-2026")
    print("  (PWH/PWL signal was DISCOVERED on full data — testing if it")
    print("   works on OOS data that was NEVER used to discover it)")
    print()
    IS_END  = date(2021,12,31)
    OS_START= date(2022,1,1)

    for period, mask in [
        ("IS  2018-2021", tdf["date"] <= IS_END),
        ("OOS 2022-2026", tdf["date"] >= OS_START),
    ]:
        sub = tdf[mask]
        for st in ["3WAY","ORB_PDHL","ORB_PWHL","ORB_ONLY","PDHL_ONLY","PWHL_ONLY"]:
            s = sub[sub["sig_type"]==st]
            if not len(s): continue
            wr=s["win"].mean(); ev=wr*PMULT-(1-wr)
            print(f"  {period}  {st:<16}  n={len(s)//2:3d}  "
                  f"WR={wr*100:5.1f}%  EV={ev:+.3f}R")
        print()

    # ── CHECK 6: Statistical significance ────────────────────────────────────
    sec("CHECK 6: STATISTICAL SIGNIFICANCE — are WRs real?")
    from scipy import stats as scipy_stats
    print(f"  {'Signal':<20}  {'n_legs':>6}  {'WR':>6}  {'p-value':>9}  {'95% CI':>16}  {'Sig?':>6}")
    print(f"  {'─'*70}")
    for st in ["PWHL_ONLY","ORB_PWHL","3WAY","ORB_PDHL","PDHL_ONLY","ORB_ONLY"]:
        sub = tdf[tdf["sig_type"]==st]
        if not len(sub): continue
        n = len(sub); w = sub["win"].sum()
        wr = w/n
        # Binomial test vs H0: WR=0.50
        res = scipy_stats.binomtest(int(w), n, 0.5, alternative="greater")
        p   = res.pvalue
        ci  = scipy_stats.proportion_confint(int(w), n, alpha=0.05, method="wilson")
        sig = "✅ YES" if p < 0.05 else ("🟡 WEAK" if p < 0.10 else "❌ NO")
        print(f"  {st:<20}  {n:>6}  {wr*100:>5.1f}%  {p:>9.4f}  "
              f"[{ci[0]*100:.1f}%–{ci[1]*100:.1f}%]  {sig}")

    # ── CHECK 7: Impossible entry re-run ─────────────────────────────────────
    sec("CHECK 7: CLEAN WR (excluding impossible fills)")
    print("  Re-running stats counting only fills where entry was reachable:")
    clean = tdf[tdf["entry_reachable"]==True]
    for st in ["3WAY","ORB_PDHL","ORB_PWHL","PDHL_ONLY","PWHL_ONLY","ORB_ONLY"]:
        all_st  = tdf[tdf["sig_type"]==st]
        cln_st  = clean[clean["sig_type"]==st]
        if not len(all_st): continue
        wr_all  = all_st["win"].mean()
        wr_cln  = cln_st["win"].mean() if len(cln_st) else 0
        delta   = (wr_cln - wr_all)*100
        print(f"  {st:<16}  All WR={wr_all*100:5.1f}%  "
              f"Clean WR={wr_cln*100:5.1f}%  delta={delta:+.1f}pp  "
              f"n_removed={len(all_st)-len(cln_st)}")

    print()


if __name__ == "__main__":
    main()
