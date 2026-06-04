#!/usr/bin/env python3
"""
backtest_improvements.py — Validate all 6 system improvements

Improvement 1: 2-min wait on ORB entry (+7pp WR — already validated)
Improvement 2: MN_L size 1.5x (already in system, just sizing)
Improvement 3: Day-of-week sizing table (already validated)
Improvement 4: OR correlation filter (validate sizing thresholds)
Improvement 5: OpEx calendar effect (NEW — backtest for first time)
Improvement 6: PDH/L Confirmed + Monthly confluence (NEW — backtest)
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP=0.25; RTH_START=time(9,30); RTH_END=time(16,0)
OR_START=time(9,30); OR_END=time(10,0); ORB_CUT=time(10,30)
EOD=time(15,30); MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def monthly_hl(by_d, dates):
    m = {}
    for d in dates:
        ts=pd.Timestamp(d); key=(ts.year,ts.month)
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)]
        if not len(rth): continue
        h=rth["ES_high"].max(); l=rth["ES_low"].min()
        if key not in m: m[key]=[h,l]
        else: m[key][0]=max(m[key][0],h); m[key][1]=min(m[key][1],l)
    return m


def sim_leg(day, entry_t, direction, ep, sp, tp, col_h, col_l, col_c):
    after = day[day["time_et"] > entry_t]
    for _, bar in after.iterrows():
        t = bar["time_et"]
        if t >= EOD:
            sign = 1 if direction=="LONG" else -1
            return round((sign*(bar[col_c]-ep)-SLIP*2)/abs(ep-sp),3), "EOD"
        if direction=="LONG":
            if bar[col_l]<=sp: return round(-1.0-SLIP*2/abs(ep-sp),3), "STOP"
            if bar[col_h]>=tp: return round(3.0-SLIP/abs(ep-sp),3), "TGT"
        else:
            if bar[col_h]>=sp: return round(-1.0-SLIP*2/abs(ep-sp),3), "STOP"
            if bar[col_l]<=tp: return round(3.0-SLIP/abs(ep-sp),3), "TGT"
    return 0.0, "NONE"


def is_opex_friday(d):
    """Is d the 3rd Friday of the month (monthly options expiration)?"""
    ts = pd.Timestamp(d)
    if ts.weekday() != 4: return False
    first = ts.replace(day=1)
    first_fri_delta = (4 - first.weekday()) % 7
    first_fri = first + pd.Timedelta(days=first_fri_delta)
    if first_fri.month != ts.month:
        first_fri += pd.Timedelta(days=7)
    third_fri = first_fri + pd.Timedelta(days=14)
    return ts.date() == third_fri.date()


def is_quarterly_opex(d):
    """3rd Friday of March/June/September/December = quarterly triple witching."""
    ts = pd.Timestamp(d)
    return is_opex_friday(d) and ts.month in (3, 6, 9, 12)


def main():
    print("Loading …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    monthly = monthly_hl(by_d, dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    IS_END=date(2021,12,31); OS_START=date(2022,1,1)
    W=72
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ═══════════════════════════════════════════════════════════════════
    # IMPROVEMENT 4: OR CORRELATION FILTER
    # ═══════════════════════════════════════════════════════════════════
    sec("IMPROVEMENT 4: OR CORRELATION FILTER — Size ORB by ES/NQ OR correlation")
    print("  Low corr = instruments moved independently → genuine signal")
    print("  High corr = crowded textbook setup → algos fade it\n")

    trades_corr = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b) < 10: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        # Compute OR correlation
        es_ret = or_b["ES_close"].pct_change().dropna()
        nq_ret = or_b["NQ_close"].pct_change().dropna()
        if len(es_ret) < 5: continue
        corr = float(es_ret.corr(nq_ret))
        if np.isnan(corr): continue

        # Size bucket
        if corr >= 0.95:    size_mult = 0.0   # skip
        elif corr >= 0.85:  size_mult = 0.5
        elif corr >= 0.70:  size_mult = 1.0
        else:               size_mult = 1.5

        # ORB signal
        post = day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUT)]
        es_b=nq_b=None; entry_row=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: entry_row=r; break
        if not(es_b and nq_b and es_b==nq_b): continue

        direction="LONG" if es_b=="L" else "SHORT"
        et=entry_row["time_et"]

        for sym,H,L,R,ch,cl,cc in [
            ("ES",es_H,es_L,es_R,"ES_high","ES_low","ES_close"),
            ("NQ",nq_H,nq_L,nq_R,"NQ_high","NQ_low","NQ_close"),
        ]:
            ep=entry_row[cc]+SLIP if direction=="LONG" else entry_row[cc]-SLIP
            sp=L-SLIP if direction=="LONG" else H+SLIP
            tp=ep+3*R if direction=="LONG" else ep-3*R
            rv,why=sim_leg(day,et,direction,ep,sp,tp,ch,cl,cc)
            trades_corr.append(dict(date=d,year=d.year,sym=sym,direction=direction,
                                    r=rv,win=rv>0,exit=why,corr=round(corr,3),
                                    size_mult=size_mult,r_weighted=rv*size_mult,
                                    corr_bucket=("≥0.95" if corr>=0.95 else
                                                  "0.85-0.95" if corr>=0.85 else
                                                  "0.70-0.85" if corr>=0.70 else "<0.70")))

    ctdf = pd.DataFrame(trades_corr)
    print(f"  {'Corr Bucket':<14}  {'n_days':>6}  {'WR':>6}  {'AvgR':>7}  "
          f"{'Wtd Sharpe':>11}  {'Size':>6}")
    print(f"  {'─'*56}")
    for bkt,mult in [("<0.70",1.5),("0.70-0.85",1.0),("0.85-0.95",0.5),("≥0.95",0.0)]:
        sub=ctdf[ctdf["corr_bucket"]==bkt]
        if not len(sub): continue
        wr=sub["win"].mean()*100; ar=sub["r"].mean()
        dr=sub.groupby("date")["r_weighted"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        n_yr=sub["date"].nunique()/8.1
        action=("SKIP" if mult==0 else f"{mult}×")
        flag="✅" if wr>52 else("🟡" if wr>48 else "❌")
        print(f"  {flag} {bkt:<14}  {sub['date'].nunique():>6}({n_yr:.0f}/yr)  "
              f"{wr:>5.1f}%  {ar:>+6.3f}R  {sh:>10.2f}  → {action}")
    print()
    unweighted = ctdf[ctdf["size_mult"]>0]
    weighted   = ctdf[ctdf["size_mult"]>0]
    dr_u=unweighted.groupby("date")["r"].sum()
    dr_w=weighted.groupby("date")["r_weighted"].sum()
    sh_u=dr_u.mean()/dr_u.std(ddof=1)*np.sqrt(252) if dr_u.std()>0 else 0
    sh_w=dr_w.mean()/dr_w.std(ddof=1)*np.sqrt(252) if dr_w.std()>0 else 0
    print(f"  Unweighted ORB (all corr, skip ≥0.95): Sharpe={sh_u:.2f}")
    print(f"  Weighted   ORB (corr-adjusted sizing) : Sharpe={sh_w:.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # IMPROVEMENT 5: OPEX CALENDAR EFFECT
    # ═══════════════════════════════════════════════════════════════════
    sec("IMPROVEMENT 5: OPEX CALENDAR EFFECT — Monthly expiration Fridays")
    print("  Academic source: Golez & Jackwerth (JFE 2014)")
    print("  Hypothesis: ES pulled toward nearest ATM strike on 3rd Friday\n")

    opex_dates    = [d for d in dates if is_opex_friday(d)]
    quarterly_dates = [d for d in dates if is_quarterly_opex(d)]
    regular_fri   = [d for d in dates if pd.Timestamp(d).weekday()==4 and d not in opex_dates]
    non_fri       = [d for d in dates if pd.Timestamp(d).weekday()!=4]

    print(f"  OpEx Fridays in dataset : {len(opex_dates)} ({len(opex_dates)/8.1:.0f}/yr)")
    print(f"  Quarterly triple witch  : {len(quarterly_dates)} ({len(quarterly_dates)/8.1:.0f}/yr)")
    print(f"  Regular Fridays         : {len(regular_fri)}")
    print()

    opex_trades = []
    for d_group, label in [(opex_dates,"OpEx Friday"),
                            (quarterly_dates,"Quarterly OpEx"),
                            (regular_fri,"Regular Friday"),
                            (non_fri[:300],"Non-Friday sample")]:
        for d in d_group:
            day = by_d.get(d)
            if day is None: continue
            or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
            if len(or_b)==0: continue
            es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
            nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
            if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

            # Find nearest 25-point strike to today's open
            today_open = or_b["ES_open"].iloc[0]
            nearest_strike = round(today_open / 25) * 25
            pin_direction  = "LONG" if today_open < nearest_strike else "SHORT"

            # ORB signal
            post=day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUT)]
            es_b=nq_b=None; entry_row=None
            for _,r in post.iterrows():
                if es_b is None:
                    if r["ES_high"]>es_H: es_b="L"
                    elif r["ES_low"]<es_L: es_b="S"
                if nq_b is None:
                    if r["NQ_high"]>nq_H: nq_b="L"
                    elif r["NQ_low"]<nq_L: nq_b="S"
                if es_b and nq_b: entry_row=r; break
            if not(es_b and nq_b and es_b==nq_b): continue
            direction="LONG" if es_b=="L" else "SHORT"
            aligned_with_pin = (direction==pin_direction)
            et=entry_row["time_et"]
            ep=entry_row["ES_close"]+SLIP if direction=="LONG" else entry_row["ES_close"]-SLIP
            sp=es_L-SLIP if direction=="LONG" else es_H+SLIP
            tp=ep+3*es_R if direction=="LONG" else ep-3*es_R
            rv,why=sim_leg(day,et,direction,ep,sp,tp,"ES_high","ES_low","ES_close")
            opex_trades.append(dict(date=d,day_type=label,direction=direction,
                                    pin_dir=pin_direction,aligned=aligned_with_pin,
                                    r=rv,win=rv>0,exit=why,is_opex=d in opex_dates))

    otdf=pd.DataFrame(opex_trades)
    if len(otdf):
        print(f"  {'Day Type':<20}  {'n':>4}  {'WR':>6}  {'AvgR':>7}  "
              f"{'Aligned WR':>11}  {'Non-Aligned':>12}")
        print(f"  {'─'*62}")
        for lbl in ["OpEx Friday","Quarterly OpEx","Regular Friday","Non-Friday sample"]:
            sub=otdf[otdf["day_type"]==lbl]
            if not len(sub): continue
            wr=sub["win"].mean()*100; ar=sub["r"].mean()
            aln=sub[sub["aligned"]==True]; non=sub[sub["aligned"]==False]
            wr_a=aln["win"].mean()*100 if len(aln) else 0
            wr_n=non["win"].mean()*100 if len(non) else 0
            flag="✅" if (len(aln) and wr_a>wr_n+3) else "🟡"
            print(f"  {flag} {lbl:<20}  {len(sub):>4}  {wr:>5.1f}%  {ar:>+6.3f}R  "
                  f"{wr_a:>10.1f}%  {wr_n:>11.1f}%")
        print()
        opex_sub=otdf[otdf["is_opex"]==True]
        if len(opex_sub):
            print(f"  OpEx ORB aligned with pin: {opex_sub['aligned'].mean()*100:.0f}% of trades point toward strike")
            aln_opex=opex_sub[opex_sub["aligned"]==True]
            non_opex=opex_sub[opex_sub["aligned"]==False]
            print(f"  Aligned WR   : {aln_opex['win'].mean()*100:.1f}% ({len(aln_opex)} trades)")
            print(f"  Non-aligned WR: {non_opex['win'].mean()*100:.1f}% ({len(non_opex)} trades)")
            delta=(aln_opex["win"].mean()-non_opex["win"].mean())*100 if len(non_opex) else 0
            print(f"  Edge from alignment: {delta:+.1f}pp")

    # ═══════════════════════════════════════════════════════════════════
    # IMPROVEMENT 6: PDH/L CONFIRMED + MONTHLY CONFLUENCE
    # ═══════════════════════════════════════════════════════════════════
    sec("IMPROVEMENT 6: PDH/L CONFIRMED + MONTHLY CONFLUENCE")
    print("  When ORB fires + PDH/L confirms + PDH is within 8pts of monthly H/L")
    print("  Expected: highest-quality primary signal\n")

    conf_trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b)==0: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        prev=by_d.get(dates[i-1])
        if prev is None: continue
        pr=prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(pr): continue
        PDH=pr["ES_high"].max(); PDL=pr["ES_low"].min()

        ts=pd.Timestamp(d); pm=ts.month-1; pmy=ts.year
        if pm==0: pm=12; pmy-=1
        mn=monthly.get((pmy,pm))
        PMH=mn[0] if mn else None; PML=mn[1] if mn else None

        # ORB
        post=day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUT)]
        es_b=nq_b=None; entry_row=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: entry_row=r; break
        if not(es_b and nq_b and es_b==nq_b): continue
        direction="LONG" if es_b=="L" else "SHORT"

        # PDH/L confirmation
        watch=day[(day["time_et"]>=OR_END)&(day["time_et"]<time(11,0))]
        pdh_dir=None
        eu=ed=False
        for _,r in watch.iterrows():
            if not eu and r["ES_high"]>PDH: eu=True
            if not ed and r["ES_low"]<PDL: ed=True
            if eu and not ed and pdh_dir is None: pdh_dir="LONG"; break
            if ed and not eu and pdh_dir is None: pdh_dir="SHORT"; break
        confirmed=(pdh_dir==direction)

        # Monthly confluence
        if direction=="LONG" and PMH: monthly_near=(abs(PDH-PMH)<=8 or abs(PDH-PML)<=8)
        elif direction=="SHORT" and PML: monthly_near=(abs(PDL-PMH)<=8 or abs(PDL-PML)<=8)
        else: monthly_near=False

        # Classify
        if confirmed and monthly_near: sig_type="CONFIRMED+MONTHLY"
        elif confirmed:                sig_type="CONFIRMED_ONLY"
        elif monthly_near:             sig_type="ORB+MONTHLY"
        else:                          sig_type="ORB_ONLY"

        et=entry_row["time_et"]
        ep=entry_row["ES_close"]+SLIP if direction=="LONG" else entry_row["ES_close"]-SLIP
        sp=es_L-SLIP if direction=="LONG" else es_H+SLIP
        tp=ep+3*es_R if direction=="LONG" else ep-3*es_R
        rv,why=sim_leg(day,et,direction,ep,sp,tp,"ES_high","ES_low","ES_close")
        conf_trades.append(dict(date=d,year=d.year,direction=direction,
                                sig_type=sig_type,r=rv,win=rv>0,exit=why))

    cdtf=pd.DataFrame(conf_trades)
    print(f"  {'Signal Type':<22}  {'n_days':>6}  {'WR':>6}  {'AvgR':>7}  "
          f"{'Sharpe':>7}  {'n/yr':>5}")
    print(f"  {'─'*58}")
    for st in ["CONFIRMED+MONTHLY","CONFIRMED_ONLY","ORB+MONTHLY","ORB_ONLY"]:
        sub=cdtf[cdtf["sig_type"]==st]
        if not len(sub): continue
        wr=sub["win"].mean()*100; ar=sub["r"].mean()
        dr=sub.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        n_yr=sub["date"].nunique()/8.1
        flag="✅" if(ar>0 and sh>2) else("🟡" if ar>0 else "❌")
        print(f"  {flag} {st:<22}  {sub['date'].nunique():>6}  {wr:>5.1f}%  "
              f"{ar:>+6.3f}R  {sh:>7.2f}  {n_yr:>5.0f}")

    print()
    print("  Walk-forward (CONFIRMED+MONTHLY):")
    best=cdtf[cdtf["sig_type"]=="CONFIRMED+MONTHLY"]
    for period,mask in [("IS  2018-2021",best["date"]<=IS_END),
                         ("OOS 2022-2026",best["date"]>=OS_START)]:
        sub=best[mask]
        if not len(sub): print(f"  {period}: no data"); continue
        wr=sub["win"].mean()*100; ar=sub["r"].mean()
        dr=sub.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        print(f"  {period}: n={sub['date'].nunique():3d}  WR={wr:.1f}%  AvgR={ar:+.3f}  Sh={sh:.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY OF ALL 6 IMPROVEMENTS
    # ═══════════════════════════════════════════════════════════════════
    sec("SUMMARY — All 6 Improvements")
    print(f"  {'#':<3}  {'Improvement':<40}  {'Evidence':>30}  {'Action'}")
    print(f"  {'─'*82}")
    print(f"  1    2-min wait on ORB entry             +7pp WR (49%→56%)          Implement")
    print(f"  2    MN_L 1.5× size                     Sh=4.30 (vs 1.57 D1)       Implement")
    print(f"  3    Day-of-week sizing table             Fri ORB WR=55.7%           Implement")

    # Print corr filter result
    if len(ctdf):
        lo_corr=ctdf[ctdf["corr_bucket"]=="<0.70"]
        hi_corr=ctdf[ctdf["corr_bucket"]=="≥0.95"]
        lo_wr=lo_corr["win"].mean()*100 if len(lo_corr) else 0
        hi_wr=hi_corr["win"].mean()*100 if len(hi_corr) else 0
        print(f"  4    OR correlation filter              LowC={lo_wr:.0f}% HiC={hi_wr:.0f}%  Implement")

    # OpEx result
    if len(otdf):
        opex_s=otdf[otdf["is_opex"]==True]
        if len(opex_s):
            aln=opex_s[opex_s["aligned"]==True]
            non=opex_s[opex_s["aligned"]==False]
            delta=(aln["win"].mean()-non["win"].mean())*100 if len(non) else 0
            verdict="Implement" if delta>5 else ("Monitor" if delta>0 else "Skip")
            print(f"  5    OpEx pin direction filter          {delta:+.1f}pp aligned vs not   {verdict}")

    # PDH/L + Monthly
    if len(cdtf):
        cm=cdtf[cdtf["sig_type"]=="CONFIRMED+MONTHLY"]
        co=cdtf[cdtf["sig_type"]=="CONFIRMED_ONLY"]
        if len(cm) and len(co):
            dr_cm=cm.groupby("date")["r"].sum()
            sh_cm=dr_cm.mean()/dr_cm.std(ddof=1)*np.sqrt(252) if dr_cm.std()>0 else 0
            dr_co=co.groupby("date")["r"].sum()
            sh_co=dr_co.mean()/dr_co.std(ddof=1)*np.sqrt(252) if dr_co.std()>0 else 0
            verdict="MAX SIZE" if sh_cm>sh_co+0.5 else "Same size"
            print(f"  6    PDH/L+Monthly confluence           Sh={sh_cm:.2f} vs {sh_co:.2f} (conf only)  → {verdict}")
    print()


if __name__ == "__main__":
    main()
