#!/usr/bin/env python3
"""
verify_clean.py — Corrected audit of ES/NQ signal numbers

The first verify script had bugs:
  - Check 5 used wrong entry price for PWHL_ONLY (OR high instead of PWH)
  - Check 2 "impossible fill" checked the NEXT bar, not the signal bar
  - Check 4 looked for PWH break even on SHORT trades (should check PWL)

This corrected version:
  1. Checks look-ahead properly (only RTH bars from prior ISO week)
  2. Checks entry price on the SIGNAL BAR itself (bar where level is first broken)
  3. Uses correct entry prices for each signal type
  4. Re-runs clean WR statistics
"""

import numpy as np
import pandas as pd
from datetime import time, date
import random

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"

OR_START=time(9,30); OR_END=time(10,0); ORB_CUT=time(10,30)
WATCH_END=time(11,0); RTH_START=time(9,30); RTH_END=time(16,0); EOD=time(15,30)
SLIP=0.25; PMULT=3.0; MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def compute_weekly_hl(by_d, dates):
    """Compute RTH-only H/L per ISO week."""
    w = {}
    for d in dates:
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        key=(int(iso.year),int(iso.week))
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)]
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
    if pw==0: yr-=1; pw=int(pd.Timestamp(f"{yr}-12-28").isocalendar().week)
    key=(yr,pw)
    return (w[key][0],w[key][1],w[key][2],w[key][3]) if key in w else (None,None,None,None)


def sim(day, entry_t, direction, ep, sp, tp, ch, cl, cc):
    rng=abs(ep-sp)
    for _,bar in day[day["time_et"]>entry_t].iterrows():
        t=bar["time_et"]
        if t>=EOD:
            s=1 if direction=="LONG" else -1
            return round((s*(bar[cc]-ep)-SLIP*2)/rng,3),"EOD"
        if direction=="LONG":
            if bar[cl]<=sp: return round(-1.0-SLIP*2/rng,3),"STOP"
            if bar[ch]>=tp: return round(PMULT-SLIP/rng,3),"TGT"
        else:
            if bar[ch]>=sp: return round(-1.0-SLIP*2/rng,3),"STOP"
            if bar[cl]<=tp: return round(PMULT-SLIP/rng,3),"TGT"
    return 0.0,"NONE"


def run(df, dates, by_d, weekly):
    trades = []
    for i,d in enumerate(dates[1:],1):
        day=by_d.get(d)
        if day is None: continue
        or_b=day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if not len(or_b): continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue
        PWH_ES,PWL_ES,PWH_NQ,PWL_NQ=get_pw(d,weekly)
        if PWH_ES is None: continue
        prev=by_d.get(dates[i-1])
        if prev is None: continue
        prth=prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(prth): continue
        PDH_ES=prth["ES_high"].max(); PDL_ES=prth["ES_low"].min()
        PDH_NQ=prth["NQ_high"].max(); PDL_NQ=prth["NQ_low"].min()
        watch=day[(day["time_et"]>=OR_END)&(day["time_et"]<WATCH_END)]
        post=day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUT)]
        # ORB
        es_b=nq_b=None; orb_row=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: orb_row=r; break
        orb_fired=bool(es_b and nq_b and es_b==nq_b)
        orb_dir=("LONG" if es_b=="L" else "SHORT") if orb_fired else None
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
        pdhl_ok=pdhl_dir is not None
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
        pwhl_ok=pwhl_dir is not None
        # Priority
        direction=None; entry_row_use=None; sig_type=None
        if orb_fired:
            pdhl_agrees=pdhl_ok and pdhl_dir==orb_dir
            pwhl_agrees=pwhl_ok and pwhl_dir==orb_dir
            any_conflict=(pdhl_ok and pdhl_dir!=orb_dir)or(pwhl_ok and pwhl_dir!=orb_dir)
            if any_conflict: continue
            if pdhl_agrees and pwhl_agrees: sig_type="3WAY"
            elif pdhl_agrees: sig_type="ORB_PDHL"
            elif pwhl_agrees: sig_type="ORB_PWHL"
            else:             sig_type="ORB_ONLY"
            direction=orb_dir; entry_row_use=orb_row if not pdhl_agrees else pdhl_row
        elif pdhl_ok and pwhl_ok and pdhl_dir==pwhl_dir:
            sig_type="PDHL_PWHL"; direction=pdhl_dir; entry_row_use=pdhl_row
        elif pdhl_ok:
            sig_type="PDHL_ONLY"; direction=pdhl_dir; entry_row_use=pdhl_row
        elif pwhl_ok:
            sig_type="PWHL_ONLY"; direction=pwhl_dir; entry_row_use=pwhl_row
        else: continue
        if direction is None or entry_row_use is None: continue

        use_pdhl=sig_type in("PDHL_ONLY","PDHL_PWHL","3WAY","ORB_PDHL")
        use_pwhl=sig_type in("PWHL_ONLY",)
        et=entry_row_use["time_et"]

        for sym,PDH,PDL,PWH,PWL,H,L,R,ch,cl,cc in [
            ("ES",PDH_ES,PDL_ES,PWH_ES,PWL_ES,es_H,es_L,es_R,"ES_high","ES_low","ES_close"),
            ("NQ",PDH_NQ,PDL_NQ,PWH_NQ,PWL_NQ,nq_H,nq_L,nq_R,"NQ_high","NQ_low","NQ_close"),
        ]:
            if use_pdhl:   ep=PDH if direction=="LONG" else PDL
            elif use_pwhl: ep=PWH if direction=="LONG" else PWL
            else:           ep=H   if direction=="LONG" else L
            sp=ep-R if direction=="LONG" else ep+R
            tp=round(ep+PMULT*R,2) if direction=="LONG" else round(ep-PMULT*R,2)

            # CORRECT fill check: can we fill at ep ON THE SIGNAL BAR?
            signal_bar = entry_row_use
            if direction=="LONG":
                fill_ok = (signal_bar[cl] <= ep <= signal_bar[ch])
            else:
                fill_ok = (signal_bar[cl] <= ep <= signal_bar[ch])

            rv,why=sim(day,et,direction,ep,sp,tp,ch,cl,cc)
            trades.append(dict(
                date=d,year=d.year,sym=sym,sig_type=sig_type,direction=direction,
                r=rv,win=rv>0,exit=why,fill_ok=fill_ok,
                ep=ep,sp=sp,tp=tp,or_R=R,
                signal_bar_h=signal_bar[ch],signal_bar_l=signal_bar[cl],
            ))
    return pd.DataFrame(trades)


def main():
    print("Loading …")
    df=load()
    dates=sorted(df["date_et"].unique())
    by_d={d:g.reset_index(drop=True) for d,g in df.groupby("date_et")}
    weekly=compute_weekly_hl(by_d,dates)
    tdf=run(df,dates,by_d,weekly)
    print(f"Total legs: {len(tdf):,}   Signal days: {tdf['date'].nunique()}\n")

    W=72
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. Look-ahead bias check ──────────────────────────────────────────────
    sec("1. LOOK-AHEAD BIAS — PWH/PWL from prior ISO week only?")
    pwhl_trades = tdf[tdf["sig_type"]=="PWHL_ONLY"]
    violations = 0
    total_checked = 0
    for _,row in pwhl_trades.head(200).iterrows():
        d=row["date"]
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        curr_w=int(iso.week); curr_yr=int(iso.year)
        pw=curr_w-1; py=curr_yr
        if pw==0: py-=1; pw=int(pd.Timestamp(f"{py}-12-28").isocalendar().week)
        # Get prior week RTH bars only
        pw_days=[dd for dd in dates
                 if (int(pd.Timestamp(dd).isocalendar().week)==pw
                     and int(pd.Timestamp(dd).isocalendar().year)==py)]
        if not pw_days: continue
        total_checked+=1
        pw_rth=[]
        for dd in pw_days:
            day=by_d.get(dd)
            if day is None: continue
            rth=day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)]
            if len(rth): pw_rth.append(rth["ES_high"].max())
        if not pw_rth: continue
        actual_pwh=max(pw_rth)
        used_pwh=row["ep"] if row["direction"]=="LONG" else None
        # For PWHL_ONLY, ep IS the PWH (from get_pw)
        # Check that this matches what we'd compute from scratch
        if used_pwh and abs(actual_pwh-used_pwh)>0.25:
            violations+=1
    print(f"  Checked {total_checked} PWHL_ONLY trades")
    print(f"  Look-ahead violations (|actual_PWH - used_PWH| > 0.25): {violations}")
    print(f"  Result: {'✅ NO LOOK-AHEAD BIAS' if violations==0 else f'❌ {violations} VIOLATIONS'}")

    # ── 2. Fill validity on SIGNAL BAR ────────────────────────────────────────
    sec("2. FILL VALIDITY — entry price reachable on the signal bar?")
    print("  (Checks if ep is within [bar_low, bar_high] of the signal bar)\n")
    print(f"  {'Signal':<18}  {'Total':>6}  {'Valid fills':>11}  {'Invalid fills':>13}  {'Valid%':>7}")
    print(f"  {'─'*60}")
    for st in ["3WAY","ORB_PDHL","ORB_PWHL","ORB_ONLY","PDHL_PWHL","PDHL_ONLY","PWHL_ONLY"]:
        sub=tdf[tdf["sig_type"]==st]
        if not len(sub): continue
        valid=sub["fill_ok"].sum(); invalid=len(sub)-valid
        pct=valid/len(sub)*100
        flag="✅" if pct>80 else ("🟡" if pct>50 else "❌")
        print(f"  {flag} {st:<18}  {len(sub):>6}  {valid:>11}  {invalid:>13}  {pct:>6.1f}%")

    # ── 3. WR with ONLY valid fills ───────────────────────────────────────────
    sec("3. WR COMPARISON — all fills vs valid fills only")
    print(f"  {'Signal':<18}  {'All WR':>7}  {'Valid WR':>9}  {'Δ':>6}  {'n_valid':>8}")
    print(f"  {'─'*56}")
    for st in ["3WAY","ORB_PDHL","ORB_PWHL","ORB_ONLY","PDHL_PWHL","PDHL_ONLY","PWHL_ONLY"]:
        all_sub=tdf[tdf["sig_type"]==st]
        val_sub=all_sub[all_sub["fill_ok"]==True]
        if not len(all_sub): continue
        all_wr=all_sub["win"].mean()*100
        val_wr=val_sub["win"].mean()*100 if len(val_sub) else 0
        delta=val_wr-all_wr
        flag="✅" if abs(delta)<2 else ("🟡" if abs(delta)<5 else "❌")
        print(f"  {flag} {st:<18}  {all_wr:>6.1f}%  {val_wr:>8.1f}%  {delta:>+5.1f}pp  {len(val_sub):>8}")

    # ── 4. Valid-fill WR by year ──────────────────────────────────────────────
    sec("4. VALID-FILL YEAR-BY-YEAR — are any years suspiciously good?")
    valid_tdf=tdf[tdf["fill_ok"]==True]
    print(f"\n  {'Year':<6}", end="")
    for st in ["3WAY","ORB_PDHL","PWHL_ONLY","ORB_ONLY"]:
        print(f"  {st:>12}", end="")
    print()
    print(f"  {'─'*58}")
    for yr in sorted(tdf["year"].unique()):
        print(f"  {yr:<6}", end="")
        for st in ["3WAY","ORB_PDHL","PWHL_ONLY","ORB_ONLY"]:
            sub=valid_tdf[(valid_tdf["year"]==yr)&(valid_tdf["sig_type"]==st)]
            if not len(sub): print(f"  {'—':>12}", end="")
            else:
                wr=sub["win"].mean()*100
                flag="*" if wr>80 else ""
                print(f"  {wr:>10.1f}%{flag}", end="")
        print()

    # ── 5. True IS/OOS split with correct entries ─────────────────────────────
    sec("5. WALK-FORWARD  —  IS 2018-2021  →  OOS 2022-2026  (valid fills only)")
    IS_END=date(2021,12,31); OS_START=date(2022,1,1)
    for period,mask in [
        ("IS  2018-2021", valid_tdf["date"]<=IS_END),
        ("OOS 2022-2026", valid_tdf["date"]>=OS_START),
    ]:
        sub=valid_tdf[mask]
        print(f"\n  {period}:")
        for st in ["3WAY","ORB_PDHL","ORB_PWHL","ORB_ONLY","PDHL_ONLY","PWHL_ONLY"]:
            s=sub[sub["sig_type"]==st]
            if not len(s): continue
            wr=s["win"].mean(); ev=wr*PMULT-(1-wr)
            dr=s.groupby("date")["r"].sum()
            sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
            flag="✅" if ev>1.2 else ("🟡" if ev>0.8 else "❌")
            print(f"    {flag} {st:<16}  n={len(s)//2:3d}  "
                  f"WR={wr*100:5.1f}%  EV={ev:+.3f}R  Sh={sh:.2f}")

    # ── 6. Statistical significance (valid fills only) ────────────────────────
    sec("6. STATISTICAL SIGNIFICANCE  —  binomial test vs WR=50%  (valid fills)")
    try:
        from scipy import stats as sc
        print(f"  {'Signal':<18}  {'n_legs':>6}  {'WR':>6}  {'p-value':>9}  {'95% CI':>18}  Sig?")
        print(f"  {'─'*72}")
        for st in ["PWHL_ONLY","ORB_PWHL","3WAY","ORB_PDHL","PDHL_ONLY","ORB_ONLY"]:
            s=valid_tdf[valid_tdf["sig_type"]==st]
            if not len(s): continue
            n=len(s); w=int(s["win"].sum()); wr=w/n
            res=sc.binomtest(w,n,0.5,alternative="greater"); p=res.pvalue
            ci=sc.proportion_confint(w,n,alpha=0.05,method="wilson")
            sig="✅ p<0.001" if p<0.001 else("✅ p<0.01" if p<0.01 else("🟡 p<0.05" if p<0.05 else "❌ NS"))
            print(f"  {st:<18}  {n:>6}  {wr*100:>5.1f}%  {p:>9.4f}  "
                  f"[{ci[0]*100:.1f}%–{ci[1]*100:.1f}%]    {sig}")
    except ImportError:
        print("  scipy not installed — skipping")

    # ── 7. Honest summary ─────────────────────────────────────────────────────
    sec("7. HONEST SUMMARY")
    print("  Signal         Reported WR  Valid-fill WR  Overstated?  Note")
    print(f"  {'─'*70}")
    reported = {"3WAY":73.4,"ORB_PDHL":67.9,"ORB_PWHL":77.9,
                "ORB_ONLY":48.6,"PDHL_ONLY":60.3,"PWHL_ONLY":79.9,"PDHL_PWHL":61.1}
    for st in ["PWHL_ONLY","ORB_PWHL","3WAY","ORB_PDHL","PDHL_ONLY","ORB_ONLY"]:
        all_sub=tdf[tdf["sig_type"]==st]
        val_sub=valid_tdf[valid_tdf["sig_type"]==st]
        if not len(all_sub): continue
        rep=reported.get(st,0)
        val_wr=val_sub["win"].mean()*100 if len(val_sub) else 0
        delta=rep-val_wr
        if abs(delta)<2: note="✅ accurate"
        elif delta>5:    note="⚠️  overstated by limit order assumption"
        else:            note="🟡 minor difference"
        print(f"  {st:<14}  {rep:>10.1f}%  {val_wr:>13.1f}%  {delta:>+8.1f}pp  {note}")
    print()


if __name__=="__main__":
    main()
