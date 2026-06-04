#!/usr/bin/env python3
"""
backtest_partial_v2.py — User's correct design: quick partial + patient hold

EXACT DESIGN (from user sketch + clarification):
  1. SHORT at PDH rejection (or LONG at PDL rejection)
  2. Stop:  above rejection wick (original stop, FIXED — NOT moved on remainder)
  3. Leg 1: close PARTIAL_PCT% at QUICK_TARGET (e.g. 0.75R)  ← quick profit
  4. Leg 2: keep remaining (1-PARTIAL_PCT)% open
             - stop stays at ORIGINAL level (room to breathe through retracements)
             - no tight trailing — give it patience
             - target: 1.5R, 2R, or 2.5R from ENTRY
  5. EOD close at 15:30 ET

Why this is different from what we tested before:
  - Previous: tight 3pt trailing stop — stopped out quickly on retracements
  - This:     original stop held — trade can retrace and still hit 2R
  - The quick partial removes psychological pressure while remainder runs

Tests across:
  - Quick target: 0.5R, 0.75R, 1.0R
  - Partial size: 33%, 50%, 67%
  - Final target: 1.25R, 1.5R, 2.0R, 2.5R
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25; REJECTION_THRESH = 1.0
RTH_START=time(9,30); RTH_END=time(16,0)
ENTRY_START=time(10,30); ENTRY_END=time(14,30)
EOD=time(15,30); MIN_RANGE=10.0


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def sim(fwd_closes, direction, ep, orig_stop, quick_tgt_px, final_tgt_px,
        partial_pct):
    """
    Leg 1 (partial_pct): exits at quick_tgt_px (first small target)
    Leg 2 (1-partial_pct): exits at final_tgt_px OR orig_stop
    Stop on BOTH legs = orig_stop (original, not moved — gives room to breathe)

    Returns (total_pnl_in_R, outcome, leg1_outcome, leg2_outcome)
    where R = abs(ep - orig_stop) = initial risk
    """
    init_risk = abs(ep - orig_stop)
    if init_risk < 0.01: return 0.0, "SKIP", "SKIP", "SKIP"

    leg1_done = False
    leg1_pnl  = 0.0

    for close in fwd_closes:
        # ── Check stop (both legs still open until partial) ──────────────────
        if not leg1_done:
            if direction == "SHORT" and close >= orig_stop:
                full_pnl = (ep - orig_stop - SLIP*2) / init_risk
                return full_pnl, "FULL_STOP", "STOP", "STOP"
            elif direction == "LONG" and close <= orig_stop:
                full_pnl = (orig_stop - ep - SLIP*2) / init_risk
                return full_pnl, "FULL_STOP", "STOP", "STOP"

            # ── Check quick target (leg 1) ───────────────────────────────────
            if direction == "SHORT" and close <= quick_tgt_px:
                leg1_pnl = (ep - quick_tgt_px - SLIP*2) * partial_pct / init_risk
                leg1_done = True
            elif direction == "LONG" and close >= quick_tgt_px:
                leg1_pnl = (quick_tgt_px - ep - SLIP*2) * partial_pct / init_risk
                leg1_done = True

        else:
            # Leg 2 still running — original stop, waiting for final target
            # ── Check stop on leg 2 ──────────────────────────────────────────
            if direction == "SHORT" and close >= orig_stop:
                leg2_pnl = (ep - orig_stop - SLIP*2) * (1-partial_pct) / init_risk
                total = leg1_pnl + leg2_pnl
                return total, "L1_WIN_L2_STOP", "TARGET", "STOP"
            elif direction == "LONG" and close <= orig_stop:
                leg2_pnl = (orig_stop - ep - SLIP*2) * (1-partial_pct) / init_risk
                total = leg1_pnl + leg2_pnl
                return total, "L1_WIN_L2_STOP", "TARGET", "STOP"

            # ── Check final target on leg 2 ──────────────────────────────────
            if direction == "SHORT" and close <= final_tgt_px:
                leg2_pnl = (ep - final_tgt_px - SLIP*2) * (1-partial_pct) / init_risk
                return leg1_pnl + leg2_pnl, "BOTH_HIT", "TARGET", "TARGET"
            elif direction == "LONG" and close >= final_tgt_px:
                leg2_pnl = (final_tgt_px - ep - SLIP*2) * (1-partial_pct) / init_risk
                return leg1_pnl + leg2_pnl, "BOTH_HIT", "TARGET", "TARGET"

    # EOD — exit whatever is open
    last = fwd_closes[-1] if len(fwd_closes) else ep
    if not leg1_done:
        pnl = ((ep-last-SLIP*2) if direction=="SHORT" else (last-ep-SLIP*2)) / init_risk
        return pnl, "EOD_FULL", "EOD", "EOD"
    else:
        leg2_pnl = ((ep-last-SLIP*2) if direction=="SHORT" else (last-ep-SLIP*2)) * (1-partial_pct) / init_risk
        return leg1_pnl + leg2_pnl, "EOD_PARTIAL", "TARGET", "EOD"


def run(df, dates, by_d, quick_r=0.75, partial_pct=0.50, final_r=2.0):
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"]>=RTH_START)&(day["time_et"]<=EOD)].reset_index(drop=True)
        if len(rth) < 20: continue
        opens=rth["ES_open"].values; closes=rth["ES_close"].values
        highs=rth["ES_high"].values; lows=rth["ES_low"].values; times=rth["time_et"].values
        prev = by_d.get(dates[i-1])
        if prev is None: continue
        pr = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(pr): continue
        PDH=pr["ES_high"].max(); PDL=pr["ES_low"].min()
        if PDH-PDL<MIN_RANGE: continue

        for fd, lv in [("SHORT",PDH),("LONG",PDL)]:
            for j in range(1,len(closes)-15):
                if times[j]<ENTRY_START: continue
                if times[j]>ENTRY_END: break
                if fd=="SHORT":
                    if not(closes[j-1]<PDH-SLIP and highs[j]>=PDH and
                           closes[j]<=PDH-REJECTION_THRESH): continue
                    if j+1>=len(closes)-5: break
                    ep=opens[j+1]+SLIP; s0=highs[j]+SLIP*2
                else:
                    if not(closes[j-1]>PDL+SLIP and lows[j]<=PDL and
                           closes[j]>=PDL+REJECTION_THRESH): continue
                    if j+1>=len(closes)-5: break
                    ep=opens[j+1]-SLIP; s0=lows[j]-SLIP*2

                ir=abs(ep-s0)
                if ir<0.5: continue
                pot=abs(ep-(PDL if fd=="SHORT" else PDH))
                if pot<MIN_RANGE: continue

                # Calculate target prices
                if fd=="SHORT":
                    quick_px = ep - quick_r * ir
                    final_px = ep - final_r * ir
                else:
                    quick_px = ep + quick_r * ir
                    final_px = ep + final_r * ir

                pnl_r, outcome, o1, o2 = sim(
                    closes[j+1:], fd, ep, s0,
                    quick_px, final_px, partial_pct)

                trades.append(dict(
                    date=d, year=d.year, direction=fd,
                    init_risk=round(ir,2),
                    pot_reward=round(pot,2),
                    pnl_r=round(pnl_r,3),
                    pnl_pts=round(pnl_r*ir,3),
                    win=pnl_r>0, outcome=outcome,
                    l1=o1, l2=o2,
                    dow=pd.Timestamp(d).day_name(),
                ))
                break

    return pd.DataFrame(trades)


def prow(label, sub, pad=46):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean()*100; ar=sub["pnl_pts"].mean()
    dr=sub.groupby("date")["pnl_pts"].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=sub["date"].nunique()/8.1
    wins=sub[sub["win"]]["pnl_pts"]; losses=sub[~sub["win"]]["pnl_pts"]
    wl=abs(wins.mean()/losses.mean()) if len(losses) and losses.mean()!=0 else 0
    flag="✅" if (ar>0 and sh>1.5) else ("🟡" if ar>0 else "❌")
    print(f"  {flag} {label:<{pad}}  n={sub['date'].nunique():3d}({n_yr:.0f}/yr)  "
          f"WR={wr:5.1f}%  AvgPnL={ar:+.3f}pts  W/L={wl:.2f}x  Sh={sh:.2f}")


def main():
    print("Loading …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    IS_END=date(2021,12,31); OS_START=date(2022,1,1)
    W=74
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. Quick target sensitivity ───────────────────────────────────────────
    sec("1. QUICK TARGET — How early to take the partial? (50% size, final=2R)")
    print("  Original stop held on remainder — room to breathe through retracements\n")
    for qr in [0.25, 0.50, 0.75, 1.00, 1.25]:
        tdf = run(df, dates, by_d, quick_r=qr, partial_pct=0.50, final_r=2.0)
        prow(f"  Quick at {qr}R  final at 2R", tdf)

    # ── 2. Partial size ───────────────────────────────────────────────────────
    sec("2. PARTIAL SIZE — How much to close at quick target? (quick=0.75R, final=2R)")
    for pct in [0.25, 0.33, 0.50, 0.67, 0.75]:
        tdf = run(df, dates, by_d, quick_r=0.75, partial_pct=pct, final_r=2.0)
        prow(f"  Close {int(pct*100)}% at 0.75R", tdf)

    # ── 3. Final target ───────────────────────────────────────────────────────
    sec("3. FINAL TARGET — How patient on the remainder? (quick=0.75R, 50% partial)")
    for fr in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        tdf = run(df, dates, by_d, quick_r=0.75, partial_pct=0.50, final_r=fr)
        prow(f"  Final at {fr}R", tdf)

    # ── 4. Grid search ────────────────────────────────────────────────────────
    sec("4. GRID SEARCH — Best combination")
    best_sh=-999; best_cfg=None; best_tdf=None
    results=[]
    for qr in [0.50, 0.75, 1.0]:
        for pct in [0.33, 0.50, 0.67]:
            for fr in [1.5, 2.0, 2.5, 3.0]:
                tdf=run(df,dates,by_d,quick_r=qr,partial_pct=pct,final_r=fr)
                if not len(tdf): continue
                dr=tdf.groupby("date")["pnl_pts"].sum()
                sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
                ar=tdf["pnl_pts"].mean()
                results.append((qr,pct,fr,sh,ar,tdf))
                if sh>best_sh: best_sh=sh; best_cfg=(qr,pct,fr); best_tdf=tdf

    results.sort(key=lambda x:-x[3])
    print(f"\n  Top 8 configs:")
    print(f"  {'Quick':>6}  {'Partial':>8}  {'Final':>6}  {'Sharpe':>8}  {'AvgPnL':>8}")
    print(f"  {'─'*46}")
    for qr,pct,fr,sh,ar,_ in results[:8]:
        flag="✅" if sh>1.5 else ("🟡" if sh>0.5 else "❌")
        print(f"  {flag} {qr:>5.2f}R  {pct*100:>6.0f}%    {fr:>5.1f}R  "
              f"{sh:>8.2f}  {ar:>+7.3f}pts")

    # ── 5. Deep dive on best ──────────────────────────────────────────────────
    if best_tdf is not None:
        qr,pct,fr = best_cfg
        sec(f"5. DEEP DIVE — Best: quick={qr}R / {int(pct*100)}% partial / final={fr}R")
        print()

        # Outcome breakdown
        print("  What actually happens to each trade:")
        for out in ["FULL_STOP","L1_WIN_L2_STOP","BOTH_HIT","EOD_FULL","EOD_PARTIAL"]:
            s=best_tdf[best_tdf["outcome"]==out]
            if not len(s): continue
            wr=(s["pnl_pts"]>0).mean()*100; ar=s["pnl_pts"].mean()
            pct_t=len(s)/len(best_tdf)*100
            print(f"    {out:<20}  {len(s):4d} ({pct_t:5.1f}%)  "
                  f"WR={wr:5.1f}%  AvgPnL={ar:+.3f}pts")
        print()

        wins=best_tdf[best_tdf["win"]]; losses=best_tdf[~best_tdf["win"]]
        print(f"  Overall WR    : {best_tdf['win'].mean()*100:.1f}%")
        print(f"  Avg win       : {wins['pnl_pts'].mean():+.3f}pts  = ${wins['pnl_pts'].mean()*5:+.2f}/MES")
        print(f"  Avg loss      : {losses['pnl_pts'].mean():+.3f}pts  = ${losses['pnl_pts'].mean()*5:+.2f}/MES")
        print(f"  Win/Loss ratio: {abs(wins['pnl_pts'].mean()/losses['pnl_pts'].mean()):.2f}x")
        print(f"  Best trade    : {best_tdf['pnl_pts'].max():+.2f}pts")
        print(f"  Worst trade   : {best_tdf['pnl_pts'].min():+.2f}pts")

        # Year-by-year
        print()
        print("  Year-by-year:")
        print(f"  {'Year':<6}  {'n':>4}  {'WR':>6}  {'AvgPnL':>9}  {'Sh':>6}  IS/OOS")
        print(f"  {'─'*46}")
        for yr, sub in best_tdf.groupby("year"):
            wr=sub["win"].mean()*100; ar=sub["pnl_pts"].mean()
            dr=sub.groupby("date")["pnl_pts"].sum()
            sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
            lbl="← IS" if yr<=2021 else "← OOS"
            flag="✅" if ar>0 else "❌"
            print(f"  {flag} {yr}  {len(sub):>4}  {wr:>5.1f}%  {ar:>+8.3f}pts  "
                  f"{sh:>6.2f}  {lbl}")

        # Walk-forward
        print()
        prow(f"IS  2018-2021  ({qr}R/{int(pct*100)}%/{fr}R)",
             best_tdf[best_tdf["date"]<=IS_END])
        prow(f"OOS 2022-2026  ({qr}R/{int(pct*100)}%/{fr}R)",
             best_tdf[best_tdf["date"]>=OS_START])

        # Direction
        print()
        prow("SHORT (PDH rejection)", best_tdf[best_tdf["direction"]=="SHORT"])
        prow("LONG  (PDL rejection)", best_tdf[best_tdf["direction"]=="LONG"])

        # Dollar summary
        print()
        annual = best_tdf["pnl_pts"].sum() / 8.1
        print("  Dollar P&L at different sizes:")
        for lbl,pv,n in [("1 MES",5,1),("3 MES",5,3),("5 MES",5,5),("1 ES",50,1)]:
            print(f"    {lbl:8s}: avg/trade=${best_tdf['pnl_pts'].mean()*pv*n:+.2f}  "
                  f"annual=${annual*pv*n:+.0f}  "
                  f"max_risk=${best_tdf['init_risk'].mean()*pv*n:.2f}")

    # ── 6. Final comparison ───────────────────────────────────────────────────
    sec("6. FINAL COMPARISON — All approaches")
    print(f"  {'System':<48}  {'Sharpe':>7}  {'AvgPnL':>8}  {'n/yr':>5}")
    print(f"  {'─'*68}")
    print(f"  ORB (primary, verified)                            3.08   +0.92pts    22")
    print(f"  PDH/L Confirmed (verified)                         3.14   +1.18pts    36")
    print(f"  Single trail 3pts (best previous)                  1.53   +0.54pts    66")
    if best_tdf is not None and best_cfg is not None:
        qr,pct,fr=best_cfg
        dr=best_tdf.groupby("date")["pnl_pts"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        ar=best_tdf["pnl_pts"].mean()
        n_yr=best_tdf["date"].nunique()/8.1
        flag="✅" if sh>1.5 else ("🟡" if sh>0.5 else "❌")
        print(f"  {flag} Quick {qr}R/{int(pct*100)}%partial/final {fr}R              "
              f"{sh:>7.2f}   {ar:>+7.3f}pts  {n_yr:>5.0f}")
    print()


if __name__=="__main__":
    main()
