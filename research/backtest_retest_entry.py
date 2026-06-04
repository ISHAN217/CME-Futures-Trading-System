#!/usr/bin/env python3
"""
backtest_retest_entry.py — Breakout + Retest Entry

Current system: Enter at close of breakout bar (chasing the move)
New idea: Wait for price to pull back to OR level (which becomes support)
          Enter on the retest → tighter stop, better R:R

LONG setup:
  1. Both ES+NQ break OR high (ORB fires LONG)
  2. Price is now ABOVE OR high
  3. Price retraces back to within RETEST_ZONE pts of OR high
  4. Enter LONG at the retest bar
  5. Stop: OR high − STOP_BELOW pts  (tight — just below support)
  6. Target: PMULT × (entry − stop) above entry

SHORT setup: mirror image with OR low

Why this might be better:
  - Stop is much tighter (1-3pts vs full OR range)
  - Entry is at a structural level (OR high = new support)
  - Better R:R even with same target multiple
  - Avoids chasing overextended bars

Trade-off:
  - Misses trades that run without pulling back
  - Needs patience after breakout fires
  - False retests (comes back through level) = stop

Tests:
  - Retest zone: ±1pt, ±2pt, ±3pt from OR level
  - Stop distance: 0.5pt, 1pt, 2pt below OR level
  - Retest window: 10, 20, 30 bars after breakout
  - With and without PDH/L confirmation
  - Comparison: current entry vs retest entry
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP=0.25; RTH_S=time(9,30); RTH_E=time(16,0)
OR_S=time(9,30); OR_E=time(10,0); ORB_C=time(10,30)
EOD=time(15,30); MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175


def load():
    df=pd.read_parquet(DATA)
    df["ts_et"]=pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"]=df["ts_et"].dt.date; df["time_et"]=df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def sim(day, entry_t, direction, ep, sp, tp, col_h, col_l, col_c):
    rng=abs(ep-sp)
    if rng < 0.01: return 0.0, "NONE"
    for _,bar in day[day["time_et"]>entry_t].iterrows():
        t=bar["time_et"]
        if t>=EOD:
            sign=1 if direction=="LONG" else -1
            return round((sign*(bar[col_c]-ep)-SLIP*2)/rng,3), "EOD"
        if direction=="LONG":
            if bar[col_l]<=sp: return round(-1.0-SLIP*2/rng,3), "STOP"
            if bar[col_h]>=tp: return round(3.0-SLIP/rng,3), "TGT"
        else:
            if bar[col_h]>=sp: return round(-1.0-SLIP*2/rng,3), "STOP"
            if bar[col_l]<=tp: return round(3.0-SLIP/rng,3), "TGT"
    return 0.0, "NONE"


def run_retest(df, dates, by_d,
               retest_zone=2.0,    # enter when price within X pts of OR level
               stop_below=1.5,     # stop X pts beyond OR level
               pmult=3.0,          # target multiple
               retest_window=30,   # max bars to wait for retest
               require_pdhl=True,  # require PDH/L confirmation
               max_break_pct=0.20, # max break distance filter
               max_pdh_R=2.0):     # max PDH distance filter

    trades = []

    for i, d in enumerate(dates[1:], 1):
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_S)&(day["time_et"]<=EOD)].reset_index(drop=True)
        if len(rth)<20: continue
        or_b=rth[(rth["time_et"]>=OR_S)&(rth["time_et"]<OR_E)]
        if not len(or_b): continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_or=day[(day["time_et"]>=OR_S)&(day["time_et"]<OR_E)]
        nq_H=nq_or["NQ_high"].max(); nq_L=nq_or["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        prev=by_d.get(dates[i-1])
        if prev is None: continue
        pr=prev[(prev["time_et"]>=RTH_S)&(prev["time_et"]<=RTH_E)]
        if not len(pr): continue
        PDH=pr["ES_high"].max(); PDL=pr["ES_low"].min()

        # ORB detection
        post=rth[(rth["time_et"]>=OR_E)&(rth["time_et"]<ORB_C)]
        nq_post=day[(day["time_et"]>=OR_E)&(day["time_et"]<ORB_C)]
        es_b=nq_b=None; break_row=None; break_dist_es=break_dist_nq=0
        for _,r in post.iterrows():
            if r["time_et"]<time(10,2): continue
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"; break_dist_es=(r["ES_high"]-es_H)/es_R
                elif r["ES_low"]<es_L: es_b="S"; break_dist_es=(es_L-r["ES_low"])/es_R
            nqr=nq_post[nq_post["time_et"]==r["time_et"]]
            if nq_b is None and len(nqr):
                nr=nqr.iloc[0]
                if nr["NQ_high"]>nq_H: nq_b="L"; break_dist_nq=(nr["NQ_high"]-nq_H)/nq_R
                elif nr["NQ_low"]<nq_L: nq_b="S"; break_dist_nq=(nq_L-nr["NQ_low"])/nq_R
            if es_b and nq_b: break_row=r; break
        if not(es_b and nq_b and es_b==nq_b): continue
        if break_dist_es>max_break_pct or break_dist_nq>max_break_pct: continue

        direction="LONG" if es_b=="L" else "SHORT"

        # PDH/L confirmation check
        if require_pdhl:
            watch=rth[(rth["time_et"]>=OR_E)&(rth["time_et"]<time(11,0))]
            pdh_dir=None; eu=ed=False
            for _,r in watch.iterrows():
                if not eu and r["ES_high"]>PDH: eu=True
                if not ed and r["ES_low"]<PDL: ed=True
                if eu and not ed and pdh_dir is None: pdh_dir="LONG"; break
                if ed and not eu and pdh_dir is None: pdh_dir="SHORT"; break
            if pdh_dir!=direction: continue

        # PDH proximity filter
        if require_pdhl:
            ep_initial=break_row["ES_close"]
            pdh_dist=abs(PDH-ep_initial)/es_R if direction=="LONG" else abs(PDL-ep_initial)/es_R
            if pdh_dist>max_pdh_R: continue

        # ── RETEST DETECTION ───────────────────────────────────────────────────
        # After breakout, watch for price to pull back to OR level
        after_break = rth[rth["time_et"]>break_row["time_et"]].reset_index(drop=True)

        retest_bar = None
        for j in range(min(retest_window, len(after_break))):
            bar = after_break.iloc[j]
            if bar["time_et"] >= time(11,30): break   # stop waiting after 11:30

            if direction == "LONG":
                # Price should be above OR high, then pull back to within retest_zone
                # Retest = bar LOW touches OR high ± retest_zone
                if bar["ES_low"] <= es_H + retest_zone:
                    # Price touched the retest zone
                    retest_bar = bar
                    break
            else:
                # SHORT: price should be below OR low, pull back up to within retest_zone
                if bar["ES_high"] >= es_L - retest_zone:
                    retest_bar = bar
                    break

        if retest_bar is None: continue   # no retest within window

        # Entry: close of retest bar (where price touched the level again)
        retest_t = retest_bar["time_et"]
        if direction == "LONG":
            ep = retest_bar["ES_close"] + SLIP   # buy as price touches OR high from above
            sp = es_H - stop_below - SLIP         # stop just below OR high
            tp = ep + pmult * (ep - sp)           # target = pmult × risk
        else:
            ep = retest_bar["ES_close"] - SLIP
            sp = es_L + stop_below + SLIP
            tp = ep - pmult * (ep - sp)

        if abs(ep-sp) < 0.25: continue   # degenerate

        # Check fill validity: entry price must be achievable on the retest bar
        fill_ok = (retest_bar["ES_low"] <= ep <= retest_bar["ES_high"])

        # Simulate
        rv, why = sim(day, retest_t, direction, ep, sp, tp,
                      "ES_high", "ES_low", "ES_close")

        trades.append(dict(
            date=d, year=d.year, direction=direction,
            ep=round(ep,2), sp=round(sp,2), tp=round(tp,2),
            stop_dist=round(abs(ep-sp),2),
            r=rv, win=rv>0, exit=why,
            fill_ok=fill_ok,
            or_R=es_R, dow=pd.Timestamp(d).day_name(),
        ))

    return pd.DataFrame(trades)


def run_baseline(df, dates, by_d, max_break_pct=0.20, max_pdh_R=2.0, pmult=3.0):
    """Current system: enter at close of breakout bar."""
    trades=[]
    for i,d in enumerate(dates[1:],1):
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_S)&(day["time_et"]<=EOD)].reset_index(drop=True)
        if len(rth)<20: continue
        or_b=rth[(rth["time_et"]>=OR_S)&(rth["time_et"]<OR_E)]
        if not len(or_b): continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_or=day[(day["time_et"]>=OR_S)&(day["time_et"]<OR_E)]
        nq_H=nq_or["NQ_high"].max(); nq_L=nq_or["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue
        prev=by_d.get(dates[i-1])
        if prev is None: continue
        pr=prev[(prev["time_et"]>=RTH_S)&(prev["time_et"]<=RTH_E)]
        if not len(pr): continue
        PDH=pr["ES_high"].max(); PDL=pr["ES_low"].min()
        post=rth[(rth["time_et"]>=OR_E)&(rth["time_et"]<ORB_C)]
        nq_post=day[(day["time_et"]>=OR_E)&(day["time_et"]<ORB_C)]
        es_b=nq_b=None; entry_row=None; bd_es=bd_nq=0
        for _,r in post.iterrows():
            if r["time_et"]<time(10,2): continue
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"; bd_es=(r["ES_high"]-es_H)/es_R
                elif r["ES_low"]<es_L: es_b="S"; bd_es=(es_L-r["ES_low"])/es_R
            nqr=nq_post[nq_post["time_et"]==r["time_et"]]
            if nq_b is None and len(nqr):
                nr=nqr.iloc[0]
                if nr["NQ_high"]>nq_H: nq_b="L"; bd_nq=(nr["NQ_high"]-nq_H)/nq_R
                elif nr["NQ_low"]<nq_L: nq_b="S"; bd_nq=(nq_L-nr["NQ_low"])/nq_R
            if es_b and nq_b: entry_row=r; break
        if not(es_b and nq_b and es_b==nq_b): continue
        if bd_es>max_break_pct or bd_nq>max_break_pct: continue
        direction="LONG" if es_b=="L" else "SHORT"
        watch=rth[(rth["time_et"]>=OR_E)&(rth["time_et"]<time(11,0))]
        pdh_dir=None; eu=ed=False
        for _,r in watch.iterrows():
            if not eu and r["ES_high"]>PDH: eu=True
            if not ed and r["ES_low"]<PDL: ed=True
            if eu and not ed and pdh_dir is None: pdh_dir="LONG"; break
            if ed and not eu and pdh_dir is None: pdh_dir="SHORT"; break
        if pdh_dir!=direction: continue
        ep_i=entry_row["ES_close"]
        pdh_d=abs(PDH-ep_i)/es_R if direction=="LONG" else abs(PDL-ep_i)/es_R
        if pdh_d>max_pdh_R: continue
        et=entry_row["time_et"]
        ep=ep_i+SLIP if direction=="LONG" else ep_i-SLIP
        sp=es_L-SLIP if direction=="LONG" else es_H+SLIP
        tp=ep+pmult*es_R if direction=="LONG" else ep-pmult*es_R
        rv,why=sim(day,et,direction,ep,sp,tp,"ES_high","ES_low","ES_close")
        trades.append(dict(date=d,year=d.year,direction=direction,r=rv,win=rv>0,
                           exit=why,stop_dist=es_R,or_R=es_R,dow=pd.Timestamp(d).day_name()))
    return pd.DataFrame(trades)


def prow(label, tdf, r_col="r", pad=44):
    if not len(tdf): print(f"  {label:<{pad}}  n=0"); return
    wr=tdf["win"].mean()*100; ar=tdf[r_col].mean()
    dr=tdf.groupby("date")[r_col].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=tdf["date"].nunique()/8.1
    wins=tdf[tdf["win"]]; losses=tdf[~tdf["win"]]
    wl=abs(wins[r_col].mean()/losses[r_col].mean()) if len(losses) and losses[r_col].mean()!=0 else 0
    flag="✅" if(ar>0 and sh>3) else("🟡" if(ar>0 and sh>2) else("🔵" if ar>0 else "❌"))
    print(f"  {flag} {label:<{pad}}  n={n_yr:.0f}/yr  WR={wr:5.1f}%  "
          f"AvgR={ar:+.3f}  W/L={wl:.2f}x  Sh={sh:.2f}")


def main():
    print("Loading …")
    df=load()
    dates=sorted(df["date_et"].unique())
    by_d={d:g.reset_index(drop=True) for d,g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")
    IS_END=date(2021,12,31); OS_START=date(2022,1,1)

    W=74
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── Baseline (current system, best filters) ───────────────────────────────
    sec("BASELINE — Current system (enter at breakout bar close)")
    base=run_baseline(df,dates,by_d)
    prow("Current (breakout bar entry)", base)
    prow("  IS  2018-2021", base[base["date"]<=IS_END])
    prow("  OOS 2022-2026", base[base["date"]>=OS_START])
    print(f"  Avg stop distance : {base['stop_dist'].mean():.1f}pts  (= OR range)")

    # ── Retest entry — parameter sweep ────────────────────────────────────────
    sec("RETEST ENTRY — Wait for price to pull back to OR level")
    print("  After breakout: wait for price to retrace to OR level (now support)")
    print("  Enter on the retest → tighter stop, better R:R\n")

    print("  Retest zone sensitivity (stop=1.5pt, window=30 bars):")
    for zone in [1.0, 2.0, 3.0, 4.0, 5.0]:
        tdf=run_retest(df,dates,by_d,retest_zone=zone,stop_below=1.5,retest_window=30)
        if len(tdf):
            pct_retest=len(tdf)/len(base)*100 if len(base) else 0
            valid=tdf["fill_ok"].mean()*100
            print(f"  zone=±{zone}pt", end="  ")
            prow(f"[{pct_retest:.0f}% of breakouts retest, {valid:.0f}% valid fills]",
                 tdf, pad=48)

    print()
    print("  Stop distance sensitivity (zone=2pt, window=30 bars):")
    for stop in [0.5, 1.0, 1.5, 2.0, 3.0]:
        tdf=run_retest(df,dates,by_d,retest_zone=2.0,stop_below=stop,retest_window=30)
        if len(tdf):
            prow(f"  stop={stop}pt below OR level", tdf)

    print()
    print("  Retest window (zone=2pt, stop=1.5pt):")
    for win in [10, 15, 20, 30, 45]:
        tdf=run_retest(df,dates,by_d,retest_zone=2.0,stop_below=1.5,retest_window=win)
        if len(tdf):
            prow(f"  wait up to {win} bars", tdf)

    # ── Best config deep dive ─────────────────────────────────────────────────
    sec("BEST RETEST CONFIG — Deep dive")
    # Find best by Sharpe
    best_sh=-999; best_tdf=None; best_cfg=""
    for zone in [1.0,2.0,3.0]:
        for stop in [1.0,1.5,2.0]:
            for win in [20,30]:
                tdf=run_retest(df,dates,by_d,retest_zone=zone,stop_below=stop,retest_window=win)
                if not len(tdf): continue
                dr=tdf.groupby("date")["r"].sum()
                sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if dr.std()>0 else 0
                if sh>best_sh:
                    best_sh=sh; best_tdf=tdf
                    best_cfg=f"zone={zone}pt, stop={stop}pt, window={win}bars"

    if best_tdf is not None:
        print(f"  Best config: {best_cfg}\n")
        prow("  Full period",   best_tdf)
        prow("  IS  2018-2021", best_tdf[best_tdf["date"]<=IS_END])
        prow("  OOS 2022-2026", best_tdf[best_tdf["date"]>=OS_START])
        print()
        print(f"  Avg stop distance : {best_tdf['stop_dist'].mean():.2f}pts")
        print(f"  Fill validity     : {best_tdf['fill_ok'].mean()*100:.0f}%")
        print()
        for ex in ["STOP","TGT","EOD"]:
            s=best_tdf[best_tdf["exit"]==ex]
            if not len(s): continue
            print(f"  {ex}: {len(s):3d}({len(s)/len(best_tdf)*100:.1f}%)  "
                  f"WR={s['win'].mean()*100:.1f}%  AvgR={s['r'].mean():+.3f}")
        print()
        print("  Year-by-year:")
        for yr,sub in best_tdf.groupby("year"):
            wr=sub["win"].mean()*100; ar=sub["r"].mean()
            dr=sub.groupby("date")["r"].sum()
            sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
            flag="✅" if ar>0 else "❌"
            print(f"  {flag} {yr}: WR={wr:.1f}%  AvgR={ar:+.3f}  Sh={sh:.2f}"
                  + (" ← IS" if yr<=2021 else " ← OOS"))

    # ── Without PDH/L confirmation ────────────────────────────────────────────
    sec("WITHOUT PDH/L — Retest entry on ORB alone (more trades)")
    print("  No PDH/L confirmation required — just ORB + retest\n")
    tdf_no_pdhl=run_retest(df,dates,by_d,retest_zone=2.0,stop_below=1.5,
                            retest_window=30,require_pdhl=False)
    prow("  No PDH/L, retest zone=2pt", tdf_no_pdhl)
    prow("  IS  2018-2021", tdf_no_pdhl[tdf_no_pdhl["date"]<=IS_END])
    prow("  OOS 2022-2026", tdf_no_pdhl[tdf_no_pdhl["date"]>=OS_START])

    # ── Final comparison ──────────────────────────────────────────────────────
    sec("FINAL COMPARISON")
    print(f"  {'System':<48}  {'n/yr':>5}  {'WR':>6}  {'AvgR':>8}  {'Sh':>6}")
    print(f"  {'─'*72}")
    print(f"  Current (breakout bar, OR-range stop)           "
          f"{len(base)/8.1:>5.0f}/yr  {base['win'].mean()*100:>5.1f}%  "
          f"{base['r'].mean():>+7.3f}R  "
          f"{(lambda d: d.mean()/d.std(ddof=1)*np.sqrt(252) if d.std()>0 else 0)(base.groupby('date')['r'].sum()):>6.2f}")
    if best_tdf is not None:
        dr=best_tdf.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if dr.std()>0 else 0
        wr=best_tdf["win"].mean()*100; ar=best_tdf["r"].mean()
        flag="✅" if sh>3 else "🟡"
        print(f"  {flag} Retest ({best_cfg})  "
              f"{len(best_tdf)/8.1:>5.0f}/yr  {wr:>5.1f}%  {ar:>+7.3f}R  {sh:>6.2f}")
    if len(tdf_no_pdhl):
        dr=tdf_no_pdhl.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if dr.std()>0 else 0
        wr=tdf_no_pdhl["win"].mean()*100; ar=tdf_no_pdhl["r"].mean()
        flag="✅" if sh>3 else "🟡"
        print(f"  {flag} Retest no-PDH/L (zone=2pt, stop=1.5pt)     "
              f"{len(tdf_no_pdhl)/8.1:>5.0f}/yr  {wr:>5.1f}%  {ar:>+7.3f}R  {sh:>6.2f}")
    print()


if __name__=="__main__":
    main()
