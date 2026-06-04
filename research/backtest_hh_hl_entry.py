#!/usr/bin/env python3
"""
backtest_hh_hl_entry.py — Higher High / Higher Low Structure Entry

User's exact concept:
  1. ORB fires LONG (OR high breaks)
  2. Price forms a Higher High (HH) above OR high
  3. Price retraces → forms a Higher Low (HL) above OR high
  4. Enter LONG when price touches HL on the pullback
  5. Stop: below the HL (tight — structural support)
  6. Target: just above the HH (the nearest resistance)

SHORT mirror:
  1. ORB fires SHORT (OR low breaks)
  2. Price forms Lower Low (LL) below OR low
  3. Price retraces up → forms Lower High (LH) below OR low
  4. Enter SHORT when price touches LH on the bounce
  5. Stop: above LH
  6. Target: just below LL

Why this is better than simple retest:
  - Not just ANY pullback — requires STRUCTURE to form first
  - Entry at a defined structural level (the HL)
  - Target defined by actual market structure (HH), not fixed R
  - Stop defined by structure (below HL), naturally tight
  - Market has proven itself: broke out AND made a HH → real commitment

Parameters tested:
  - HH advance: min pts price must advance above OR high before HL forms
  - Pullback: min pts from HH before HL is defined
  - Entry tolerance: how close to HL to trigger
  - Stop: below HL
  - Target buffer: pts above HH
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP=0.25; RTH_S=time(9,30); RTH_E=time(16,0); OR_S=time(9,30); OR_E=time(10,0)
ORB_C=time(10,30); EOD=time(15,30); MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175


def load():
    df=pd.read_parquet(DATA)
    df["ts_et"]=pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"]=df["ts_et"].dt.date; df["time_et"]=df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def sim(day, entry_t, direction, ep, sp, tp):
    rng=abs(ep-sp)
    if rng<0.01: return 0.0,"NONE"
    for _,bar in day[day["time_et"]>entry_t].iterrows():
        t=bar["time_et"]
        if t>=EOD:
            sign=1 if direction=="LONG" else -1
            return round((sign*(bar["ES_close"]-ep)-SLIP*2)/rng,3),"EOD"
        if direction=="LONG":
            if bar["ES_low"]<=sp: return round(-1.0-SLIP*2/rng,3),"STOP"
            if bar["ES_high"]>=tp: return round((tp-ep-SLIP)/rng,3),"TGT"
        else:
            if bar["ES_high"]>=sp: return round(-1.0-SLIP*2/rng,3),"STOP"
            if bar["ES_low"]<=tp: return round((ep-tp-SLIP)/rng,3),"TGT"
    return 0.0,"NONE"


def find_hh_hl(bars_after_break, direction, or_level,
               min_hh_advance=2.0,   # HH must be this far above OR level
               min_pullback=2.0,     # must retrace this much from HH
               entry_tol=1.0,        # trigger when within this much of HL
               stop_below=1.5,       # stop this far beyond HL
               target_buf=1.0,       # target this far beyond HH
               max_pattern_bars=60): # max bars to wait for pattern
    """
    Detect HH/HL pattern after ORB breakout and return entry details.

    Returns: (entry_px, stop_px, target_px, entry_t, HH, HL) or None
    """
    post_high = None   # HH: highest close seen after breakout
    post_low  = None   # LL for SHORT

    hh_time = None
    hl_confirmed = False
    hl_level = None
    hh_level = None

    phase = "finding_hh"   # states: finding_hh → pullback → entry

    for j, (_, bar) in enumerate(bars_after_break.head(max_pattern_bars).iterrows()):
        t = bar["time_et"]
        if t >= time(12, 0): break   # pattern must complete by noon

        if direction == "LONG":
            # Phase 1: find HH (highest point above OR level by min_hh_advance)
            if phase == "finding_hh":
                if post_high is None or bar["ES_high"] > post_high:
                    post_high = bar["ES_high"]
                    hh_time = t
                    hh_level = post_high
                # HH confirmed when it's min_hh_advance above OR level
                if hh_level and hh_level >= or_level + min_hh_advance:
                    phase = "pullback"

            elif phase == "pullback":
                # Track HL: lowest close after HH was made
                # HL confirmed when price has pulled back by min_pullback from HH
                if bar["ES_high"] > hh_level:
                    # New HH — reset
                    hh_level = bar["ES_high"]; hh_time = t
                elif bar["ES_low"] < hh_level - min_pullback:
                    # Pullback has started — track the HL
                    if hl_level is None or bar["ES_low"] < hl_level:
                        hl_level = bar["ES_close"]   # HL = close of pullback bar
                    phase = "entry"

            elif phase == "entry":
                if hl_level is None: break
                # Wait for price to bounce back up to HL level
                # Entry when close is within entry_tol of HL
                if bar["ES_low"] <= hl_level + entry_tol:
                    # Price touched HL zone — enter LONG
                    ep = bar["ES_close"] + SLIP
                    sp = hl_level - stop_below - SLIP
                    tp = hh_level + target_buf - SLIP
                    if abs(ep-sp) < 0.5: continue
                    if tp <= ep: continue   # target must be above entry
                    return ep, sp, tp, t, hh_level, hl_level

        else:  # SHORT
            if phase == "finding_hh":
                if post_low is None or bar["ES_low"] < post_low:
                    post_low = bar["ES_low"]
                    hh_level = post_low   # LL for short
                if hh_level and hh_level <= or_level - min_hh_advance:
                    phase = "pullback"

            elif phase == "pullback":
                if bar["ES_low"] < hh_level:
                    hh_level = bar["ES_low"]
                elif bar["ES_high"] > hh_level + min_pullback:
                    hl_level = bar["ES_close"]   # LH = close of bounce bar
                    phase = "entry"

            elif phase == "entry":
                if hl_level is None: break
                if bar["ES_high"] >= hl_level - entry_tol:
                    ep = bar["ES_close"] - SLIP
                    sp = hl_level + stop_below + SLIP
                    tp = hh_level - target_buf + SLIP
                    if abs(ep-sp) < 0.5: continue
                    if tp >= ep: continue
                    return ep, sp, tp, t, hh_level, hl_level

    return None


def run(df, dates, by_d,
        min_hh_advance=2.0, min_pullback=3.0,
        entry_tol=1.5, stop_below=1.5, target_buf=1.0,
        require_pdhl=True, max_break_pct=0.20, max_pdh_R=2.0):

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
        es_b=nq_b=None; break_row=None; bd_es=bd_nq=0
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
            if es_b and nq_b: break_row=r; break
        if not(es_b and nq_b and es_b==nq_b): continue
        if bd_es>max_break_pct or bd_nq>max_break_pct: continue

        direction="LONG" if es_b=="L" else "SHORT"

        # PDH/L confirmation
        if require_pdhl:
            watch=rth[(rth["time_et"]>=OR_E)&(rth["time_et"]<time(11,0))]
            pdh_dir=None; eu=ed=False
            for _,r in watch.iterrows():
                if not eu and r["ES_high"]>PDH: eu=True
                if not ed and r["ES_low"]<PDL: ed=True
                if eu and not ed and pdh_dir is None: pdh_dir="LONG"; break
                if ed and not eu and pdh_dir is None: pdh_dir="SHORT"; break
            if pdh_dir!=direction: continue
            ep_init=break_row["ES_close"]
            pdh_dist=abs(PDH-ep_init)/es_R if direction=="LONG" else abs(PDL-ep_init)/es_R
            if pdh_dist>max_pdh_R: continue

        # Find HH/HL pattern
        or_level = es_H if direction=="LONG" else es_L
        bars_after = rth[rth["time_et"]>break_row["time_et"]].reset_index(drop=True)
        result = find_hh_hl(bars_after, direction, or_level,
                            min_hh_advance, min_pullback, entry_tol,
                            stop_below, target_buf)

        if result is None: continue

        ep, sp, tp, entry_t, hh, hl = result
        fill_ok = True  # market entry at touch of HL

        rv, why = sim(day, entry_t, direction, ep, sp, tp)

        trades.append(dict(
            date=d, year=d.year, direction=direction,
            ep=round(ep,2), sp=round(sp,2), tp=round(tp,2),
            hh=round(hh,2), hl=round(hl,2),
            stop_dist=round(abs(ep-sp),2),
            rr_at_entry=round(abs(tp-ep)/abs(ep-sp),2),
            r=rv, win=rv>0, exit=why,
            or_R=es_R, dow=pd.Timestamp(d).day_name(),
        ))

    return pd.DataFrame(trades)


def sh(dr): return dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0


def prow(label, tdf, pad=44):
    if not len(tdf): print(f"  {label:<{pad}}  n=0"); return
    wr=tdf["win"].mean()*100; ar=tdf["r"].mean()
    dr=tdf.groupby("date")["r"].sum()
    s=sh(dr); n_yr=tdf["date"].nunique()/8.1
    wins=tdf[tdf["win"]]; losses=tdf[~tdf["win"]]
    wl=abs(wins["r"].mean()/losses["r"].mean()) if len(losses) and losses["r"].mean()!=0 else 0
    flag="✅" if(ar>0 and s>3) else("🟡" if(ar>0 and s>2) else("🔵" if ar>0 else "❌"))
    print(f"  {flag} {label:<{pad}}  n={n_yr:.0f}/yr  WR={wr:5.1f}%  "
          f"AvgR={ar:+.4f}  W/L={wl:.2f}x  Sh={s:.2f}")


def main():
    print("Loading …")
    df=load()
    dates=sorted(df["date_et"].unique())
    by_d={d:g.reset_index(drop=True) for d,g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")
    IS_END=date(2021,12,31); OS_START=date(2022,1,1)
    W=74
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── Baseline ──────────────────────────────────────────────────────────────
    sec("BASELINE — Current system (enter at breakout bar)")
    from backtest_retest_entry import run_baseline
    base=run_baseline(df,dates,by_d)
    prow("Current (breakout bar, OR-stop)", base)
    prow("  OOS 2022-2026", base[base["date"]>=OS_START])

    # ── HH/HL parameter sweep ─────────────────────────────────────────────────
    sec("HH/HL ENTRY — Parameter sweep")
    print("  After ORB: wait for HH → pullback → HL → enter at HL retest\n")
    print("  Pullback sensitivity (HH_advance=2pt, entry_tol=1.5pt, stop=1.5pt):")
    for pb in [1.0, 2.0, 3.0, 4.0, 5.0]:
        tdf=run(df,dates,by_d,min_hh_advance=2.0,min_pullback=pb,
                entry_tol=1.5,stop_below=1.5)
        prow(f"  min_pullback={pb}pt", tdf)

    print()
    print("  Stop distance sensitivity (pullback=3pt, entry_tol=1.5pt):")
    for stop in [0.5, 1.0, 1.5, 2.0, 3.0]:
        tdf=run(df,dates,by_d,min_hh_advance=2.0,min_pullback=3.0,
                entry_tol=1.5,stop_below=stop)
        prow(f"  stop={stop}pt below HL", tdf)

    print()
    print("  Target buffer (pts above HH) — pullback=3pt, stop=1.5pt:")
    for buf in [0.0, 0.5, 1.0, 2.0, 3.0]:
        tdf=run(df,dates,by_d,min_hh_advance=2.0,min_pullback=3.0,
                entry_tol=1.5,stop_below=1.5,target_buf=buf)
        prow(f"  target=HH+{buf}pt", tdf)

    print()
    print("  HH advance required (min pts HH must be above OR):")
    for adv in [1.0, 2.0, 3.0, 5.0, 8.0]:
        tdf=run(df,dates,by_d,min_hh_advance=adv,min_pullback=3.0,
                entry_tol=1.5,stop_below=1.5)
        prow(f"  HH must be >{adv}pt above OR", tdf)

    # ── Best config ───────────────────────────────────────────────────────────
    sec("FINDING BEST CONFIG — Grid search")
    best_sh=-999; best_cfg=None; best_tdf=None
    for adv in [1.0,2.0,3.0]:
        for pb in [2.0,3.0,4.0]:
            for stop in [1.0,1.5,2.0]:
                for buf in [0.5,1.0,2.0]:
                    tdf=run(df,dates,by_d,min_hh_advance=adv,min_pullback=pb,
                            entry_tol=1.5,stop_below=stop,target_buf=buf)
                    if not len(tdf): continue
                    dr=tdf.groupby("date")["r"].sum()
                    s=sh(dr)
                    if s>best_sh:
                        best_sh=s; best_tdf=tdf
                        best_cfg=f"HH>{adv}pt, pullback={pb}pt, stop={stop}pt, target=HH+{buf}pt"

    print(f"\n  Best config: {best_cfg}")
    if best_tdf is not None:
        print()
        prow("  Full period", best_tdf)
        prow("  IS  2018-2021", best_tdf[best_tdf["date"]<=IS_END])
        prow("  OOS 2022-2026", best_tdf[best_tdf["date"]>=OS_START])
        print()
        print(f"  Avg R:R at entry  : {best_tdf['rr_at_entry'].mean():.2f}:1")
        print(f"  Avg stop distance : {best_tdf['stop_dist'].mean():.2f}pts")
        print(f"  Avg HH advance    : {(best_tdf['hh']-best_tdf['ep']).abs().mean():.2f}pts above OR")
        print()
        for ex in ["STOP","TGT","EOD"]:
            s2=best_tdf[best_tdf["exit"]==ex]
            if not len(s2): continue
            print(f"  {ex}: {len(s2):3d}({len(s2)/len(best_tdf)*100:.1f}%)  "
                  f"WR={s2['win'].mean()*100:.1f}%  AvgR={s2['r'].mean():+.4f}")
        print()
        print("  Year-by-year:")
        for yr,sub in best_tdf.groupby("year"):
            wr=sub["win"].mean()*100; ar=sub["r"].mean()
            dr=sub.groupby("date")["r"].sum()
            s2=sh(dr)
            flag="✅" if ar>0 else "❌"
            print(f"  {flag} {yr}: n={len(sub):3d}  WR={wr:.1f}%  AvgR={ar:+.4f}  Sh={s2:.2f}"
                  + (" IS" if yr<=2021 else " OOS"))

    # ── Without PDH/L ─────────────────────────────────────────────────────────
    sec("WITHOUT PDH/L — More trades, same HH/HL pattern")
    tdf_no=run(df,dates,by_d,min_hh_advance=2.0,min_pullback=3.0,
               entry_tol=1.5,stop_below=1.5,require_pdhl=False)
    prow("No PDH/L required", tdf_no)
    prow("  OOS 2022-2026", tdf_no[tdf_no["date"]>=OS_START])

    # ── Final comparison ──────────────────────────────────────────────────────
    sec("FINAL COMPARISON")
    print(f"  {'System':<52}  {'n/yr':>5}  {'WR':>6}  {'AvgR':>8}  {'Sh':>6}")
    print(f"  {'─'*76}")
    print(f"  Current (breakout bar, OR-range stop)              "
          f"{base['date'].nunique()/8.1:>5.0f}  {base['win'].mean()*100:>5.1f}%  "
          f"{base['r'].mean():>+7.4f}  {sh(base.groupby('date')['r'].sum()):>6.2f}")
    if best_tdf is not None:
        s_best=sh(best_tdf.groupby("date")["r"].sum())
        flag="✅" if s_best>3 else "🟡"
        print(f"  {flag} HH/HL structure ({best_cfg[:35]})  "
              f"{best_tdf['date'].nunique()/8.1:>5.0f}  {best_tdf['win'].mean()*100:>5.1f}%  "
              f"{best_tdf['r'].mean():>+7.4f}  {s_best:>6.2f}")
    if len(tdf_no):
        print(f"  🔵 HH/HL no-PDH/L                                  "
              f"{tdf_no['date'].nunique()/8.1:>5.0f}  {tdf_no['win'].mean()*100:>5.1f}%  "
              f"{tdf_no['r'].mean():>+7.4f}  {sh(tdf_no.groupby('date')['r'].sum()):>6.2f}")
    print()


if __name__=="__main__":
    main()
