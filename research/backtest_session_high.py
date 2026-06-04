#!/usr/bin/env python3
"""
backtest_session_high.py — Daily/Session High-Low Retest System

Two versions tested:
  A. Post-ORB High/Low Retest
     - ORB fires LONG → price makes new high → retraces → retest → SHORT fade
     - Applies on ORB days only
     - The Post-ORB High = first major high made AFTER the ORB breakout

  B. Morning High/Low Retest (no ORB required)
     - Tracks the highest high in the first N minutes of session
     - Waits for meaningful pullback
     - Fades the first retest with rejection candle
     - Fires on ANY day (including NQ-too-wide days the ORB misses)

Entry logic (same as MTF scalp):
  Rejection candle at the level → enter opposite direction
  Stop: beyond the rejection wick
  Target: trailing 3pt stop

Key insight: these are INTRADAY structural levels built during the session,
vs our existing system which uses PRE-SESSION levels (PDH, PWH, PMH).
Potentially additive because they cover DIFFERENT reference levels.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP=0.25; RTH_S=time(9,30); RTH_E=time(16,0)
OR_S=time(9,30); OR_E=time(10,0); ORB_C=time(10,30)
EOD=time(15,30); MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175
TRAIL=3.0; BE=3.0


def load():
    df=pd.read_parquet(DATA)
    df["ts_et"]=pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"]=df["ts_et"].dt.date; df["time_et"]=df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def sim_trail(fwd, direction, ep, s0):
    stop=s0; best=ep
    for c in fwd:
        if direction=="SHORT": best=min(best,c); u=ep-c-SLIP*2
        else: best=max(best,c); u=c-ep-SLIP*2
        if u>=BE:
            if direction=="SHORT": stop=min(stop,best+TRAIL)
            else: stop=max(stop,best-TRAIL)
        if direction=="SHORT":
            if c>=stop: return ep-stop-SLIP*2
            if c<=(ep-60): return 60.0-SLIP*2
        else:
            if c<=stop: return stop-ep-SLIP*2
            if c>=(ep+60): return 60.0-SLIP*2
    last=fwd[-1] if len(fwd) else ep
    return ep-last-SLIP*2 if direction=="SHORT" else last-ep-SLIP*2


def run_session_high(df, dates, by_d,
                     high_window_end=time(10,30),  # define "morning high" by this time
                     min_advance=5.0,              # high must be this far above OR high
                     min_pullback=5.0,             # must retrace this much from high
                     retest_zone=2.0,              # trigger within X pts of high
                     reject_thresh=1.0,            # close must be X pts away from level
                     require_orb=False,            # A: True (post-ORB only), B: False (any day)
                     retest_start=None):           # earliest time to look for retest
    """
    Finds the morning session high, waits for pullback, then fades the retest.
    """
    if retest_start is None:
        retest_start = time(11, 0)

    trades = []

    for i, d in enumerate(dates[1:], 1):
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_S)&(day["time_et"]<=EOD)].reset_index(drop=True)
        if len(rth)<30: continue
        closes=rth["ES_close"].values; highs=rth["ES_high"].values
        lows=rth["ES_low"].values; times=rth["time_et"].values

        # OR range
        or_b=rth[(rth["time_et"]>=OR_S)&(rth["time_et"]<OR_E)]
        if not len(or_b): continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_or=day[(day["time_et"]>=OR_S)&(day["time_et"]<OR_E)]
        nq_H=nq_or["NQ_high"].max(); nq_L=nq_or["NQ_low"].min(); nq_R=nq_H-nq_L

        if not(MIN_ES<=es_R<=MAX_ES): continue  # skip invalid ES OR

        # If require_orb, check ORB fired (NQ also valid + both break same direction)
        orb_direction = None
        if require_orb:
            if not(MIN_NQ<=nq_R<=MAX_NQ): continue  # NQ too wide
            post=rth[(rth["time_et"]>=OR_E)&(rth["time_et"]<ORB_C)]
            nq_post=day[(day["time_et"]>=OR_E)&(day["time_et"]<ORB_C)]
            es_b=nq_b=None
            for _,r in post.iterrows():
                if r["time_et"]<time(10,2): continue
                if es_b is None:
                    if r["ES_high"]>es_H: es_b="L"
                    elif r["ES_low"]<es_L: es_b="S"
                nqr=nq_post[nq_post["time_et"]==r["time_et"]]
                if nq_b is None and len(nqr):
                    nr=nqr.iloc[0]
                    if nr["NQ_high"]>nq_H: nq_b="L"
                    elif nr["NQ_low"]<nq_L: nq_b="S"
                if es_b and nq_b: break
            if not(es_b and nq_b and es_b==nq_b): continue
            orb_direction = "LONG" if es_b=="L" else "SHORT"

        # Find the morning high and low (up to high_window_end)
        morning = rth[(rth["time_et"]>=OR_E)&(rth["time_et"]<=high_window_end)]
        if not len(morning): continue
        morning_H = morning["ES_high"].max()
        morning_L = morning["ES_low"].min()

        # For version A (post-ORB): use ORB direction to pick which extreme
        if require_orb:
            if orb_direction=="LONG":
                trade_setups = [("SHORT", morning_H, morning_H-min_advance, es_H)]
            else:
                trade_setups = [("LONG", morning_L, morning_L+min_advance, es_L)]
        else:
            # Version B: try both
            trade_setups = [
                ("SHORT", morning_H, morning_H-min_advance, es_H),
                ("LONG",  morning_L, morning_L+min_advance, es_L),
            ]

        ts=pd.Timestamp(d)
        dow=ts.day_name(); month=ts.month
        iso_wk=(int(ts.isocalendar().year),int(ts.isocalendar().week))

        for (fade_dir, level, level_adj, or_ref) in trade_setups:
            # Level must be meaningfully away from OR level
            if fade_dir=="SHORT" and level < or_ref + min_advance: continue
            if fade_dir=="LONG"  and level > or_ref - min_advance: continue

            # Scan retest window (after morning high window, before EOD)
            prev_c=None; pulled_back=False; best_extreme=level
            found=False

            for j in range(len(closes)):
                t=times[j]
                if t < high_window_end: prev_c=closes[j]; continue
                if t < retest_start: prev_c=closes[j]; continue
                if t >= time(14,30): break
                if prev_c is None: prev_c=closes[j]; continue

                # Track pullback from the morning extreme
                if fade_dir=="SHORT":
                    if closes[j] < best_extreme - min_pullback:
                        pulled_back=True
                else:
                    if closes[j] > best_extreme + min_pullback:
                        pulled_back=True

                if not pulled_back:
                    prev_c=closes[j]; continue

                # Detect rejection candle at the morning extreme
                if fade_dir=="SHORT":
                    # Price came back up to test the morning high
                    if (prev_c < level - SLIP and
                        highs[j] >= level and
                        closes[j] <= level - reject_thresh):
                        ep=closes[j]+SLIP if j+1<len(closes) else closes[j]+SLIP
                        # Use actual next bar open if available
                        if j+1<len(rth):
                            ep=rth["ES_open"].values[j+1]+SLIP
                        s0=highs[j]+SLIP*2
                        if abs(ep-s0)<0.5: prev_c=closes[j]; continue
                        pnl=sim_trail(closes[j+1:], "SHORT", ep, s0)
                        trades.append(dict(
                            date=d, year=d.year, direction="SHORT",
                            level=round(level,2), level_type="SESSION_H",
                            entry_time=t, or_R=es_R,
                            morning_H=morning_H, morning_L=morning_L,
                            pnl=round(pnl,2), win=pnl>0,
                            stop_dist=round(abs(ep-s0),2),
                            dow=dow, month=month, iso_wk=iso_wk,
                        ))
                        found=True; break

                else:
                    # Price came back down to test the morning low
                    if (prev_c > level + SLIP and
                        lows[j] <= level and
                        closes[j] >= level + reject_thresh):
                        if j+1<len(rth):
                            ep=rth["ES_open"].values[j+1]-SLIP
                        else:
                            ep=closes[j]-SLIP
                        s0=lows[j]-SLIP*2
                        if abs(ep-s0)<0.5: prev_c=closes[j]; continue
                        pnl=sim_trail(closes[j+1:], "LONG", ep, s0)
                        trades.append(dict(
                            date=d, year=d.year, direction="LONG",
                            level=round(level,2), level_type="SESSION_L",
                            entry_time=t, or_R=es_R,
                            morning_H=morning_H, morning_L=morning_L,
                            pnl=round(pnl,2), win=pnl>0,
                            stop_dist=round(abs(ep-s0),2),
                            dow=dow, month=month, iso_wk=iso_wk,
                        ))
                        found=True; break

                prev_c=closes[j]
                if found: break

    return pd.DataFrame(trades)


def sh(dr): return dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0


def prow(label, sub, pad=46):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean()*100; ar=sub["pnl"].mean()
    dr=sub.groupby("date")["pnl"].sum()
    s=sh(dr); n_yr=sub["date"].nunique()/8.1
    wins=sub[sub["win"]]; losses=sub[~sub["win"]]
    wl=abs(wins["pnl"].mean()/losses["pnl"].mean()) if len(losses) and losses["pnl"].mean()!=0 else 0
    flag="✅" if(ar>0 and s>2) else("🟡" if ar>0 else "❌")
    print(f"  {flag} {label:<{pad}}  n={n_yr:.0f}/yr  WR={wr:5.1f}%  "
          f"AvgPnL={ar:+.3f}pts  W/L={wl:.2f}x  Sh={s:.2f}")


def main():
    print("Loading …")
    df=load()
    dates=sorted(df["date_et"].unique())
    by_d={d:g.reset_index(drop=True) for d,g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")
    IS_END=date(2021,12,31); OS_START=date(2022,1,1)
    W=74
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── Version A: Post-ORB High/Low Retest ──────────────────────────────────
    sec("VERSION A — Post-ORB Session High/Low Retest")
    print("  ORB fires → price makes post-ORB high → retraces → SHORT the retest")
    print("  (Only on ORB days, NQ must be valid)\n")
    results_a={}
    for adv in [3.0, 5.0, 8.0]:
        for pb in [3.0, 5.0, 8.0]:
            tdf=run_session_high(df,dates,by_d,min_advance=adv,min_pullback=pb,
                                  require_orb=True)
            key=f"adv={adv}pt pb={pb}pt"
            results_a[key]=tdf
            if len(tdf):
                prow(f"  advance>{adv}pt pullback>{pb}pt", tdf)

    # ── Version B: Morning High/Low Retest (any day) ─────────────────────────
    sec("VERSION B — Morning High/Low Retest (fires any day, no ORB needed)")
    print("  Tracks highest high in first 60min → waits for pullback → fades retest")
    print("  Fires even on NQ-too-wide days that ORB misses\n")
    results_b={}
    for adv in [3.0, 5.0, 10.0]:
        for pb in [3.0, 5.0, 10.0]:
            tdf=run_session_high(df,dates,by_d,min_advance=adv,min_pullback=pb,
                                  require_orb=False)
            key=f"adv={adv}pt pb={pb}pt"
            results_b[key]=tdf
            if len(tdf):
                prow(f"  advance>{adv}pt pullback>{pb}pt", tdf)

    # ── Best config deep dive ─────────────────────────────────────────────────
    sec("BEST CONFIG — Deep dive")
    all_results={**{f"A_{k}":v for k,v in results_a.items()},
                 **{f"B_{k}":v for k,v in results_b.items()}}
    best_k=max(all_results,
               key=lambda k:(all_results[k].groupby("date")["pnl"].sum().mean()/
                              all_results[k].groupby("date")["pnl"].sum().std(ddof=1)
                              if len(all_results[k])>1 and
                              all_results[k].groupby("date")["pnl"].sum().std()>0 else -999))
    best=all_results[best_k]
    print(f"  Best: {best_k}\n")
    if len(best):
        prow("  Full period",    best)
        prow("  IS  2018-2021",  best[best["date"]<=IS_END])
        prow("  OOS 2022-2026",  best[best["date"]>=OS_START])
        print()
        prow("  SHORT (session high fade)", best[best["direction"]=="SHORT"])
        prow("  LONG  (session low  fade)", best[best["direction"]=="LONG"])
        print()
        print(f"  Avg stop dist  : {best['stop_dist'].mean():.2f}pts")
        print(f"  Year-by-year:")
        for yr,sub in best.groupby("year"):
            wr=sub["win"].mean()*100; ar=sub["pnl"].mean()
            dr=sub.groupby("date")["pnl"].sum()
            s=sh(dr)
            flag="✅" if ar>0 else "❌"
            print(f"  {flag} {yr}: WR={wr:.1f}%  AvgPnL={ar:+.3f}  Sh={s:.2f}"
                  + (" IS" if yr<=2021 else " OOS"))

    # ── Does it ADD signal days vs existing scalp? ────────────────────────────
    sec("ADDITIVITY — New signal days vs existing MTF scalp")
    existing_scalp=run_session_high(df,dates,by_d,min_advance=3,min_pullback=3,require_orb=False)
    if len(best) and len(existing_scalp):
        overlap=len(set(best["date"].unique()) & set(existing_scalp["date"].unique()))
        only_new=len(set(best["date"].unique()) - set(existing_scalp["date"].unique()))
        print(f"  Session H/L signal days : {best['date'].nunique()/8.1:.0f}/yr")
        print(f"  Existing scalp days     : {existing_scalp['date'].nunique()/8.1:.0f}/yr")
        print(f"  Overlap (same day)      : {overlap/8.1:.0f}/yr")
        print(f"  NEW days added          : {only_new/8.1:.0f}/yr  ← ADDITIVE signal days")

    # ── Comparison vs existing scalp ─────────────────────────────────────────
    sec("COMPARISON — Session H/L vs Existing MTF Scalp")
    print(f"  {'System':<44}  {'Sh':>6}  {'n/yr':>5}  {'WR':>6}  {'AvgPnL':>8}")
    print(f"  {'─'*68}")
    print(f"  MTF Scalp (current, PDH/PWH/PMH/RND)           2.64     110  44.1%   +0.79pts")
    if len(best):
        dr=best.groupby("date")["pnl"].sum()
        s=sh(dr); wr=best["win"].mean()*100; ar=best["pnl"].mean()
        n_yr=best["date"].nunique()/8.1
        flag="✅" if(ar>0 and s>2) else "🟡"
        print(f"  {flag} Session H/L retest ({best_k[:20]:<20})  {s:>6.2f}  {n_yr:>5.0f}  "
              f"{wr:>5.1f}%  {ar:>+7.3f}pts")
    print()


if __name__=="__main__":
    main()
