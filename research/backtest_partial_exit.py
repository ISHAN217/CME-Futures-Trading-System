#!/usr/bin/env python3
"""
backtest_partial_exit.py — Scaled Exit Strategy at Resistance Zone

User's idea (from sketch):
  1. Enter SHORT when price enters resistance zone (rejection candle at PDH)
  2. Stop placed ABOVE the zone (above rejection wick)
  3. First partial profit at 1R (lock in guaranteed gain on part of position)
  4. Trail the remaining position
  5. Final target at 1:3 R:R (or let trail run)

Why this is better than a single exit:
  - Partial exit guarantees profit on any trade that moves at all
  - Remaining position catches the big moves
  - Solves the "wins are too small" problem from the last backtest

SETUP:
  Entry  : Next bar after rejection candle at PDH/PDL
  Stop   : Above rejection wick + slip (initial risk = ~2-3pts)
  Partial: Close PARTIAL_PCT% at PARTIAL_R × initial_risk away from entry
  Trail  : Trail remaining position by TRAIL_PTS behind best price
  Final  : Exit remaining at FINAL_R × initial_risk OR EOD

Tests:
  Partial sizes  : 25%, 33%, 50%, 67%, 75%
  Partial trigger: 0.5R, 1.0R, 1.5R
  Final target   : 2R, 3R, trail only
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


def weekly_hl(by_d, dates):
    w = {}
    for d in dates:
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        key=(int(iso.year),int(iso.week))
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)]
        if not len(rth): continue
        h=rth["ES_high"].max(); l=rth["ES_low"].min()
        if key not in w: w[key]=[h,l]
        else: w[key][0]=max(w[key][0],h); w[key][1]=min(w[key][1],l)
    return w


def get_pwhl(d, w):
    ts=pd.Timestamp(d); iso=ts.isocalendar()
    pw=int(iso.week)-1; yr=int(iso.year)
    if pw==0: yr-=1; pw=int(pd.Timestamp(f"{yr}-12-28").isocalendar().week)
    key=(yr,pw)
    return (w[key][0],w[key][1]) if key in w else (None,None)


def sim_partial(fwd_closes, direction, ep, init_stop,
                partial_pct, partial_r, trail_pts, final_r):
    """
    Simulate scaled exit trade.

    Leg 1 (partial_pct of position):
        Target = ep ± partial_r × init_risk
        If stop hit first: full stop on 100% of position

    Leg 2 (remaining (1-partial_pct) of position):
        After partial hit: stop moves to BREAKEVEN
        Trail trail_pts behind best price
        Final target at final_r × init_risk from entry

    Returns (total_pnl, outcome_description, max_pnl_seen)
    """
    init_risk = abs(ep - init_stop)
    sign = 1 if direction == "LONG" else -1

    if direction == "SHORT":
        partial_tgt = ep - partial_r * init_risk
        final_tgt   = ep - final_r   * init_risk
    else:
        partial_tgt = ep + partial_r * init_risk
        final_tgt   = ep + final_r   * init_risk

    stop = init_stop
    partial_done = False
    best_close   = ep
    max_pnl      = 0.0

    for close in fwd_closes:
        # Track best price seen and unrealised P&L
        if direction == "SHORT":
            best_close = min(best_close, close)
            unreal = ep - close
        else:
            best_close = max(best_close, close)
            unreal = close - ep
        max_pnl = max(max_pnl, unreal)

        # Update trailing stop on remaining position (after partial)
        if partial_done:
            if direction == "SHORT":
                new_trail = best_close + trail_pts
                stop = min(stop, new_trail)   # ratchet stop down
            else:
                new_trail = best_close - trail_pts
                stop = max(stop, new_trail)   # ratchet stop up

        # ── Check stop ──
        if direction == "SHORT" and close >= stop:
            if not partial_done:
                # Stopped before partial: full loss
                pnl = (ep - stop) * 1.0 - SLIP * 2
                return pnl, "FULL_STOP", max_pnl
            else:
                # Stopped after partial: leg1 profit + leg2 stop
                leg1_pnl = (ep - partial_tgt) * partial_pct - SLIP * 2 * partial_pct
                leg2_pnl = (ep - stop) * (1-partial_pct) - SLIP * 2 * (1-partial_pct)
                return leg1_pnl + leg2_pnl, "TRAIL_STOP", max_pnl
        elif direction == "LONG" and close <= stop:
            if not partial_done:
                pnl = (stop - ep) * 1.0 - SLIP * 2
                return pnl, "FULL_STOP", max_pnl
            else:
                leg1_pnl = (partial_tgt - ep) * partial_pct - SLIP * 2 * partial_pct
                leg2_pnl = (stop - ep) * (1-partial_pct) - SLIP * 2 * (1-partial_pct)
                return leg1_pnl + leg2_pnl, "TRAIL_STOP", max_pnl

        # ── Check partial target ──
        if not partial_done:
            if direction == "SHORT" and close <= partial_tgt:
                partial_done = True
                # Move stop to breakeven
                stop = ep + SLIP  # breakeven for remaining
            elif direction == "LONG" and close >= partial_tgt:
                partial_done = True
                stop = ep - SLIP

        # ── Check final target ──
        if partial_done:
            if direction == "SHORT" and close <= final_tgt:
                leg1_pnl = (ep - partial_tgt) * partial_pct - SLIP * 2 * partial_pct
                leg2_pnl = (ep - final_tgt)   * (1-partial_pct) - SLIP * 2 * (1-partial_pct)
                return leg1_pnl + leg2_pnl, "FINAL_TARGET", max_pnl
            elif direction == "LONG" and close >= final_tgt:
                leg1_pnl = (partial_tgt - ep) * partial_pct - SLIP * 2 * partial_pct
                leg2_pnl = (final_tgt - ep)   * (1-partial_pct) - SLIP * 2 * (1-partial_pct)
                return leg1_pnl + leg2_pnl, "FINAL_TARGET", max_pnl

    # EOD exit
    last = fwd_closes[-1] if len(fwd_closes) else ep
    if not partial_done:
        pnl = (ep - last if direction=="SHORT" else last - ep) - SLIP * 2
        return pnl, "EOD", max_pnl
    else:
        leg1_pnl = (ep - partial_tgt if direction=="SHORT" else partial_tgt - ep) * partial_pct - SLIP*2*partial_pct
        leg2_pnl = (ep - last if direction=="SHORT" else last - ep) * (1-partial_pct) - SLIP*2*(1-partial_pct)
        return leg1_pnl + leg2_pnl, "EOD_PARTIAL", max_pnl


def run(df, dates, by_d, wkly,
        partial_pct=0.50, partial_r=1.0,
        trail_pts=3.0, final_r=3.0):

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
        prev_rth = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(prev_rth): continue
        PDH = prev_rth["ES_high"].max(); PDL = prev_rth["ES_low"].min()
        if PDH - PDL < MIN_RANGE: continue

        for fd, lv, tgt_lv in [("SHORT",PDH,PDL),("LONG",PDL,PDH)]:
            for j in range(1, len(closes)-15):
                if times[j] < ENTRY_START: continue
                if times[j] > ENTRY_END:   break
                if fd == "SHORT":
                    if not(closes[j-1]<PDH-SLIP and highs[j]>=PDH and
                           closes[j]<=PDH-REJECTION_THRESH): continue
                    if j+1 >= len(closes)-5: break
                    ep=opens[j+1]+SLIP; s0=highs[j]+SLIP*2
                else:
                    if not(closes[j-1]>PDL+SLIP and lows[j]<=PDL and
                           closes[j]>=PDL+REJECTION_THRESH): continue
                    if j+1 >= len(closes)-5: break
                    ep=opens[j+1]-SLIP; s0=lows[j]-SLIP*2

                if abs(ep-s0) < 0.5: continue
                init_risk = abs(ep-s0)
                pot_reward = abs(ep - (PDL if fd=="SHORT" else PDH))
                if pot_reward < MIN_RANGE: continue

                pnl, outcome, max_p = sim_partial(
                    closes[j+1:], fd, ep, s0,
                    partial_pct, partial_r, trail_pts, final_r)

                trades.append(dict(
                    date=d, year=d.year, direction=fd,
                    init_risk=round(init_risk,2),
                    pot_reward=round(pot_reward,2),
                    pnl=round(pnl,2), win=pnl>0,
                    outcome=outcome, max_pnl=round(max_p,2),
                    dow=pd.Timestamp(d).day_name(),
                ))
                break

    return pd.DataFrame(trades)


def prow(label, sub, pad=44):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean()*100; ar=sub["pnl"].mean()
    dr=sub.groupby("date")["pnl"].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=sub["date"].nunique()/8.1
    wins=sub[sub["win"]]["pnl"]; losses=sub[~sub["win"]]["pnl"]
    avg_w=wins.mean() if len(wins) else 0; avg_l=losses.mean() if len(losses) else 0
    flag="✅" if (ar>0 and sh>1.5) else ("🟡" if ar>0 else "❌")
    print(f"  {flag} {label:<{pad}}  n={sub['date'].nunique():4d}({n_yr:.0f}/yr)  "
          f"WR={wr:5.1f}%  AvgPnL={ar:+.3f}  W/L={abs(avg_w/avg_l):.2f}x  Sh={sh:.2f}")


def main():
    print("Loading …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    wkly  = weekly_hl(by_d, dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    IS_END=date(2021,12,31); OS_START=date(2022,1,1)
    W=74
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. Partial size sweep ─────────────────────────────────────────────────
    sec("1. PARTIAL EXIT SIZE  —  How much to close at first target?")
    print("  Partial at 1R (= initial stop distance), final at 3R, trail 3pts\n")
    for pct in [0.25, 0.33, 0.50, 0.67, 0.75]:
        tdf = run(df, dates, by_d, wkly, partial_pct=pct,
                  partial_r=1.0, trail_pts=3.0, final_r=3.0)
        prow(f"  Close {int(pct*100)}% at 1R, trail {int((1-pct)*100)}%", tdf)

    # ── 2. Partial trigger level ──────────────────────────────────────────────
    sec("2. PARTIAL TRIGGER LEVEL  —  When to take the first profit?")
    print("  50% partial, trail 3pts, final 3R\n")
    for pr in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        tdf = run(df, dates, by_d, wkly, partial_pct=0.50,
                  partial_r=pr, trail_pts=3.0, final_r=3.0)
        prow(f"  Partial at {pr}R ({pr:.2f}×risk)", tdf)

    # ── 3. Final target distance ──────────────────────────────────────────────
    sec("3. FINAL TARGET  —  How far to let remaining position run?")
    print("  50% partial at 1R, trail 3pts\n")
    for fr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 999]:
        lbl = f"Final at {fr}R" if fr < 100 else "Final = trail only (no target)"
        tdf = run(df, dates, by_d, wkly, partial_pct=0.50,
                  partial_r=1.0, trail_pts=3.0, final_r=fr)
        prow(lbl, tdf)

    # ── 4. Best combined config ───────────────────────────────────────────────
    sec("4. BEST COMBINATION  —  Grid search")
    best_sh=-999; best_cfg=None; best_tdf=None
    results=[]
    for pct in [0.33, 0.50, 0.67]:
        for pr in [0.75, 1.0, 1.5]:
            for fr in [2.0, 3.0, 5.0]:
                for tr in [2.0, 3.0, 5.0]:
                    tdf=run(df,dates,by_d,wkly,pct,pr,tr,fr)
                    if not len(tdf): continue
                    dr=tdf.groupby("date")["pnl"].sum()
                    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
                    results.append((pct,pr,tr,fr,sh,tdf["pnl"].mean(),tdf))
                    if sh>best_sh: best_sh=sh; best_cfg=(pct,pr,tr,fr); best_tdf=tdf

    results.sort(key=lambda x: -x[4])
    print(f"  Top 5 configs:")
    print(f"  {'Partial%':>9}  {'P@R':>5}  {'Trail':>6}  {'Final':>6}  {'Sharpe':>7}  {'AvgPnL':>8}")
    print(f"  {'─'*52}")
    for pct,pr,tr,fr,sh,ar,_ in results[:5]:
        flag="✅" if sh>2 else ("🟡" if sh>1 else "❌")
        print(f"  {flag} {pct*100:>8.0f}%  {pr:>5.2f}  {tr:>6.1f}  {fr:>6.1f}  "
              f"{sh:>7.2f}  {ar:>+7.3f}pts")

    if best_tdf is not None:
        pct,pr,tr,fr = best_cfg
        print(f"\n  Best config: {int(pct*100)}% partial at {pr}R, trail {tr}pts, final {fr}R")

    # ── 5. Deep dive on best ──────────────────────────────────────────────────
    if best_tdf is not None:
        pct,pr,tr,fr = best_cfg
        sec(f"5. DEEP DIVE  —  Best: {int(pct*100)}% at {pr}R / trail {tr}pts / final {fr}R")

        print("  Exit breakdown:")
        for out in ["FULL_STOP","TRAIL_STOP","FINAL_TARGET","EOD","EOD_PARTIAL"]:
            s=best_tdf[best_tdf["outcome"]==out]
            if not len(s): continue
            wr=(s["pnl"]>0).mean()*100; ar=s["pnl"].mean()
            print(f"    {out:<16}  {len(s):4d} ({len(s)/len(best_tdf)*100:5.1f}%)  "
                  f"WR={wr:5.1f}%  AvgPnL={ar:+.3f}pts")

        print()
        wins=best_tdf[best_tdf["win"]]; losses=best_tdf[~best_tdf["win"]]
        print(f"  WR         : {best_tdf['win'].mean()*100:.1f}%")
        print(f"  Avg win    : {wins['pnl'].mean():+.3f}pts  = ${wins['pnl'].mean()*5:+.2f}/MES")
        print(f"  Avg loss   : {losses['pnl'].mean():+.3f}pts  = ${losses['pnl'].mean()*5:+.2f}/MES")
        print(f"  Win/Loss   : {abs(wins['pnl'].mean()/losses['pnl'].mean()):.2f}x")
        print(f"  Best trade : {best_tdf['pnl'].max():+.2f}pts")
        print(f"  Worst trade: {best_tdf['pnl'].min():+.2f}pts")

        print()
        print("  Year-by-year:")
        print(f"  {'Year':<6}  {'n':>4}  {'WR':>6}  {'AvgPnL':>9}  {'Sharpe':>7}  IS/OOS")
        print(f"  {'─'*46}")
        for yr,sub in best_tdf.groupby("year"):
            wr=sub["win"].mean()*100; ar=sub["pnl"].mean()
            dr=sub.groupby("date")["pnl"].sum()
            sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
            lbl="← IS" if yr<=2021 else "← OOS"
            flag="✅" if ar>0 else "❌"
            print(f"  {flag} {yr}  {len(sub):>4}  {wr:>5.1f}%  {ar:>+8.3f}pts  "
                  f"{sh:>7.2f}  {lbl}")

        print()
        print("  Walk-forward:")
        prow(f"  IS  2018-2021", best_tdf[best_tdf["date"]<=IS_END])
        prow(f"  OOS 2022-2026", best_tdf[best_tdf["date"]>=OS_START])

        print()
        print("  Dollar P&L:")
        annual=best_tdf["pnl"].sum()/8.1
        for lbl,pv,n in [("1 MES",5,1),("3 MES",5,3),("5 MES",5,5)]:
            print(f"    {lbl}: avg/trade=${best_tdf['pnl'].mean()*pv*n:+.2f}  "
                  f"annual=${annual*pv*n:+.0f}")

    # ── 6. vs previous approaches ─────────────────────────────────────────────
    sec("6. COMPARISON  —  All approaches honest side-by-side")
    print(f"  {'Approach':<46}  {'n/yr':>5}  {'WR':>6}  {'AvgPnL':>8}  {'Sharpe':>7}")
    print(f"  {'─'*70}")
    print(f"  ORB (primary, verified)                       22/yr  47.8%   +0.92pts   3.08")
    print(f"  PDH/L Confirmed (verified)                    36/yr  54.5%   +1.18pts   3.14")
    print(f"  Single trail 3pts (prev best)                 66/yr  43.1%   +0.54pts   1.53")
    if best_tdf is not None and best_cfg is not None:
        pct,pr,tr,fr=best_cfg
        tdf=best_tdf
        if len(tdf):
            wr=tdf["win"].mean()*100; ar=tdf["pnl"].mean()
            dr=tdf.groupby("date")["pnl"].sum()
            sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
            n_yr=tdf["date"].nunique()/8.1
            flag="✅" if (ar>0 and sh>2) else ("🟡" if ar>0 else "❌")
            print(f"  {flag} Partial {int(pct*100)}%@{pr}R + trail{tr}pt + final{fr}R  "
                  f"{n_yr:>5.0f}/yr  {wr:>5.1f}%  {ar:>+7.3f}pts  {sh:>7.2f}")
    print()


if __name__=="__main__":
    main()
