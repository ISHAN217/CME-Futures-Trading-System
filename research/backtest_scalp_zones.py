#!/usr/bin/env python3
"""
backtest_scalp_zones.py — S&R Zone Scalp System

Built on the validated finding that structural levels are respected
68-87% of the time. Harvest that edge with short-duration fade trades.

SCALP MECHANICS:
  Entry  : Close of the bar that FIRST enters the zone from outside.
           Fade the level: SHORT if approaching from below (resistance),
           LONG if approaching from above (support).
  Stop   : First bar that CLOSES beyond the far side of the zone.
           Exit at the far zone boundary + SLIP.
  Target : First bar that CLOSES back outside the zone (entry side).
           Exit at near zone boundary - SLIP.
  Time   : 10-bar time stop (10 minutes). Exit at market if neither hit.

ZONES TRADED (avoids conflict with ORB window 10:00-10:30):
  PDH/PDL  — traded 10:30 onwards (after OR locks, ORB done)
  PWH/PWL  — traded any time during RTH
  OR H/L   — traded 10:30-15:30 (after ORB window)

COMBINED PORTFOLIO:
  Main (ORB + PDH-L Confirmed): 5% per leg
  Scalp overlay: 1% per touch (small, quick, frequent)

Shows standalone scalp stats AND combined portfolio with ORB.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"

RTH_START = time(9,30); RTH_END  = time(16,0)
OR_START  = time(9,30); OR_END   = time(10,0)
SCALP_START_PDH = time(10,30)   # after ORB window
SCALP_START_OR  = time(10,30)   # OR levels only valid post-ORB
SCALP_END       = time(15,30)
SLIP  = 0.25
N_BARS = 10
ZONE_WIDTH = 5     # ±5pt zone (best balance from S&R test)


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


def simulate_scalp(closes, entry_close, approach_dir, zone_lo, zone_hi):
    """
    Simulate ONE scalp trade from bar after entry.
    Returns (pnl_pts, outcome) where pnl is in raw ES points.

    direction:
      approach_dir='below' → we FADE SHORT (expecting rejection at resistance)
      approach_dir='above' → we FADE LONG  (expecting support holds)
    """
    if approach_dir == 'below':
        # SHORT: sold near zone top, profit if price drops back below zone
        ep = entry_close + SLIP          # our sell entry
        stop_price  = zone_hi + SLIP     # stop: price breaks above zone top
        target_price= zone_lo - SLIP     # target: price falls back below zone bottom
        for close in closes:
            if close >= zone_hi:   # stop hit (close beyond far side)
                return -(stop_price - ep), 'STOP'
            if close <= zone_lo:   # target hit (reversed out)
                return ep - target_price, 'TARGET'
        # Time stop: exit at last bar close
        return ep - (closes[-1] + SLIP), 'TIME'
    else:
        # LONG: bought near zone bottom, profit if price rises back above zone
        ep = entry_close - SLIP
        stop_price  = zone_lo - SLIP
        target_price= zone_hi + SLIP
        for close in closes:
            if close <= zone_lo:
                return -(ep - stop_price), 'STOP'
            if close >= zone_hi:
                return target_price - ep, 'TARGET'
        return (closes[-1] - SLIP) - ep, 'TIME'


def run_scalp(df, dates, by_d, wkly):
    scalps = []

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"]>=RTH_START)&(day["time_et"]<=SCALP_END)
                  ].reset_index(drop=True)
        if len(rth) < 20: continue

        closes = rth["ES_close"].values
        times  = rth["time_et"].values

        # Build levels
        or_b = rth[(rth["time_et"]>=OR_START)&(rth["time_et"]<OR_END)]
        if not len(or_b): continue
        OR_H = or_b["ES_high"].max(); OR_L = or_b["ES_low"].min()

        prev = by_d.get(dates[i-1])
        if prev is None: continue
        prev_rth = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(prev_rth): continue
        PDH = prev_rth["ES_high"].max(); PDL = prev_rth["ES_low"].min()

        PWH, PWL = get_pwhl(d, wkly)

        # Define levels with their earliest valid trade time
        levels = [
            ("PDH",  PDH,       SCALP_START_PDH, 'resist'),
            ("PDL",  PDL,       SCALP_START_PDH, 'support'),
            ("OR_H", OR_H,      SCALP_START_OR,  'resist'),
            ("OR_L", OR_L,      SCALP_START_OR,  'support'),
        ]
        if PWH: levels += [
            ("PWH",  PWH, RTH_START, 'resist'),
            ("PWL",  PWL, RTH_START, 'support'),
        ]

        for level_type, level, earliest, bias in levels:
            if level is None: continue
            zlo = level - ZONE_WIDTH; zhi = level + ZONE_WIDTH
            in_zone = (closes >= zlo) & (closes <= zhi)

            # Find FIRST approach to this zone after earliest time
            fired = False
            for j in range(1, len(closes) - N_BARS - 1):
                if times[j] < earliest: continue
                if times[j] >= SCALP_END: break
                if not in_zone[j] or in_zone[j-1]: continue  # only first entry

                # Approach direction
                if closes[j-1] < zlo:
                    approach_dir = 'below'   # came from below → resistance
                elif closes[j-1] > zhi:
                    approach_dir = 'above'   # came from above → support
                else:
                    continue

                # Match approach to level bias
                correct_dir = (
                    (bias == 'resist' and approach_dir == 'below') or
                    (bias == 'support' and approach_dir == 'above')
                )
                # Also allow opposite (breakout test from wrong side)
                # For maximum data, trade all approaches

                entry_close = closes[j]
                fwd_closes = closes[j+1:j+1+N_BARS]
                if len(fwd_closes) < 3: break

                pnl, outcome = simulate_scalp(
                    fwd_closes, entry_close, approach_dir, zlo, zhi)

                scalps.append(dict(
                    date=d, year=d.year,
                    level_type=level_type,
                    level=round(level,2),
                    approach_dir=approach_dir,
                    correct_dir=correct_dir,
                    entry_close=round(entry_close,2),
                    pnl_pts=round(pnl,2),
                    win=pnl>0,
                    outcome=outcome,
                    time_et=times[j],
                    dow=pd.Timestamp(d).day_name(),
                ))
                fired = True
                break   # one scalp per level per day

    return pd.DataFrame(scalps)


def run_orb_pnl(df, dates, by_d):
    """
    Simplified ORB + PDH-L Confirmed P&L for portfolio combination.
    Returns daily P&L in ES points per leg.
    """
    orb_days = {}
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if not len(or_b): continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(8<=es_R<=60 and 30<=nq_R<=175): continue

        # ORB
        post=day[(day["time_et"]>=OR_END)&(day["time_et"]<time(10,30))]
        es_b=nq_b=None; orb_row=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: orb_row=r; break
        if not(es_b and nq_b and es_b==nq_b): continue

        direction="LONG" if es_b=="L" else "SHORT"
        et=orb_row["time_et"]; sign=1 if direction=="LONG" else -1

        # ES exit
        ep=orb_row["ES_close"]+0.25*sign
        sp=ep-es_R*sign; tp=ep+3*es_R*sign
        after=day[day["time_et"]>et]
        pnl=0.0
        for _,bar in after.iterrows():
            t=bar["time_et"]
            if t>=time(15,30): pnl=sign*(bar["ES_close"]-ep)-0.5; break
            if direction=="LONG":
                if bar["ES_low"]<=sp: pnl=-(ep-sp)-0.5; break
                if bar["ES_high"]>=tp: pnl=tp-ep-0.25; break
            else:
                if bar["ES_high"]>=sp: pnl=-(sp-ep)-0.5; break
                if bar["ES_low"]<=tp: pnl=ep-tp-0.25; break

        orb_days[d] = round(pnl, 2)

    return orb_days


def prow(label, sub, r_col="r", pad=36):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean() if "win" in sub.columns else (sub[r_col]>0).mean()
    ar=sub[r_col].mean()
    dr=sub.groupby("date")[r_col].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=sub["date"].nunique()/8.1
    flag="✅" if (wr>0.60 or sh>2) else ("🟡" if wr>0.52 else "❌")
    print(f"  {flag} {label:<{pad}}  n={sub['date'].nunique():4d} ({n_yr:.0f}/yr)  "
          f"WR={wr*100:5.1f}%  AvgPnL={ar:+.2f}pts  Sh={sh:.2f}")


def main():
    print("Loading …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    wkly  = weekly_hl(by_d, dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    print("Running scalp system …")
    sc = run_scalp(df, dates, by_d, wkly)
    sc["r"] = sc["pnl_pts"]
    print(f"  {len(sc):,} scalp trades  on  {sc['date'].nunique()} signal days\n")

    W=72
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)
    IS_END=date(2021,12,31); OS_START=date(2022,1,1)

    # ── 1. Overall performance ─────────────────────────────────────────────────
    sec("1. SCALP SYSTEM — Overall (zone ±5pts, 10-bar time stop)")
    prow("All scalp trades",   sc)
    prow("  Correct-dir only", sc[sc["correct_dir"]==True])
    prow("  Wrong-dir only",   sc[sc["correct_dir"]==False])
    print()
    prow("  PDH fades",  sc[sc["level_type"]=="PDH"])
    prow("  PDL fades",  sc[sc["level_type"]=="PDL"])
    prow("  OR_H fades", sc[sc["level_type"]=="OR_H"])
    prow("  OR_L fades", sc[sc["level_type"]=="OR_L"])
    prow("  PWH fades",  sc[sc["level_type"]=="PWH"])
    prow("  PWL fades",  sc[sc["level_type"]=="PWL"])

    # ── 2. Exit breakdown ──────────────────────────────────────────────────────
    sec("2. EXIT BREAKDOWN — How do scalp trades resolve?")
    for outcome in ["TARGET","STOP","TIME"]:
        sub=sc[sc["outcome"]==outcome]
        if not len(sub): continue
        wr=(sub["pnl_pts"]>0).mean()*100
        ar=sub["pnl_pts"].mean()
        pct=len(sub)/len(sc)*100
        print(f"  {outcome:<8}  {len(sub):5d} trades ({pct:5.1f}%)  "
              f"WR={wr:5.1f}%  AvgPnL={ar:+.2f}pts")

    # ── 3. Year-by-year ────────────────────────────────────────────────────────
    sec("3. YEAR-BY-YEAR SCALP PERFORMANCE")
    print(f"  {'Year':<6}  {'Trades':>6}  {'WR':>6}  {'AvgPnL':>8}  {'Sharpe':>7}  {'IS/OOS':>6}")
    print(f"  {'─'*48}")
    for yr, sub in sc.groupby("year"):
        wr=(sub["pnl_pts"]>0).mean()*100; ar=sub["pnl_pts"].mean()
        dr=sub.groupby("date")["pnl_pts"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        lbl="← IS" if yr<=2021 else "← OOS"
        flag="✅" if ar>0 else "❌"
        print(f"  {flag} {yr:<6}  {len(sub):>6}  {wr:>5.1f}%  {ar:>+7.2f}pts  "
              f"{sh:>7.2f}  {lbl}")

    # ── 4. Walk-forward ────────────────────────────────────────────────────────
    sec("4. WALK-FORWARD  IS 2018-2021  →  OOS 2022-2026")
    prow("IS  2018-2021", sc[sc["date"]<=IS_END])
    prow("OOS 2022-2026", sc[sc["date"]>=OS_START])

    # ── 5. Dollar impact ───────────────────────────────────────────────────────
    sec("5. DOLLAR IMPACT  —  1 MES per scalp trade ($5/pt)")
    total_pts  = sc["pnl_pts"].sum()
    daily_avg  = sc.groupby("date")["pnl_pts"].sum().mean()
    weekly_avg = daily_avg * 5
    annual_avg = daily_avg * 252
    print(f"  Total points P&L         : {total_pts:+.1f} pts over 8.1 years")
    print(f"  Avg daily P&L (1 MES)    : {daily_avg*5:+.2f} ($5/pt × 1 MES)")
    print(f"  Avg weekly P&L (1 MES)   : {weekly_avg*5:+.2f}")
    print(f"  Avg annual P&L (1 MES)   : {annual_avg*5:+.2f}")
    print(f"  At 5 MES per scalp:       ${annual_avg*5*5:+.0f}/year  "
          f"${weekly_avg*5*5:+.2f}/week")
    print()
    print(f"  vs ORB (verified): ~$2,000/yr at 1% risk per leg on $25k")

    # ── 6. COMBINED PORTFOLIO ──────────────────────────────────────────────────
    sec("6. COMBINED PORTFOLIO  —  ORB + Scalp simultaneously")
    print("  ORB: 5% per leg on $25k (primary system)")
    print("  Scalp: 1% per touch on $25k (overlay)")
    print("  These trade DIFFERENT windows — no conflict\n")

    print("  Running ORB P&L …")
    orb_pnl = run_orb_pnl(df, dates, by_d)

    # Convert to daily R-units for Sharpe comparison
    # ORB: 1R = ES OR range; Scalp: 1R = 5pts zone
    all_dates = sorted(set(list(orb_pnl.keys()) + list(sc["date"].unique())))
    combined = []
    for d in all_dates:
        orb_p   = orb_pnl.get(d, 0.0)   # ES points from ORB
        scalp_p = sc[sc["date"]==d]["pnl_pts"].sum() if d in sc["date"].values else 0.0
        # Normalise: ORB uses 5%/leg risk, scalp uses 1%/leg → weight 5:1
        combined.append(dict(
            date=d,
            orb_pnl=orb_p,
            scalp_pnl=scalp_p,
            # Combined daily P&L in points (unweighted for Sharpe comparison)
            combined=orb_p + scalp_p * 0.2,   # scalp at 1/5 weight
        ))
    cdf = pd.DataFrame(combined)

    orb_dr   = cdf["orb_pnl"]
    scalp_dr = cdf.groupby("date")["scalp_pnl"].first() if False else cdf["scalp_pnl"]
    comb_dr  = cdf["combined"]

    def sharpe(s):
        return s.mean()/s.std(ddof=1)*np.sqrt(252) if s.std()>0 else 0

    print(f"  ORB standalone Sharpe    : {sharpe(orb_dr):.2f}")
    print(f"  Scalp standalone Sharpe  : {sharpe(cdf['scalp_pnl']):.2f}")
    print(f"  Combined Sharpe          : {sharpe(comb_dr):.2f}")
    print()
    print(f"  ORB positive days        : {(orb_dr>0).mean()*100:.1f}%")
    print(f"  Scalp positive days      : {(cdf['scalp_pnl']>0).mean()*100:.1f}%")
    print(f"  Combined positive days   : {(comb_dr>0).mean()*100:.1f}%")

    # Days ORB loses, scalp wins
    orb_loss_scalp_win = ((orb_dr<0) & (cdf["scalp_pnl"]>0)).mean()*100
    orb_win_scalp_win  = ((orb_dr>0) & (cdf["scalp_pnl"]>0)).mean()*100
    print()
    print(f"  Days ORB loses, scalp wins : {orb_loss_scalp_win:.1f}%  ← diversification")
    print(f"  Days both win              : {orb_win_scalp_win:.1f}%")

    # ── 7. Honest verdict ─────────────────────────────────────────────────────
    sec("7. HONEST VERDICT")
    sc_dr = sc.groupby("date")["pnl_pts"].sum()
    sc_sh = sc_dr.mean()/sc_dr.std(ddof=1)*np.sqrt(252) if sc_dr.std()>0 else 0
    sc_wr = (sc["pnl_pts"]>0).mean()*100
    sc_avg = sc["pnl_pts"].mean()
    print(f"  Scalp WR              : {sc_wr:.1f}%")
    print(f"  Scalp avg P&L/trade   : {sc_avg:+.2f} ES pts")
    print(f"  Scalp Sharpe          : {sc_sh:.2f}")
    print(f"  Scalp signal days/yr  : {sc['date'].nunique()/8.1:.0f}")
    print()
    if sc_sh > 2.0:
        print("  ✅ Scalp has strong standalone edge — implement as overlay")
    elif sc_sh > 1.0:
        print("  🟡 Scalp has moderate edge — worth implementing as small overlay")
    elif sc_avg > 0:
        print("  ⚠️  Scalp is marginally positive — size very small if trading")
    else:
        print("  ❌ Scalp does not show positive P&L — do not implement")
    print()


if __name__=="__main__":
    main()
