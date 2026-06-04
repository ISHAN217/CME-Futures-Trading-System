#!/usr/bin/env python3
"""
backtest_mtf_sr.py — Multi-Timeframe S&R Confluence Scalp

Core idea: single-timeframe rejection candles have 47% stop rate.
When MULTIPLE timeframes agree on the same level (within N points),
institutional orders from multiple participant types cluster there.
Rejection from a multi-timeframe zone should be more reliable.

Timeframes tested:
  D1  — Prior Day High/Low    (day traders)
  W1  — Prior Week High/Low   (swing traders)
  MN  — Prior Month High/Low  (position traders / monthly options)
  RND — Round numbers         (algos + psychological)

Confluence definition:
  A level has "2-TF confluence" if another TF level is within DIST pts
  A level has "3-TF confluence" if TWO other TF levels are within DIST pts

Entry: rejection candle at the most confluent level
Stop:  tight (above wick), with SINGLE TRAIL 3pts (best exit found)
"""

import numpy as np
import pandas as pd
from datetime import time, date
from collections import defaultdict

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25; REJECTION_THRESH = 1.0
RTH_START=time(9,30); RTH_END=time(16,0)
ENTRY_START=time(10,30); ENTRY_END=time(14,30)
EOD=time(15,30); MIN_RANGE=10.0
TRAIL_DIST=3.0; BE_THRESH=3.0


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def compute_weekly_hl(by_d, dates):
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


def compute_monthly_hl(by_d, dates):
    """Prior calendar month RTH H/L."""
    m = {}
    for d in dates:
        ts=pd.Timestamp(d)
        key=(ts.year, ts.month)
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)]
        if not len(rth): continue
        h=rth["ES_high"].max(); l=rth["ES_low"].min()
        if key not in m: m[key]=[h,l]
        else: m[key][0]=max(m[key][0],h); m[key][1]=min(m[key][1],l)
    return m


def get_pwhl(d, w):
    ts=pd.Timestamp(d); iso=ts.isocalendar()
    pw=int(iso.week)-1; yr=int(iso.year)
    if pw==0: yr-=1; pw=int(pd.Timestamp(f"{yr}-12-28").isocalendar().week)
    key=(yr,pw)
    return (w[key][0],w[key][1]) if key in w else (None,None)


def get_pmhl(d, m):
    ts=pd.Timestamp(d)
    pm=ts.month-1; yr=ts.year
    if pm==0: pm=12; yr-=1
    key=(yr,pm)
    return (m[key][0],m[key][1]) if key in m else (None,None)


def count_confluence(level, all_levels, dist=5.0):
    """Count how many OTHER levels are within dist pts of this level."""
    return sum(1 for (_, lv) in all_levels if abs(lv - level) <= dist)


def sim_trail(fwd_closes, direction, ep, s0):
    stop=s0; best=ep; max_pnl=0.0
    for c in fwd_closes:
        if direction=="SHORT": best=min(best,c); u=ep-c-SLIP*2
        else:                   best=max(best,c); u=c-ep-SLIP*2
        max_pnl=max(max_pnl,u)
        if u>=BE_THRESH:
            if direction=="SHORT": stop=min(stop,best+TRAIL_DIST)
            else:                   stop=max(stop,best-TRAIL_DIST)
        if direction=="SHORT":
            if c>=stop: return ep-stop-SLIP*2,"STOP"
            if c<=(ep-50): return ep-(ep-50)-SLIP*2,"FAR_TGT"  # safety
        else:
            if c<=stop: return stop-ep-SLIP*2,"STOP"
            if c>=(ep+50): return (ep+50)-ep-SLIP*2,"FAR_TGT"
    last=fwd_closes[-1] if len(fwd_closes) else ep
    return (ep-last-SLIP*2 if direction=="SHORT" else last-ep-SLIP*2),"EOD"


def run_mtf(df, dates, by_d, wkly, monthly,
            conf_dist=5.0, min_tf=1):
    """
    Run scalp with multi-timeframe confluence filter.
    min_tf: minimum number of timeframes that must agree
      1 = any single level (baseline)
      2 = at least 2 TFs within conf_dist
      3 = at least 3 TFs within conf_dist
    """
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

        PWH, PWL = get_pwhl(d, wkly)
        PMH, PML = get_pmhl(d, monthly)

        # Round numbers near current day's range
        day_mid = (rth["ES_high"].max() + rth["ES_low"].min()) / 2
        round_lvls = [round(day_mid/50)*50 + k*50 for k in range(-3,4)]

        # Build ALL levels available today with their TF label
        all_resist = [("D1_H", PDH)]
        all_support = [("D1_L", PDL)]
        if PWH: all_resist.append(("W1_H", PWH))
        if PWL: all_support.append(("W1_L", PWL))
        if PMH: all_resist.append(("MN_H", PMH))
        if PML: all_support.append(("MN_L", PML))
        for rl in round_lvls:
            all_resist.append(("RND", float(rl)))
            all_support.append(("RND", float(rl)))

        # For each level, compute TF confluence score
        # Then only trade levels that meet min_tf requirement

        def get_conf_score(level, level_list):
            """Count how many other levels are within conf_dist."""
            return sum(1 for (nm, lv) in level_list
                       if abs(lv - level) <= conf_dist and abs(lv - level) > 0.01)

        # Scan for rejection candles at high-confluence levels
        def scan(fade_dir, lvl_type, level, lvl_list):
            for j in range(1, len(closes)-15):
                if times[j] < ENTRY_START: continue
                if times[j] > ENTRY_END:   break
                if fade_dir == "SHORT":
                    if not(closes[j-1] < level - SLIP and
                           highs[j] >= level and
                           closes[j] <= level - REJECTION_THRESH): continue
                    if j+1 >= len(closes)-5: break
                    ep = opens[j+1] + SLIP; s0 = highs[j] + SLIP*2
                else:
                    if not(closes[j-1] > level + SLIP and
                           lows[j] <= level and
                           closes[j] >= level + REJECTION_THRESH): continue
                    if j+1 >= len(closes)-5: break
                    ep = opens[j+1] - SLIP; s0 = lows[j] - SLIP*2
                if abs(ep-s0) < 0.5: continue

                # Check confluence
                conf = get_conf_score(level, lvl_list)
                if conf + 1 < min_tf: continue  # conf+1 counts this level itself

                pnl, outcome = sim_trail(closes[j+1:], fade_dir, ep, s0)
                return dict(
                    date=d, year=d.year, direction=fade_dir,
                    level_type=lvl_type, level=round(level,2),
                    confluence=conf+1,  # this level + agreeing levels
                    init_risk=round(abs(ep-s0),2),
                    pnl=round(pnl,2), win=pnl>0, outcome=outcome,
                    dow=pd.Timestamp(d).day_name(),
                )
            return None

        # Trade one SHORT and one LONG per day (at most)
        # Pick the level with HIGHEST confluence
        best_short = None; best_short_conf = 0
        best_long  = None; best_long_conf  = 0

        for nm, lv in all_resist:
            conf = get_conf_score(lv, all_resist) + 1
            if conf >= min_tf and conf > best_short_conf:
                rec = scan("SHORT", nm, lv, all_resist)
                if rec:
                    best_short = rec; best_short_conf = conf

        for nm, lv in all_support:
            conf = get_conf_score(lv, all_support) + 1
            if conf >= min_tf and conf > best_long_conf:
                rec = scan("LONG", nm, lv, all_support)
                if rec:
                    best_long = rec; best_long_conf = conf

        if best_short: trades.append(best_short)
        if best_long:  trades.append(best_long)

    return pd.DataFrame(trades)


def prow(label, sub, pad=46):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean()*100; ar=sub["pnl"].mean()
    dr=sub.groupby("date")["pnl"].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=sub["date"].nunique()/8.1
    wins=sub[sub["win"]]["pnl"]; losses=sub[~sub["win"]]["pnl"]
    wl=abs(wins.mean()/losses.mean()) if len(losses) and losses.mean()!=0 else 0
    flag="✅" if(ar>0 and sh>1.5) else("🟡" if ar>0 else "❌")
    print(f"  {flag} {label:<{pad}}  n={sub['date'].nunique():4d}({n_yr:.0f}/yr)  "
          f"WR={wr:5.1f}%  AvgPnL={ar:+.3f}pts  W/L={wl:.2f}x  Sh={sh:.2f}")


def main():
    print("Loading …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print("Computing weekly H/L …")
    wkly = compute_weekly_hl(by_d, dates)
    print("Computing monthly H/L …")
    monthly = compute_monthly_hl(by_d, dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    IS_END=date(2021,12,31); OS_START=date(2022,1,1)
    W=74
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. Baseline (single TF) vs multi-TF ──────────────────────────────────
    sec("1. SINGLE vs MULTI-TIMEFRAME CONFLUENCE")
    print("  Entry: rejection candle at most confluent level")
    print("  Exit: single trail 3pts (best proven exit)\n")

    all_results = {}
    for min_tf in [1, 2, 3]:
        for dist in [3, 5, 8]:
            key = f"min{min_tf}_dist{dist}"
            tdf = run_mtf(df, dates, by_d, wkly, monthly,
                          conf_dist=dist, min_tf=min_tf)
            all_results[key] = tdf

    print(f"  {'Config':<30}  {'n/yr':>5}  {'WR':>6}  {'AvgPnL':>8}  "
          f"{'W/L':>5}  {'Sharpe':>7}")
    print(f"  {'─'*62}")
    print(f"  Single TF (baseline)              66/yr  43.1%  +0.540pts  1.73x   1.53")
    for min_tf in [1, 2, 3]:
        for dist in [3, 5, 8]:
            key=f"min{min_tf}_dist{dist}"
            tdf=all_results[key]
            if not len(tdf): continue
            wr=tdf["win"].mean()*100; ar=tdf["pnl"].mean()
            dr=tdf.groupby("date")["pnl"].sum()
            sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
            n_yr=tdf["date"].nunique()/8.1
            wins=tdf[tdf["win"]]["pnl"]; losses=tdf[~tdf["win"]]["pnl"]
            wl=abs(wins.mean()/losses.mean()) if len(losses) and losses.mean()!=0 else 0
            flag="✅" if(ar>0 and sh>1.5) else("🟡" if ar>0 else "❌")
            print(f"  {flag} ≥{min_tf} TF within ±{dist}pts            "
                  f"{n_yr:>5.0f}/yr  {wr:>5.1f}%  {ar:>+7.3f}pts  {wl:.2f}x  {sh:>7.2f}")
        print()

    # ── 2. Deep dive on best configuration ────────────────────────────────────
    best_key = max(all_results,
                   key=lambda k: (
                       all_results[k].groupby("date")["pnl"].sum().mean() /
                       all_results[k].groupby("date")["pnl"].sum().std(ddof=1)
                       if len(all_results[k])>1 and
                       all_results[k].groupby("date")["pnl"].sum().std()>0 else -999))
    best = all_results[best_key]
    min_tf_best = int(best_key[3]); dist_best = int(best_key.split("dist")[1])

    sec(f"2. BEST CONFIG: ≥{min_tf_best} TF within ±{dist_best}pts")
    if len(best):
        # Confluence distribution
        print("  Signal distribution by confluence level:")
        for c in sorted(best["confluence"].unique()):
            sub=best[best["confluence"]==c]
            wr=sub["win"].mean()*100; ar=sub["pnl"].mean()
            dr=sub.groupby("date")["pnl"].sum()
            sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
            flag="✅" if(ar>0 and sh>1.5) else("🟡" if ar>0 else "❌")
            print(f"  {flag}  {c}-TF confluence  n={sub['date'].nunique():4d}  "
                  f"WR={wr:.1f}%  AvgPnL={ar:+.3f}pts  Sh={sh:.2f}")

        print()
        # Level type breakdown
        print("  By level type:")
        for lt in sorted(best["level_type"].unique()):
            sub=best[best["level_type"]==lt]
            if not len(sub): continue
            prow(f"    {lt}", sub, pad=20)

        # Year-by-year
        print()
        print("  Year-by-year:")
        print(f"  {'Year':<6}  {'n':>4}  {'WR':>6}  {'AvgPnL':>9}  {'Sh':>7}  IS/OOS")
        print(f"  {'─'*48}")
        for yr,sub in best.groupby("year"):
            wr=sub["win"].mean()*100; ar=sub["pnl"].mean()
            dr=sub.groupby("date")["pnl"].sum()
            sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
            lbl="← IS" if yr<=2021 else "← OOS"
            flag="✅" if ar>0 else "❌"
            print(f"  {flag} {yr}  {len(sub):>4}  {wr:>5.1f}%  {ar:>+8.3f}pts  "
                  f"{sh:>7.2f}  {lbl}")

        print()
        prow(f"IS  2018-2021", best[best["date"]<=IS_END])
        prow(f"OOS 2022-2026", best[best["date"]>=OS_START])

        print()
        prow("SHORT (resistance)", best[best["direction"]=="SHORT"])
        prow("LONG  (support)",    best[best["direction"]=="LONG"])

        # Exit breakdown
        print()
        for out in ["STOP","EOD","FAR_TGT"]:
            s=best[best["outcome"]==out]
            if not len(s): continue
            print(f"  {out:<10}  {len(s):4d}({len(s)/len(best)*100:.1f}%)  "
                  f"WR={(s['pnl']>0).mean()*100:.1f}%  AvgPnL={s['pnl'].mean():+.3f}pts")

        # Dollar summary
        ann = best["pnl"].sum() / 8.1
        print()
        print("  Dollar P&L:")
        for lbl,pv,n in [("1 MES",5,1),("3 MES",5,3),("5 MES",5,5)]:
            print(f"    {lbl}: avg/trade=${best['pnl'].mean()*pv*n:+.2f}  "
                  f"annual=${ann*pv*n:+.0f}")

    # ── 3. Monthly H/L specific test ──────────────────────────────────────────
    sec("3. DOES MONTHLY H/L ADD VALUE?  (isolation test)")
    print("  Monthly H/L: watched by position traders, fund managers, monthly options\n")
    monthly_only = run_mtf(df,dates,by_d,wkly,monthly,conf_dist=5,min_tf=1)
    mn_trades = monthly_only[monthly_only["level_type"].isin(["MN_H","MN_L"])]
    d1_trades = monthly_only[monthly_only["level_type"]=="D1_H"]
    d1_l_trades = monthly_only[monthly_only["level_type"]=="D1_L"]
    prow("Monthly H/L (MN_H/MN_L) alone",      mn_trades)
    prow("D1_H/PDH alone (comparison)",          pd.concat([d1_trades,d1_l_trades]))

    # ── 4. Summary comparison ─────────────────────────────────────────────────
    sec("4. FULL COMPARISON")
    print(f"  {'System':<48}  {'Sharpe':>7}  {'n/yr':>5}  {'WR':>6}")
    print(f"  {'─'*66}")
    print(f"  ORB primary                                          3.08     22  47.8%")
    print(f"  PDH/L Confirmed                                      3.14     36  54.5%")
    print(f"  Single-TF trail (baseline scalp)                     1.53     66  43.1%")
    for key,tdf in sorted(all_results.items(),
                          key=lambda x: -(x[1].groupby("date")["pnl"].sum().mean()/
                                          x[1].groupby("date")["pnl"].sum().std(ddof=1)
                                          if len(x[1])>1 and x[1].groupby("date")["pnl"].sum().std()>0
                                          else -999))[:6]:
        if not len(tdf): continue
        mt=int(key[3]); dt=int(key.split("dist")[1])
        dr=tdf.groupby("date")["pnl"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        wr=tdf["win"].mean()*100; n_yr=tdf["date"].nunique()/8.1
        flag="✅" if sh>1.5 else("🟡" if sh>0.5 else "❌")
        print(f"  {flag} MTF ≥{mt}TF ±{dt}pt                              "
              f"{sh:>7.2f}  {n_yr:>5.0f}  {wr:>5.1f}%")
    print()


if __name__=="__main__":
    main()
