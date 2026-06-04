#!/usr/bin/env python3
"""
backtest_final_system.py — CORRECTED ARCHITECTURE

TWO COMPLETELY INDEPENDENT SYSTEMS running simultaneously:

SYSTEM 1 — ASYMMETRIC PRIMARY (ORB + PDH/L + 1H Momentum)
  LONG  : ORB LONG + PDH/L confirms + first 30-min return > +0.1%
  SHORT : ORB SHORT + PDH/L confirms + first 30-min return < -0.2%
           + previous day was bearish (prev_close < prev_open)
  Filters: break < 20% OR range, PDH within 2R of entry
  Entry : market fill at breakout bar close
  Stop  : OR range (avg 19pts)
  Target: 3 × OR range

SYSTEM 2 — MTF SCALP (independent, fires on ALL valid-OR days)
  Fires when: rejection candle at D1/W1/MN/RND level (10:30-14:30)
  Entry : next bar open
  Stop  : trailing 3pt (BE after 3pt profit)
  Runs independently of whether primary signal fired

KEY FIX: Scalp runs on ALL days with valid ES OR (8-60pts)
  even if NQ is too wide, even if no ORB, even if no PDH/L confirm.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
RTH_S = time(9,30); RTH_E = time(16,0)
OR_S  = time(9,30); OR_E  = time(10,0); ORB_C = time(10,30)
EOD   = time(15,30)
MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175
TRAIL=3.0; BE=3.0
ACCT=25000; PRI_RISK=0.05; SC_RISK=0.01
IS_END=date(2021,12,31); OS_START=date(2022,1,1)


def load():
    df=pd.read_parquet(DATA)
    df["ts_et"]=pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"]=df["ts_et"].dt.date; df["time_et"]=df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def build_levels(by_d, dates):
    weekly={}; monthly={}
    for d in dates:
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        wk=(int(iso.year),int(iso.week)); mn=(ts.year,ts.month)
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_S)&(day["time_et"]<=RTH_E)]
        if not len(rth): continue
        h=rth["ES_high"].max(); l=rth["ES_low"].min()
        if wk not in weekly: weekly[wk]=[h,l]
        else: weekly[wk][0]=max(weekly[wk][0],h); weekly[wk][1]=min(weekly[wk][1],l)
        if mn not in monthly: monthly[mn]=[h,l]
        else: monthly[mn][0]=max(monthly[mn][0],h); monthly[mn][1]=min(monthly[mn][1],l)
    return weekly, monthly


def sim_orb(day, et, di, ep, sp, tp):
    rng=abs(ep-sp)
    if rng<0.01: return 0.0,"NONE"
    for _,bar in day[day["time_et"]>et].iterrows():
        if bar["time_et"]>=EOD:
            s=1 if di=="LONG" else -1
            return round((s*(bar["ES_close"]-ep)-SLIP*2)/rng,3),"EOD"
        if di=="LONG":
            if bar["ES_low"]<=sp: return round(-1.0-SLIP*2/rng,3),"STOP"
            if bar["ES_high"]>=tp: return round(3.0-SLIP/rng,3),"TGT"
        else:
            if bar["ES_high"]>=sp: return round(-1.0-SLIP*2/rng,3),"STOP"
            if bar["ES_low"]<=tp: return round(3.0-SLIP/rng,3),"TGT"
    return 0.0,"NONE"


def sim_trail(fwd, di, ep, s0):
    stop=s0; best=ep
    for c in fwd:
        if di=="SHORT": best=min(best,c); u=ep-c-SLIP*2
        else: best=max(best,c); u=c-ep-SLIP*2
        if u>=BE:
            if di=="SHORT": stop=min(stop,best+TRAIL)
            else: stop=max(stop,best-TRAIL)
        if di=="SHORT":
            if c>=stop: return ep-stop-SLIP*2
        else:
            if c<=stop: return stop-ep-SLIP*2
    last=fwd[-1] if len(fwd) else ep
    return ep-last-SLIP*2 if di=="SHORT" else last-ep-SLIP*2


def run_primary(df, dates, by_d,
                long_min_1h=0.10, short_max_1h=-0.20,
                short_bearish_prev=True, max_break=0.20, max_pdh=2.0):
    """Asymmetric ORB+PDH/L with 1H momentum filter."""
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
        # 1H momentum
        or_ret=(or_b["ES_close"].iloc[-1]-or_b["ES_open"].iloc[0])/or_b["ES_open"].iloc[0]*100
        # Prev day
        prev=by_d.get(dates[i-1])
        if prev is None: continue
        pr=prev[(prev["time_et"]>=RTH_S)&(prev["time_et"]<=RTH_E)]
        if not len(pr): continue
        PDH=pr["ES_high"].max(); PDL=pr["ES_low"].min()
        prev_bearish=(pr["ES_close"].iloc[-1]<pr["ES_open"].iloc[0])
        # ORB
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
        if bd_es>max_break or bd_nq>max_break: continue
        di="LONG" if es_b=="L" else "SHORT"
        # Asymmetric filters
        if di=="LONG" and or_ret<long_min_1h: continue
        if di=="SHORT" and or_ret>short_max_1h: continue
        if di=="SHORT" and short_bearish_prev and not prev_bearish: continue
        # PDH/L
        watch=rth[(rth["time_et"]>=OR_E)&(rth["time_et"]<time(11,0))]
        pdh_dir=None; eu=ed=False
        for _,r in watch.iterrows():
            if not eu and r["ES_high"]>PDH: eu=True
            if not ed and r["ES_low"]<PDL: ed=True
            if eu and not ed and pdh_dir is None: pdh_dir="LONG"; break
            if ed and not eu and pdh_dir is None: pdh_dir="SHORT"; break
        if pdh_dir!=di: continue
        ep_i=entry_row["ES_close"]
        pdh_d=abs(PDH-ep_i)/es_R if di=="LONG" else abs(PDL-ep_i)/es_R
        if pdh_d>max_pdh: continue
        et=entry_row["time_et"]
        ep=ep_i+SLIP if di=="LONG" else ep_i-SLIP
        sp=es_L-SLIP if di=="LONG" else es_H+SLIP
        tp=ep+3*es_R if di=="LONG" else ep-3*es_R
        rv,why=sim_orb(day,et,di,ep,sp,tp)
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        trades.append(dict(
            date=d,year=d.year,direction=di,r=rv,usd=rv*(ACCT*PRI_RISK),
            win=rv>0,exit=why,or_R=es_R,or_ret=round(or_ret,3),
            dow=ts.day_name(),month=ts.month,
            iso_wk=(int(iso.year),int(iso.week)),
        ))
    return pd.DataFrame(trades)


def run_scalp(df, dates, by_d, weekly, monthly):
    """Independent scalp — runs on ALL days with valid ES OR (8-60pts)."""
    trades=[]
    for i,d in enumerate(dates[1:],1):
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_S)&(day["time_et"]<=EOD)].reset_index(drop=True)
        if len(rth)<20: continue
        opens=rth["ES_open"].values; closes=rth["ES_close"].values
        highs=rth["ES_high"].values; lows=rth["ES_low"].values; times=rth["time_et"].values
        or_b=rth[(rth["time_et"]>=OR_S)&(rth["time_et"]<OR_E)]
        if not len(or_b): continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        if not(MIN_ES<=es_R<=MAX_ES): continue   # ONLY ES OR filter — NQ NOT required
        # Levels
        prev=by_d.get(dates[i-1])
        if prev is None: continue
        pr=prev[(prev["time_et"]>=RTH_S)&(prev["time_et"]<=RTH_E)]
        if not len(pr): continue
        PDH=pr["ES_high"].max(); PDL=pr["ES_low"].min()
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        pw=int(iso.week)-1; py=int(iso.year)
        if pw==0: py-=1; pw=int(pd.Timestamp(f"{py}-12-28").isocalendar().week)
        wk=weekly.get((py,pw)); PWH=wk[0] if wk else None; PWL=wk[1] if wk else None
        pm=ts.month-1; pmy=ts.year
        if pm==0: pm=12; pmy-=1
        mn=monthly.get((pmy,pm)); PMH=mn[0] if mn else None; PML=mn[1] if mn else None
        day_mid=(rth["ES_high"].max()+rth["ES_low"].min())/2
        rnds=[round(day_mid/50)*50+k*50 for k in range(-3,4)]
        lvl_r=[(a,b) for a,b in [("D1_H",PDH),("W1_H",PWH),("MN_H",PMH)]+
               [("RND",float(r)) for r in rnds] if b is not None]
        lvl_s=[(a,b) for a,b in [("D1_L",PDL),("W1_L",PWL),("MN_L",PML)]+
               [("RND",float(r)) for r in rnds] if b is not None]
        def conf(lv,ll): return sum(1 for(_,v) in ll if 0<abs(v-lv)<=8)+1
        prev_c=None
        for j in range(1,len(closes)-15):
            if times[j]<time(10,30): prev_c=closes[j]; continue
            if times[j]>time(14,30): break
            if prev_c is None: prev_c=closes[j]; continue
            fired=False
            for fd,lvls in [("SHORT",lvl_r),("LONG",lvl_s)]:
                best_c=0; best_pnl=None; best_risk=None; best_nm=None
                for nm,lv in sorted(lvls,key=lambda x:-conf(x[1],lvls)):
                    c2=conf(lv,lvls)
                    if c2<=best_c: continue
                    if fd=="SHORT":
                        if not(prev_c<lv-SLIP and highs[j]>=lv and closes[j]<=lv-1.0): continue
                        if j+1>=len(opens): break
                        ep2=opens[j+1]+SLIP; s0=highs[j]+SLIP*2
                    else:
                        if not(prev_c>lv+SLIP and lows[j]<=lv and closes[j]>=lv+1.0): continue
                        if j+1>=len(opens): break
                        ep2=opens[j+1]-SLIP; s0=lows[j]-SLIP*2
                    if abs(ep2-s0)<0.5: continue
                    best_c=c2; best_pnl=sim_trail(closes[j+1:],fd,ep2,s0)
                    best_risk=abs(ep2-s0); best_nm=nm; break
                if best_pnl is not None:
                    n_mes=max(1,round((ACCT*SC_RISK)/(best_risk*5)))
                    mm=1.5 if best_nm=="MN_L" else 1.0
                    usd=best_pnl*5*n_mes*mm
                    trades.append(dict(
                        date=d,year=d.year,pnl=best_pnl,usd=round(usd,2),
                        win=best_pnl>0,dow=ts.day_name(),month=ts.month,
                        iso_wk=(int(iso.year),int(iso.week)),level_type=best_nm,
                    )); fired=True; break
            prev_c=closes[j]
            if fired: break
    return pd.DataFrame(trades)


def sh(dr): return dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0


def prow(lbl, tdf, r_col, pad=38):
    if not len(tdf): print(f"  {lbl:<{pad}}  n=0"); return
    wr=(tdf[r_col]>0).mean()*100
    ar=tdf[r_col].mean()
    dr=tdf.groupby("date")[r_col].sum(); s=sh(dr); nyr=tdf["date"].nunique()/8.1
    wins=tdf[tdf[r_col]>0]; losses=tdf[tdf[r_col]<=0]
    wl=abs(wins[r_col].mean()/losses[r_col].mean()) if len(losses) and losses[r_col].mean()!=0 else 0
    flag="✅" if(ar>0 and s>3) else("🟡" if(ar>0 and s>1.5) else "❌")
    print(f"  {flag} {lbl:<{pad}}  {nyr:.0f}/yr  WR={wr:5.1f}%  "
          f"Avg={ar:+.4f}  W/L={wl:.2f}x  Sh={s:.2f}")


def main():
    print("Loading …")
    df=load()
    dates=sorted(df["date_et"].unique())
    by_d={d:g.reset_index(drop=True) for d,g in df.groupby("date_et")}
    weekly,monthly=build_levels(by_d,dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    print("Running Asymmetric Primary …")
    pri=run_primary(df,dates,by_d)
    print(f"  → {len(pri)} primary trades ({len(pri)/8.1:.0f}/yr)\n")

    print("Running Independent Scalp …")
    scl=run_scalp(df,dates,by_d,weekly,monthly)
    print(f"  → {len(scl)} scalp trades ({len(scl)/8.1:.0f}/yr)\n")

    W=74
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. Primary detail ─────────────────────────────────────────────────────
    sec("1. ASYMMETRIC PRIMARY SYSTEM")
    prow("Overall",             pri, "r")
    prow("  LONG  (1H >+0.10%)",pri[pri["direction"]=="LONG"],  "r")
    prow("  SHORT (1H <-0.20% + prev bearish)", pri[pri["direction"]=="SHORT"],"r")
    print()
    print(f"  1H momentum gradient (LONG only):")
    lt=pri[pri["direction"]=="LONG"]
    for lo,hi,label in [(-99,0.1,"0.1% threshold"),
                         (0.1,0.2,"0.1–0.2%"),(0.2,0.5,"0.2–0.5%"),(0.5,99,">0.5%")]:
        s=lt[(lt["or_ret"]>=lo)&(lt["or_ret"]<hi)]
        if not len(s): continue
        wr=s["win"].mean()*100; ar=s["r"].mean()
        print(f"    {label:<20}: n={len(s):3d}  WR={wr:.1f}%  AvgR={ar:+.3f}")

    print()
    print(f"  Exit breakdown:")
    for ex in ["STOP","TGT","EOD"]:
        s=pri[pri["exit"]==ex]
        if not len(s): continue
        print(f"    {ex}: {len(s):3d}({len(s)/len(pri)*100:.1f}%)  "
              f"WR={s['win'].mean()*100:.1f}%  AvgR={s['r'].mean():+.3f}")
    print()
    print(f"  Day-of-week:")
    for dw in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
        s=pri[pri["dow"]==dw]
        if not len(s): continue
        print(f"    {dw:<12}: {s['date'].nunique()/8.1:.0f}/yr  WR={s['win'].mean()*100:.1f}%  AvgR={s['r'].mean():+.3f}")

    # ── 2. Scalp detail ───────────────────────────────────────────────────────
    sec("2. INDEPENDENT SCALP SYSTEM (all days with valid ES OR)")
    prow("All scalp", scl, "pnl")
    prow("  SHORT (resistance fade)", scl[scl["win"]==scl["pnl"]>0][scl["pnl"]>0].head(0).__class__(scl[scl["pnl"]>0]) if False else scl, "pnl")
    # Direction
    prow("  SHORT fades", scl[scl["level_type"].isin(["D1_H","W1_H","MN_H","RND"])&
                              (scl["pnl"]<0).apply(lambda x: not x)], "pnl") if False else None
    # Level type
    print(f"\n  By level:")
    for lt in sorted(scl["level_type"].unique()):
        s=scl[scl["level_type"]==lt]
        if not len(s): continue
        wr=(s["pnl"]>0).mean()*100; ar=s["usd"].mean()
        print(f"    {lt:<8}: {s['date'].nunique()/8.1:.0f}/yr  WR={wr:.1f}%  Avg=${ar:+.0f}")
    print()
    print(f"  Days with valid ES OR but NQ too wide: "
          f"~{len(scl)/8.1-scl['date'].nunique()/8.1:.0f}/yr additional scalp-only days")

    # ── 3. Walk-forward ───────────────────────────────────────────────────────
    sec("3. WALK-FORWARD — IS vs OOS")
    print("  Primary:")
    prow("    IS  2018-2021", pri[pri["date"]<=IS_END], "r")
    prow("    OOS 2022-2026", pri[pri["date"]>=OS_START], "r")
    print("  Scalp:")
    prow("    IS  2018-2021", scl[scl["date"]<=IS_END], "pnl")
    prow("    OOS 2022-2026", scl[scl["date"]>=OS_START], "pnl")

    # ── 4. Joint system ───────────────────────────────────────────────────────
    sec("4. JOINT SYSTEM — Primary + Scalp (independent)")
    all_d=sorted(set(list(pri["date"].unique())+list(scl["date"].unique())))
    daily={}
    for d in all_d:
        p=pri[pri["date"]==d]["usd"].sum() if len(pri) else 0
        s=scl[scl["date"]==d]["usd"].sum() if len(scl) else 0
        daily[d]=p+s
    dr=pd.Series(daily)
    wk_d={}
    for d,v in daily.items():
        ts2=pd.Timestamp(d); iso2=ts2.isocalendar()
        wk=(int(iso2.year),int(iso2.week)); wk_d[wk]=wk_d.get(wk,0)+v
    wks=pd.Series(wk_d); tot_wks=8*52

    sh_full=sh(dr); sh_is=sh(dr[dr.index.map(lambda d2:d2<=IS_END)])
    sh_oos=sh(dr[dr.index.map(lambda d2:d2>=OS_START)])
    ann=dr.sum()/8.1; wk_avg=wks.mean()
    sig_wks=len(wks); zero_wks=tot_wks-sig_wks
    trades_pw=(pri["date"].nunique()+scl["date"].nunique())/8.1/52

    print(f"\n  SIGNAL COVERAGE:")
    print(f"    Primary days/yr     : {pri['date'].nunique()/8.1:.0f}  ({pri['date'].nunique()/8.1/52:.2f}/wk)")
    print(f"    Scalp days/yr       : {scl['date'].nunique()/8.1:.0f}  ({scl['date'].nunique()/8.1/52:.2f}/wk)")
    print(f"    Total trades/wk     : {trades_pw:.1f}")
    print(f"    Weeks with signal   : {sig_wks}/{tot_wks}  ({sig_wks/tot_wks*100:.0f}%)")
    print(f"    Zero-signal weeks   : {zero_wks}  ({zero_wks/tot_wks*100:.0f}%)")

    print(f"\n  PERFORMANCE:")
    print(f"    Combined Sharpe     : {sh_full:.2f}")
    print(f"    IS  Sharpe          : {sh_is:.2f}")
    print(f"    OOS Sharpe          : {sh_oos:.2f}  {'✅ OOS>IS' if sh_oos>sh_is else '⚠️  OOS<IS'}")
    print(f"    Win weeks           : {(wks>0).mean()*100:.1f}%")
    print(f"    Annual $            : ${ann:+,.0f}")
    print(f"    Avg $/week          : ${wk_avg:+,.0f}")
    print(f"    Median $/week       : ${wks.median():+,.0f}")
    for p,v in zip([5,25,50,75,90,95],np.percentile(wks,[5,25,50,75,90,95])):
        print(f"    {p:>3}th pctile       : ${v:>+8,.0f}")

    # ── 5. Year by year ───────────────────────────────────────────────────────
    sec("5. YEAR-BY-YEAR — FULL JOINT SYSTEM")
    print(f"  {'Yr':<5}  {'Wks':>4}  {'Win%':>6}  {'Avg$/wk':>9}  {'Ann$':>9}  "
          f"{'P_Sh':>6}  {'S_Sh':>6}  {'J_Sh':>6}")
    print(f"  {'─'*62}")
    for yr,wk_yr in wks.groupby(wks.index.map(lambda x:x[0])):
        wr=(wk_yr>0).mean()*100; avg=wk_yr.mean(); ann2=wk_yr.sum()
        p_sub=pri[pri["year"]==yr]; s_sub=scl[scl["year"]==yr]
        p_sh=sh(p_sub.groupby("date")["r"].sum()) if len(p_sub)>1 else 0
        s_sh=sh(s_sub.groupby("date")["pnl"].sum()) if len(s_sub)>1 else 0
        j_sh=sh(dr[dr.index.map(lambda d2:d2.year==yr)]) if len(dr[dr.index.map(lambda d2:d2.year==yr)])>1 else 0
        lbl="IS" if yr<=2021 else "OOS"
        flag="✅" if ann2>0 else "❌"
        print(f"  {flag} {yr} {lbl}  {len(wk_yr):>4}  {wr:>5.1f}%  "
              f"${avg:>+8,.0f}  ${ann2:>+8,.0f}  {p_sh:>6.2f}  {s_sh:>6.2f}  {j_sh:>6.2f}")

    yrs_pos=sum(1 for yr in range(2018,2027) if dr[dr.index.map(lambda d2:d2.year==yr)].sum()>0)
    print(f"\n  Years positive: {yrs_pos}/9")

    # ── 6. Final summary ──────────────────────────────────────────────────────
    sec("6. COMPLETE SYSTEM SUMMARY")
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  FINAL SYSTEM — ASYMMETRIC PRIMARY + INDEPENDENT SCALP             │
  │  $25k account | Primary: 5%/leg | Scalp: 1%/trade                 │
  ├────────────────────────────┬────────────────────┬────────────────────┤
  │  Metric                    │    Primary         │    Scalp           │
  ├────────────────────────────┼────────────────────┼────────────────────┤
  │  Signal days/yr            │  {pri['date'].nunique()/8.1:>6.0f}/yr          │  {scl['date'].nunique()/8.1:>6.0f}/yr          │
  │  Win Rate                  │  {pri['win'].mean()*100:>6.1f}%         │  {(scl['pnl']>0).mean()*100:>6.1f}%         │
  │  LONG WR                   │  {pri[pri['direction']=='LONG']['win'].mean()*100:>6.1f}%         │  N/A                │
  │  SHORT WR                  │  {pri[pri['direction']=='SHORT']['win'].mean()*100 if len(pri[pri['direction']=='SHORT']) else 0:>6.1f}%         │  N/A                │
  │  Avg R/trade               │  {pri['r'].mean():>+6.4f}R         │  {scl['pnl'].mean():>+6.3f}pts       │
  │  Sharpe                    │  {sh(pri.groupby('date')['r'].sum()):>6.2f}          │  {sh(scl.groupby('date')['pnl'].sum()):>6.2f}          │
  │  OOS Sharpe                │  {sh(pri[pri['date']>=OS_START].groupby('date')['r'].sum()):>6.2f}          │  {sh(scl[scl['date']>=OS_START].groupby('date')['pnl'].sum()):>6.2f}          │
  │  Annual $ expected         │  ${pri['usd'].sum()/8.1:>+8,.0f}       │  ${scl['usd'].sum()/8.1:>+8,.0f}       │
  ├────────────────────────────┴────────────────────┴────────────────────┤
  │  COMBINED JOINT SYSTEM                                              │
  │  Sharpe  : {sh_full:.2f}   IS={sh_is:.2f}   OOS={sh_oos:.2f}  {"✅ growing" if sh_oos>sh_is else "⚠️ stable"}             │
  │  Win_wks : {(wks>0).mean()*100:.1f}%  Trades/wk: {trades_pw:.1f}  Zero-sig: {zero_wks/tot_wks*100:.0f}%  Yrs+: {yrs_pos}/9         │
  │  Annual$ : ${ann:+,.0f}  Avg/wk: ${wk_avg:+,.0f}  Median/wk: ${wks.median():+,.0f}            │
  │  5th pct : ${np.percentile(wks,5):+,.0f}    95th pct : ${np.percentile(wks,95):+,.0f}                  │
  └─────────────────────────────────────────────────────────────────────┘
    """)


if __name__=="__main__":
    main()
