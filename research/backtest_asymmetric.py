#!/usr/bin/env python3
"""
backtest_asymmetric.py — Asymmetric LONG/SHORT System

KEY INSIGHT from diagnostics:
  LONG signals: WR=61.5% when 1H momentum aligns
  SHORT signals: WR=36.7% when against trend → losing money

ROOT CAUSE: ES/NQ are structurally bullish (2018-2026 bull market).
  Short breakdowns frequently fail as institutions buy dips.

ASYMMETRIC DESIGN:
  LONG (relaxed conditions):
    - ORB LONG + PDH/L confirms
    - First 30-min return > +0.10% (market had upward momentum)
    - Break distance < 20% OR range
    - PDH within 2R of entry

  SHORT (strict conditions):
    - ORB SHORT + PDH/L confirms
    - First 30-min return < -0.20% (real selling, not just drift)
    - Previous day close < previous day open (bearish prior day)
    - Break distance < 20% OR range
    - PDH within 2R of entry

  SCALP: Unchanged (MTF rejection candle, trail 3pts)

EXPECTED IMPROVEMENT:
  LONG WR: 54% → 61%+
  SHORT WR: 45% → 55%+ (or skip most shorts)
  Combined WR: 54% → 59%+
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25; RTH_S=time(9,30); RTH_E=time(16,0)
OR_S=time(9,30); OR_E=time(10,0); ORB_C=time(10,30)
EOD=time(15,30); MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175
TRAIL=3.0; BE=3.0; ACCT=25000; PRI_RISK=0.05; SC_RISK=0.01


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


def sh(dr): return dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0


def run_full(df, dates, by_d, weekly, monthly,
             # LONG filters
             long_min_1h_ret=0.10,     # first 30min must be up this %
             # SHORT filters
             short_max_1h_ret=-0.20,   # first 30min must be down this %
             short_require_bearish_prev=True,  # prev day must have been down
             # Common filters
             max_break_pct=0.20,
             max_pdh_R=2.0):

    pri_trades=[]; scl_trades=[]

    for i,d in enumerate(dates[1:],1):
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_S)&(day["time_et"]<=EOD)].reset_index(drop=True)
        if len(rth)<20: continue
        opens=rth["ES_open"].values; closes=rth["ES_close"].values
        highs=rth["ES_high"].values; lows=rth["ES_low"].values; times=rth["time_et"].values

        ts=pd.Timestamp(d); iso=ts.isocalendar(); dow=ts.day_name(); month=ts.month
        iso_wk=(int(iso.year),int(iso.week))

        or_b=rth[(rth["time_et"]>=OR_S)&(rth["time_et"]<OR_E)]
        if not len(or_b): continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_or=day[(day["time_et"]>=OR_S)&(day["time_et"]<OR_E)]
        nq_H=nq_or["NQ_high"].max(); nq_L=nq_or["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        # First 30-min return (momentum signal)
        or_open = or_b["ES_open"].iloc[0]
        or_close = or_b["ES_close"].iloc[-1]
        first_30_ret = (or_close - or_open) / or_open * 100

        # Previous day direction
        prev=by_d.get(dates[i-1])
        if prev is None: continue
        pr=prev[(prev["time_et"]>=RTH_S)&(prev["time_et"]<=RTH_E)]
        if not len(pr): continue
        PDH=pr["ES_high"].max(); PDL=pr["ES_low"].min()
        prev_open=pr["ES_open"].iloc[0]; prev_close=pr["ES_close"].iloc[-1]
        prev_bullish=(prev_close > prev_open)   # prev day was up
        prev_bearish=(prev_close < prev_open)   # prev day was down

        # Levels
        pw=int(iso.week)-1; py=int(iso.year)
        if pw==0: py-=1; pw=int(pd.Timestamp(f"{py}-12-28").isocalendar().week)
        wk=weekly.get((py,pw)); PWH=wk[0] if wk else None; PWL=wk[1] if wk else None
        pm=ts.month-1; pmy=ts.year
        if pm==0: pm=12; pmy-=1
        mn=monthly.get((pmy,pm)); PMH=mn[0] if mn else None; PML=mn[1] if mn else None
        day_mid=(rth["ES_high"].max()+rth["ES_low"].min())/2
        rnds=[round(day_mid/50)*50+k*50 for k in range(-3,4)]

        # ORB detection
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

        # ── ASYMMETRIC FILTERS ─────────────────────────────────────────
        if direction=="LONG":
            if first_30_ret < long_min_1h_ret: continue   # need positive momentum
        else:  # SHORT
            if first_30_ret > short_max_1h_ret: continue  # need negative momentum
            if short_require_bearish_prev and prev_bullish: continue  # prev day bearish

        # PDH/L confirmation
        watch=rth[(rth["time_et"]>=OR_E)&(rth["time_et"]<time(11,0))]
        pdh_dir=None; eu=ed=False
        for _,r in watch.iterrows():
            if not eu and r["ES_high"]>PDH: eu=True
            if not ed and r["ES_low"]<PDL: ed=True
            if eu and not ed and pdh_dir is None: pdh_dir="LONG"; break
            if ed and not eu and pdh_dir is None: pdh_dir="SHORT"; break
        if pdh_dir!=direction: continue

        # PDH proximity
        ep_i=entry_row["ES_close"]
        pdh_d=abs(PDH-ep_i)/es_R if direction=="LONG" else abs(PDL-ep_i)/es_R
        if pdh_d>max_pdh_R: continue

        et=entry_row["time_et"]
        ep=ep_i+SLIP if direction=="LONG" else ep_i-SLIP
        sp=es_L-SLIP if direction=="LONG" else es_H+SLIP
        tp=ep+3*es_R if direction=="LONG" else ep-3*es_R
        rv,why=sim_orb(day,et,direction,ep,sp,tp)
        pri_trades.append(dict(
            date=d,year=d.year,direction=direction,
            r=rv,usd=rv*(ACCT*PRI_RISK),win=rv>0,exit=why,
            or_R=es_R,first_30_ret=round(first_30_ret,3),
            dow=dow,month=month,iso_wk=iso_wk,
        ))

        # Scalp (unchanged)
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
                    scl_trades.append(dict(
                        date=d,year=d.year,pnl=best_pnl,usd=round(usd,2),
                        win=best_pnl>0,dow=dow,month=month,iso_wk=iso_wk,
                        level_type=best_nm,
                    )); fired=True; break
            prev_c=closes[j]
            if fired: break

    return pd.DataFrame(pri_trades), pd.DataFrame(scl_trades)


def main():
    print("Loading …")
    df=load()
    dates=sorted(df["date_et"].unique())
    by_d={d:g.reset_index(drop=True) for d,g in df.groupby("date_et")}
    weekly,monthly=build_levels(by_d,dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")
    IS_END=date(2021,12,31); OS_START=date(2022,1,1)

    W=76
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── Run both systems ──────────────────────────────────────────────────────
    print("Running Asymmetric System …")
    pri_a, scl_a = run_full(df,dates,by_d,weekly,monthly)

    print("Running Baseline (original) for comparison …")
    pri_b, scl_b = run_full(df,dates,by_d,weekly,monthly,
                             long_min_1h_ret=-99,   # no filter
                             short_max_1h_ret=99,
                             short_require_bearish_prev=False)

    sec("1. PRIMARY SYSTEM — Asymmetric vs Baseline")
    def prow(label,tdf,r_col="r",pad=36):
        if not len(tdf): print(f"  {label:<{pad}}  n=0"); return
        wr=(tdf[r_col]>0).mean()*100 if r_col in ["usd","pnl"] else tdf["win"].mean()*100
        ar=tdf[r_col].mean(); dr=tdf.groupby("date")[r_col].sum()
        s=sh(dr); nyr=tdf["date"].nunique()/8.1
        wins=tdf[tdf["win"]]; losses=tdf[~tdf["win"]]
        wl=abs(wins[r_col].mean()/losses[r_col].mean()) if len(losses) and losses[r_col].mean()!=0 else 0
        flag="✅" if(ar>0 and s>3) else("🟡" if(ar>0 and s>2) else "❌")
        print(f"  {flag} {label:<{pad}}  n={nyr:.0f}/yr  WR={wr:5.1f}%  "
              f"AvgR={ar:+.4f}  W/L={wl:.2f}x  Sh={s:.2f}")

    print(f"\n  {'─'*72}")
    prow("BASELINE (no asymmetric filter)", pri_b)
    prow("ASYMMETRIC (new system)",          pri_a)
    print(f"\n  Direction breakdown (Asymmetric):")
    prow("  LONG  trades", pri_a[pri_a["direction"]=="LONG"])
    prow("  SHORT trades", pri_a[pri_a["direction"]=="SHORT"])
    print(f"\n  Direction breakdown (Baseline):")
    prow("  LONG  trades", pri_b[pri_b["direction"]=="LONG"])
    prow("  SHORT trades", pri_b[pri_b["direction"]=="SHORT"])

    sec("2. WALK-FORWARD — IS 2018-2021 vs OOS 2022-2026")
    for tdf,name in [(pri_b,"Baseline"),(pri_a,"Asymmetric")]:
        print(f"  {name}:")
        prow(f"    IS  2018-2021", tdf[tdf["date"]<=IS_END])
        prow(f"    OOS 2022-2026", tdf[tdf["date"]>=OS_START])

    sec("3. YEAR-BY-YEAR COMPARISON")
    print(f"  {'Year':<6}  {'Base WR':>8}  {'Base Sh':>8}  │  {'Asym WR':>8}  {'Asym Sh':>8}  {'Δ WR':>6}")
    print(f"  {'─'*58}")
    for yr in sorted(set(list(pri_a["year"].unique())+list(pri_b["year"].unique()))):
        ba=pri_b[pri_b["year"]==yr]; as2=pri_a[pri_a["year"]==yr]
        bwr=ba["win"].mean()*100 if len(ba) else 0
        awr=as2["win"].mean()*100 if len(as2) else 0
        bsh=sh(ba.groupby("date")["r"].sum()) if len(ba)>1 else 0
        ash=sh(as2.groupby("date")["r"].sum()) if len(as2)>1 else 0
        lbl="IS" if yr<=2021 else "OOS"
        flag="✅" if awr>bwr+1 else("🟡" if awr>bwr else "❌")
        print(f"  {flag} {yr} {lbl}  {bwr:>7.1f}%  {bsh:>8.2f}  │  "
              f"{awr:>7.1f}%  {ash:>8.2f}  {awr-bwr:>+5.1f}pp")

    sec("4. SIGNAL FREQUENCY — What changed?")
    print(f"  Baseline  : {len(pri_b)/8.1:.0f}/yr total  "
          f"({len(pri_b[pri_b['direction']=='LONG'])/8.1:.0f} LONG + "
          f"{len(pri_b[pri_b['direction']=='SHORT'])/8.1:.0f} SHORT)")
    print(f"  Asymmetric: {len(pri_a)/8.1:.0f}/yr total  "
          f"({len(pri_a[pri_a['direction']=='LONG'])/8.1:.0f} LONG + "
          f"{len(pri_a[pri_a['direction']=='SHORT'])/8.1:.0f} SHORT)")
    print(f"  Lost {(len(pri_b)-len(pri_a))/8.1:.0f}/yr signals from stricter filters")
    print(f"  Gained {(pri_a['win'].mean()-pri_b['win'].mean())*100:+.1f}pp WR")

    sec("5. JOINT SYSTEM — Asymmetric Primary + MTF Scalp")
    all_dates=sorted(set(list(pri_a["date"].unique())+list(scl_a["date"].unique())))
    daily_usd={}
    for d in all_dates:
        p=pri_a[pri_a["date"]==d]["usd"].sum() if len(pri_a) else 0
        s=scl_a[scl_a["date"]==d]["usd"].sum() if len(scl_a) else 0
        daily_usd[d]=p+s
    dr=pd.Series(daily_usd)
    wk_pnl={}
    for d,v in daily_usd.items():
        ts2=pd.Timestamp(d); iso2=ts2.isocalendar()
        wk=(int(iso2.year),int(iso2.week)); wk_pnl[wk]=wk_pnl.get(wk,0)+v
    wks=pd.Series(wk_pnl); tot_wks=8*52

    print(f"\n  Combined annual $ : ${dr.sum()/8.1:+,.0f}")
    print(f"  Combined Sharpe   : {sh(dr):.2f}")
    print(f"  IS  Sharpe        : {sh(dr[dr.index.map(lambda d2:d2<=IS_END)]):.2f}")
    print(f"  OOS Sharpe        : {sh(dr[dr.index.map(lambda d2:d2>=OS_START)]):.2f}")
    print(f"  Win weeks         : {(wks>0).mean()*100:.1f}%")
    print(f"  Zero-sig weeks    : {(tot_wks-len(wks))/tot_wks*100:.0f}%")
    print(f"  Trades/week       : {(len(pri_a)+len(scl_a))/8.1/52:.1f}")
    print(f"  Avg $/week        : ${wks.mean():+,.0f}")
    print(f"  Median $/week     : ${wks.median():+,.0f}")
    print(f"\n  Year-by-year joint:")
    for yr,wk_yr in wks.groupby(wks.index.map(lambda x:x[0])):
        wr=(wk_yr>0).mean()*100; ann=wk_yr.sum()
        flag="✅" if ann>0 else "❌"
        print(f"  {flag} {yr}: Win_wks={wr:.0f}%  Ann=${ann:+,.0f}")

    sec("6. FULL DETAILED BREAKDOWN — Asymmetric Primary")
    print(f"\n  Exit breakdown:")
    for ex in ["STOP","TGT","EOD"]:
        s=pri_a[pri_a["exit"]==ex]
        if not len(s): continue
        print(f"    {ex}: {len(s):3d}({len(s)/len(pri_a)*100:.1f}%)  "
              f"WR={s['win'].mean()*100:.1f}%  AvgR={s['r'].mean():+.3f}")

    print(f"\n  By day of week:")
    for dw in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
        s=pri_a[pri_a["dow"]==dw]
        if not len(s): continue
        wr=s["win"].mean()*100; ar=s["r"].mean()
        print(f"    {dw:<12}: n={s['date'].nunique():2d}/yr  WR={wr:.1f}%  AvgR={ar:+.3f}")

    print(f"\n  By month:")
    months=["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in range(1,13):
        s=pri_a[pri_a["month"]==m]
        if not len(s): continue
        wr=s["win"].mean()*100; ar=s["r"].mean()
        flag="🔥" if wr>65 else("✅" if ar>0 else "❌")
        print(f"    {flag} {months[m]}: WR={wr:.1f}%  AvgR={ar:+.3f}  n={len(s)/8.1:.0f}/yr")

    print(f"\n  First 30-min return distribution (LONG trades):")
    long_t=pri_a[pri_a["direction"]=="LONG"]
    for lo,hi in [(-99,0),(0,0.2),(0.2,0.5),(0.5,99)]:
        s=long_t[(long_t["first_30_ret"]>=lo)&(long_t["first_30_ret"]<hi)]
        if not len(s): continue
        wr=s["win"].mean()*100; ar=s["r"].mean()
        label=f"1H ret {lo:.1f} to {hi:.1f}%" if hi<99 else f"1H ret >{lo:.1f}%"
        print(f"    {label:<22}: n={len(s):3d}  WR={wr:.1f}%  AvgR={ar:+.3f}")

    sec("7. FINAL HONEST COMPARISON")
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  SYSTEM COMPARISON (Primary only, $25k, 5% risk/leg)               │
  ├─────────────────────────────────────────────────────────────────────┤
  │  Metric              │  Baseline       │  Asymmetric    │  Change   │
  ├─────────────────────────────────────────────────────────────────────┤
  │  Signal days/yr      │  {len(pri_b)/8.1:>5.0f}/yr        │  {len(pri_a)/8.1:>5.0f}/yr       │           │
  │  Win Rate            │  {pri_b['win'].mean()*100:>5.1f}%        │  {pri_a['win'].mean()*100:>5.1f}%       │  {(pri_a['win'].mean()-pri_b['win'].mean())*100:>+4.1f}pp    │
  │  Avg R/trade         │  {pri_b['r'].mean():>+6.3f}R       │  {pri_a['r'].mean():>+6.3f}R      │           │
  │  Sharpe              │  {sh(pri_b.groupby('date')['r'].sum()):>5.2f}          │  {sh(pri_a.groupby('date')['r'].sum()):>5.2f}         │           │
  │  IS  Sharpe          │  {sh(pri_b[pri_b['date']<=IS_END].groupby('date')['r'].sum()):>5.2f}          │  {sh(pri_a[pri_a['date']<=IS_END].groupby('date')['r'].sum()):>5.2f}         │           │
  │  OOS Sharpe          │  {sh(pri_b[pri_b['date']>=OS_START].groupby('date')['r'].sum()):>5.2f}          │  {sh(pri_a[pri_a['date']>=OS_START].groupby('date')['r'].sum()):>5.2f}         │           │
  │  LONG WR             │  {pri_b[pri_b['direction']=='LONG']['win'].mean()*100:>5.1f}%        │  {pri_a[pri_a['direction']=='LONG']['win'].mean()*100:>5.1f}%       │  {(pri_a[pri_a['direction']=='LONG']['win'].mean()-pri_b[pri_b['direction']=='LONG']['win'].mean())*100:>+4.1f}pp    │
  │  SHORT WR            │  {pri_b[pri_b['direction']=='SHORT']['win'].mean()*100:>5.1f}%        │  {pri_a[pri_a['direction']=='SHORT']['win'].mean()*100:>5.1f}%       │  {(pri_a[pri_a['direction']=='SHORT']['win'].mean()-pri_b[pri_b['direction']=='SHORT']['win'].mean())*100:>+4.1f}pp    │
  └─────────────────────────────────────────────────────────────────────┘
    """)


if __name__=="__main__":
    main()
