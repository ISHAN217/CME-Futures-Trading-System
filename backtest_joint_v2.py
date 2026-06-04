#!/usr/bin/env python3
"""
backtest_joint_v2.py — Four-System Joint Backtest

SYSTEM 1 — ASYMMETRIC PRIMARY   (ORB + PDH/L + 1H Momentum)     5% risk = $1,250/trade
SYSTEM 2 — MTF SCALP            (rejection candles at S/R)        1% risk ≈  $250/trade
SYSTEM 3 — DOUBLE TEST          (two failed level tests)          2% risk =  $500/trade
SYSTEM 4 — GAP + ORB ALIGNMENT  (gap direction matches ORB)       2% risk =  $500/trade

New in v2: System 4 fires when overnight gap direction MATCHES the ORB
breakout — dual independent confirmation. Validated: WR=53.9%, OOS Sh=2.77,
8/9 years positive.

Shows 3-tier progression: P+S → P+S+DT → P+S+DT+GAP

Includes:
  • Verification that all individual systems reproduce prior numbers
  • Overlap analysis between System 1 and System 4
  • Year-by-year full table
  • Competition-week projection
"""

import numpy as np
import pandas as pd
from datetime import time, date, datetime

DATA     = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP     = 0.25
RTH_S    = time(9,30); RTH_E = time(16,0)
OR_S     = time(9,30); OR_E  = time(10,0); ORB_C = time(10,30)
EOD      = time(15,30)
MIN_ES   = 8;  MAX_ES = 60
MIN_NQ   = 30; MAX_NQ = 175
TRAIL    = 3.0; BE = 3.0
ACCT     = 25_000
PRI_RISK = 0.05; SC_RISK = 0.01; DT_RISK = 0.02; GAP_RISK = 0.02
IS_END   = date(2021,12,31); OS_START = date(2022,1,1)

KNOWN = {
    "primary_wr":60.6, "primary_sh":4.06, "primary_nyr":18,
    "scalp_wr":44.7,   "scalp_sh":2.28,   "scalp_nyr":127,
    "dt_wr":38.7,      "dt_sh":4.23,      "dt_nyr":65,
    "gap_wr":53.9,     "gap_sh":2.61,     "gap_nyr":33,
}


# ── data ─────────────────────────────────────────────────────────────────────
def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
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


# ── simulators ───────────────────────────────────────────────────────────────
def sim_orb(day, et, di, ep, sp, tp):
    rng=abs(ep-sp)
    if rng<0.01: return 0.0,"NONE"
    for _,bar in day[day["time_et"]>et].iterrows():
        if bar["time_et"]>=EOD:
            pnl=(bar["ES_close"]-ep-SLIP*2) if di=="LONG" else (ep-bar["ES_close"]-SLIP*2)
            return round(pnl/rng,3),"EOD"
        if di=="LONG":
            if bar["ES_low"]<=sp:  return round(-1.0-SLIP*2/rng,3),"STOP"
            if bar["ES_high"]>=tp: return round(3.0-SLIP/rng,3),"TGT"
        else:
            if bar["ES_high"]>=sp: return round(-1.0-SLIP*2/rng,3),"STOP"
            if bar["ES_low"]<=tp:  return round(3.0-SLIP/rng,3),"TGT"
    return 0.0,"NONE"

def sim_trail(fwd,di,ep,s0):
    stop=s0; best=ep
    for c in fwd:
        if di=="SHORT": best=min(best,c); u=ep-c-SLIP*2
        else:           best=max(best,c); u=c-ep-SLIP*2
        if u>=BE:
            if di=="SHORT": stop=min(stop,best+TRAIL)
            else:           stop=max(stop,best-TRAIL)
        if di=="SHORT":
            if c>=stop: return ep-stop-SLIP*2
        else:
            if c<=stop: return stop-ep-SLIP*2
    last=fwd[-1] if len(fwd) else ep
    return ep-last-SLIP*2 if di=="SHORT" else last-ep-SLIP*2

def tdm(t1,t2):
    d1=datetime(2000,1,1,t1.hour,t1.minute); d2=datetime(2000,1,1,t2.hour,t2.minute)
    return (d2-d1).total_seconds()/60


# ── system 1 ─────────────────────────────────────────────────────────────────
def run_primary(df,dates,by_d):
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
        or_ret=(or_b["ES_close"].iloc[-1]-or_b["ES_open"].iloc[0])/or_b["ES_open"].iloc[0]*100
        prev=by_d.get(dates[i-1])
        if prev is None: continue
        pr=prev[(prev["time_et"]>=RTH_S)&(prev["time_et"]<=RTH_E)]
        if not len(pr): continue
        PDH=pr["ES_high"].max(); PDL=pr["ES_low"].min()
        prev_bearish=(pr["ES_close"].iloc[-1]<pr["ES_open"].iloc[0])
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
        if bd_es>0.20 or bd_nq>0.20: continue
        di="LONG" if es_b=="L" else "SHORT"
        if di=="LONG"  and or_ret<0.10: continue
        if di=="SHORT" and or_ret>-0.20: continue
        if di=="SHORT" and not prev_bearish: continue
        watch=rth[(rth["time_et"]>=OR_E)&(rth["time_et"]<time(11,0))]
        pdh_dir=None; eu=ed=False
        for _,r in watch.iterrows():
            if not eu and r["ES_high"]>PDH: eu=True
            if not ed and r["ES_low"]<PDL: ed=True
            if eu and not ed and pdh_dir is None: pdh_dir="LONG";  break
            if ed and not eu and pdh_dir is None: pdh_dir="SHORT"; break
        if pdh_dir!=di: continue
        ep_i=entry_row["ES_close"]
        pdh_d=abs(PDH-ep_i)/es_R if di=="LONG" else abs(PDL-ep_i)/es_R
        if pdh_d>2.0: continue
        ep=ep_i+SLIP if di=="LONG" else ep_i-SLIP
        sp=es_L-SLIP if di=="LONG" else es_H+SLIP
        tp=ep+3*es_R if di=="LONG" else ep-3*es_R
        rv,why=sim_orb(day,entry_row["time_et"],di,ep,sp,tp)
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        trades.append(dict(date=d,year=d.year,system="primary",direction=di,
                           r=rv,usd=round(rv*(ACCT*PRI_RISK),2),win=rv>0,exit=why,
                           iso_wk=(int(iso.year),int(iso.week))))
    return pd.DataFrame(trades)


# ── system 2 ─────────────────────────────────────────────────────────────────
def run_scalp(df,dates,by_d,weekly,monthly):
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
        if not(MIN_ES<=es_R<=MAX_ES): continue
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
                best_c=0; best_pnl=best_risk=best_nm=None
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
                    trades.append(dict(date=d,year=d.year,system="scalp",
                                       pnl=best_pnl,usd=round(usd,2),win=best_pnl>0,
                                       level_type=best_nm,
                                       iso_wk=(int(iso.year),int(iso.week))))
                    fired=True; break
            prev_c=closes[j]
            if fired: break
    return pd.DataFrame(trades)


# ── system 3 ─────────────────────────────────────────────────────────────────
def run_double_test(df,dates,by_d,weekly,monthly,
                    min_pullback=3.0,max_window_min=90,
                    level_zone=1.0,reject_min=0.5,target_R=3.0):
    trades=[]
    for i,d in enumerate(dates[1:],1):
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_S)&(day["time_et"]<=EOD)].reset_index(drop=True)
        if len(rth)<20: continue
        or_b=rth[(rth["time_et"]>=OR_S)&(rth["time_et"]<OR_E)]
        if not len(or_b): continue
        es_R=or_b["ES_high"].max()-or_b["ES_low"].min()
        if not(MIN_ES<=es_R<=MAX_ES): continue
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
        res_levels=[(PDH,"D1_H")]
        if PWH: res_levels.append((PWH,"W1_H"))
        if PMH: res_levels.append((PMH,"MN_H"))
        sup_levels=[(PDL,"D1_L")]
        if PWL: sup_levels.append((PWL,"W1_L"))
        if PML: sup_levels.append((PML,"MN_L"))
        scan=rth[(rth["time_et"]>=time(10,30))&(rth["time_et"]<=time(14,0))].reset_index(drop=True)
        def try_dir(is_short):
            lvls=res_levels if is_short else sup_levels
            state="watching"; first_lv=first_close=first_t=best_since=None
            for _,bar in scan.iterrows():
                t=bar["time_et"]
                if state=="watching":
                    for lv,nm in lvls:
                        cond=(bar["ES_high"]>=lv and bar["ES_close"]<=lv-reject_min) if is_short \
                             else(bar["ES_low"]<=lv and bar["ES_close"]>=lv+reject_min)
                        if cond:
                            state="first_done"; first_lv=lv
                            first_close=bar["ES_close"]; first_t=t; best_since=first_close; break
                elif state=="first_done":
                    if tdm(first_t,t)>max_window_min: state="watching"; first_lv=None; continue
                    best_since=min(best_since,bar["ES_close"]) if is_short else max(best_since,bar["ES_close"])
                    if(first_close-best_since if is_short else best_since-first_close)>=min_pullback:
                        state="pulled_back"
                elif state=="pulled_back":
                    if tdm(first_t,t)>max_window_min: state="watching"; continue
                    cond2=(bar["ES_high"]>=first_lv-level_zone and bar["ES_close"]<=first_lv-reject_min) if is_short \
                          else(bar["ES_low"]<=first_lv+level_zone and bar["ES_close"]>=first_lv+reject_min)
                    if cond2:
                        ep=bar["ES_close"]-SLIP if is_short else bar["ES_close"]+SLIP
                        sp=first_lv+1.0+SLIP if is_short else first_lv-1.0-SLIP
                        risk=abs(ep-sp)
                        if 0.5<=risk<=25:
                            tp=ep-target_R*risk if is_short else ep+target_R*risk
                            fwd=rth[rth.index>bar.name].reset_index(drop=True)
                            rv=0.0; why="NONE"
                            for _,fb in fwd.iterrows():
                                if fb["time_et"]>=EOD:
                                    rv=(ep-fb["ES_close"]-SLIP*2)/risk if is_short else(fb["ES_close"]-ep-SLIP*2)/risk
                                    why="EOD"; break
                                if is_short:
                                    if fb["ES_high"]>=sp: rv=-1.0; why="STOP"; break
                                    if fb["ES_low"]<=tp: rv=target_R-SLIP/risk; why="TGT"; break
                                else:
                                    if fb["ES_low"]<=sp: rv=-1.0; why="STOP"; break
                                    if fb["ES_high"]>=tp: rv=target_R-SLIP/risk; why="TGT"; break
                            return dict(date=d,year=d.year,system="double_test",
                                        direction="SHORT" if is_short else "LONG",
                                        r=rv,usd=round(rv*(ACCT*DT_RISK),2),win=rv>0,
                                        exit=why,iso_wk=(int(iso.year),int(iso.week)))
            return None
        result=try_dir(True) or try_dir(False)
        if result: trades.append(result)
    return pd.DataFrame(trades)


# ── system 4 — GAP + ORB ALIGNMENT ───────────────────────────────────────────
def run_gap_orb(df, dates, by_d, min_gap_pct=0.10, max_break=0.20):
    """
    Gap + ORB Alignment.
    Overnight gap direction MUST match the ORB breakout direction.
    Both ES + NQ must confirm. No PDH/L or momentum filter required.
    This is a SEPARATE, INDEPENDENT signal from the primary (different
    days will fire, some overlap expected).
    Risk: 2% = $500/trade, same as Double Test.
    """
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"]>=RTH_S)&(day["time_et"]<=EOD)].reset_index(drop=True)
        if len(rth)<20: continue
        prev = by_d.get(dates[i-1])
        if prev is None: continue
        pr = prev[(prev["time_et"]>=RTH_S)&(prev["time_et"]<=RTH_E)]
        if not len(pr): continue
        prev_close = pr["ES_close"].iloc[-1]
        or_b = rth[(rth["time_et"]>=OR_S)&(rth["time_et"]<OR_E)]
        if not len(or_b): continue
        rth_open = or_b.iloc[0]["ES_open"]
        gap_pct  = (rth_open - prev_close) / prev_close * 100
        if abs(gap_pct) < min_gap_pct: continue
        gap_dir = "L" if gap_pct > 0 else "S"
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_or=day[(day["time_et"]>=OR_S)&(day["time_et"]<OR_E)]
        nq_H=nq_or["NQ_high"].max(); nq_L=nq_or["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue
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
        if es_b!=gap_dir: continue
        di="LONG" if es_b=="L" else "SHORT"
        ep=entry_row["ES_close"]+SLIP if di=="LONG" else entry_row["ES_close"]-SLIP
        sp=es_L-SLIP if di=="LONG" else es_H+SLIP
        tp=ep+3*es_R if di=="LONG" else ep-3*es_R
        risk=abs(ep-sp)
        fwd=rth[rth["time_et"]>entry_row["time_et"]].reset_index(drop=True)
        rv=0.0; why="NONE"
        for _,bar in fwd.iterrows():
            if bar["time_et"]>=EOD:
                pnl=(bar["ES_close"]-ep-SLIP*2) if di=="LONG" else(ep-bar["ES_close"]-SLIP*2)
                rv=pnl/risk; why="EOD"; break
            if di=="LONG":
                if bar["ES_low"]<=sp: rv=-1.0; why="STOP"; break
                if bar["ES_high"]>=tp: rv=3.0-SLIP/risk; why="TGT"; break
            else:
                if bar["ES_high"]>=sp: rv=-1.0; why="STOP"; break
                if bar["ES_low"]<=tp: rv=3.0-SLIP/risk; why="TGT"; break
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        trades.append(dict(date=d,year=d.year,system="gap_orb",direction=di,
                           r=rv,usd=round(rv*(ACCT*GAP_RISK),2),win=rv>0,exit=why,
                           gap_pct=round(gap_pct,3),
                           iso_wk=(int(iso.year),int(iso.week))))
    return pd.DataFrame(trades)


# ── helpers ───────────────────────────────────────────────────────────────────
def sh(dr):
    return dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0.0

def max_dd(equity):
    return (equity - equity.cummax()).min()

def daily_sh(daily_usd):
    ret = daily_usd / ACCT
    return ret.mean()/ret.std(ddof=1)*np.sqrt(252) if ret.std()>0 else 0.0

def build_daily(trade_dfs, usd_cols):
    """Sum daily USD P&L across multiple trade DataFrames."""
    all_d = sorted(set(d for tdf in trade_dfs for d in tdf["date"].unique()))
    series_list = []
    for tdf, col in zip(trade_dfs, usd_cols):
        series_list.append(tdf.groupby("date")[col].sum())
    combined = {}
    for d in all_d:
        combined[d] = sum(s.get(d, 0) for s in series_list)
    return pd.Series(combined)


def print_system_block(label, tdf, r_col="usd"):
    IS_t = tdf[tdf["year"]<=2021]; OOS_t = tdf[tdf["year"]>=2022]
    n = len(tdf); nyr = n/8.1
    wr = (tdf[r_col]>0).mean()*100
    avg = tdf[r_col].mean()
    dr = tdf.groupby("date")[r_col].sum()
    s = daily_sh(dr)
    s_is = daily_sh(IS_t.groupby("date")[r_col].sum()) if len(IS_t) else 0
    s_oo = daily_sh(OOS_t.groupby("date")[r_col].sum()) if len(OOS_t) else 0
    wr_is = (IS_t[r_col]>0).mean()*100 if len(IS_t) else 0
    wr_oo = (OOS_t[r_col]>0).mean()*100 if len(OOS_t) else 0
    wins = tdf[tdf[r_col]>0]; losses = tdf[tdf[r_col]<=0]
    wl = abs(wins[r_col].mean()/losses[r_col].mean()) if len(losses) and losses[r_col].mean()!=0 else 99
    flag = "✅" if (avg>0 and s>2.5) else ("🟡" if(avg>0 and s>1.5) else "❌")
    print(f"  {flag} {label:<38} {nyr:4.0f}/yr  WR={wr:5.1f}%  Avg=${avg:+5.0f}  "
          f"W/L={wl:.2f}x  Sh={s:.2f}")
    print(f"     IS  2018-21: WR={wr_is:5.1f}%  Sh={s_is:.2f}   "
          f"OOS 2022-26: WR={wr_oo:5.1f}%  Sh={s_oo:.2f}")


def print_joint_block(label, daily_usd):
    IS_d  = daily_usd[daily_usd.index <= IS_END]
    OOS_d = daily_usd[daily_usd.index >= OS_START]
    s     = daily_sh(daily_usd)
    s_is  = daily_sh(IS_d)  if len(IS_d)>1  else 0
    s_oo  = daily_sh(OOS_d) if len(OOS_d)>1 else 0
    ann   = daily_usd.sum() / 8.1
    mdd   = max_dd(daily_usd.cumsum())
    wk_d  = {}
    for d,v in daily_usd.items():
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        wk=(int(iso.year),int(iso.week)); wk_d[wk]=wk_d.get(wk,0)+v
    wks   = pd.Series(wk_d)
    win_w = (wks>0).sum(); tot_w = len(wks)
    sig_w_pct = tot_w / (8.1*52) * 100
    flag = "✅" if (s>3.0) else ("🟡" if s>2.0 else "❌")
    print(f"  {flag} {label:<42}  Sh={s:.2f}  IS={s_is:.2f}  OOS={s_oo:.2f}  "
          f"Ann=${ann:,.0f}  MaxDD=${mdd:,.0f}  WinWk={win_w}/{tot_w}({win_w/tot_w*100:.0f}%)")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading …")
    df    = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d,g in df.groupby("date_et")}
    weekly, monthly = build_levels(by_d, dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    print("Running all 4 systems …")
    pri = run_primary(df, dates, by_d)
    scl = run_scalp(df, dates, by_d, weekly, monthly)
    dbt = run_double_test(df, dates, by_d, weekly, monthly)
    gap = run_gap_orb(df, dates, by_d)
    for nm, t in [("Primary",pri),("Scalp",scl),("Double Test",dbt),("Gap+ORB",gap)]:
        print(f"  {nm:<14}: {len(t):4d} trades ({len(t)/8.1:.0f}/yr)")
    print()

    W = 76

    # ── VERIFICATION ─────────────────────────────────────────────────────────
    print("=" * W)
    print("  VERIFICATION (±5% WR, ±10% Sh, ±5% freq)")
    print("=" * W)
    def verify(name, tdf, r_col, k_wr, k_sh, k_nyr):
        wr  = (tdf[r_col]>0).mean()*100
        s   = sh(tdf.groupby("date")[r_col].sum())
        nyr = len(tdf)/8.1
        ok_wr  = abs(wr-k_wr)/k_wr < 0.05
        ok_sh  = abs(s-k_sh)/(abs(k_sh)+0.01) < 0.10
        ok_nyr = abs(nyr-k_nyr)/k_nyr < 0.05
        flag   = "✅" if(ok_wr and ok_sh and ok_nyr) else "⚠️ "
        print(f"  {flag} {name:<14}  WR:{wr:.1f}%(exp{k_wr:.1f})  "
              f"Sh:{s:.2f}(exp{k_sh:.2f})  n/yr:{nyr:.0f}(exp{k_nyr})")
    print()
    verify("Primary",    pri, "usd", KNOWN["primary_wr"], KNOWN["primary_sh"], KNOWN["primary_nyr"])
    verify("Scalp",      scl, "usd", KNOWN["scalp_wr"],   KNOWN["scalp_sh"],   KNOWN["scalp_nyr"])
    verify("DoubleTest", dbt, "usd", KNOWN["dt_wr"],      KNOWN["dt_sh"],      KNOWN["dt_nyr"])
    verify("Gap+ORB",    gap, "r",   KNOWN["gap_wr"],     KNOWN["gap_sh"],     KNOWN["gap_nyr"])

    # ── INDIVIDUAL SYSTEMS ────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  INDIVIDUAL SYSTEM STATS")
    print("=" * W + "\n")
    print_system_block("System 1 — Primary ORB ($1,250/trade)",     pri, "usd")
    print()
    print_system_block("System 2 — MTF Scalp ($250/trade)",         scl, "usd")
    print()
    print_system_block("System 3 — Double Test ($500/trade)",       dbt, "usd")
    print()
    print_system_block("System 4 — Gap+ORB Align ($500/trade)",     gap, "usd")

    # ── OVERLAP ANALYSIS ─────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  OVERLAP ANALYSIS — System 1 (Primary) vs System 4 (Gap+ORB)")
    print("=" * W)
    pri_dates = set(pri["date"].unique())
    gap_dates = set(gap["date"].unique())
    overlap   = pri_dates & gap_dates
    print(f"\n  Primary fires on  : {len(pri_dates)} days  ({len(pri_dates)/8.1:.0f}/yr)")
    print(f"  Gap+ORB fires on  : {len(gap_dates)} days  ({len(gap_dates)/8.1:.0f}/yr)")
    print(f"  Same-day overlap  : {len(overlap)} days  ({len(overlap)/8.1:.0f}/yr)")
    print(f"  Overlap rate      : {len(overlap)/len(gap_dates)*100:.0f}% of Gap+ORB days also have Primary")
    print(f"  Gap-only days     : {len(gap_dates-pri_dates)} days  ({len(gap_dates-pri_dates)/8.1:.0f}/yr)  ← purely additive")
    if len(overlap):
        ov_pri = pri[pri["date"].isin(overlap)]
        ov_gap = gap[gap["date"].isin(overlap)]
        dir_agree = sum(1 for d in overlap
                        if len(pri[pri["date"]==d])>0 and len(gap[gap["date"]==d])>0
                        and pri[pri["date"]==d]["direction"].iloc[0]==gap[gap["date"]==d]["direction"].iloc[0])
        print(f"  Direction match   : {dir_agree}/{len(overlap)} ({dir_agree/len(overlap)*100:.0f}%) overlap days have same direction")
        print(f"  Combined risk when both fire: 5%+2% = 7% = $1,750 (same direction)")
        print(f"  Overlap WR — Primary : {(ov_pri['usd']>0).mean()*100:.1f}%")
        print(f"  Overlap WR — Gap+ORB : {(ov_gap['usd']>0).mean()*100:.1f}%")

    # ── JOINT SYSTEM PROGRESSION ──────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  JOINT SYSTEM PROGRESSION (cumulative systems added)")
    print("=" * W + "\n")

    d_pri = pri.groupby("date")["usd"].sum()
    d_scl = scl.groupby("date")["usd"].sum()
    d_dbt = dbt.groupby("date")["usd"].sum()
    d_gap = gap.groupby("date")["usd"].sum()

    all_d_ps   = sorted(set(list(pri["date"])+list(scl["date"])))
    all_d_psd  = sorted(set(all_d_ps+list(dbt["date"])))
    all_d_psdg = sorted(set(all_d_psd+list(gap["date"])))

    daily_ps   = pd.Series({d: d_pri.get(d,0)+d_scl.get(d,0) for d in all_d_ps})
    daily_psd  = pd.Series({d: d_pri.get(d,0)+d_scl.get(d,0)+d_dbt.get(d,0) for d in all_d_psd})
    daily_psdg = pd.Series({d: d_pri.get(d,0)+d_scl.get(d,0)+d_dbt.get(d,0)+d_gap.get(d,0) for d in all_d_psdg})

    print_joint_block("P + S   (existing system)",           daily_ps)
    print_joint_block("P + S + Double Test",                 daily_psd)
    print_joint_block("P + S + Double Test + Gap+ORB  ★",   daily_psdg)

    # ── YEAR-BY-YEAR TABLE ────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  YEAR-BY-YEAR — FULL 4-SYSTEM JOINT")
    print("=" * W)
    print(f"\n  {'Yr':<5} {'Pri':>5} {'Scl':>5} {'DblT':>5} {'Gap':>5} "
          f"{'Tot/wk':>7} {'P&L':>9} {'Ret%':>6} {'Sh':>6}  tag")
    print("  " + "─" * 64)

    for yr in sorted(set(list(pri["year"])+list(scl["year"])+list(dbt["year"])+list(gap["year"]))):
        py=pri[pri["year"]==yr]; sy=scl[scl["year"]==yr]
        dy=dbt[dbt["year"]==yr]; gy=gap[gap["year"]==yr]
        yr_usd = py["usd"].sum()+sy["usd"].sum()+dy["usd"].sum()+gy["usd"].sum()
        yr_ret = yr_usd/ACCT*100
        all_d_yr = sorted(set(list(py["date"])+list(sy["date"])+list(dy["date"])+list(gy["date"])))
        dr_y = {d: d_pri.get(d,0)+d_scl.get(d,0)+d_dbt.get(d,0)+d_gap.get(d,0) for d in all_d_yr}
        ds_y = pd.Series(dr_y)
        sh_y = daily_sh(ds_y) if len(ds_y)>1 else 0
        tag  = "IS " if yr<=2021 else "OOS"
        flag = "✅" if yr_usd>0 else "❌"
        n_wks = 52 if yr<2026 else 22
        ppw=len(py)/n_wks; spw=len(sy)/n_wks; dpw=len(dy)/n_wks; gpw=len(gy)/n_wks
        tot_pw=ppw+spw+dpw+gpw
        print(f"  {flag} {yr}  {ppw:>4.2f}  {spw:>4.2f}  {dpw:>4.2f}  {gpw:>4.2f}  "
              f"{tot_pw:>5.2f}/wk  ${yr_usd:>7,.0f}  {yr_ret:>5.1f}%  {sh_y:>5.2f}  {tag}")

    # ── FINAL SUMMARY TABLE ───────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  FINAL COMPARISON TABLE")
    print("=" * W)

    def joint_stats(dr):
        ann   = dr.sum()/8.1
        s     = daily_sh(dr)
        IS_d  = dr[dr.index<=IS_END]; OOS_d=dr[dr.index>=OS_START]
        s_oo  = daily_sh(OOS_d) if len(OOS_d)>1 else 0
        mdd   = max_dd(dr.cumsum())
        wk_d  = {}
        for d,v in dr.items():
            ts=pd.Timestamp(d); iso=ts.isocalendar()
            wk=(int(iso.year),int(iso.week)); wk_d[wk]=wk_d.get(wk,0)+v
        wks=pd.Series(wk_d)
        win_w=(wks>0).sum(); tot_w=len(wks)
        n_sig = len(dr[dr!=0])
        return ann, s, s_oo, mdd, win_w, tot_w, n_sig

    configs = [
        ("P + S (baseline)",           daily_ps),
        ("P + S + Double Test",        daily_psd),
        ("P + S + DT + Gap+ORB ★",     daily_psdg),
    ]

    print(f"\n  {'System':<30} {'Ann$':>8}  {'Ann%':>6}  {'Sh':>5}  "
          f"{'OOS_Sh':>7}  {'MaxDD':>8}  {'WinWk':>10}  {'Sig/wk':>7}")
    print("  " + "─" * 76)

    for lbl, dr in configs:
        ann, s, s_oo, mdd, win_w, tot_w, n_sig = joint_stats(dr)
        print(f"  {lbl:<30} ${ann:>7,.0f}  {ann/ACCT*100:>5.1f}%  {s:>4.2f}  "
              f"OOS={s_oo:>4.2f}  ${mdd:>7,.0f}  {win_w}/{tot_w}({win_w/tot_w*100:.0f}%)  "
              f"{n_sig/8.1/52:>5.2f}/wk")

    # contribution
    print(f"\n  CONTRIBUTION BREAKDOWN (4-system joint)")
    print(f"  {'─' * 60}")
    total_pnl = pri["usd"].sum()+scl["usd"].sum()+dbt["usd"].sum()+gap["usd"].sum()
    for nm, tdf, col in [("Primary",pri,"usd"),("Scalp",scl,"usd"),
                          ("Double Test",dbt,"usd"),("Gap+ORB",gap,"usd")]:
        t_usd=tdf[col].sum(); share=t_usd/total_pnl*100 if total_pnl else 0
        avg_t=tdf[col].mean()
        dr_t=tdf.groupby("date")[col].sum()
        sh_t=daily_sh(dr_t)
        print(f"  {nm:<14}  {len(tdf):4d} trades  ${t_usd:>8,.0f}  {share:5.1f}%  "
              f"${avg_t:>6.0f}/trade  Sh={sh_t:.2f}")
    print(f"  {'TOTAL':<14}  {len(pri)+len(scl)+len(dbt)+len(gap):4d} trades  ${total_pnl:>8,.0f}")

    # competition projection
    print(f"\n  COMPETITION WEEK PROJECTION (June 9–13, 2026 — 5 days)")
    print(f"  {'─' * 60}")
    for nm, tdf, col in [("Primary",pri,"usd"),("Scalp",scl,"usd"),
                          ("Double Test",dbt,"usd"),("Gap+ORB",gap,"usd")]:
        y26 = tdf[tdf["year"]==2026]
        pw  = len(y26)/22 if len(y26) else 0
        wr  = (y26[col]>0).mean()*100 if len(y26) else 0
        w   = y26[y26[col]>0][col].mean() if len(y26[y26[col]>0]) else 0
        l   = y26[y26[col]<=0][col].mean() if len(y26[y26[col]<=0]) else 0
        exp = pw*(wr/100*w+(1-wr/100)*l)
        print(f"  {nm:<14}  {pw:.2f}/wk  WR={wr:.0f}%  exp=${exp:+.0f}/wk")
    total_exp = sum(
        (len(tdf[tdf["year"]==2026])/22)*
        ((tdf[tdf["year"]==2026][col]>0).mean()*100/100*
         (tdf[tdf["year"]==2026][tdf[tdf["year"]==2026][col]>0][col].mean() or 0)+
         (1-(tdf[tdf["year"]==2026][col]>0).mean())*
         (tdf[tdf["year"]==2026][tdf[tdf["year"]==2026][col]<=0][col].mean() or 0))
        for nm,tdf,col in [("P",pri,"usd"),("S",scl,"usd"),("DT",dbt,"usd"),("G",gap,"usd")]
        if len(tdf[tdf["year"]==2026])
    )
    print(f"\n  ★ Expected total competition week P&L: ${total_exp:+,.0f}")
    print()


if __name__ == "__main__":
    main()
